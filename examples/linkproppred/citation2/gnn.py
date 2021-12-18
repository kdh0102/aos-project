import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from logger import Logger

import numpy as np
import tensorflow as tf
import sys, time

from save_features import *
from reorganize import *

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False, cached=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    predictor.eval()

    t1 = time.time()
    h = model(data.x, data.adj_t)
    
    print("Original model latency: %.5fs" % (time.time()-t1))

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)[:num_edge_to_test]
        target = split_edge[split]['target_node'].to(h.device)[:num_edge_to_test]
        target_neg = split_edge[split]['target_node_neg'].to(h.device)[:num_edge_to_test][:, 0]

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)
        
        neg_preds = []

        for perm in DataLoader(range(source.size(0)), batch_size):    
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1)
        
        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    # train_mrr = test_split('eval_train')
    # valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return -1, -1, test_mrr
    SparseTensor(row=[], col=[], sparse_sizes=(1, 2))

@torch.no_grad()
def inference():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    split_edge = dataset.get_edge_split()

    '''source_node_list = split_edge['test']['source_node'].numpy()[:num_edge_to_test]
    source_node = set(source_node_list)
    target_node_list = split_edge['test']['target_node'].numpy()[:num_edge_to_test]
    target_node = set(target_node_list)
    target_neg_list = split_edge['test']['target_node_neg'].numpy()[:num_edge_to_test][:, 0]
    target_neg = set(target_neg_list)
    nodes = source_node.union(target_node)
    nodes = list(nodes.union(target_neg))
    nodes.sort()'''

    num_nodes = data.adj_t.size(0)
    # save working_set (adj)
    print("Start saving adj_t")
    adj = data.adj_t.set_diag()
    save_raw("adj_t.txt", data.adj_t, num_nodes)
    #two_hop = adj.matmul(adj)
    #save_raw("two-hop.txt", two_hop, num_nodes)

    # Reorganize code
    sorted_degree_path = "sorted_degree.txt"
    working_set_path = "adj_t.txt" 

    print("Started Sorting")
    degree = []
    for i in range(num_nodes):
        degree.append( (i, data.adj_t[i].nnz()) )
    degree.sort(key = lambda degree:degree[1], reverse=True)
    save_raw("sorted_degree.txt", degree, num_nodes)

    threshold = 1000
    use_GLIST = True
    use_topk = True

    print("Started Reorganization")
    if not use_GLIST:
        low = 100
        new_index_table, new_index_sorted = greedy_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold, low)
    else:
        topk = 20
        low = threshold - 10
        new_index_table, new_index_sorted = GLIST_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold, topk, low, use_topk)

    print("Finished Reorganization")
    if functionality_check(new_index_sorted, num_nodes):
        print("Saving")
        save_new_graph(data, new_index_table, new_index_sorted, num_nodes)


    # save 3-hop neighbors
    # adj = data.adj_t.set_diag()
    # print(adj)
    # two_hop = adj[nodes].matmul(adj)
    # print(two_hop)
    # three_hop = two_hop.matmul(adj)
    # print(three_hop)

    # out = open("three-hop.txt", "w")
    # for (src, dst, neg) in zip(source_node_list, target_node_list, target_neg_list):
    #     src_i = nodes.index(src)
    #     dst_i = nodes.index(dst)
    #     neg_i = nodes.index(neg)
    #     neighbors = set(map(int, three_hop[src_i].coo()[1].numpy()))
    #     neighbors = neighbors.union(set(map(int, three_hop[dst_i].coo()[1].numpy())))
    #     neighbors = neighbors.union(set(map(int, three_hop[neg_i].coo()[1].numpy())))
    #     neighbors_list = list(neighbors)
    #     neighbors_list.sort()
    #     nlist = list(map(str, neighbors_list))
    #     print(" ".join(nlist), file=out)
    # sys.exit()

    '''predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)
    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
        model.load_state_dict(torch.load("model-sage.pt"))
        predictor.load_state_dict(torch.load("predictor-sage.pt"))
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
        model.load_state_dict(torch.load("model.pt"))
        predictor.load_state_dict(torch.load("predictor.pt"))

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
    
    
    predictor.eval()

    evaluator = Evaluator(name='ogbl-citation2')
    

    ## load 3-hop neighbors and run test
    three_hop = open("three-hop.txt", "r")
    lines = three_hop.readlines()
    pos_preds = []
    neg_preds = []
    
    total_latency = 0.0
    for idx, (src, dst, neg, line) in enumerate(zip(source_node_list, target_node_list, target_neg_list, lines)):
        neighbors_list = list(map(int, line.strip().split(" ")))
        count = len(neighbors_list)
        neighbors = torch.tensor(neighbors_list)
        
        x_sub = torch.zeros(count, data.num_features)
        x_sub[:count] = data.x[neighbors]

        adj_t_sub = data.adj_t[neighbors][:, neighbors]
    
        src_i = neighbors_list.index(src)
        dst_i = neighbors_list.index(dst)
        neg_i = neighbors_list.index(neg)

        x_sub = x_sub.to(device)
        adj_t_sub = adj_t_sub.to(device)
        t1 = time.time()
        out = model(x_sub, adj_t_sub)
        total_latency += (time.time() - t1)
        pos_preds += [predictor(out[src_i], out[dst_i]).cpu()]
        neg_preds += [predictor(out[src_i], out[neg_i]).cpu()]

    pos_pred = torch.cat(pos_preds, dim=0)
    neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1)

    test_acc = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['mrr_list'].mean().item()
    print("1-node inference acc: %.5f" % test_acc)
    print("1-node inference total latency: %.5fs" % (total_latency))
    print("1-node inference per node avg latency: %.5fs" % (total_latency/num_edge_to_test))'''


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    # torch.manual_seed(12345)
    # idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    # split_edge['eval_train'] = {
    #     'source_node': split_edge['train']['source_node'][idx],
    #     'target_node': split_edge['train']['target_node'][idx],
    #     'target_node_neg': split_edge['valid']['target_node_neg'],
    # }

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)
    

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
        model.load_state_dict(torch.load("model-sage.pt"))
        predictor.load_state_dict(torch.load("predictor-sage.pt"))
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)
        model.load_state_dict(torch.load("model.pt"))
        predictor.load_state_dict(torch.load("predictor.pt"))

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    #np.savetxt('embedding.txt', data.x.numpy())
    #np.savetxt('adjacent_row.txt', data.adj_t.coo()[0].numpy(), fmt='%i')
    #np.savetxt('adjacent_col.txt', data.adj_t.coo()[1].numpy(), fmt='%i')
    #np.savetxt('train_idx_source.txt', split_edge['train']['source_node'].numpy(), fmt='%i')
    #np.savetxt('train_idx_source.txt', split_edge['train']['target_node'].numpy(), fmt='%i')
    #np.savetxt('valid_idx_source.txt', split_edge['valid']['source_node'].numpy(), fmt='%i')
    #np.savetxt('valid_idx_target.txt', split_edge['valid']['target_node'].numpy(), fmt='%i')
    #np.savetxt('test_idx_source.txt', split_edge['test']['source_node'].numpy(), fmt='%i')
    #np.savetxt('test_idx_traget.txt', split_edge['test']['target_node'].numpy(), fmt='%i')

    
    evaluator = Evaluator(name='ogbl-citation2')
    _, _, test_acc = test(model, predictor, data, split_edge, evaluator,
                              args.batch_size)
    print("Original model acc: %.5f" % test_acc)
    # logger = Logger(args.runs, args)

    # for run in range(args.runs):
    #     model.reset_parameters()
    #     predictor.reset_parameters()
    #     optimizer = torch.optim.Adam(
    #         list(model.parameters()) + list(predictor.parameters()),
    #         lr=args.lr)

    #     for epoch in range(1, 1 + args.epochs):
    #         loss = train(model, predictor, data, split_edge, optimizer,
    #                      args.batch_size)
    #         print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

    #         # if epoch % args.eval_steps == 0:
    #         result = test(model, predictor, data, split_edge, evaluator,
    #                           args.batch_size)
    #             # logger.add_result(run, result)

    #             # if epoch % args.log_steps == 0:
    #         train_mrr, valid_mrr, test_mrr = result
    #         print(f'Run: {run + 1:02d}, '
    #                       f'Epoch: {epoch:02d}, '
    #                       f'Loss: {loss:.4f}, '
    #                       f'Train: {train_mrr:.4f}, '
    #                       f'Valid: {valid_mrr:.4f}, '
    #                       f'Test: {test_mrr:.4f}')
    #     torch.save(model.state_dict(), "model.pt")
    #     torch.save(predictor.state_dict(), "predictor.pt")

    #     print('GraphSAGE' if args.use_sage else 'GCN')
    #     logger.print_statistics(run)
    # print('GraphSAGE' if args.use_sage else 'GCN')
    # logger.print_statistics()


if __name__ == "__main__":
    global num_edge_to_test
    num_edge_to_test = 2
    #main()
    inference()
