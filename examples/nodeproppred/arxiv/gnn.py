import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import sys
import numpy as np
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch import Tensor
import time
import copy
from cupy.cuda import nvtx
from reorganize import *
from save_features import *
import random

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=False, normalize=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    ## gcn_norm is done in advance
    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    t1 = time.time()
    out = model(data.x, data.adj_t)
    print("Original model latency: %.5fs" % (time.time()-t1))
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test'][:10000]],
        'y_pred': y_pred[split_idx['test'][:10000]],
    })['acc']

    return train_acc, valid_acc, test_acc

def reorganize():
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()

    # Reorganize code
    num_nodes = data.adj_t.size(0)

    sorted_degree_path = "sorted_degree.txt"
    working_set_path = "adj_t.txt"

    # Save sorted/reverse sorted degree
    degree = []
    for i in range(num_nodes):
        degree.append( (i, data.adj_t[i].nnz()) )

    degree.sort(key = lambda degree:degree[1], reverse=True)
    save_raw("sorted_degree.txt", degree, num_nodes)

    threshold = 100
    use_GLIST = True
    use_topk = False

    if not use_GLIST:
        low = 90
        new_index_table, new_index_sorted = greedy_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold, low)
    else:
        topk = 20
        low = 90
        new_index_table, new_index_sorted = GLIST_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold, topk, low, use_topk)

    if functionality_check(new_index_sorted, num_nodes):
        save_new_graph(data, new_index_table, new_index_sorted, num_nodes)

def sweep():
    num_nodes = 169343

    sorted_degree_path = "sorted_degree.txt"
    working_set_path = "adj_t.txt"

    use_GLIST = True
    conditions = [(300, False, -1, 200), (300, False, -1, 100), (300, False, -1, 50)]
    # conditions = [(30, False, -1, 20), (100, False, -1, 60), (300, False, -1, 200)] # GLIST low
    # conditions = [(1000, True, 60, -1), (300, True, 200, -1), (100, True, 600, -1)] # GLIST topk
    # conditions = [(30, False, -1, 20), (100, False, -1, 60), (300, False, -1, 200)] # Greedy
    
    if use_GLIST:
        for condition in conditions:
            threshold, use_topk, topk, low = condition
            if use_topk:
                filename = "trace_%d_top%d.txt" % (threshold, topk)
            else:
                filename = "trace_%d_%d.txt" % (threshold, low)
            new_index_table, new_index_sorted = GLIST_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold, topk, low, use_topk)
            if functionality_check(new_index_sorted, num_nodes):
                save_remapped_index(filename, new_index_sorted)
            print(filename, "saved")

    else:
        for condition in conditions:
            threshold, use_topk, topk, low = condition
            filename = "trace_%d_%d_greedy.txt" % (threshold, low)
            new_index_table, new_index_sorted = greedy_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold, low)

            if functionality_check(new_index_sorted, num_nodes):
                save_remapped_index(filename, new_index_sorted)
                print(filename, "saved")

def save_neighbors():
    print("save neighbors")
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()

    num_nodes = data.adj_t.size(0)
    adj = data.adj_t.set_diag()
    print(adj)

    split_idx = dataset.get_idx_split()
    test_idx = list(split_idx['test'].numpy())
    
    sort = open("sorted_degree.txt", "r")
    sort_lines = sort.readlines()
    threshold = 30
    selected = []
    for line in sort_lines:
        idx, degree = line.strip().split(" ")
        if int(degree) >= threshold and (int(idx) in test_idx):
            selected.append(int(idx))
        if int(degree) < threshold:
            break
    print(len(selected))
    selected = random.sample(selected, k=100)
    print(selected)
    # selected.sort()

    # two_hop = adj[selected].matmul(adj)
    # print(two_hop)
    # three_hop = two_hop.matmul(adj)
    # print(three_hop)

    under = open("over30.txt", "w")
    for idx in selected:
        neighbors = list(map(str, adj[idx].coo()[1].numpy()))
        print(" ".join(neighbors), file=under)
    # save_raw("three-hop.txt", three_hop, num_nodes)

def remap_neighbors():
    mapping = open("new_index.txt", "r")
    lines = mapping.readlines()
    old_to_new = {}
    for old_idx, line in enumerate(lines):
        new_idx = line.strip()
        old_to_new[old_idx] = int(new_idx)
    mapping.close()

    three_hop = open("under300.txt", "r")
    three_hop_remapped = open("under300-remapped.txt", "w")
    lines = three_hop.readlines()
    for line in lines:
        old_neighbors = line.strip().split(" ")
        new_neighbors = []
        for neighbor in old_neighbors:
            new_neighbors.append(int(old_to_new[int(neighbor)]))
        new_neighbors.sort()
        print(" ".join(list(map(str, new_neighbors))), file=three_hop_remapped)
    three_hop.close()
    three_hop_remapped.close()

def inference(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    split_idx = dataset.get_idx_split()

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
        model.load_state_dict(torch.load("model-sage.pt"))
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
        model.load_state_dict(torch.load("model.pt"))
        data.adj_t = gcn_norm(data.adj_t, None, total_nodes, False, True)

    evaluator = Evaluator(name='ogbn-arxiv')
    model.eval()
    
    ## load 3-hop neighbors and run test
    data = data.to(device)
    three_hop = open("over30.txt", "r")
    lines = three_hop.readlines()
    
    total_latency = 0.0
    for idx, (node, line) in enumerate(zip(split_idx['test'], lines)):
        neighbors_list = list(map(int, line.strip().split(" ")))

        count = len(neighbors_list)
        neighbors = torch.tensor(neighbors_list)

        x_sub = torch.zeros(count, data.num_features)
        x_sub[:count] = data.x[neighbors]

        adj_t_sub = data.adj_t[neighbors][:, neighbors]
        nvtx.RangePush("Start One Node")
        x_sub = x_sub.to(device)
        adj_t_sub = adj_t_sub.to(device)
        t1 = time.time()
        out = model(x_sub, adj_t_sub)
        total_latency += (time.time() - t1)
        nvtx.RangePop()
        
    print("1-node inference total latency: %.5fs" % (total_latency))

def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
        model.load_state_dict(torch.load("model-sage.pt"))
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
        model.load_state_dict(torch.load("model.pt"))
        data.adj_t = gcn_norm(data.adj_t, None, total_nodes, False, True)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    # data.x[split_idx['train']] = 0
    # data.x[split_idx['valid']] = 0
    _, _, test_acc = test(model, data, split_idx, evaluator)
    print("Original model acc: %.5f" % test_acc)

    # for run in range(args.runs):
    #     model.reset_parameters()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #     for epoch in range(1, 1 + args.epochs):
    #         loss = train(model, data, train_idx, optimizer)
    #         result = test(model, data, split_idx, evaluator)
    #         # logger.add_result(run, result)

    #         # if epoch % args.log_steps == 0:
    #         train_acc, valid_acc, test_acc = result
    #         print(f'Run: {run + 1:02d}, '
    #                 f'Epoch: {epoch:02d}, '
    #                 f'Loss: {loss:.4f}, '
    #                 f'Train: {100 * train_acc:.2f}%, '
    #                 f'Valid: {100 * valid_acc:.2f}% '
    #                 f'Test: {100 * test_acc:.2f}%')
    #     torch.save(model.state_dict(), "model-sage.pt")
    #     logger.print_statistics(run)
    # logger.print_statistics()


if __name__ == "__main__":
    # sort = open("sorted_degree.txt", "r")
    # sort_lines = sort.readlines()
    # summ = 0
    # for line in sort_lines:
    #     idx, degree = list(map(int, line.strip().split(" ")))
    #     summ += degree
    # print(summ/len(sort_lines))
    # sys.exit()

    global total_nodes
    total_nodes = 169343

    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--mode', type=str, default="inference")
    args = parser.parse_args()
    print(args)

    if args.mode == 'original':
        main(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'reorganize':
        reorganize()
    elif args.mode == 'neighbor':
        save_neighbors()
    elif args.mode == 'remapping':
        remap_neighbors()
    elif args.mode == 'sweep':
        sweep()
    else:
        sys.exit()

