def save_raw(path, data, num_nodes, split_idx=None):
    out = open(path, "w")

    if path == "adj_t.txt" or path == "two-hop.txt" or path == "three-hop.txt":
        if split_idx == None:
            for node_idx in range(num_nodes):
                neighbors = list(map(str, data[node_idx].coo()[1].numpy()))
                print(" ".join(neighbors), file=out)
        else:
            for node in split_idx:
                node_idx = int(node.numpy())
                neighbors = list(map(str, data[node_idx].coo()[1].numpy()))
                print(" ".join(neighbors), file=out)
    else:
        for i in range(num_nodes):
            save_data = map(str, data[i])
            print(" ".join(save_data), file=out)

    out.close()

def save_new_graph(data, new_index_table, new_index_sorted, num_nodes):

    new_adj_t = []
    new_embedding = []
    embedding = data.x.numpy()

    for index in new_index_sorted:
        new_embedding.append( embedding[index[0]] )

        neighbors = data.adj_t[index[0]].coo()[1].numpy() # list of neighbors

        new_neighbors = []
        for neighbor in reversed(neighbors):
            new_neighbors.append( new_index_table[neighbor][1] )
        new_adj_t.append(new_neighbors)

    # Store reorganized embedding, adj matrix
    out = open("new_embedding.txt", "w")
    for i in range(num_nodes):
        new_embedding[i] = list(map(str, new_embedding[i]))
        print(" ".join(new_embedding[i]), file=out)
    out.close()

    out = open("new_adj_t.txt", "w")
    for i in range(num_nodes):
        new_adj_t[i] = list(map(str, new_adj_t[i]))
        print(" ".join(new_adj_t[i]), file=out)
    out.close()


