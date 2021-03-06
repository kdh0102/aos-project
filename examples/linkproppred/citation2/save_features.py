import struct

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
    out = open("new_index.txt", "w")
    for i in range(num_nodes):
        new_index = new_index_sorted[i][1]
        print(new_index, file=out)
    out.close()

    # out = open("new_embedding.bin", "wb")
    # for i in range(num_nodes):
    #     for j in range(8):
    #         for embedding in new_embedding[i]:
    #             data = struct.pack("f", embedding)
    #             out.write(data)
    #     #new_embedding[i] = list(map(str, new_embedding[i]))
    #     #print(" ".join(new_embedding[i]), file=out)
    #     out.write(b"\n")
    # out.close()

    # out = open("new_adj_t.txt", "w")
    # for i in range(num_nodes):
    #     new_adj_t[i] = list(map(str, new_adj_t[i]))
    #     print(" ".join(new_adj_t[i]), file=out)
    # out.close()

def save_remapped_index(filename, new_index_sorted):
    three_hop = open("under300.txt", "r")
    three_hop_remapped = open(filename, "w")
    lines = three_hop.readlines()
    for line in lines:
        old_neighbors = line.strip().split(" ")
        new_neighbors = []
        for neighbor in old_neighbors:
            new_neighbors.append(int(new_index_sorted[int(neighbor)][1]))
        new_neighbors.sort()
        print(" ".join(list(map(str, new_neighbors))), file=three_hop_remapped)
    three_hop.close()
    three_hop_remapped.close()