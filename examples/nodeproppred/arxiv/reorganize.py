import copy

def GLIST_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold=100, topk = 20, low=90, use_topk=True):
    sorted_degree_file = open(sorted_degree_path, "r")
    working_set_file = open(working_set_path, "r")

    sorted_degree_lines = sorted_degree_file.readlines()
    working_set_lines = working_set_file.readlines()
    
    traversed = [False] * num_nodes
    popular = [False] * num_nodes

    # Naive reorganization algorithm
    count = 0
    new_index = 0
    new_index_table = []

    important_vertice = []
    working_set = []

    k = 0
    for degree_line in sorted_degree_lines:
        degree = list(map(int, degree_line.strip().split(" ")))
        
        if degree[1] > threshold:
            continue

        if use_topk:
            if k < topk:
                neighbors_list = list(map(int, working_set_lines[degree[0]].strip().split(" ")))
                important_vertice.append(degree[0])
                working_set.append(set(neighbors_list))
                k = k+1
                continue
        else:
            if degree[1] > low:
                neighbors_list = list(map(int, working_set_lines[degree[0]].strip().split(" ")))
                important_vertice.append(degree[0])
                working_set.append(set(neighbors_list))    
                continue

        break


    important_vertex = important_vertice[0]
    index_list = { vertex : i for i, vertex in enumerate(important_vertice)}

    while important_vertice != []:
        important_vertice.remove(important_vertex)

        if not traversed[important_vertex]:
            traversed[important_vertex] = True
            
            while popular[new_index]:
                new_index += 1
            popular[new_index] = True
 
            new_index_table.append( (important_vertex, new_index) )
            count += 1
        
        for x in working_set[index_list[important_vertex]]:
            if not traversed[x]:
                traversed[x] = True

                while popular[new_index]:
                    new_index += 1
                popular[new_index] = True

                new_index_table.append( (x, new_index) )
                count += 1

        # Next Important Vertex
        intersection = -1
        for j in important_vertice:
            if intersection < len(working_set[index_list[important_vertex]] & working_set[index_list[j]]):
                intersection = len(working_set[index_list[important_vertex]] & working_set[index_list[j]])
                important_vertex = j

    for i, degree_line in enumerate(sorted_degree_lines):
        degree = list(map(int, degree_line.strip().split(" ")))

        if not traversed[degree[0]]:
            while popular[new_index]:
                new_index += 1
            popular[new_index] = True
            traversed[degree[0]] = True
            new_index_table.append( (degree[0], new_index) )
            count += 1

    sorted_degree_file.close()
    working_set_file.close()

    new_index_sorted = copy.copy(new_index_table)
    new_index_sorted.sort(key = lambda new_index_sorted:new_index_sorted[0])

    return new_index_table, new_index_sorted

def greedy_algorithm(sorted_degree_path, working_set_path, num_nodes, threshold=100, low=10):
    sorted_degree_file = open(sorted_degree_path, "r")
    working_set_file = open(working_set_path, "r")

    sorted_degree_lines = sorted_degree_file.readlines()
    working_set_lines = working_set_file.readlines()
    
    traversed = [False] * num_nodes
    popular = [False] * num_nodes

    # Naive reorganization algorithm
    count = 0
    new_index = 0
    new_index_table = []

    for i, degree_line in enumerate(sorted_degree_lines):
        degree = list(map(int, degree_line.strip().split(" ")))
        
        if degree[1] > threshold:
            popular[degree[0]] = True
            traversed[degree[0]] = True
            new_index_table.append( (degree[0], degree[0]) )
            count += 1
            continue

        if degree[1] < low:
            break

        if not traversed[degree[0]]:
            traversed[degree[0]] = True
            
            while popular[new_index]:
                new_index += 1
            popular[new_index] = True
 
            new_index_table.append( (degree[0], new_index) )
            count += 1

        # read working-set of degree[0] node
        neighbors_list = list(map(int, working_set_lines[degree[0]].strip().split(" ")))

        for x in neighbors_list:
            if not traversed[x]:
                traversed[x] = True

                while popular[new_index]:
                    new_index += 1
                popular[new_index] = True

                new_index_table.append( (x, new_index) )
                count += 1
                
    for i, degree_line in enumerate(reversed(sorted_degree_lines)):
        degree = list(map(int, degree_line.strip().split(" ")))

        if degree[1] == low:
            break
        
        if not traversed[degree[0]]:
            while popular[new_index]:
                new_index += 1
            popular[new_index] = True
            traversed[degree[0]] = True
            new_index_table.append( (degree[0], new_index) )
            count += 1

    sorted_degree_file.close()
    working_set_file.close()

    new_index_sorted = copy.copy(new_index_table)
    new_index_sorted.sort(key = lambda new_index_sorted:new_index_sorted[0])

    return new_index_table, new_index_sorted

def functionality_check(new_index_sorted, num_nodes):
    orig_sum = 0
    new_sum = 0
    ref_sum = 0

    for i in range(num_nodes):
        ref_sum += i
        orig_sum += new_index_sorted[i][1]
        new_sum += new_index_sorted[i][0]

    if ref_sum == orig_sum and orig_sum == new_sum:
        print("Remapping succeeded")
        return True
    else:
        print("Remapping failed")
        return False


    







