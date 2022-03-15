## Spanning tree for most weighted product Instance in the batch###
import networkx as nx


def create_weightedPI_tree(G, pos, PI_sequence):
    edge_list = []
    remain_node = []
    span_edges = []
    reduced_span = []
    full_elist = list(G.edges())
    full_nlist = G.nodes()
    for i in range(len(PI_sequence) - 1):
        e = [PI_sequence[i], PI_sequence[i + 1]]
        edge_list.append(e)

    ### enlist nodes to be added to complete the spanning tree####
    for node in full_nlist:
        if not node in PI_sequence:
            remain_node.append(node)
    # print("Remaining node:", remain_node)
    # print(full_elist)
    ### enlist pair of edges from global edge list for remaining nodes
    for node in remain_node:
        for i in range(len(full_elist)):
            for j in range(len(full_elist[i])):
                if full_elist[i][j] == node:
                    #print(node, full_elist[i])
                    d = node, full_elist[i], G[full_elist[i][0]][full_elist[i][1]][0]['weight']
                    span_edges.append(d)

    #print(span_edges)

    ### remove edges from prospective list which has both the nodes not present in the current graph
    for e in span_edges:
        #print(e[0], e[1][0], e[1][1], e[2])
        if not (e[1][0] in remain_node and e[1][1] in remain_node):
            reduced_span.append(e)

    #print("deleted span edges:", reduced_span)

    ### pick edge pair from the reduced prospective list with minimum path to the current graph
    visited = []
    weight = []
    edge = []
    for e in reduced_span:
        a = []
        if e[0] in visited and e[1] == min(weight):
            weight.clear()
            edge.clear()
            weight.append(e[2])
            a = [e[1][0], e[1][1]]
            edge.append(a)
        if not e[0] in visited:
            visited.append(e[0])
            weight.append(e[2])
            a = [e[1][0], e[1][1]]
            edge.append(a)

    #print(weight)
    #print(edge)

    edge_list.extend(edge)

    #print("Spanning tree edge list:", edge_list)

    S = nx.MultiGraph()
    S.add_nodes_from(PI_sequence)
    S.add_edges_from(edge_list)

    #print("The graph is a tree?", nx.is_tree(S))

    return S