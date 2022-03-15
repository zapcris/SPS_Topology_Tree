import math
from collections import Counter
import numpy as np
from gird import *
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import tree, SpanningTreeIterator
import pygraphviz
import scipy as sp
import pydot
import itertools
import sys

G_pos = {1: (10, 7), 2: (19, 13), 3: (21, 10), 4: (21, 17), 5: (13, 0),
          6: (17, 3), 7: (21, 25), 8: (11, 4), 9: (15, 10), 10: (17, 19),
          11: (28, 7), 12: (33, 12), 13: (24, 13), 14: (0, 3), 15: (25, 22),
          17: (27, 25), 19: (32, 31), 20: (26, 33)}

Batch_sequence = [[1, 5, 9, 10, 2, 11, 13, 15, 7, 20],
                  [1, 2, 7, 3, 5, 6, 8, 9, 13, 15, 19, 20],
                  [1, 5, 8, 6, 3, 2, 4, 10, 15, 17, 20],
                  [1, 8, 9, 10, 2, 11, 13, 15, 7, 20],
                  [1, 4, 17, 3, 8, 9, 13, 15, 19, 20],
                  [1, 6, 8, 6, 3, 12, 4, 10, 15, 17, 20],
                  [1, 14, 8, 6, 13, 2, 4, 10, 15, 17, 20]]

Qty_order = [10, 30, 50, 20, 60, 20, 40]

PI_weight = []
for i in range(len(Qty_order)):
    PI_weight.append(Qty_order[i] * len(Batch_sequence[i]))

print("weight list:", PI_weight)



matrix = np.array(
    [[-2, 5, 3, 2],
     [9, -6, 5, 1],
     [3, 2, 7, 3],
     [-1, 8, -4, 8]])

diags = [matrix[::-1, :].diagonal(i) for i in range(-3, 4)]
diags.extend(matrix.diagonal(i) for i in range(3, -4, -1))

# for n in diags:
#     print(n.tolist())


Grid = np.zeros((35, 35))
print(Grid)

max_index = PI_weight.index(max(PI_weight))
print(max_index)

nList_diag = Batch_sequence[max_index]
print("Diagnoally prefered sequence", nList_diag)

np.fill_diagonal(Grid, 99)

di = np.diag_indices_from(Grid)
# print(di)

# create list to be filled in diagonal of grid matrix

interval = len(Grid.diagonal()) // len(nList_diag)

# print(interval)


diag_seq = [0 for n in Grid.diagonal()]

for i in range(0, len(Grid.diagonal()), 3):
    # print(i)
    diag_seq[i] = 6

# print(diag_seq)

# Initiliaze the grid map
grid_map = GridMap(35, 35, 1, 17.5, 17.5)

# grid_map.plot_grid_map()

# Place diagonals workstations on the grid MAP
for i in range(len(nList_diag)):
    x = i * interval
    y = i * interval
    grid_map.set_value_from_xy_pos(x, y, 1)

    label = f"WK({nList_diag[i]})"
    # plt.annotate(label,  # this is the text
    #              (x, y),  # these are the coordinates to position the label
    #              textcoords="offset points",  # how to position the text
    #              xytext=(0, 0),  # distance from text to points (x,y)
    #              ha='left')  # horizontal alignment can be left, right or cente


## Minimum Spanning Tree algorithm

def unique_values_in_list_of_lists(lst):
    result = set(x for l in lst for x in l)
    return list(result)


def euclidean_dist(x1, y1, x2, y2):
    dist = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0)
    return round(dist)


node_list = unique_values_in_list_of_lists(Batch_sequence)
edge_list = []
raw_elist = []

for i in range(len(Batch_sequence)):
    for j in range(len(Batch_sequence[i]) - 1):
        # print(graph[i][j], graph[i][j+1])
        edge = [Batch_sequence[i][j], Batch_sequence[i][j + 1]]
        raw_elist.append(edge)
        if not edge in edge_list:
            edge_list.append(edge) #### edge list of non repeating edges

### Generate a graph from the Genetic STage 1 Force directed output####
G = nx.MultiGraph()
G.add_nodes_from(node_list)
G.add_edges_from(raw_elist)
width_dict = Counter(G.edges())
edge_dict = [(u, v, {'weight': euclidean_dist(G_pos[u][0], G_pos[u][1], G_pos[v][0], G_pos[v][1]), 'Flow': value})
             for ((u, v), value) in width_dict.items()]
G.remove_edges_from(raw_elist)
G.add_edges_from(edge_dict)
print(G.edges())
print(G.edges.data())


mst = tree.minimum_spanning_tree(G, algorithm="prim")

T = nx.minimum_spanning_tree(G)

print("edges:", sorted(T.edges(data=True)))
#
# pos1 = nx.nx_pydot.pydot_layout(G, prog="fdp")
# pos3 = nx.nx_pydot.graphviz_layout(G, prog="dot")
# print(pos1)
# nx.draw(T, pos3, with_labels=True)


# p = nx.drawing.nx_pydot.to_pydot(T)

# print("tree position:", p)
# plt.show()
# grid_map.plot_grid_map()
# plt.grid(which='minor', alpha=0.2)
# plt.grid(which='major', alpha=0.5)

# plt.show()

### function to calculate spanning trees#
main_edge = []
for i in range(len(nList_diag) - 1):
    # print(graph[i][j], graph[i][j+1])
    edge1 = [nList_diag[i], nList_diag[i + 1]]
    main_edge.append(edge1)

print(main_edge)


def draw_hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"

    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL: 0, CURRENT: 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels = make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1 / levels[currentLevel][TOTAL]
        left = dx / 2
        pos[node] = (round((left + dx * levels[currentLevel][CURRENT]) * width), vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc - vert_gap)
        return pos

    if levels is None:
        levels = make_levels({})
    else:
        levels = {l: {TOTAL: levels[l], CURRENT: 0} for l in levels}
    vert_gap = height / (max([l for l in levels]) + 1)
    return make_pos({})


## draw hierarchial graph

h_pos = draw_hierarchy_pos(T, root=1, width=40, height=40)

print("printed h pos:", h_pos)


# nx.draw(T, h_pos, with_labels=True)
# plt.grid(visible=True, color='r', linestyle='-', linewidth=2)
# plt.show()


def fitness_function(T, batch_list):
    PI_cost = []

    for i in range(len(batch_list)):
        cost = 0.0
        for j in range(len(batch_list[i]) - 1):
            cost += nx.dijkstra_path_length(T, batch_list[i][j], batch_list[i][j + 1])

        PI_cost.append(round(cost * (PI_weight[i] / 100)))

    batch_cost = sum(PI_cost)
    return PI_cost, batch_cost


print(h_pos)
print(T.edges())
print(h_pos[20][0], h_pos[20][1])

## add weights to the spanning tree edges
for e in T.edges():
    dist1 = 0.0

    first_node_x = h_pos[e[0]][0]
    second_node_x = h_pos[e[1]][0]
    first_node_y = h_pos[e[0]][1]
    second_node_y = h_pos[e[1]][1]

    dist1 += round(
        math.sqrt(math.pow(second_node_x - first_node_x, 2) + math.pow(second_node_y - first_node_y, 2) * 1.0))
    # print("edges pair:", e[0], e[1], dist1)
    T[e[0]][e[1]][0]['weight'] = dist1


## function to add edge weight####

def set_edge_weight(graph, pos):
    for e in graph.edges():
        dist = 0.0
        first_node_x = pos[e[0]][0]
        second_node_x = pos[e[1]][0]
        first_node_y = pos[e[0]][1]
        second_node_y = pos[e[1]][1]
        dist += round(
            math.sqrt(math.pow(second_node_x - first_node_x, 2) + math.pow(first_node_y - second_node_y, 2) * 1.0))
        graph[e[0]][e[1]][0]['weight'] = dist


print("length source to target", nx.dijkstra_path_length(T, 1, 5))
print("length source to target", nx.dijkstra_path_length(T, 1, 11))

print(fitness_function(T, Batch_sequence))


# print(w_map)
## Spanning tree for most weighted product Instance in the batch###

def create_weightedPI_tree(G, pos, PI_sequence, full_elist, full_nlist):
    edge_list = []
    remain_node = []
    span_edges = []
    reduced_span = []
    for i in range(len(PI_sequence) - 1):
        e = [PI_sequence[i], PI_sequence[i + 1]]
        edge_list.append(e)

    ### enlist nodes to be added to complete the spanning tree####
    for node in full_nlist:
        if not node in PI_sequence:
            remain_node.append(node)
    print("Remaining node:", remain_node)
    print(full_elist)
    ### enlist pair of edges from global edge list for remaining nodes
    for node in remain_node:
        for i in range(len(full_elist)):
            for j in range(len(full_elist[i])):
                if full_elist[i][j] == node:
                    print(node, full_elist[i])
                    d = node, full_elist[i], G[full_elist[i][0]][full_elist[i][1]][0]['weight']
                    span_edges.append(d)

    print(span_edges)

    ### remove edges from prospective list which has both the nodes not present in the current graph
    for e in span_edges:
        print(e[0], e[1][0], e[1][1], e[2])
        if not (e[1][0] in remain_node and e[1][1] in remain_node):
            reduced_span.append(e)

    print("deleted span edges:", reduced_span)

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

    print(weight)
    print(edge)

    edge_list.extend(edge)

    print("Spanning tree edge list:", edge_list)

    S = nx.MultiGraph()
    S.add_nodes_from(PI_sequence)
    S.add_edges_from(edge_list)

    print("The graph is a tree?", nx.is_tree(S))

    return mst


MST = create_weightedPI_tree(G, G_pos, Batch_sequence[2], edge_list, node_list)
print(MST)
mst_pos = draw_hierarchy_pos(MST, root=1, width=40, height=40)
# nx.draw(MST, mst_pos, with_labels=True)
# plt.grid(visible=True, color='r', linestyle='-', linewidth=2)
# plt.show()

print(G.edges.data("weight", default=1))
print(G.get_edge_data(1, 5, 0))
print(G[1][5][0]['weight'])

##### Genetic algorithm for Optimizing the Spanning tree problem######

### Create population here####
random_pop = []
grid_size = 40

for i in range(10):
    random_pop.append(SpanningTreeIterator(G, minimum=True, ignore_nan=True))

for i in range(10):
    random_pop.append(SpanningTreeIterator(G, minimum=False, ignore_nan=True))

random_pop.append(
    create_weightedPI_tree(G, G_pos, Batch_sequence[PI_weight.index(max(PI_weight))], edge_list, node_list))

ST_min = iter(SpanningTreeIterator(G, minimum=True, ignore_nan=True))
ST_max = iter(SpanningTreeIterator(G, minimum=False, ignore_nan=True))
ST = SpanningTreeIterator(G, minimum=False, ignore_nan=True)

# test_lst = []
# for x in range(10):
#     test_lst.append(next(ST))
# print(len(test_lst))


print("Spanning iterator:", next(ST_min).edges())
print("Spanning iterator:", next(ST_min).edges())


PT = tree.partition_spanning_tree(G, minimum=True, weight="weight", partition="partition", ignore_nan=False)
print(PT.edges())
# print("The graph is a tree?", nx.is_tree(ST))

## convert to tree all population###3
tree_pos = []
tree_pop = []
# for pop in random_pop:
#     t = draw_hierarchy_pos(pop, root=1, width=grid_size, height=grid_size)
#     tree_pos.append(t)
#     nx.draw(pop, t, with_labels=True)
#     plt.pause(0.05)
#     print("number of tree graph")


##### calculate fitness function for random population

random_fitness = []
for pop in tree_pop:
    random_fitness.append(fitness_function(pop, Batch_sequence))

# print(random_fitness)
