import math
import time
from collections import Counter
import numpy as np

from Draw_heirachial_graph import draw_hierarchy_pos
from Fitness_Function import fitness_function
from grid_map import *
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import tree, SpanningTreeIterator
import pygraphviz
import scipy as sp
import pydot
import itertools
import sys
from Func_weighted_spanning_tree import *

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


## Start of the Spanning Tree solution to the problem#####


node_list = unique_values_in_list_of_lists(Batch_sequence)
edge_list = []
raw_elist = []

for i in range(len(Batch_sequence)):
    for j in range(len(Batch_sequence[i]) - 1):
        # print(graph[i][j], graph[i][j+1])
        edge = [Batch_sequence[i][j], Batch_sequence[i][j + 1]]
        raw_elist.append(edge)
        if not edge in edge_list:
            edge_list.append(edge)  #### edge list of non repeating edges

### Generate a graph from the Genetic STage 1 Force directed output####
G = nx.MultiGraph()
G.add_nodes_from(node_list)
G.add_edges_from(raw_elist)
width_dict = Counter(G.edges())

### reliable edge list with distance and flow in dictionary####
edge_dict = [(u, v, {'weight': euclidean_dist(G_pos[u][0], G_pos[u][1], G_pos[v][0], G_pos[v][1]), 'Flow': value})
             for ((u, v), value) in width_dict.items()]
G.remove_edges_from(raw_elist)
G.add_edges_from(edge_dict)
# print(G.edges())
# print(G.edges.data())

# mst = tree.minimum_spanning_tree(G, algorithm="prim")
#
# T = nx.minimum_spanning_tree(G)
#
# print("edges:", sorted(T.edges(data=True)))
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
# main_edge = []
# for i in range(len(nList_diag) - 1):
#     # print(graph[i][j], graph[i][j+1])
#     edge1 = [nList_diag[i], nList_diag[i + 1]]
#     main_edge.append(edge1)
#
# print(main_edge)

## draw hierarchial graph
# h_pos = draw_hierarchy_pos(T, root=1, width=40, height=40)
# print("printed h pos:", h_pos)
# # nx.draw(T, h_pos, with_labels=True)
# # plt.grid(visible=True, color='r', linestyle='-', linewidth=2)
# # plt.show()
# print(h_pos)
#
# print(h_pos[20][0], h_pos[20][1])

# ## add weights to the spanning tree edges
# for e in T.edges():
#     dist1 = 0.0
#
#     first_node_x = h_pos[e[0]][0]
#     second_node_x = h_pos[e[1]][0]
#     first_node_y = h_pos[e[0]][1]
#     second_node_y = h_pos[e[1]][1]
#
#     dist1 += round(
#         math.sqrt(math.pow(second_node_x - first_node_x, 2) + math.pow(second_node_y - first_node_y, 2) * 1.0))
#     # print("edges pair:", e[0], e[1], dist1)
#     T[e[0]][e[1]][0]['weight'] = dist1


# print("length source to target", nx.dijkstra_path_length(T, 1, 5))
# print("length source to target", nx.dijkstra_path_length(T, 1, 11))
#
# print(fitness_function(T, Batch_sequence, PI_weight))

# print(w_map)


### test spaning tree out of  a
# MST = create_weightedPI_tree(G, G_pos, Batch_sequence[2])
# print(MST)
# mst_pos = draw_hierarchy_pos(MST, root=1, width=40, height=40)
# nx.draw(MST, mst_pos, with_labels=True)
# plt.grid(visible=True, color='r', linestyle='-', linewidth=2)
# plt.show()


print(G[1][5][0]['weight'])

##### Genetic algorithm for Optimizing the Spanning tree problem######

print("Start of the Genetic Algorithm")

### Create population here####
random_pop = []
grid_size = 40

for i in range(1):
    iter_class = iter(SpanningTreeIterator(G, minimum=True, ignore_nan=True))
    random_pop.append(next(iter_class))

for i in range(1):
    iter_class = iter(SpanningTreeIterator(G, minimum=False, ignore_nan=True))
    random_pop.append(next(iter_class))
#
# random_pop.append(
#     create_weightedPI_tree(G, G_pos, Batch_sequence[PI_weight.index(max(PI_weight))]))
for pseq in Batch_sequence:
    random_pop.append(create_weightedPI_tree(G, G_pos, pseq))
    print("The new batch sequenc:", pseq)

## Draw hierarchy tree psoitions all population###
tree_pos = []
for i, chr_Tree in enumerate(random_pop):
    pos = draw_hierarchy_pos(chr_Tree, root=1, width=grid_size, height=grid_size)
    tree_pos.append(pos)
    plt.figure()
    plt.title(f"The plot belongs to tree {i} ")
    nx.draw(chr_Tree, pos, with_labels=True)
    plt.grid(True)
    plt.pause(0.05)
    plt.show()

##### calculate fitness function for random population

random_fitness = []

print("Taking a pause")
time.sleep(1)  # Pause 5.5 seconds
print("pause ended")

for i, (chr_Tree, pos) in enumerate(zip(random_pop, tree_pos)):
    random_fitness.append(fitness_function(chr_Tree, Batch_sequence, PI_weight))
    time.sleep(0.05)
    print(i, random_fitness[i])

# print(random_pop[20].edges())
## Choose the fittest spanning tree####

print(random_fitness)
sorted_fitness = sorted(random_fitness)
pIndex_1 = random_fitness.index(sorted_fitness[0])
pIndex_2 = random_fitness.index(sorted_fitness[1])


parent1 = random_pop[pIndex_1]
parent2 = random_pop[pIndex_2]

print(parent1)
print(parent2)
##### crossover function#######

# print(len(list(G.neighbors(1))))
