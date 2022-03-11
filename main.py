import numpy as np
from gird import *
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import tree, SpanningTreeIterator
import pygraphviz
import scipy as sp
import pydot

pos2 = {1: (10, 7), 2: (19, 13), 3: (21, 10), 4: (21, 17), 5: (13, 0), 6: (17, 3), 7: (21, 25), 8: (11, 4), 9: (15, 10),
        10: (17, 19), 11: (28, 7), 12: (33, 12), 13: (24, 13), 14: (0, 3), 15: (25, 22), 17: (27, 25), 19: (32, 31),
        20: (26, 33)}
Batch_sequence = [[1, 5, 9, 10, 2, 11, 13, 15, 7, 20],
                  [1, 2, 7, 3, 5, 6, 8, 9, 13, 15, 19, 20],
                  [1, 5, 8, 6, 3, 2, 4, 10, 15, 17, 20],
                  [1, 8, 9, 10, 2, 11, 13, 15, 7, 20],
                  [1, 4, 17, 3, 8, 9, 13, 15, 19, 20],
                  [1, 6, 8, 6, 3, 12, 4, 10, 15, 17, 20],
                  [1, 14, 8, 6, 13, 2, 4, 10, 15, 17, 20]]

Qty_order = [10, 30, 50, 20, 60, 20, 40]

weight = []
for i in range(len(Qty_order)):
    weight.append(Qty_order[i] * len(Batch_sequence[i]))

print(weight)
ip_positions = {1: (10, 7), 2: (19, 13), 3: (21, 10), 4: (21, 17), 5: (13, 0),
                6: (17, 3), 7: (21, 25), 8: (11, 4), 9: (15, 10), 10: (17, 19),
                11: (28, 7), 12: (33, 12), 13: (24, 13), 14: (0, 3), 15: (25, 22),
                17: (27, 25), 19: (32, 31), 20: (26, 33)}

edge_frequencies = [[1, 5, {'frequency': 2}], [1, 2, {'frequency': 1}], [1, 8, {'frequency': 1}],
                    [1, 4, {'frequency': 1}], [1, 6, {'frequency': 1}], [1, 14, {'frequency': 1}],
                    [2, 10, {'frequency': 2}], [2, 11, {'frequency': 2}], [2, 7, {'frequency': 1}],
                    [2, 3, {'frequency': 1}], [2, 4, {'frequency': 2}], [2, 13, {'frequency': 1}],
                    [3, 7, {'frequency': 1}], [3, 5, {'frequency': 1}], [3, 6, {'frequency': 2}],
                    [3, 17, {'frequency': 1}], [3, 8, {'frequency': 1}], [3, 12, {'frequency': 1}],
                    [4, 10, {'frequency': 3}], [4, 17, {'frequency': 1}], [4, 12, {'frequency': 1}],
                    [5, 9, {'frequency': 1}], [5, 6, {'frequency': 1}], [5, 8, {'frequency': 1}],
                    [6, 8, {'frequency': 5}], [6, 13, {'frequency': 1}], [7, 15, {'frequency': 2}],
                    [7, 20, {'frequency': 2}], [8, 9, {'frequency': 3}], [8, 14, {'frequency': 1}],
                    [9, 10, {'frequency': 2}], [9, 13, {'frequency': 2}], [10, 15, {'frequency': 3}],
                    [11, 13, {'frequency': 2}], [13, 15, {'frequency': 4}], [15, 19, {'frequency': 2}],
                    [15, 17, {'frequency': 3}], [17, 20, {'frequency': 3}], [19, 20, {'frequency': 2}]]

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

max_index = weight.index(max(weight))
print(max_index)

nList_diag = Batch_sequence[max_index]
print("Diagnoally prefered sequence", nList_diag)

np.fill_diagonal(Grid, 99)

di = np.diag_indices_from(Grid)
# print(di)

# create list to be filled in diagonal of grid matrix

interval = len(Grid.diagonal()) // len(nList_diag)

# print(interval)

print(len(Grid.diagonal()))

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


## Minimum Spanninf Tree algorithm

def unique_values_in_list_of_lists(lst):
    result = set(x for l in lst for x in l)
    return list(result)


node_list = unique_values_in_list_of_lists(Batch_sequence)
edge_list = []
for i in range(len(Batch_sequence)):
    for j in range(len(Batch_sequence[i]) - 1):
        # print(graph[i][j], graph[i][j+1])
        edge = [Batch_sequence[i][j], Batch_sequence[i][j + 1]]
        edge_list.append(edge)

print("edge_list:", edge_list)
new_edge_list = [[1, 8], [1, 2], [1, 4], [1, 5], [1, 6], [1, 4], [2, 11], [2, 7], [2, 10], [2, 13], [4, 7], [5, 3],
                 [5, 9], [7, 20], [13, ]]
G = nx.MultiGraph()
G.add_nodes_from(node_list)
G.add_edges_from(edge_list)

mst = tree.minimum_spanning_tree(G, algorithm="prim")
ST = SpanningTreeIterator(G, minimum=True, ignore_nan=True)

T = nx.minimum_spanning_tree(G)

print("edges:", sorted(T.edges(data=True)))

pos1 = nx.nx_pydot.pydot_layout(G, prog="fdp")
pos3 = nx.nx_pydot.graphviz_layout(G, prog="dot")
print(pos1)
# nx.draw(T, pos3, with_labels=True)


p = nx.drawing.nx_pydot.to_pydot(T)

print("tree position:", p)
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


def hierarchy_pos(G, root, levels=None, width=1., height=1.):
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

h_pos = hierarchy_pos(T, root=1, width=40, height=40)

print("printed h pos:", h_pos)
nx.draw(T, h_pos, with_labels=True)
plt.grid(visible=True, color='r', linestyle='-', linewidth=2)
plt.show()
