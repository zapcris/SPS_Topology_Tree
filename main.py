import random
import random
import sys
import time
from random import randint
import matplotlib.pyplot as plt
from networkx.algorithms import SpanningTreeIterator
from Draw_heirachial_graph import draw_hierarchy_pos
from Fitness_Function import fitness_function
from Func_weighted_spanning_tree import *
from grid_map import *
import EoN


# recursion_limit = sys.getrecursionlimit()
#
# sys.setrecursionlimit(1500)


def rand_index(gen):
    r1 = (random.randint(2, 6))
    r2 = (random.randint(6, 10))
    r3 = (random.randint(11, 15))
    index = [r1, r2, r3]
    print(index)

    if gen >= 10:
        print("stopped")
    else:
        rand_index(gen + 1)


#
# G_pos = {1: (10, 7), 2: (19, 13), 3: (21, 10), 4: (21, 17), 5: (13, 0),
#          6: (17, 3), 7: (21, 25), 8: (11, 4), 9: (15, 10), 10: (17, 19),
#          11: (28, 7), 12: (33, 12), 13: (24, 13), 14: (0, 3), 15: (25, 22),
#          17: (27, 25), 19: (32, 31), 20: (26, 33)}

# G_pos = {1: (10, 7), 2: (19, 13), 3: (21, 10), 4: (21, 17), 5: (13, 0),
#          6: (17, 3), 7: (21, 25), 8: (11, 4), 9: (15, 10), 10: (17, 19),
#          11: (28, 7), 12: (33, 12), 13: (24, 13), 14: (0, 3), 15: (25, 22),
#          16: (12, 14), 17: (27, 25), 18: (21, 22), 19: (32, 31), 20: (26, 33)}

G_pos = {1: (9, 4), 2: (18, 13), 3: (22, 8), 4: (18, 10), 5: (13, 0), 6: (15, 4), 7: (24, 22), 8: (12, 6), 9: (10, 11), 10: (14, 16), 11: (26, 15), 12: (30, 4), 13: (17, 17), 14: (0, 2), 15: (19, 24), 16: (22, 12), 17: (26, 20), 19: (23, 33), 20: (27, 28)}




Batch_sequence = [[1, 5, 9, 10, 2, 11, 13, 15, 7, 20],
                  [1, 2, 7, 3, 5, 6, 8, 9, 13, 15, 19, 20],
                  [1, 5, 8, 6, 3, 2, 4, 10, 15, 17, 20],
                  [1, 8, 9, 10, 2, 11, 13, 15, 7, 20],
                  [1, 4, 17, 3, 8, 9, 13, 15, 19, 20],
                  [1, 6, 8, 6, 3, 12, 4, 10, 15, 17, 20],
                  [1, 14, 8, 6, 13, 2, 4, 10, 15, 17, 20]]

# Batch_sequence = [[1, 5, 9, 10, 2, 11, 13, 15, 7, 20],
#                   [1, 2, 7, 3, 5, 6, 8, 9, 13, 15, 19, 20],
#                   [1, 5, 8, 6, 3, 2, 4, 10, 15, 17, 20],
#                   [1, 8, 9, 10, 2, 11, 13, 15, 7, 20],
#                   [1, 4, 17, 3, 8, 9, 13, 15, 19, 20],
#                   [1, 6, 8, 6, 3, 12, 18, 10, 15, 17, 20],
#                   [1, 14, 8, 6, 13, 2, 4, 10, 16, 17, 20],
#                   [1, 3, 8, 6, 13, 2, 15, 10, 16, 17, 19]]

Qty_order = [10, 30, 50, 20, 60, 20, 40]

PI_weight = []
for i in range(len(Qty_order)):
    PI_weight.append(Qty_order[i] * len(Batch_sequence[i]))

print("weighted list for Product Instances:", PI_weight)

## Start of the Spanning Tree solution to the problem#####


node_list = unique_values_in_list_of_lists(Batch_sequence)
print("Total nodes in batch:", node_list)
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

##### Genetic algorithm for Optimizing the Spanning tree problem######

print("Start of the Genetic Algorithm")

### Create population here####
random_pop = []
grid_size = 28

L = nx.MultiGraph()
L.add_nodes_from(node_list)
L.add_edges_from(raw_elist)

for i in range(1):
    iter_class1 = iter(SpanningTreeIterator(G, minimum=True, ignore_nan=True))
    random_pop.append(next(iter_class1))

for i in range(1):
    iter_class2 = iter(SpanningTreeIterator(G, minimum=False, ignore_nan=True))
    random_pop.append(next(iter_class2))

# for i in range(1):
#     iter_class3 = iter(SpanningTreeIterator(L, minimum=True, ignore_nan=True))
#     random_pop.append(next(iter_class3))

#     create_weightedPI_tree(G, G_pos, Batch_sequence[PI_weight.index(max(PI_weight))]))
for pseq in Batch_sequence:
    random_pop.append(create_weightedPI_tree(G, G_pos, pseq))
    # print("The new batch sequenc:", pseq)

##Draw hierarchy tree psoitions all population###
tree_pos = []
for i, chr_Tree in enumerate(random_pop):
    # pos= G_pos
    pos = draw_hierarchy_pos(chr_Tree, root=1, width=grid_size, height=grid_size)
    # pos = EoN.hierarchy_pos(chr_Tree, root=1, width=40)
    tree_pos.append(pos)
    plt.figure()
    plt.title(f"The plot belongs to initial population {i + 1} ")
    nx.draw(chr_Tree, pos, with_labels=True)
    plt.grid(True)
    plt.pause(0.05)
    plt.show()

##### calculate fitness function for random population
cross_gen_fitness = []
random_fitness = []

print("Taking a pause")
time.sleep(1)  # Pause 1 seconds
print("pause ended")

for i, (chr_Tree, pos) in enumerate(zip(random_pop, tree_pos)):
    random_fitness.append(fitness_function(chr_Tree, Batch_sequence, PI_weight,pos))
    time.sleep(0.05)
    # print(i, random_fitness[i])

cross_gen_fitness.append(random_fitness)

# print(random_pop[20].edges())
## Choose the fittest spanning tree####

print("TREE Fitness values for population", random_fitness)
print("Euclidean Fitness values for population", random_fitness)
sys.exit()
sorted_fitness = sorted(random_fitness)
pIndex_1 = random_fitness.index(sorted_fitness[0])
pIndex_2 = random_fitness.index(sorted_fitness[1])

parent1 = random_pop[pIndex_1]
parent2 = random_pop[pIndex_2]

##### crossover function#######
### convert the global map G to prufer suitable
prufer_map = {}
for i, j in enumerate(set(G)):
    prufer_map[j] = i

print("Prufer global map:", prufer_map)

### convert back to original map
origin_map = dict([(value, key) for key, value in prufer_map.items()])
print("original map:", origin_map)


def get_prufer_sequence(parent, map):
    # map = {}
    # for i, j in enumerate(set(parent)):
    #     map[j] = i
    # print("map of offsrping1 :", map)
    parent = nx.relabel_nodes(parent, map)
    # print(set(parent))
    # print(set(range(len(parent))))
    pruf_seq = nx.to_prufer_sequence(parent)
    return pruf_seq, parent


def prufer_to_tree(pruf_seq, map):
    graph = nx.from_prufer_sequence(pruf_seq)

    graph = nx.relabel_nodes(graph, map)
    print("The graph is a tree?", nx.is_tree(graph))
    return graph


def uniform_crossover(A, B):
    for i in range(len(A)):
        # if P[i] < 0.01:
        temp = A[i]
        A[i] = B[i]
        B[i] = temp
    a1 = A
    b1 = B
    return A, B


def crossover(a, b, index):
    return b[:index] + a[index:], a[:index] + b[index:]


def multi_point_crossver(parent1, parent2, index_arr):
    # create segments first
    child_A = []
    child_B = []
    A_seg = [0] * (len(index_arr) + 1)
    B_seg = [0] * (len(index_arr) + 1)
    tmp_a = []
    tmp_b = []
    for i in range(len(index_arr) + 1):
        if i == 0:
            tmp_a.append(parent1[:index_arr[i]])
            tmp_b.append(parent2[:index_arr[i]])
        elif i != 0 and i != len(index_arr):
            tmp_a.append(parent1[index_arr[i - 1]:index_arr[i]])
            tmp_b.append(parent2[index_arr[i - 1]:index_arr[i]])
        elif i == len(index_arr):
            tmp_a.append(parent1[index_arr[i - 1]:])
            tmp_b.append(parent2[index_arr[i - 1]:])
    # print(tmp_a)
    # print(tmp_b)
    ### exchange odd segments to other parent ######
    for i, (Aseg, Bseg) in enumerate(zip(tmp_a, tmp_b)):

        if (i % 2) == 0:
            A_seg[i] = Aseg
            B_seg[i] = Bseg
        else:
            tmp = Aseg
            A_seg[i] = Bseg
            B_seg[i] = tmp
    # print(A_seg)
    # print(B_seg)
    for a, b in zip(A_seg, B_seg):
        child_A.extend(a)
        child_B.extend(b)
    # print(child_A)
    # print(child_B)

    return child_A, child_B


def breed_crossover(parent1, parent2):
    index = int(np.random.uniform(low=1, high=len(parent1) - 1))  # random point between 1 and 1 is always 1
    return np.hstack([parent1[:index], parent2[index:]])


A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
B = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 12, 13, 14, 15, 16]
index = [2, 6, 12, 14]
multi_point_crossver(A, B, index)

children = []
# print("multipoint corssover test:", breed_crossover(A, B))
children.append(breed_crossover(A, B))
children.append(breed_crossover(B, A))

print("multipoint corssover test:", children)
test = [1, 2, 3, 4, 5, 6, 7, 8, 9, ]
random.shuffle(test)


# print("A shuffled list", test)


def genetic_stage2(parent1, parent2, gen):
    print("Started recurrsion with generation number", gen)
    parent_fitness = []
    offspring_fitness = []
    offspring_fitness = []
    pruf_parent1 = get_prufer_sequence(parent1, prufer_map)
    pruf_parent2 = get_prufer_sequence(parent2, prufer_map)
    # print("parent1 prufer", pruf_parent1[0])
    # print("parent2 prufer", pruf_parent2[0])
    middle_idx1 = round((len(pruf_parent1[0]) - 1) / 2)
    middle_idx2 = round((len(pruf_parent2[0]) - 1) / 2)

    # off_pruf1 = pruf_parent1[0][:middle_idx1] + pruf_parent2[0][middle_idx2:]
    # off_pruf2 = pruf_parent2[0][:middle_idx2] + pruf_parent1[0][middle_idx1:]
    # off_pruf3 = pruf_parent1[0][:middle_idx1] + pruf_parent2[0][:middle_idx2]
    # off_pruf4 = pruf_parent2[0][:middle_idx2] + pruf_parent1[0][:middle_idx1]

    index2 = [randint(1, len(pruf_parent1[0]) - 1), randint(1, len(pruf_parent1[0]) - 1)]

    cut_points = [(random.randint(2, 5)), (random.randint(6, 10)), (random.randint(11, 15))]
    Off = multi_point_crossver(pruf_parent1[0], pruf_parent2[0], cut_points)

    # random.shuffle(Off[0])
    # random.shuffle(Off[1])
    # # Off = crossover(pruf_parent1[0], pruf_parent2[0], index)
    # # Off = multi_point_crossover(pruf_parent1[0], pruf_parent2[0], index2)
    #
    # print("offsprinf prufer sequences:", Off)
    # Off_2 = uniform_crossover(pruf_parent1[0], pruf_parent2[0])

    off1_tree = prufer_to_tree(Off[0], origin_map)
    off2_tree = prufer_to_tree(Off[1], origin_map)

    logical_off = [off1_tree, off2_tree]  # off3_tree, off4_tree]
    offspring_trees = []
    # print("pause started")
    # time.sleep(5)
    # print("resumed")
    ### Draw and calculate fitness function ####
    for i, off_tree in enumerate(logical_off):
        pos_off = draw_hierarchy_pos(off_tree, root=1, width=grid_size, height=grid_size)
        # pos_off = EoN.hierarchy_pos(off_tree, root=1, width=40)
        plt.figure()
        plt.title(f"The plot belongs to offsrping {i + 1} in generation {gen} ")
        nx.draw(off_tree, pos_off, with_labels=True)
        plt.grid(True)
        plt.pause(0.05)
        plt.show()
        offspring_trees.append(convert_logical_spatial(off_tree, pos_off))

    for i, off_top in enumerate(offspring_trees):
        offspring_fitness.append(fitness_function(off_top, Batch_sequence, PI_weight))

    cross_gen_fitness.append(offspring_fitness)
    print(f'The fitness list of generation {gen} is {offspring_fitness}')

    if min(offspring_fitness) <= 2000 or gen >= 40:
        print("Recurssion Ended ")


    else:
        # time.sleep(1)
        gen = gen + 1

        sorted_fitness = sorted(offspring_fitness)
        pIndex_1 = offspring_fitness.index(sorted_fitness[0])
        pIndex_2 = offspring_fitness.index(sorted_fitness[1])
        new_parent1 = offspring_trees[pIndex_1]
        new_parent2 = offspring_trees[pIndex_2]
        genetic_stage2(new_parent1, new_parent2, gen)


genetic_stage2(parent1, parent2, 1)

min_fit = 0
gen_fit = 0
for i, fit in enumerate(cross_gen_fitness):
    print(fit)
    if i == 0:
        min_fit = min(fit)
        gen_fit = 1
    elif min(fit) < min_fit:
        min_fit = min(fit)
        gen_fit = i + 1

if gen_fit <= 1:
    print(
        f" Fittest value found is : {min_fit} in initial population chromosome no: {(random_fitness.index(min_fit)) + 1}")
else:
    print(f" Fittest value found is : {min_fit} in generation {gen_fit}")

#### GRAPH TO GRID MAPP#####

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
# print(Grid)

max_index = PI_weight.index(max(PI_weight))
# print(max_index)

nList_diag = Batch_sequence[max_index]
# print("Diagnoally prefered sequence", nList_diag)

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

### OLD CODE ###
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


# print(G[1][5][0]['weight'])

# pruf_parent1 = get_prufer_sequence(parent1, prufer_map)
# pruf_parent2 = get_prufer_sequence(parent2, prufer_map)
#
# middle_idx1 = round((len(pruf_parent1[0]) - 1) / 2)
# middle_idx2 = round((len(pruf_parent2[0]) - 1) / 2)
#
# # off_pruf1_1 = pruf_parent1[0][:middle_idx1]
# # off_pruf1_2 = pruf_parent1[0][middle_idx1:]
# # off_pruf2_1 = pruf_parent2[0][:middle_idx2]
# # off_pruf2_2 = pruf_parent2[0][middle_idx2:]
# #
# # off_pruf1 = off_pruf1_1 + off_pruf2_2
# # off_pruf2 = off_pruf2_1 + off_pruf1_2
# # off_pruf3 = off_pruf1_1 + off_pruf2_1
# # off_pruf4 = off_pruf2_1 + off_pruf1_1
#
# off_pruf1 = pruf_parent1[0][:middle_idx1] + pruf_parent2[0][middle_idx2:]
# off_pruf2 = pruf_parent2[0][:middle_idx2] + pruf_parent1[0][middle_idx1:]
# off_pruf3 = pruf_parent1[0][:middle_idx1] + pruf_parent2[0][:middle_idx2]
# off_pruf4 = pruf_parent2[0][:middle_idx2] + pruf_parent1[0][:middle_idx1]
#
# print(pruf_parent1)
# print(pruf_parent2)
#
# print("offspring 1 prufer sequence:", off_pruf1)
# print("offspring 2 prufer sequence:", off_pruf2)
#
# off1_tree = prufer_to_tree(off_pruf1, origin_map)
# off2_tree = prufer_to_tree(off_pruf2, origin_map)
# off3_tree = prufer_to_tree(off_pruf3, origin_map)
# off4_tree = prufer_to_tree(off_pruf4, origin_map)
#
# logical_off = [off1_tree, off2_tree, off3_tree, off4_tree]
#
# print("The graph is a tree?", nx.is_tree(off1_tree))
#
#
# offspring_trees= []
# ### Draw and calculate fitness function ####
# for i, off_tree in enumerate(logical_off):
#     pos_off = draw_hierarchy_pos(off_tree, root=1, width=grid_size, height=grid_size)
#     plt.figure()
#     plt.title(f"The plot belongs to offsrping {i + 1} ")
#     nx.draw(off_tree, pos_off, with_labels=True)
#     plt.grid(True)
#     plt.pause(0.05)
#     plt.show()
#     offspring_trees.append(convert_logical_spatial(off_tree, pos_off))
#
# offspring_fitness= []
# for i, off_top in enumerate(offspring_trees):
#     offspring_fitness.append(fitness_function(off_top, Batch_sequence, PI_weight))
#
# print(offspring_fitness)
