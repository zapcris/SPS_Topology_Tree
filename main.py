import numpy as np
from gird import *
import matplotlib.pyplot as plt

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
print(di)

# create list to be filled in diagonal of grid matrix

interval = len(Grid.diagonal()) // len(nList_diag)

print(interval)

print(len(Grid.diagonal()))

diag_seq = [0 for n in Grid.diagonal()]

for i in range(0, len(Grid.diagonal()), 3):
    print(i)
    diag_seq[i] = 6

# for n in nList_diag:
#     for i in range(0, len(Grid.diagonal()) , 3):
#         diag_seq[i] = nList_diag[i]

print(diag_seq)
