import networkx as nx

from Func_weighted_spanning_tree import euclidean_dist


def fitness_function(T, batch_list, PI_weight,pos):
    PI_cost = []
    EU_cost = []

    for i in range(len(batch_list)):
        cost = 0.0
        cost2 = 0.0
        for j in range(len(batch_list[i]) - 1):
            #print(f"source{batch_list[i][j]}and target {batch_list[i][j + 1]}")
            cost += nx.dijkstra_path_length(T, batch_list[i][j], batch_list[i][j + 1])
            cost2 += euclidean_dist(pos[batch_list[i][j]][0],pos[batch_list[i][j]][1],pos[batch_list[i][j+1]][0], pos[batch_list[i][j+1]][1])


        #PI_cost.append(round(cost * (PI_weight[i] / 100)))
        PI_cost.append(cost)
        EU_cost.append(round(cost2 * (PI_weight[i] / 100)))

    batch_tree_cost = sum(PI_cost)
    batch_eu_cost = sum(EU_cost)
    return batch_tree_cost, PI_cost # batch_eu_cost   # ,PI_cost
