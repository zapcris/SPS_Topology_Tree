import networkx as nx


def fitness_function(T, batch_list, PI_weight):
    PI_cost = []

    for i in range(len(batch_list)):
        cost = 0.0
        for j in range(len(batch_list[i]) - 1):
            #print(f"source{batch_list[i][j]}and target {batch_list[i][j + 1]}")
            cost += nx.dijkstra_path_length(T, batch_list[i][j], batch_list[i][j + 1])

        PI_cost.append(round(cost * (PI_weight[i] / 100)))

    batch_cost = sum(PI_cost)
    return batch_cost  # ,PI_cost
