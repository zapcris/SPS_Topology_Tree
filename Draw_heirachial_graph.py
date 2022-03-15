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


# def hierarchy_pos2(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
#     '''If there is a cycle that is reachable from root, then result will not be a hierarchy.
#
#        G: the graph
#        root: the root node of current branch
#        width: horizontal space allocated for this branch - avoids overlap with other branches
#        vert_gap: gap between levels of hierarchy
#        vert_loc: vertical location of root
#        xcenter: horizontal location of root
#     '''
#
#     def h_recur(G, node=root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
#                 pos=None, parent=None, parsed=[]):
#         if (node not in parsed):
#             parsed.append(root)
#             if pos == None:
#                 pos = {node: (xcenter, vert_loc)}
#             else:
#                 pos[node] = (xcenter, vert_loc)
#             neighbors = G.neighbors(node)
#             if parent != None:
#                 neighbors.remove(parent)
#             if len(list(neighbors)) != 0:
#                 print("list of neighbors:", root, len(list(neighbors)))
#                 dx = width / len(list(neighbors))
#                 nextx = xcenter - width / 2 - dx / 2
#                 for neighbor in neighbors:
#                     nextx += dx
#                     pos = h_recur(G, neighbor, width=dx, vert_gap=vert_gap,
#                                   vert_loc = vert_loc - vert_gap, xcenter=nextx, pos=pos,
#                                   parent=root, parsed=parsed)
#         return pos
#
#     return h_recur(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5)
