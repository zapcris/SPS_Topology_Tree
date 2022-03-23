from shapely.geometry import MultiLineString, LineString
from itertools import combinations

multiline = MultiLineString([[(614633.1598889811, 6614684.232110311), (614585.0239559432, 6615176.69973293), (614244.3696605981, 6615210.024609649), (614174.0171430812, 6615058.211282375)],
                             [(614849.2836035677, 6614574.273030049), (615163.3363697577, 6614591.624011607), (615477.7302093033, 6614608.993836996), (615475.8039105758, 6614892.159749944),
                              (615474.6318041045, 6615064.459401229), (614967.3343471865, 6615119.389699113)],
                             [(615054.1363645532, 6614185.399992246), (615163.3363697577, 6614591.624011607), (615227.7403992868, 6614831.207001455), (615475.8039105758, 6614892.159749944),
                              (615835.3545208545, 6614980.506471327), (615867.958614701, 6615021.869873968), (615474.6318041045, 6615064.459401229), (615474.2581286087, 6615119.389699113),
                              (615286.7657710963, 6615227.024200648)],
                             [(616057.5676853136, 6615001.338955494), (615867.958614701, 6615021.869873968), (616067.9839273975, 6615275.633330373)]])


for line1, line2 in combinations([line for line in multiline],2):
    if line1.intersects(line2):
        print(line1.intersection(line2))


line1 = LineString([(0,0), (5,5)])

line2 = LineString([(5,0), (0,5)])

line3 = LineString([(1,0), (5,4)])

line4 = LineString([(4,0), (0,4)])

#print(line1.intersection(line4))
if line1.intersects(line3):
    print("done")


"Testing the edges "

total_edges = MultiLineString([[(0,0), (5,5)],
                                [(5,0), (0,5)],
                                [(1,0), (5,4)],
                                [(4,0), (0,4)],
                                ])
count = 0
for line1, line2 in combinations([line for line in total_edges],2):

    if line1.intersects(line2):
        print(line1.intersection(line2))
        count += 1

print(count)



G_pos = {1: (10, 7), 2: (19, 13), 3: (21, 10), 4: (21, 17), 5: (13, 0),
         6: (17, 3), 7: (21, 25), 8: (11, 4), 9: (15, 10), 10: (17, 19),
         11: (28, 7), 12: (33, 12), 13: (24, 13), 14: (0, 3), 15: (25, 22),
         17: (27, 25), 19: (32, 31), 20: (26, 33)}

Batch_sequence = [1, 5, 9, 10, 2, 11, 13, 15, 7, 20]


edge_list = []
edge_pos_list = []
for i in range(len(Batch_sequence)-1):
    edge = [Batch_sequence[i], Batch_sequence[i+1]]
    edge_pos = [G_pos[Batch_sequence[i]], G_pos[Batch_sequence[i+1]]]
    edge_list.append(edge)
    edge_pos_list.append(edge_pos)

print(edge_list)
print(edge_pos_list)

new_edges = MultiLineString(edge_pos_list)

c = 0

for line1, line2 in combinations([line for line in new_edges],2):

    if line1.intersects(line2):
        print(line1.intersection(line2))
        c += 1

print(c)
