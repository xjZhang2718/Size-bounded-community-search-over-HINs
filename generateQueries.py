from random import shuffle

from buildHINs import buildDBLP,buildAmazon,buildAminer,buildDoubanMovie,buildFreebase
from enumerateVertex import getCenter2end

def getPneigbors(q):
    global center2ends,end2centers
    neigbors = set()
    if q in end2centers.keys():
        for center in end2centers[q]:
            neigbors = neigbors.union(center2ends[center])
        return neigbors-{q}
    else:
        return set()

dataname = "Amazon"
G = buildAmazon()
type2nodes = dict()
for n in G.nodes:
    if G.nodes[n]['type'] not in type2nodes:
        type2nodes[G.nodes[n]['type']] = []
        type2nodes[G.nodes[n]['type']].append(n)
    else:
        type2nodes[G.nodes[n]['type']].append(n)
queries = []
with open("datasets/{}/P_pool.txt".format(dataname),"r") as f:
    for line in f.readlines():
        line = line.strip().split(",")
        print(line)
        shuffle(type2nodes[line[0]])
        center2ends, end2centers = getCenter2end(G, line)
        Gp = dict()
        for i in type2nodes[line[0]]:
            P = getPneigbors(i)
            if not P:
                continue
            Gp[i] = P
        nodes = list(Gp.keys())
        shuffle(nodes)
        querynodes = nodes[:10]
        for q in querynodes:
            for s in [9,12,15,18,21]:
                queries.append("{} {} {} {}".format(dataname,",".join(line),q,s))
with open("{}_queries.txt".format(dataname),"w") as f:
    for query in queries:
        f.write(query+"\n")