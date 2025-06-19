from collections import deque
import heapq
import networkx as nx
from heapq import heapify, heappop, heappush
import time
import math
import sys
from buildHINs import buildDBLP, buildAmazon, buildAminer, buildDoubanMovie, buildFreebase


def normalized_edge(u, v):
    return (u, v) if u < v else (v, u)

def is_triangle_connected_all_edges(G):
    all_edges = set()
    for u in G.keys():
        for v in G[u]:
            all_edges.add(normalized_edge(u, v))
    start_edge = next(iter(all_edges))
    visited = {start_edge}
    queue = deque([start_edge])
    while queue:
        u, v = queue.popleft()
        common_neighbors = G[u].intersection(G[v])
        for w in common_neighbors:
            e1 = normalized_edge(u, w)
            e2 = normalized_edge(v, w)
            if e1 in all_edges and e1 not in visited:
                visited.add(e1)
                queue.append(e1)
            if e2 in all_edges and e2 not in visited:
                visited.add(e2)
                queue.append(e2)
    return len(visited) == len(all_edges)


def TCtruss(G):
    H = G.copy()
    all_edges = set()
    for u in G.keys():
        for v in G[u]:
            all_edges.add(normalized_edge(u, v))
    support = {
        normalized_edge(u, v): len(set(H[u]) & set(H[v]))
        for u, v in all_edges
    }
    truss = {}
    k = 3
    n = len(H.keys())
    last_support = dict()
    while len(H.keys()) == n and is_triangle_connected_all_edges(H):
        last_support = support.copy()
        removal_queue = deque(
            (u, v) for u, v in all_edges if support[normalized_edge(u, v)] < k - 2
        )
        while removal_queue:
            u, v = removal_queue.popleft()
            if not normalized_edge(u,v) in all_edges:
                continue
            del support[normalized_edge(u, v)]
            common_neighbors = set(H[u]) & set(H[v])
            for w in common_neighbors:
                e1 = normalized_edge(u, w)
                e2 = normalized_edge(v, w)
                if w in H[u] and e1 in support:
                    support[e1] -= 1
                    if support[e1] < k - 2:
                        removal_queue.append((u, w))
                if w in H[v] and e2 in support:
                    support[e2] -= 1
                    if support[e2] < k - 2:
                        removal_queue.append((v, w))
            H[u].remove(v)
            H[v].remove(u)
            all_edges.remove(normalized_edge(u, v))
            truss[normalized_edge(u, v)] = k - 1
        for i in list(H.keys()):
            if len(H[i]) == 0:
                del H[i]
        k += 1
    return k-2, last_support


def getCenternodes(G, q, P):
    X = {q}
    for t in P:
        X = {neighbor for node in X for neighbor in G.neighbors(node) if G.nodes[neighbor]['type'] == t}
        if not X:
            break
    return X

def getCenter2end(G,P):
    if len(P) % 2 == 1:
        P = P[:math.ceil(len(P)/2)]
        allCenters = []
        for n in G.nodes():
            if G.nodes[n]['type'] == P[-1]:
                allCenters.append(n)
        center2ends = dict()
        end2centers = dict()
        reversed_P = P[::-1][1:]
        for center in allCenters:
            ends = getCenternodes(G, center, reversed_P)
            center2ends[center] = ends
            for end in ends:
                if end not in end2centers.keys():
                    end2centers[end] = set()
                    end2centers[end].add(center)
                else:
                    end2centers[end].add(center)
    else:
        end2centersP = P[:int(len(P) / 2)]
        i = 1
        Gcopy = G.copy()
        allCenters = []
        for n in Gcopy.nodes():
            if not Gcopy.nodes[n]["type"] == end2centersP[-1]:
                continue
            for nei in Gcopy.neighbors(n):
                if not Gcopy.nodes[nei]["type"] == end2centersP[-1]:
                    continue
                G.add_node("I{}".format(i),type="center")
                allCenters.append("I{}".format(i))
                G.add_edge(n, "I{}".format(i))
                G.add_edge("I{}".format(i), nei)
                i+=1
        P = P[:math.ceil(len(P)/2)]
        P.append("center")
        center2ends = dict()
        end2centers = dict()
        reversed_P = P[::-1][1:]
        for center in allCenters:
            ends = getCenternodes(G, center, reversed_P)
            center2ends[center] = ends
            for end in ends:
                if end not in end2centers.keys():
                    end2centers[end] = set()
                    end2centers[end].add(center)
                else:
                    end2centers[end].add(center)
    return center2ends, end2centers

def getPneigbors(q):
    global center2ends,end2centers
    neigbors = set()
    if q in end2centers.keys():
        for center in end2centers[q]:
            neigbors = neigbors.union(center2ends[center])
        return neigbors-{q}
    else:
        return set()

def truss_decomposition(adj):
    local_adj = {u: set(neighbors) for u, neighbors in adj.items()}
    truss = {u: {} for u in local_adj}
    support = {}

    for u, neighbors in local_adj.items():
        for v in neighbors:
            if u < v:
                common = local_adj[u] & local_adj[v]
                s = len(common)
                support[(u, v)] = s
                k = s + 2
                truss[u][v] = k
                truss[v][u] = k

    heap = [(s, edge) for edge, s in support.items()]
    heapq.heapify(heap)

    def update_edge(x, y, s):
        key = (x, y) if x < y else (y, x)
        if key in support and support[key] > s:
            support[key] -= 1
            heapq.heappush(heap, (support[key], key))

    while heap:
        s, (u, v) = heapq.heappop(heap)
        if (u, v) not in support or support[(u, v)] != s:
            continue

        current_truss = s + 2
        truss[u][v] = current_truss
        truss[v][u] = current_truss

        local_adj[u].remove(v)
        local_adj[v].remove(u)
        del support[(u, v)]

        common = local_adj[u] & local_adj[v]
        for w in common:
            update_edge(u, w, s)
            update_edge(v, w, s)

    return truss


def normalize_edge(u, v):
    return (u, v) if u <= v else (v, u)

def get_common_neighbors(graph, a, b):
    if len(graph[a]) < len(graph[b]):
        return {w for w in graph[a] if w in graph[b]}
    else:
        return {w for w in graph[b] if w in graph[a]}


def find_triangle_connected_edges(C, graphC, nodeofR, graph):
    edge = C[0]
    start_edge = edge
    queue = deque([edge])
    visited_edges = {start_edge}
    graphCR = dict()
    graphCR[edge[0]] = {edge[1]}
    graphCR[edge[1]] = {edge[0]}
    CR = set(graphC.nodes()).union(nodeofR)
    nodeofR = set()
    while queue:
        a, b = queue.popleft()
        common = get_common_neighbors(graph, a, b) & CR
        for w in common:
            for edge in (normalize_edge(a, w), normalize_edge(b, w)):
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    queue.append(edge)
    for a,b in visited_edges:
        if a not in graphCR:
            graphCR[a] = set()
        if b not in graphCR:
            graphCR[b] = set()
        graphCR[a].add(b)
        graphCR[b].add(a)
        if normalize_edge(a,b) in C:
            continue
        nodeofR.add(a)
        nodeofR.add(b)
    return visited_edges.difference(set(C)), nodeofR, graphCR

def min_support(C,graph):
    min_support = float('inf')
    supports = dict()
    for u,v in C:
        support = len(set(graph[u]).intersection(set(graph[v])))
        supports[(u,v)] = support
        min_support = min(min_support, support)
    return min_support+2, supports

def getSubgraphedgeinduced(C):
    graph = nx.Graph()
    graph.add_edges_from(C)
    return graph


def branchcheck(C,Ry,graphC,graphCR):
    nodeofC = set(graphC.nodes())
    for e in Ry:
        if len(set(e)&nodeofC) == 2:
            graphC.add_edge(e[0],e[1])
        else:
            break
    supports = dict()
    for u,v in C:
        support = len(set(graphC[u]).intersection(set(graphC[v])))
        supports[(u,v)] = support
    edge_heap = [(score, edge) for edge, score in supports.items()]

    nodeofR = set(graphCR.keys()).difference(nodeofC)
    heapify(edge_heap)
    n2d = {}
    for a,b in C:
        trofnodes = graphCR[a]&graphCR[b]&nodeofR
        for node in trofnodes:
            if node not in n2d:
                n2d[node] = 0
            n2d[node] += 1
    n2d_sorted = sorted(n2d.items(), key=lambda x: x[1], reverse=True)
    for i in range(s-len(nodeofC)):
        if i >= len(n2d_sorted) or not edge_heap:
            break
        t = min(n2d_sorted[i][1], len(edge_heap))
        updated_edges = []
        for _ in range(t):
            if not edge_heap: break
            score, edge = heappop(edge_heap)
            updated_edges.append((score + 1, edge))

        for item in updated_edges:
            heappush(edge_heap, item)
    return edge_heap and edge_heap[0][0]+2 >= (ck + 1)



def enumerate_edges(C, Ry, graphCR, X):
    global ck,s,khat
    graphC = nx.Graph()
    graphC.add_edges_from(C)
    if len(graphC.nodes()) > s:
        return
    k,supports = min_support(C, graphC)
    print(k, ck, list(graphC.nodes()), graphC.number_of_edges(),len(list(graphC.nodes())), len(C), len(Ry))
    if k == khat and len(C) == s:
        global start_time
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time,ck," ".join(C)))
        sys.exit(0)

    if len(graphC.nodes()) == s:
        if k > ck:
            ck = k
    if len(set(graphC.nodes()))+len(set(graphCR.keys()).difference(set(graphC.nodes())))<=ck:
        return
    if k+len(Ry)<=ck:
        return
    edges = set(C)|set(Ry)

    GCR = dict()
    for u,v in edges:
        if u not in GCR:
            GCR[u] = set()
        GCR[u].add(v)
        if v not in GCR:
            GCR[v] = set()
        GCR[v].add(u)
    truss = truss_decomposition(graphCR)
    for u,v in C:
        try:
            if truss[u][v]<=ck:
                return
        except:
            return
    if not branchcheck(C,Ry,graphC.copy(),graphCR):
        return

    for ei in range(len(Ry)):
        e = Ry[ei]
        if not set(e) & set(graphC.nodes()):
            break
        new_C = C+[e]
        new_graphC = nx.Graph()
        new_graphC.add_edges_from(new_C)
        if len(new_graphC.nodes()) > s:
            break
        new_Ry = []
        new_nodeofR = set()
        roots = dict()
        for i in Ry[ei+1:]:
            if truss[i[0]][i[1]]<=ck:
                continue
            new_Ry.append(i)
            new_nodeofR.add(i[0])
            new_nodeofR.add(i[1])
            t = set(new_graphC.nodes())&set(i)
            if len(t) == 2:
                roots[i] = 2 + s + truss[i[0]][i[1]]/100
            elif len(t) == 1:
                tn = i[1] if i[0] in set(new_graphC.nodes()) else i[0]
                roots[i] = 1 + len(graphCR[tn]&set(new_graphC.nodes())) + truss[i[0]][i[1]] / 100
        roots = sorted(roots.items(), key=lambda x: x[1], reverse=True)
        roots = [i[0] for i in roots]
        new_R, new_nodeofRy, new_graphCR = find_triangle_connected_edges(new_C, new_graphC, new_nodeofR, graphCR.copy())
        new_Ry = list(new_R.difference(set(Ry[:ei+1])).difference(set(roots)))
        for ej in range(len(roots)-1, -1, -1):
            if roots[ej] in new_R:
                new_Ry.insert(0, roots[ej])
        enumerate_edges(new_C, new_Ry, new_graphCR.copy(), X)
        graphCR[e[0]].remove(e[1])
        graphCR[e[1]].remove(e[0])


def heuristic(q,s):
    global center2ends,end2centers,Gp
    centernodes = end2centers[q]
    Pneigbors = getPneigbors(q)
    lenlargestpstar = 0
    largestpstar = set()
    for i in centernodes:
        if len(center2ends[i])>lenlargestpstar:
            largestpstar = center2ends[i].copy()
            lenlargestpstar = len(center2ends[i])
    largestpstar.remove(q)
    largestpstar = list(largestpstar)
    if lenlargestpstar>=s:
        return s,[q]+largestpstar[:s]
    Pcenters = centernodes.copy()
    for n in Pneigbors:
        Pcenters = Pcenters.union(end2centers[n])
    usedc = set()
    C = set()
    pair_candidates = [
        (c, c2, center2ends[c] & center2ends[c2], center2ends[c] | center2ends[c2])
        for c in centernodes
        for c2 in Pcenters
        if c != c2
    ]
    cy, cyy, inter_set, union_set = max(
        pair_candidates,
        key=lambda x: (len(x[2]), len(x[3])),
        default=(None, None, set(), set())
    )

    C = union_set.copy()
    usedc.add(cy)
    usedc.add(cyy)

    while len(C) < s and not Pcenters.issubset(usedc):
        candidate = max(
            ((c2, C & center2ends[c2]) for c2 in Pcenters if c2 not in usedc),
            key=lambda x: len(x[1]),
            default=(None, set())
        )
        best_node, best_inter = candidate
        if best_node is None or len(best_inter) == 0:
            break
        usedc.add(best_node)
        C |= center2ends[best_node]
        Pcenters = set()
        for n in C:
            Pcenters = Pcenters.union(end2centers[n])
    T = dict()
    for i in C:
        neigbors = set()
        if i in end2centers.keys():
            for center in end2centers[i]:
                neigbors = neigbors.union(center2ends[center])
        neigbors = neigbors - {i}
        T[i] = neigbors&C
    while len(C) > s:
        trussness = truss_decomposition(T)
        minkv = 1000
        vk = 0
        for i in C:
            if not trussness[i]:
                kv = 0
            else:
                kv = min(trussness[i].values())
            if kv<minkv:
                minkv = kv
                vk = i
        C.remove(vk)
        del T[vk]
        for i in T:
            T[i] = T[i]-{vk}
    if len(C)<s:
        k=0
    else:
        k,support = TCtruss(T)
    return k,list(C)


def DBLP(metaPath, q, s):
    global center2ends, end2centers, ck, khat, node2upper, Gp, start_time, optC, timer
    G = buildDBLP()
    import threading,os
    def force_exit():
        print("P={},q={},s={},Runing time = {},optC={}".format(" ".join(metaPath),q, s, 3600," ".join(optC)))
        os._exit(1)
    timer = threading.Timer(3600, force_exit)
    timer.start()

    try:
        ck = 0
        start_time = time.time()
        center2ends, end2centers = getCenter2end(G, metaPath)
        ck = 0
        ck, optC = heuristic(q, s)
        if ck==s:
            end_time = time.time()
            print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time,ck," ".join(optC)))
            sys.exit(0)

        targetnodes = []
        for i in G.nodes():
            if G.nodes[i]['type'] == metaPath[0]:
                targetnodes.append(i)
        Gp = dict()
        node2upper = dict()
        R = set()
        for i in targetnodes:
            P = getPneigbors(i)
            Gp[i] = P
        truss = truss_decomposition(Gp)
        khat = min(max(truss[q].values()),s)
        roots = dict()
        for i in Gp[q]:
            roots[normalize_edge(q,i)] = truss[q][i]
        roots = sorted(roots.items(), key=lambda x: x[1], reverse=True)
        roots = [i[0] for i in roots]
        Ry = list(R)
        for ej in range(len(roots) - 1, -1, -1):
            Ry.insert(0, roots[ej])
        X = set()
        nodeofR = set(Gp.keys()).difference({q})
        for ei in range(len(roots)):
            e = roots[ei]
            if truss[e[0]][e[1]]<=ck:
                continue
            graphC = nx.Graph()
            graphC.add_edge(e[0],e[1])
            C = list()
            C.append(e)
            R, nodeofRy, graphCR = find_triangle_connected_edges(C, graphC, nodeofR, Gp)
            Ry = list(R.difference(set(roots)))
            for ej in range(len(roots)-1, ei, -1):
                if roots[ej] in R:
                    Ry.insert(0, roots[ej])
            Gp[e[0]].remove(e[1])
            Gp[e[1]].remove(e[0])
            if 2+len(nodeofRy)<=s:
                continue
            enumerate_edges(C, Ry, graphCR, X)
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time, ck,
                                                                    " ".join(optC)))
    finally:
        timer.cancel()


def Aminer(metaPath, q, s):
    global center2ends, end2centers, ck, khat, node2upper, Gp, start_time, optC, timer
    G = buildAminer()
    import threading,os
    def force_exit():
        print("P={},q={},s={},Runing time = {},optC={}".format(" ".join(metaPath),q, s, 3600," ".join(optC)))
        os._exit(1)
    timer = threading.Timer(3600, force_exit)
    timer.start()
    try:
        ck = 0
        start_time = time.time()
        center2ends, end2centers = getCenter2end(G, metaPath)
        ck = 0
        ck, optC = heuristic(q, s)
        if ck==s:
            end_time = time.time()
            print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time,ck," ".join(optC)))
            sys.exit(0)
        targetnodes = []
        for i in G.nodes():
            if G.nodes[i]['type'] == metaPath[0]:
                targetnodes.append(i)
        Gp = dict()
        node2upper = dict()
        R = set()
        for i in targetnodes:
            P = getPneigbors(i)
            Gp[i] = P
        truss = truss_decomposition(Gp)
        khat = min(max(truss[q].values()),s)
        roots = dict()
        for i in Gp[q]:
            roots[normalize_edge(q,i)] = truss[q][i]
        roots = sorted(roots.items(), key=lambda x: x[1], reverse=True)
        roots = [i[0] for i in roots]
        Ry = list(R)
        for ej in range(len(roots) - 1, -1, -1):
            Ry.insert(0, roots[ej])
        X = set()
        nodeofR = set(Gp.keys()).difference({q})
        for ei in range(len(roots)):
            e = roots[ei]
            if truss[e[0]][e[1]]<=ck:
                continue
            graphC = nx.Graph()
            graphC.add_edge(e[0],e[1])
            C = list()
            C.append(e)
            R, nodeofRy, graphCR = find_triangle_connected_edges(C, graphC, nodeofR, Gp)
            Ry = list(R.difference(set(roots)))
            for ej in range(len(roots)-1, ei, -1):
                if roots[ej] in R:
                    Ry.insert(0, roots[ej])
            Gp[e[0]].remove(e[1])
            Gp[e[1]].remove(e[0])
            if 2+len(nodeofRy)<=s:
                continue
            enumerate_edges(C, Ry, graphCR, X)
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time, ck,
                                                                    " ".join(optC)))
    finally:
        timer.cancel()

def Amazon(metaPath, q, s):
    global center2ends, end2centers, ck, khat, node2upper, Gp, start_time, optC, timer
    G = buildAmazon()
    import threading,os
    def force_exit():
        print("P={},q={},s={},Runing time = {},optC={}".format(" ".join(metaPath),q, s, 3600," ".join(optC)))
        os._exit(1)
    timer = threading.Timer(3600, force_exit)
    timer.start()
    try:
        ck = 0
        start_time = time.time()
        center2ends, end2centers = getCenter2end(G, metaPath)
        ck = 0
        ck, optC = heuristic(q, s)
        if ck==s:
            end_time = time.time()
            print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time,ck," ".join(optC)))
            sys.exit(0)
        targetnodes = []
        for i in G.nodes():
            if G.nodes[i]['type'] == metaPath[0]:
                targetnodes.append(i)
        Gp = dict()
        node2upper = dict()
        R = set()
        for i in targetnodes:
            P = getPneigbors(i)
            Gp[i] = P
        truss = truss_decomposition(Gp)
        khat = min(max(truss[q].values()),s)
        roots = dict()
        for i in Gp[q]:
            roots[normalize_edge(q,i)] = truss[q][i]
        roots = sorted(roots.items(), key=lambda x: x[1], reverse=True)
        roots = [i[0] for i in roots]
        Ry = list(R)
        for ej in range(len(roots) - 1, -1, -1):
            Ry.insert(0, roots[ej])
        X = set()
        nodeofR = set(Gp.keys()).difference({q})
        for ei in range(len(roots)):
            e = roots[ei]
            if truss[e[0]][e[1]]<=ck:
                continue
            graphC = nx.Graph()
            graphC.add_edge(e[0],e[1])
            C = list()
            C.append(e)
            R, nodeofRy, graphCR = find_triangle_connected_edges(C, graphC, nodeofR, Gp)
            Ry = list(R.difference(set(roots)))
            for ej in range(len(roots)-1, ei, -1):
                if roots[ej] in R:
                    Ry.insert(0, roots[ej])
            Gp[e[0]].remove(e[1])
            Gp[e[1]].remove(e[0])
            if 2+len(nodeofRy)<=s:
                continue
            enumerate_edges(C, Ry, graphCR, X)
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time, ck,
                                                                    " ".join(optC)))
    finally:
        timer.cancel()

def DoubanMovie(metaPath, q, s):
    global center2ends, end2centers, ck, khat, node2upper, Gp, start_time, optC, timer
    G = buildDoubanMovie()
    import threading,os
    def force_exit():
        print("P={},q={},s={},Runing time = {},optC={}".format(" ".join(metaPath),q, s, 3600," ".join(optC)))
        os._exit(1)
    timer = threading.Timer(3600, force_exit)
    timer.start()
    try:
        start_time = time.time()
        center2ends, end2centers = getCenter2end(G, metaPath)
        ck = 0
        ck, optC = heuristic(q, s)
        print(ck)
        if ck==s:
            end_time = time.time()
            print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time,ck," ".join(optC)))
            sys.exit(0)
        targetnodes = []
        for i in G.nodes():
            if G.nodes[i]['type'] == metaPath[0]:
                targetnodes.append(i)
        Gp = dict()
        node2upper = dict()
        R = set()
        for i in targetnodes:
            P = getPneigbors(i)
            Gp[i] = P
        truss = truss_decomposition(Gp)
        khat = min(max(truss[q].values()),s)
        roots = dict()
        for i in Gp[q]:
            roots[normalize_edge(q,i)] = truss[q][i]
        roots = sorted(roots.items(), key=lambda x: x[1], reverse=True)
        roots = [i[0] for i in roots]
        Ry = list(R)
        for ej in range(len(roots) - 1, -1, -1):
            Ry.insert(0, roots[ej])
        X = set()
        nodeofR = set(Gp.keys()).difference({q})
        for ei in range(len(roots)):
            e = roots[ei]
            if truss[e[0]][e[1]]<=ck:
                continue
            graphC = nx.Graph()
            graphC.add_edge(e[0],e[1])
            C = list()
            C.append(e)
            R, nodeofRy, graphCR = find_triangle_connected_edges(C, graphC, nodeofR, Gp)
            Ry = list(R.difference(set(roots)))
            for ej in range(len(roots)-1, ei, -1):
                if roots[ej] in R:
                    Ry.insert(0, roots[ej])
            Gp[e[0]].remove(e[1])
            Gp[e[1]].remove(e[0])
            if 2+len(nodeofRy)<=s:
                continue
            enumerate_edges(C, Ry, graphCR, X)
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time, ck,
                                                                    " ".join(optC)))
    finally:
        timer.cancel()

def Freebase(metaPath, q, s):
    global center2ends, end2centers, ck, khat, node2upper, Gp, start_time, optC, timer
    G = buildFreebase()
    import threading,os
    def force_exit():
        print("P={},q={},s={},Runing time = {},optC={}".format(" ".join(metaPath),q, s, 3600," ".join(optC)))
        os._exit(1)
    timer = threading.Timer(3600, force_exit)
    timer.start()
    try:
        ck = 0
        start_time = time.time()
        center2ends, end2centers = getCenter2end(G, metaPath)
        ck = 0
        ck, optC = heuristic(q, s)
        if ck==s:
            end_time = time.time()
            print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time,ck," ".join(optC)))
            sys.exit(0)
        targetnodes = []
        for i in G.nodes():
            if G.nodes[i]['type'] == metaPath[0]:
                targetnodes.append(i)
        Gp = dict()
        node2upper = dict()
        R = set()
        for i in targetnodes:
            P = getPneigbors(i)
            Gp[i] = P
        truss = truss_decomposition(Gp)
        khat = min(max(truss[q].values()),s)
        roots = dict()
        for i in Gp[q]:
            roots[normalize_edge(q,i)] = truss[q][i]
        roots = sorted(roots.items(), key=lambda x: x[1], reverse=True)
        roots = [i[0] for i in roots]
        Ry = list(R)
        for ej in range(len(roots) - 1, -1, -1):
            Ry.insert(0, roots[ej])
        X = set()
        nodeofR = set(Gp.keys()).difference({q})
        for ei in range(len(roots)):
            e = roots[ei]
            if truss[e[0]][e[1]]<=ck:
                continue
            graphC = nx.Graph()
            graphC.add_edge(e[0],e[1])
            C = list()
            C.append(e)
            R, nodeofRy, graphCR = find_triangle_connected_edges(C, graphC, nodeofR, Gp)
            Ry = list(R.difference(set(roots)))
            for ej in range(len(roots)-1, ei, -1):
                if roots[ej] in R:
                    Ry.insert(0, roots[ej])
            Gp[e[0]].remove(e[1])
            Gp[e[1]].remove(e[0])
            if 2+len(nodeofRy)<=s:
                continue
            enumerate_edges(C, Ry, graphCR, X)
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath), q, s, end_time - start_time, ck,
                                                                    " ".join(optC)))
    finally:
        timer.cancel()

if __name__ == '__main__':
    global s, metaPath, timer
    # metaPath = ["actor", "movie","actor"]
    # q = "a4791"
    # s = 15
    # DoubanMovie(metaPath, q, s)
    if sys.argv[1] == "DBLP":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        DBLP(metaPath,q,s)
    elif sys.argv[1] == "Amazon":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        Amazon(metaPath,q,s)
    elif sys.argv[1] == "Aminer":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        Aminer(metaPath,q,s)
    elif sys.argv[1] == "DoubanMovie":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        DoubanMovie(metaPath,q,s)
    elif sys.argv[1] == "Freebase":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        Freebase(metaPath,q,s)