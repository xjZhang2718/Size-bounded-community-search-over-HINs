import math
import sys
import time
import copy
from heapq import heapify, heappop, heappush
from collections import deque

from buildHINs import buildAmazon, buildDBLP, buildAminer, buildDoubanMovie, buildFreebase
from enumerateEdge import truss_decomposition,TCtruss,normalize_edge

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
        usedn = set()
        for n in Gcopy.nodes():
            if not Gcopy.nodes[n]["type"] == end2centersP[-1]:
                continue
            for nei in Gcopy.neighbors(n):
                if not Gcopy.nodes[nei]["type"] == end2centersP[-1]:
                    continue
                if nei in usedn:
                    continue
                G.add_node("I{}".format(i),type="center")
                allCenters.append("I{}".format(i))
                G.add_edge(n, "I{}".format(i))
                G.add_edge("I{}".format(i), nei)
                i+=1
            usedn.add(n)
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

def getSubgraph(inducedNodes):
    global Gp
    Ty = set(inducedNodes)
    T = dict()
    for i in Ty:
        T[i] = Gp[i]&Ty
    return T

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
    S = sorted(centernodes,key=lambda x:len(center2ends[x]), reverse=True)
    Pcenters = centernodes.copy()
    for n in Pneigbors:
        Pcenters = Pcenters.union(end2centers[n])
    usedi = set()
    ok = 0
    oC = []
    while S:
        t = S.pop(0)
        usedi.add(t)
        C = center2ends[t].copy()
        Sy = Pcenters.union(set(S)).difference(usedi)
        while len(C) < s and Sy:
            candidate = max(
                ((c2, C & center2ends[c2]) for c2 in Sy),
                key=lambda x: len(x[1]),
                default=(None, set())
            )
            best_node, best_inter = candidate
            if best_node is None or len(best_inter) == 0:
                break
            Sy.remove(best_node)
            C |= center2ends[best_node]

        T = dict()
        for i in C:
            T[i] = getPneigbors(i).intersection(C)
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
        if len(C)<s or q not in C:
            k=0
        else:
            k,support = TCtruss(T)
        if k>ok:
            ok = k
            oC = list(T.keys())
    return ok,oC

def k_truss_peeling_order(adj):
    peeling_order = []
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
    support1 = support.copy()
    heap = [(s, edge) for edge, s in support.items()]
    heapify(heap)
    def update_edge(x, y, s):
        key = (x, y) if x < y else (y, x)
        if key in support and support[key] > s:
            support[key] -= 1
            heappush(heap, (support[key], key))
    while heap:
        s, (u, v) = heappop(heap)
        if (u, v) not in support or support[(u, v)] != s:
            continue

        current_truss = s + 2
        truss[u][v] = current_truss
        truss[v][u] = current_truss

        local_adj[u].remove(v)
        local_adj[v].remove(u)
        peeling_order.append((v, u))
        del support[(u, v)]

        common = local_adj[u] & local_adj[v]
        for w in common:
            update_edge(u, w, s)
            update_edge(v, w, s)
    return truss,peeling_order

def upperbound(k,C,R):
    global Gp,s
    graphC = getSubgraph(C)
    truss, peeling_order = k_truss_peeling_order(graphC)
    setC = set(C)
    n2d = dict()
    for i in R:
        t = Gp[i]&setC
        if t:
            c = 0
            for j in t:
                c += len(graphC[j]&t)
            if c==0:
                continue
            n2d[i] = int(c/2)
    n2d = sorted(n2d.items(), key=lambda x: x[1], reverse=True)
    mink = 100
    for cnode in C:
        truss1 = copy.deepcopy(truss)
        k = max(truss1[cnode].values())
        uk = k
        c = len(C)
        m = min(s-c, len(n2d))
        sumc = c
        sntri = 0
        for i in range(m):
            n, ntri = n2d[i]
            sntri = sntri + ntri
            needtri = int((uk*uk+uk)/2)
            needtri = needtri - sumc
            a = 0
            for j in peeling_order:
                if truss1[j[0]][j[1]]>=uk+1:
                    a += 1
            if a >= needtri:
                uk = uk + 1
            else:
                a = 0
                for j in peeling_order:
                    if truss1[j[0]][j[1]] < uk + 1:
                        sntri = sntri-uk-1+truss1[j[0]][j[1]]
                        truss1[j[0]][j[1]] = uk + 1
                        truss1[j[1]][j[0]] = uk + 1
                    a += 1
                    if a == needtri and sntri>=0:
                        uk = uk + 1
                        break
            sumc = sumc+c+1
            c+=1
        if uk < mink:
            mink = uk
    return mink

def reductionbydistance(C,R):
    global Gp,ck,s
    GCR = dict()
    new_R = set(R)
    t = set(C).union(R)
    dupper = math.floor(2*s/(ck+1))
    for i in t:
        GCR[i] = Gp[i]&t
    for i in C:
        t = set()
        distance = {i: 0}
        queue = deque([i])
        while queue:
            current = queue.popleft()
            for neighbor in GCR[current]:
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)
        for j in R:
            if j in distance and distance[j]<=dupper:
                t.add(j)
        new_R = new_R&t
    return new_R

def get_common_neighbors(truss,a,b):
    global ck
    Aneighbors = set()
    for node,k in truss[a].items():
        if k<=ck:
            continue
        else:
            Aneighbors.add(node)
    Bneighbors = set()
    for node,k in truss[b].items():
        if k<=ck:
            continue
        else:
            Bneighbors.add(node)
    return Aneighbors & Bneighbors

def getTCnodes(C, R, GCRtruss):
    edge = normalize_edge(C[0], C[1])
    start_edge = edge
    queue = deque([edge])
    visited_edges = {start_edge}
    CR = set(C)|set(R)
    while queue:
        a, b = queue.popleft()
        common = get_common_neighbors(GCRtruss, a, b) & CR
        for w in common:
            for edge in (normalize_edge(a, w), normalize_edge(b, w)):
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    queue.append(edge)
        if normalize_edge(a,b) in C:
            continue
    nodes = set()
    for i in visited_edges:
        nodes = nodes.union(i)
    return nodes&set(R)

def enumerate_combinations(q, R, s):
    C = [q]
    def backtrack(C, R, X):
        global ck,optC
        if len(C)>=2:
            T = getSubgraph(C)
            k, support = TCtruss(T)
        else:
            k=0
        print(k, ck, C, len(R))
        if k == khat and len(C) == s:
            global start_time,metaPath
            end_time = time.time()
            print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath),q, s, end_time - start_time,khat," ".join(C)))
            sys.exit(0)
        GCR = dict()
        CR = set(C) | set(R)
        for _ in CR:
            GCR[_] = Gp[_] & CR
        GCRtruss = truss_decomposition(GCR)
        for i in C:
            if not GCRtruss[i]:
                return
            elif max(GCRtruss[i].values())<=ck:
                return
        if len(C)+len(R)<s:
            return
        if k+min(s-len(C),len(R)) <= ck:
            return
        if k>0:
            kk = upperbound(k,C,R)
            if kk<=ck:
                return
        if len(C) == s:
            if k>ck:
                ck = k
                optC = C
            return

        for i in range(len(R)):
            if R[i] in X:
                continue
            if not GCR[R[i]]&set(C):
                break
            new_C = C+[R[i]]

            new_R = []
            for node in R[i+1:]:
                if GCRtruss[node]:
                    h = max(GCRtruss[node].values())
                else:
                    h = 0
                if h<=ck:
                    X.add(node)
                    continue
                new_R.append(node)
            if ck>0:
                new_R = reductionbydistance(new_C,new_R)
            new_R = getTCnodes(new_C, new_R, GCRtruss)
            new_R = list(new_R.difference(set(new_C)).difference(X))
            new_Ry = []
            triangleheap = []
            vi = 0
            heapify(triangleheap)
            while vi < len(new_R):
                v = new_R[vi]
                if not Gp[v] & set(C):
                    new_Ry.append(v)
                    new_R.remove(v)
                else:
                    neighbors = Gp[v] & set(C)
                    yy = 0
                    for y in neighbors:
                        yy += len(Gp[y] & set(C))
                    if GCRtruss[v]:
                        h = max(GCRtruss[v].values())
                    else:
                        h = 0
                    heappush(triangleheap, (int(yy / 2)+h/100,v))
                    new_R.remove(v)
            while triangleheap:
                score, node = heappop(triangleheap)
                new_Ry.insert(0, node)
            backtrack(new_C, new_Ry, X.copy())
            v_centers = end2centers[R[i]]
            to_remove = set()
            for node in R[i:]:
                if node in end2centers.keys():
                    if end2centers[node].issubset(v_centers):
                        to_remove.add(node)
            for node in to_remove:
                X.add(node)
    backtrack(C, R, set())

def beginSearch(G,metaPath,q):
    global center2ends, end2centers, ck, khat, node2upper, Gp, start_time, optC, timer,s
    import threading,os
    optC = []
    def force_exit():
        print("P={},q={},s={},Runing time = {:.6f},optC={}".format(" ".join(metaPath),q, s, 3600," ".join(optC)))
        os._exit(1)
    timer = threading.Timer(3600, force_exit)
    timer.start()
    try:
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
        for i in targetnodes:
            P = getPneigbors(i)
            if not P:
                continue
            Gp[i] = P
        truss = truss_decomposition(Gp)
        khat = max(truss[q].values())
        R = list(Gp.keys())
        R.remove(q)
        newR = []
        tt = []
        heapify(tt)
        for i in R:
            if not i in Gp[q] or not truss[i]:
                newR.append(i)
            else:
                h = max(truss[i].values())
                heappush(tt, (h, i))
        while tt:
            score, node = heappop(tt)
            newR.insert(0, node)
        enumerate_combinations(q, newR, s)
        end_time = time.time()
        print("P={},q={},s={},Runing time = {:.6f},k={},optC={}".format(" ".join(metaPath),q, s, end_time - start_time,ck," ".join(optC)))
    finally:
        timer.cancel()

def Amazon(metaPath, q):
    G = buildAmazon()
    beginSearch(G,metaPath, q)

def DBLP(metaPath, q):
    G = buildDBLP()
    beginSearch(G,metaPath, q)

def Aminer(metaPath, q):
    G = buildAminer()
    beginSearch(G,metaPath, q)

def DoubanMovie(metaPath, q):
    G = buildDoubanMovie()
    beginSearch(G,metaPath, q)

def Freebase(metaPath, q):
    G = buildFreebase()
    beginSearch(G,metaPath, q)

if __name__ == '__main__':
    global s,metaPath,timer
    # metaPath = ["item","view","item"]
    # q = "i1052"
    # s = 21
    # Amazon(metaPath, q)
    if sys.argv[1] == "DBLP":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        DBLP(metaPath,q)
    elif sys.argv[1] == "Amazon":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        Amazon(metaPath,q)
    elif sys.argv[1] == "Freebase":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        Freebase(metaPath,q)
    elif sys.argv[1] == "Aminer":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        Aminer(metaPath,q)
    elif sys.argv[1] == "DoubanMovie":
        metaPath = sys.argv[2].split(",")
        q = sys.argv[3]
        s = int(sys.argv[4])
        DoubanMovie(metaPath,q)