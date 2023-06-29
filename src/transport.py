from copy import deepcopy
from random import random, choice
from ot.gromov import fused_gromov_wasserstein2

from multiprocessing import Process, Lock, Value, Array

import numpy as np

def node_dists(G1, G2, d):
    M = np.zeros((G1.number_of_nodes(), G2.number_of_nodes()))

    for i, u in enumerate(G1.nodes):
        for j, v in enumerate(G2.nodes):
            M[i, j] = d(G1.nodes[u]['x'], G2.nodes[v]['x'])
    return M

def greedy_random_transport_plan(p, q, e=1e-10):
    n = p.shape[0]
    m = q.shape[0]
    e /= n
    tp = deepcopy(p)
    tq = deepcopy(q)
    ixp = [i for i in range(n)]
    ixq = [i for i in range(m)]
    T = np.zeros((n, m))
    
    while len(ixp) > 0 and len(ixq) > 0:
        ip = choice(ixp)
        iq = choice(ixq)

        if tp[ip] < e or tq[iq] < e:
            #We transfer everything that we can
            t = min(tp[ip], tq[iq])
        else:
            t = random()*min(tp[ip], tq[iq])

        tp[ip] -= t
        tq[iq] -= t
        T[ip, iq] += t

        if tp[ip] == 0:
            ixp.pop(ixp.index(ip))
        if tq[iq] == 0:
            ixq.pop(ixq.index(iq))

    return T

def fgw(C1, C2, M, h1, h2, alpha=0.5, Niter=5000, verbose=True, text=""):

    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    #First try with the "hard coded" transport plan
    min_dist = fused_gromov_wasserstein2(M, C1, C2, h1, h2, alpha=alpha) 
    
    for i in range(Niter):
        printv(f"{text} Iteration {i+1}/{Niter}", end="\r")
        G0 = greedy_random_transport_plan(h1, h2)
        try:
            f = fused_gromov_wasserstein2(M, C1, C2, h1, h2, alpha=alpha, G0=G0)
        except AssertionError:
            printv("A greedy generated transport plan was not exact")
            continue
        if (f < min_dist or min_dist < 0) and f >= 0:
            min_dist = f
    #printv(f"FGW computation completed                                                                          ")
    return min_dist

def one_one_fgw(class1, class2, d, Cs1, Cs2, hs1, hs2, alpha=0.5, Niter=1000):
    D = np.zeros((len(class1), len(class2)))
    for i, G1 in enumerate(class1):
        for j, G2 in enumerate(class2):
            #print(f"i : {i+1}/{len(class1)} ------ j : {j+1}/{len(class2)}", end="\n")
            C1 = Cs1[i]
            C2 = Cs2[j]
            h1 = hs1[i]
            h2 = hs2[j]
            M = node_dists(G1, G2, d)
            D[i, j] = fgw(C1, C2, M, h1, h2, alpha, Niter, text=f"i : {i+1}/{len(class1)} ------ j : {j+1}/{len(class2)} => ")
        print("\r\b\r")
    print()
    return D

def one_one_parralelised(class1, class2, d, Cs1, Cs2, hs1, hs2, alpha=0.5, Niter=1000, Nprocess=5):
    D = Array('d', len(class1)*len(class2))
    i = Value('i', 0)
    j = Value('i', 0)
    coordinates_lock = Lock()

    def f(k, D, i, j):
        while i.value < len(class1):
            coordinates_lock.acquire()
            i0 = i.value
            j0 = j.value
            j.value += 1
            if j.value >= len(class2):
                j.value = 0
                i.value += 1
            coordinates_lock.release()

            """             if i0 >= len(class1):
                break """

            print(f"i : {i0+1}/{len(class1)} --- j : {j0+1}/{len(class2)}", end="\r")

            G1 = class1[i0]
            G2 = class2[j0]
            C1 = Cs1[i0]
            C2 = Cs2[j0]
            h1 = hs1[i0]
            h2 = hs2[j0]
            M = node_dists(G1, G2, d)
            D[i0*len(class2)+j0] = fgw(C1, C2, M, h1, h2, alpha, 1000, verbose=False)

    processes = []
    for k in range(Nprocess):
        p = Process(target=f, args=(k, D, i, j))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    D = np.array(D).reshape((len(class1), len(class2)))
    return D
        