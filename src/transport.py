from copy import deepcopy
from random import random, choice
from ot.gromov import fused_gromov_wasserstein2

from multiprocessing import Process, Lock, Value, Array

import numpy as np

from time import time
from datetime import timedelta

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

def fgw(C1, C2, M, h1, h2, alpha=0.5, Niter=100, verbose=True, text=""):

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
    return min_dist

def one_one_fgw(class1, class2, d, Cs1, Cs2, hs1, hs2, alpha=0.5, Niter=100):
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
        print("\r")
    print()
    return D

def one_one_parralelised(class1, class2, d, Cs1, Cs2, hs1, hs2, alpha=0.5, Niter=100, Nprocess=7):
    
    D = Array('d', len(class1)*len(class2))
    i = Value('i', 0)
    j = Value('i', 0)
    N_done = Value('i', 0)
    coordinates_lock = Lock()

    symetric = class1 == class2 and Cs1 == Cs2 and hs1 == hs2

    def f(D, i, j, N_done):
        while i.value < len(class1):
            coordinates_lock.acquire()
            i0 = i.value
            j0 = j.value
            j.value += 1
            if j.value >= len(class2):
                i.value += 1
                j.value = i.value if symetric else 0
            ndone = N_done.value
            N_done.value += 1
            coordinates_lock.release()

            if ndone == 0:
                remaining_time = "?"
            else:
                remaining_time = str(timedelta(seconds=((N_total_to_do-ndone)/ndone)*(time()-t_start))).split(':')
                remaining_time = f"{remaining_time[0]} h {remaining_time[1]} min {remaining_time[2].split('.')[0]} s"
            print(f"i : {i0+1}/{len(class1)} --- j : {j0+1}/{len(class2)}  | Estimated Remaining Time : {remaining_time}          ", end="\r")

            if symetric and i0 == j0:
                D[i0*len(class1)+j0] = 0
                continue

            G1 = class1[i0]
            G2 = class2[j0]
            C1 = Cs1[i0]
            C2 = Cs2[j0]
            h1 = hs1[i0]
            h2 = hs2[j0]
            M = node_dists(G1, G2, d)
            f = fgw(C1, C2, M, h1, h2, alpha, Niter, verbose=False)
            D[i0*len(class2)+j0] = f
            if symetric:
                D[j0*len(class2)+i0] = f

    processes = []
    for _ in range(Nprocess):
        p = Process(target=f, args=(D, i, j, N_done))
        processes.append(p)
    
    N_total_to_do = len(class1)*len(class2) if not symetric else len(class1)*(len(class1)-1)/2
    t_start = time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    D = np.array(D).reshape((len(class1), len(class2)))
    print()
    return D

