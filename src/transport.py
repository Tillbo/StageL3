from copy import deepcopy
from random import random, choice
from ot.gromov import fused_gromov_wasserstein2, fused_gromov_wasserstein

from multiprocessing import Process, Lock, Value, Array

import numpy as np

from time import time
from datetime import timedelta

def node_dists(G1, G2, d):
    """
    Return the matrix distance between two graphs

    Args :
    - G1 (nx.Graph) : first graph
    - G2 (nx.Graph) : second graph
    - d : real valued distance function
    """
    M = np.zeros((G1.number_of_nodes(), G2.number_of_nodes()))

    for i, u in enumerate(G1.nodes):
        for j, v in enumerate(G2.nodes):
            M[i, j] = d(G1.nodes[u]['x'], G2.nodes[v]['x'])
    return M

def greedy_random_transport_plan(p, q, e=1e-10):
    """
    Computes a random transport plan between two distributions with a greedy method.

    Args :
    - p : source histogram
    - q : target histogram
    - e (float, optionnal : default = 1e-10) : threshold for ending a transport

    Output :
    - T (np.array) : transport matrix
    """
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

def fgw(C1, C2, M, h1, h2, alpha=0.5, Niter=100, verbose=True):
    """
    Compute FGW distance

    Args :
    - C1 (np.array) : first structre matrix
    - C2 (np.array) : second structre matrix
    - M (np.array) : distance matrix
    - h1 (array like) : first histogram
    - h2 (array like) : second histogram
    - alpha (float, default : 0.5) : alpha coefficient for FGW
    - Niter (int, default : 100) : Number of iterations starting from a greedy random transport plan
    - verbose (boolean, default : True) : prints information about the computation (number of iteration and if a greedy transport plan is not correct)

    Output : 
    - fgw (float) : FGW distance between the two structured data
    """
    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    #First try with the "hard coded" transport plan
    min_dist = fused_gromov_wasserstein2(M, C1, C2, h1, h2, alpha=alpha)
    
    for i in range(Niter):
        printv(f"Iteration {i+1}/{Niter}", end="\r")
        G0 = greedy_random_transport_plan(h1, h2)
        try:
            f = fused_gromov_wasserstein2(M, C1, C2, h1, h2, alpha=alpha, G0=G0)
        except AssertionError:
            printv("A greedy generated transport plan was not exact")
            continue
        if (f < min_dist or min_dist < 0) and f >= 0:
            min_dist = f
    return min_dist

def one_one_parallelised(class1, class2, d, Cs1, Cs2, hs1, hs2, alpha=0.5, Niter=100, Nprocess=7, symetric=False):
    """
    Computed FGW distance between graphs of class1 and graphs of class2
    Use multiprocessing parallelization

    Args :
    - class1 : list of graphs from class 1
    - class2 : list of graphs from class 2
    - d : distance function
    - Cs1 : list of structure matrixes of graphs of class1
    - Cs2 : list of structure matrixes of graphs of class2
    - hs1 : histograms of graphs of class1
    - hs2 : histograms of graphs of class2
    - alpha (float, default : 0.5) : alpha coefficient for FGW
    - Niter (int, default : 100) : number of iterations for computing FGW
    - Nprocess (int, default : 7) : number of processes to use
    - symetric (default : False) : if True, assumes class1 = class2, hs1 = hs2 and Cs1 = Cs2 and only compute half of the distances and makes the matrix symetric. There is no check.
    """
    D = Array('d', len(class1)*len(class2))
    i = Value('i', 0)
    j = Value('i', 0)
    N_done = Value('i', 0)
    coordinates_lock = Lock()

    def f(D, i, j, N_done):
        """
        Function that will be used by each process
        Args :
        - D : shared array for one to one distance matrix
        - i : shared int for i coordinate
        - j : share int for j coordinate
        - N_done : shared int for keeping track of number of computations that have been performed
        """
        while i.value < len(class1):
            coordinates_lock.acquire()
            # Protected zone
            # ==============
            i0 = i.value
            j0 = j.value
            j.value += 1
            if j.value >= len(class2):
                i.value += 1
                j.value = i.value if symetric else 0
            ndone = N_done.value
            N_done.value += 1
            # ======================
            # End of protected zone
            coordinates_lock.release()

            #Prints information about the computation
            if ndone == 0:
                remaining_time = "?"
            else:
                remaining_time = str(timedelta(seconds=((N_total_to_do-ndone)/ndone)*(time()-t_start))).split(':')
                remaining_time = f"{remaining_time[0]} h {remaining_time[1]} min {remaining_time[2].split('.')[0]} s"
            print(f"i : {i0+1}/{len(class1)} --- j : {j0+1}/{len(class2)}  | ETA : {remaining_time}                   ", end="\r")

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
    
    #Number of computations to perform. Only usefull for computing ETA
    N_total_to_do = len(class1)*len(class2) if not symetric else len(class1)*(len(class1)-1)/2
    t_start = time()
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    #Make D into a matrix
    D = np.array(D).reshape((len(class1), len(class2)))
    print() #Prints a new line to end the "\r"
    return D

