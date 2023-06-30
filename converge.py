from src import *
from ot.gromov import fused_gromov_wasserstein2
from multiprocessing import Process, Value, Array, Lock

NMAX = 20
NPROCESS = 5
EPS = 1e-3
NSAMPLE = 10
ITERMAX = 100

Niters = np.arange(0, ITERMAX+int(ITERMAX/NSAMPLE), int(ITERMAX/NSAMPLE))
graphs, histos, d = parse_and_transform(Nmax=NMAX)

iS = Value('i', 0)
jS = Value('i', 0)

NbCouples = Array('i', NSAMPLE+1)

lock_index = Lock()
lock_array = Lock()

def compute(lock_index, lock_array, iS, jS, NbCouples):
    i = 0
    j = 0
    while i < len(graphs[0]):

        lock_index.acquire()
        i = iS.value
        j = jS.value

        jS.value += 1
        if jS.value >= len(graphs[1]):
            jS.value = 0
            iS.value += 1
        lock_index.release()

        if i >= len(graphs[0]):
            break
        print(f"{i+1}/{len(graphs[0])} --- {j+1}/{len(graphs[1])}    ", end="\r")
        G1 = graphs[0][i]
        G2 = graphs[1][j]

        C1 = all_to_all(G1)/2
        C2 = all_to_all(G2)/2

        h1 = histos[0][i]
        h2 = histos[1][j]

        M = node_dists(G1, G2, d)

        def f(n):
            #print(f"---- {n}/1000 ----")
            return fgw(C1, C2, M, h1, h2, Niter=n, verbose=False)

        vf = np.vectorize(f)

        Y = vf(Niters)

        Y -= min(Y)

        it = np.argmax(Y<EPS)

        lock_array.acquire()
        NbCouples[it] += 1
        lock_array.release()


processes = []
for _ in range(NPROCESS):
    p = Process(target=compute, args=(lock_index, lock_array, iS, jS, NbCouples))
    processes.append(p)
for p in processes:
    p.start()
for p in processes:
    p.join()

NbCouples = NbCouples[:]

plt.plot(Niters, NbCouples, 'ro')
plt.show()