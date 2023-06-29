from src import *

NMAX = -1
NPROCESS = 5

print("\n====== PARSING DATASET ======\n")
graphs, histos, d = parse_and_transform(Nmax=NMAX)

print("\n====== COMPUTING ALL TO ALL MATRIXES ======\n", end="\n")

print("Class 0...", end=" ")
C0 = [all_to_all(G)*2 for G in graphs[0]]
print("OK\nClass 1...", end=" ")
C1 = [all_to_all(G)*2 for G in graphs[1]]
print("OK\nClass 2...", end=" ")
C2 = [all_to_all(G)*2 for G in graphs[2]]
print("OK")

print("\n====== COMPUTING D01 ======\n")
D01 = one_one_parralelised(graphs[0], graphs[1], d, C0 , C1, histos[0], histos[1], NPROCESS)
np.save("./save/D01.npy", D01)

print("\n====== COMPUTING D02 ======\n")
D02 = one_one_parralelised(graphs[0], graphs[2], d, C0 , C2, histos[0], histos[2], NPROCESS)
np.save("./save/D02.npy", D01)

print("\n====== COMPUTING D12 ======\n")
D12 = one_one_parralelised(graphs[0], graphs[2], d, C0 , C2, histos[0], histos[2], NPROCESS)
np.save("./save/D02.npy", D12)

print("\n====== COMPUTING D00 ======\n")
D00 = one_one_parralelised(graphs[0], graphs[0], d, C0 , C0, histos[0], histos[0], NPROCESS)
np.save("./save/D00.npy", D00)

print("\n====== COMPUTING D11 ======\n")
D11 = one_one_parralelised(graphs[1], graphs[1], d, C1, C1, histos[1], histos[1], NPROCESS)
np.save("./save/D11.npy", D11)

print("\n====== COMPUTING D22 ======\n")
D22 = one_one_parralelised(graphs[2], graphs[2], d, C2, C2, histos[2], histos[2], NPROCESS)
np.save("./save/D22.npy", D22)