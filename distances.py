from src import *

PERCENT = config['PERCENTAGE']
NPROCESS = config['PROCESSES']
SEED = config['SEED']
PATH = f"save/{config['XP_NAME']}"

np.random.seed(SEED)

print("\n====== PARSING DATASET ======\n")

graphs, histos, d, indexes = parse_and_transform(name=config['DATA_NAME'], percent=PERCENT, connected=config['CONNECTED'], smiles=config['SMILES'])
print(f"Number of classes : {len(graphs)}")
for i, c in enumerate(graphs):
    print(f"    Class {i} : {len(graphs[i])} graphs")

print("\n====== SAVING INDEXES ======\n")

indexes = [str(i) for i in indexes]

with open(f"{PATH}/indexes.txt", "w") as f:
    f.write("\n".join(indexes))

print("\n====== COMPUTING ALL TO ALL MATRIXES ======\n", end="\n")

print("Class 0...", end=" ")
C0 = [all_to_all(G)/2 for G in graphs[0]]
print("OK\nClass 1...", end=" ")
C1 = [all_to_all(G)/2 for G in graphs[1]]
print("OK\nClass 2...", end=" ")
C2 = [all_to_all(G)/2 for G in graphs[2]]
print("OK")

print("\n====== COMPUTING D01 ======\n")
D01 = one_one_parallelised(graphs[0], graphs[1], d, C0 , C1, histos[0], histos[1], alpha=0.5, Nprocess=NPROCESS)
np.save(f"{PATH}/D01.npy", D01)

print("\n====== COMPUTING D02 ======\n")
D02 = one_one_parallelised(graphs[0], graphs[2], d, C0 , C2, histos[0], histos[2], alpha=0.5, Nprocess=NPROCESS)
np.save(f"{PATH}/D02.npy", D02)

print("\n====== COMPUTING D12 ======\n")
D12 = one_one_parallelised(graphs[1], graphs[2], d, C1 , C2, histos[1], histos[2], alpha=0.5, Nprocess=NPROCESS)
np.save(f"{PATH}/D12.npy", D12)

print("\n====== COMPUTING D00 ======\n")
D00 = one_one_parallelised(graphs[0], graphs[0], d, C0 , C0, histos[0], histos[0], alpha=0.5, Nprocess=NPROCESS, symetric=True)
np.save(f"{PATH}/D00.npy", D00)

print("\n====== COMPUTING D11 ======\n")
D11 = one_one_parallelised(graphs[1], graphs[1], d, C1, C1, histos[1], histos[1], alpha=0.5, Nprocess=NPROCESS, symetric=True)
np.save(f"{PATH}/D11.npy", D11)

print("\n====== COMPUTING D22 ======\n")
D22 = one_one_parallelised(graphs[2], graphs[2], d, C2, C2, histos[2], histos[2], alpha=0.5, Nprocess=NPROCESS, symetric=True)
np.save(f"{PATH}/D22.npy", D22)