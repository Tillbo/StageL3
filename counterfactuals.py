from src import *

PATH = f"./save/{config['XP_NAME']}"

print("Loading distances...", end=" ", flush=True)
try:
    D01 = np.load(f"{PATH}/D01.npy")
    D02 = np.load(f"{PATH}/D02.npy")
    D12 = np.load(f"{PATH}/D12.npy")
except FileNotFoundError:
    print("Distances matrixes not found.")
    exit(1)
print("OK")

print("Computing counterfactuals from 0 to 1...", end=" ", flush=True)
Counter01 = np.argmin(D01, axis=1)
print("Saving...", end=" ", flush=True)
np.save(f"{PATH}/Counter01.npy", Counter01)

print("Ok\nComputing counterfactuals from 0 to 2...", end=" ", flush=True)
Counter02 = np.argmin(D02, axis=1)
print("Saving...", end=" ", flush=True)
np.save(f"{PATH}/Counter02.npy", Counter02)

print("Ok\nComputing counterfactuals from 1 to 2...", end=" ", flush=True)
Counter12 = np.argmin(D12, axis=1)
print("Saving...", end=" ")
np.save(f"{PATH}/Counter12.npy", Counter12)

print("Ok\nComputing counterfactuals from 1 to 0...", end=" ", flush=True)
Counter10 = np.argmin(D01, axis=0)
print("Saving...", end=" ", flush=True)
np.save(f"{PATH}/Counter10.npy", Counter10)

print("Ok\nComputing counterfactuals from 2 to 0...", end=" ", flush=True)
Counter20 = np.argmin(D02, axis=0)
print("Saving...", end=" ", flush=True)
np.save(f"{PATH}/Counter20.npy", Counter20)

print("Ok\nComputing counterfactuals from 2 to 1...", end=" ", flush=True)
Counter21 = np.argmin(D12, axis=0)
print("Saving...", end=" ", flush=True)
np.save(f"{PATH}/Counter21.npy", Counter21)
print("OK")


print("Parsing...", end=" ")
try:
    graphs, indexes = parse(name=config['DATA_NAME'], percent=config['PERCENTAGE'], smiles=True, connected=False)
except KeyError:
    print("Smiles must be accessible to save counterfactual images. Exit.")
    exit(1)
print("OK")
print(f"Number of classes : {len(graphs)}")
for i, c in enumerate(graphs):
    print(f"    Class {i} : {len(graphs[i])} graphs")


for i in range(3):
    for j in range(3):
        if i == j:
            continue
        Counterfactuals = np.load(f"{PATH}/Counter{i}{j}.npy")
        indexes = np.random.choice(list(range(len(Counterfactuals))), size=min(len(Counterfactuals), config['N_PLOT_COUNTER']), replace=False)
        for k, index in enumerate(indexes):
            Gi = graphs[i][index]
            Gj = graphs[j][Counterfactuals[index]]
            save_mol_folder([Gi, Gj], f"{PATH}/Counter{i}{j}/{index}", ["Example", "Counterfactual"])
print("Saving complete.")