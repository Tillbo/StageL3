from src import *

PATH = f"./save/{config['XP_NAME']}"

with open(f"{PATH}/NCLASSES.txt", "r") as f:
    N = int(f.read())

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        print(f"Class {i} to {j}...", end=" ", flush=True)
        if i < j:
            D = np.load(f"{PATH}/D{i}{j}.npy")
            Counter = np.argmin(D, axis=1)
        else:
            D = np.load(f"{PATH}/D{j}{i}.npy")
            Counter = np.argmin(D, axis=0)
        np.save(f"{PATH}/Counter{i}{j}.npy", Counter)
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