from src import *
from random import randint

PERCENT = 0.01
NPLOT = 4

print("Parsing...", end=" ")
graphs = parse(percent=PERCENT)
print("OK")

for i in range(3):
    for j in range(3):
        if i == j:
            continue
        Counterfactuals = np.load(f"save/Counter{i}{j}.npy")
        indexes = np.random.choice(list(range(len(Counterfactuals))), size=NPLOT, replace=False)
        for k, index in enumerate(indexes):
            Gi = graphs[i][index]
            Gj = graphs[j][Counterfactuals[index]]
            save_mol_folder([Gi, Gj], f"Counter{i}{j}/{k+1}", ["Example", "Counterfactual"])