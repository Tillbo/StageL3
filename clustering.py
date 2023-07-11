from src import *

PATH = f"./save/{config['XP_NAME']}"
NPLOT_CLUSTER = config['N_PLOT_CLUSTERS']
PERCENT = config['PERCENTAGE']

D00 = np.load(f"./{PATH}/D00.npy")
D11 = np.load(f"./{PATH}/D11.npy")
D22 = np.load(f"./{PATH}/D22.npy")

try:
    mkdir(f"{PATH}/Clustering")
except FileExistsError:
    pass

print("===== Clustering for class 0 =====")
try:
    mkdir(f"{PATH}/Clustering/0")
except FileExistsError:
    pass
plt.figure(0)
plt.title("Dendogram for class 0")
T00 = dendo(D00)
plt.savefig(f"{PATH}/Clustering/0/Dendogram.png")
plt.close()

print("===== Clustering for class 1 =====")
try:
    mkdir(f"{PATH}/Clustering/1")
except FileExistsError:
    pass
plt.figure(1)
plt.title("Dendogram for class 1")
T11 = dendo(D11)
plt.savefig(f"{PATH}/Clustering/1/Dendogram.png")
plt.close()

print("===== Clustering for class 2 =====")
try:
    mkdir(f"{PATH}/Clustering/2")
except FileExistsError:
    pass
plt.figure(2)
plt.title("Dendogram for class 2")
T22 = dendo(D22)
plt.savefig(f"{PATH}/Clustering/2/Dendogram.png")
plt.close()

print("===== Plotting clusters =====")
try:
    graphs, indexes = parse(name=config['DATA_NAME'], percent=PERCENT, smiles=True, connected=False)
except KeyError:
    print("Smiles must be accessible to save molecules. Exit.")
    exit(1)

print("Class 0...", end=" ", flush=True)
n1 = 0
n2 = 0
for i, G in enumerate(graphs[0]):
    if n1 < NPLOT_CLUSTER and T00[i] == 1:
        save_mol_folder([G], f"{PATH}/Clustering/0/1", [G.graph["index"]])
        n1 += 1
    if n2 < NPLOT_CLUSTER and T00[i] == 2:
        save_mol_folder([G], f"{PATH}/Clustering/0/2", [G.graph["index"]])
        n2 += 2

print("OK\nClass 1...", end=" ", flush=True)
n1 = 0
n2 = 0
for i, G in enumerate(graphs[1]):
    if n1 < NPLOT_CLUSTER and T11[i] == 1:
        save_mol_folder([G], f"{PATH}/Clustering/1/1", [G.graph["index"]])
        n1 += 1
    if n2 < NPLOT_CLUSTER and T00[i] == 2:
        save_mol_folder([G], f"{PATH}/Clustering/1/2", [G.graph["index"]])
        n2 += 2

print("OK\nClass 2...", end=" ", flush=True)
n1 = 0
n2 = 0
for i, G in enumerate(graphs[2]):
    if n1 < NPLOT_CLUSTER and T22[i] == 1:
        save_mol_folder([G], f"{PATH}/Clustering/2/1", [G.graph["index"]])
        n1 += 1
    if n2 < NPLOT_CLUSTER and T00[i] == 2:
        save_mol_folder([G], f"{PATH}/Clustering/2/2", [G.graph["index"]])
        n2 += 2
print("OK")