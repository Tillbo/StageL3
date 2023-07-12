from src import *

PATH = f"./save/{config['XP_NAME']}"
NPLOT_CLUSTER = config['N_PLOT_CLUSTERS']
PERCENT = config['PERCENTAGE']

with open(f"{PATH}/NCLASSES.txt", "r") as f:
    N = int(f.read())

try:
    mkdir(f"{PATH}/Clustering")
except FileExistsError:
    pass

try:
    graphs, indexes = parse(name=config['DATA_NAME'], percent=PERCENT, smiles=True, connected=False)
    smiles = True
except KeyError:
    print("Smiles must be accessible to save molecules. Exit.")
    smiles = False

for i in range(N):
    print(f"\n====== Clustering for class {i} ======\n")
    D = np.load(f"./{PATH}/D{i}{i}.npy")
    try:
        mkdir(f"{PATH}/Clustering/{i}")
    except FileExistsError:
        pass
    plt.title("Dendogram for class 0")
    T = dendo(D)
    plt.savefig(f"{PATH}/Clustering/0/Dendogram.png")
    plt.close()

    if smiles:
        print("Plotting clusters...", end=" ", flush=True)
        n1 = 0
        n2 = 0
        for j, G in enumerate(graphs[i]):
            if n1 < NPLOT_CLUSTER and T[j] == 1:
                save_mol_folder([G], f"{PATH}/Clustering/0/1", [G.graph["index"]])
                n1 += 1
            if n2 < NPLOT_CLUSTER and T[j] == 2:
                save_mol_folder([G], f"{PATH}/Clustering/0/2", [G.graph["index"]])
                n2 += 2
        print("OK")