from src import *

PERCENT = config['PERCENTAGE']
NPROCESS = config['PROCESSES']
SEED = config['SEED']
PATH = f"save/{config['XP_NAME']}"

np.random.seed(SEED)

print("\n====== PARSING DATASET ======\n")

if config['ESPAM'] is None:
    graphs, histos, d, indexes = parse_and_transform(name=config['DATA_NAME'], percent=PERCENT, connected=config['CONNECTED'], smiles=config['SMILES'])
else:
    graphs, histos, d, indexes = parse_transform_espam(name=config['DATA_NAME'], espam=config['ESPAM'], attention=config['ATTENTION'], percent=config['PERCENT'], connected=config['CONNECTED'], smiles=config['SMILES'])
print(f"Number of classes : {len(graphs)}")
for i, c in enumerate(graphs):
    print(f"    Class {i} : {len(graphs[i])} graphs")

print("\n====== SAVING INDEXES ======\n")

indexes = [str(i) for i in indexes]

with open(f"{PATH}/indexes.txt", "w") as f:
    f.write("\n".join(indexes))

with open(f"{PATH}/NCLASSES.txt", 'w') as f:
    f.write(f"{len(graphs)}")

print("\n====== COMPUTING ALL TO ALL MATRIXES ======\n", end="\n")

Cs = []
for i, c in enumerate(graphs):
    print(f"Class {i}...", end=" ", flush=True)
    Cs.append([all_to_all(G)/2 for G in c])
    print("OK")

for i, c1 in enumerate(graphs):
    for j, c2 in enumerate(graphs):
        print(f"\n====== COMPUTING D{i}{j} ======\n")
        D = one_one_parallelised(c1, c2, d, Cs[i], Cs[j], histos[i], histos[j], alpha=0.5, Nprocess=NPROCESS, symetric=(i==j))
        np.save(f"{PATH}/D{i}{j}.npy", D)