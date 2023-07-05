from src import *

graphs, histos, d = parse_and_transform(Nmax=3)

D00 = np.load("./save/D00.npy")
D11 = np.load("./save/D11.npy")
D22 = np.load("./save/D22.npy")

print(graphs[0])
print(D00)

clusters = cluster_dbscan(graphs[0], D00, eps=3, min_samples=2)
print(clusters)