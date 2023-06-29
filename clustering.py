from src import *

graphs, histos, d = parse_and_transform(Nmax=10)

D00 = np.load("./save/D00.npy")
D11 = np.load("./save/D11.npy")
D22 = np.load("./save/D22.npy")

print(graphs[0])

clusters = cluster_dist(graphs[0], D00, 10)
print(clusters)