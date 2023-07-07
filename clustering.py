from src import *

#graphs, histos, d = parse_and_transform(Nmax=3)

PATH = "transfer/save0.1attention"

D00 = np.load(f"./{PATH}/D00.npy")
D11 = np.load(f"./{PATH}/D11.npy")
D22 = np.load(f"./{PATH}/D22.npy")

plt.figure(0)
plt.title("Dendogram for class 0")
Z = dendo(D00)
plt.figure(1)
plt.title("Dendogram for class 1")
Z = dendo(D11)
plt.figure(2)
plt.title("Dendogram for class 2")
Z = dendo(D22)

plt.show()
