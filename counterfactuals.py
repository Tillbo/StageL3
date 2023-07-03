from src import *

D01 = np.load("./save/D01.npy")
D02 = np.load("./save/D02.npy")
D12 = np.load("./save/D12.npy")

Counter01 = np.argmin(D01, axis=1)
np.save("./save/Counter01.npy", Counter01)
Counter02 = np.argmin(D02, axis=1)
np.save("./save/Counter02.npy", Counter02)
Counter12 = np.argmin(D12, axis=1)
np.save("./save/Counter12.npy", Counter12)
Counter10 = np.argmin(D01, axis=0)
np.save("./save/Counter10.npy", Counter10)
Counter20 = np.argmin(D02, axis=0)
np.save("./save/Counter20.npy", Counter20)
Counter21 = np.argmin(D12, axis=0)
np.save("./save/Counter21.npy", Counter21)

