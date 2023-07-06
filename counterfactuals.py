from src import *

print("Loading distances...", end=" ")
D01 = np.load("./save/D01.npy")
D02 = np.load("./save/D02.npy")
D12 = np.load("./save/D12.npy")
print("OK")

print("Computing counterfactuals from 0 to 1...", end=" ")
Counter01 = np.argmin(D01, axis=1)
print("Saving...", end=" ")
np.save("./save/Counter01.npy", Counter01)

print("Ok\nComputing counterfactuals from 0 to 2...", end=" ")
Counter02 = np.argmin(D02, axis=1)
print("Saving...", end=" ")
np.save("./save/Counter02.npy", Counter02)

print("Ok\nComputing counterfactuals from 1 to 2...", end=" ")
Counter12 = np.argmin(D12, axis=1)
print("Saving...", end=" ")
np.save("./save/Counter12.npy", Counter12)

print("Ok\nComputing counterfactuals from 1 to 0...", end=" ")
Counter10 = np.argmin(D01, axis=0)
print("Saving...", end=" ")
np.save("./save/Counter10.npy", Counter10)

print("Ok\nComputing counterfactuals from 2 to 0...", end=" ")
Counter20 = np.argmin(D02, axis=0)
print("Saving...", end=" ")
np.save("./save/Counter20.npy", Counter20)

print("Ok\nComputing counterfactuals from 2 to 1...", end=" ")
Counter21 = np.argmin(D12, axis=0)
print("Saving...", end=" ")
np.save("./save/Counter21.npy", Counter21)
print("OK")
