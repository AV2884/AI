import numpy as np


toy = np.array([0, 1, 2, 0, 1])

N = np.zeros((3,3),dtype=int)

for i in range(len(toy)-1):
    print(i,toy[i])
    N[toy[i],toy[i+1]] += 1

print(N)