import numpy as np

np1 = np.array([1,2,3,4,5,6,7,8,9])
print(np1)
print(np1.shape)

# np2 = np1.reshape()
# print(np2)
# print(np2.shape)

np3 = np.zeros((2,2,2,2))
 
y = np.where(np3 % 1 == 1)
print(f"y = {y}")