import numpy as np

A=np.zeros((9,10))
B=np.zeros(2)
print(B)

# A[1][0] = 1
for i in range(9):
    for j in range(10):
        A[i][j] = j
# A = np.append(A,[1, 2, 3])

print(A)
print(len(A[1,:]))
