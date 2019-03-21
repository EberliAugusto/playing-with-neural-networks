import  numpy as np

a = np.array([[1,2,3],[2,4,6],[4,8,12]])
b = np.array([1,0,1])

print(np.dot(a, b))
print(a @ b)