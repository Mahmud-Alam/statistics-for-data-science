import numpy as np
import pandas as pd

#%%

A = np.array([[1,2,2],[2,1,-2],[-2,2,-1]])

#%%

A = np.array([[3,-4,-2],[2,1,1],[-1,2,1]])

print('\nActual Matrix:')
print(A)

print('\ndet(A) :',np.linalg.det(A))
print('\nCo-factor(A) :')
print(np.linalg.inv(A).T * np.linalg.det(A))
print('\nAdj(A) :')
print((np.linalg.inv(A).T * np.linalg.det(A)).T)
print('\nAdj(A) * 1/det(A)')
print('\nInverse(A) : ')
print(np.linalg.inv(A))

print('\ntranspose Actual Matrix:')
print(A.T)

#%%

print('\ndeterminant, trace and rank: ')

B = np.array([[3,-4,-2],[2,1,1],[-1,2,1]])

print('\ndet(B) :',np.linalg.det(B))
print('\ntrace(B) :',np.trace(B))
print('\nrank(B) :',np.linalg.matrix_rank(B))