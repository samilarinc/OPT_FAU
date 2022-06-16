# Optimization for Engineers - Dr.Johannes Hild
# Mock Homework to check setup
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your setup is not working correctly.')
print('First we check if the math package numpy is installed.')

import numpy as np

X = np.power(2, 3)
Y = 2**3
if X == Y:
    print('numpy seems to work.\n')

print('Next we check if the function definitions in BenchmarkObjective.py are available.')

import benchmarkObjective as BM

p = np.array([[3], [2], [16]])
x = np.array([[1.05], [-0.54], [-0.03]])
myObjective = BM.benchmarkObjective(p)
f = myObjective.objective(x)
fg = myObjective.gradient(x)
fh = myObjective.hessian(x)
xdata = myObjective.getXData()
fdata = myObjective.getFData()

print('The benchmark objective without noise returns')
print(f)
print('at x and the gradient is')
print(fg)
print('and the hessian is')
print(fh,'\n')

print('The benchmark objective measure points are')
print(xdata)
print('with measure results')
print(fdata)

import LLTSolver as LLT
L = np.array([[3, 0], [1, 2]], dtype=float)

print('Now we use LLTSolver. The matrix A = ')
print(L @ L.T)
print('can obviously be decomposed into A = L @ L.T with triangle matrix L = ')
print(L)

print('We can use this to quickly solve A @ x = [[14], [9]] by calling LLTSolver. Solution: x = ') 
x = LLT.LLTSolver(L, np.array([[14], [9]]))
print(x)


print('\nEverything seems to be fine, please return your files in StudOn')
