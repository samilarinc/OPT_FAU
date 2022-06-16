# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import incompleteCholesky as IC

print('Checking IncompleteCholesky...')
alpha = 0
delta = -1
verbose = 1
A = np.array([[5, 4, 3, 2, 1], [4, 5, 2, 1, 0], [3, 2, 5, 0, 0], [2, 1, 0, 5, 0], [1, 0, 0, 0, 5]], dtype=float)
print('If you get a warning about negative delta, this is okay!')
L = IC.incompleteCholesky(A, alpha, delta, verbose)
if np.linalg.norm(A - L @ L.T) > 1.0e-3:
    raise Exception('Your full Cholesky decomposition is not working correctly.')
else:
    print('Check 01 okay')

B = np.array([[4, 1, 0], [1, 4, 0], [0, 0, 4]], dtype=float)
LBe = np.array([[2, 0, 0], [0.5, 1.9364, 0], [0, 0, 2]], dtype=float)
LB = IC.incompleteCholesky(B)
if np.linalg.norm(LB - LBe) > 1.0e-3:
    raise Exception('Your incomplete Cholesky decomposition is not working correctly.')
else:
    print('Check 02 okay')

alpha = 4
LCe = np.array([[2, 0, 0], [0.5, 2, 0], [0, 0, 2]], dtype=float)
LC = IC.incompleteCholesky(B, alpha)
if np.linalg.norm(LC - LCe) > 1.0e-3:
    raise Exception('Your incomplete Cholesky decomposition is not working correctly for alpha = 4.')
else:
    print('Check 03 okay')

alpha = 1.0e-3
delta = 1
LDe = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
LD = IC.incompleteCholesky(B, alpha, delta)
if np.linalg.norm(LD - LDe) > 1.0e-3:
    raise Exception('Your incomplete Cholesky decomposition is not working correctly for delta = 1.')
else:
    print('Check 04 okay')

print('Checking PrecCGSolver...')

import PrecCGSolver as PCG

A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
b = np.array([[5], [8], [3]], dtype=float)
delta = 1.0e-6
x = PCG.PrecCGSolver(A, b, delta, 1)
xe = np.array([[1], [1], [1]])
if np.linalg.norm(x - xe) > 1.0e-3:
    raise Exception('Your PrecCGSolver is not working correctly.')
else:
    print('Check 05 okay')

A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
x = PCG.PrecCGSolver(A, b, delta, 1)
xe = np.array([[1], [0], [2], [0], [3]])

if np.linalg.norm(x - xe) > 1.0e-3:
    raise Exception('Your PrecCGSolver is not working correctly for other dimensions.')
else:
    print('Check 06 okay')

A = np.array([[1, 2, 3, 4], [2, 4, -100, -100], [3, -100, 7, 0], [4, -100, 0, 3]], dtype=float)
b = np.array([[0], [5], [8], [3]], dtype=float)
x = PCG.PrecCGSolver(A, b, delta, 1)
print('Check 07 is okay, if you got a message that PrecCGSolver took more than 4 iterations')

if IC.matrnr() == 0:
    raise Exception('Please set your matriculation number in incompleteCholesky.py!')
elif PCG.matrnr() == 0:
    raise Exception('Please set your matriculation number in PrecCGSolver.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
