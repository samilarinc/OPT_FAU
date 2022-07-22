# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import quadraticObjective as QO
import benchmarkObjective as BO
import augmentedLagrangianObjective as AO
import augmentedLagrangianDescent as AD
import projectionInBox as PB

A = np.array([[2, 0], [0, 2]], dtype=float)
B = np.array([[0], [0]], dtype=float)
C = 1
myObjective = QO.quadraticObjective(A, B, C)
D = np.array([[2, 0], [0, 2]], dtype=float)
E = np.array([[0], [0]], dtype=float)
F = -1
myConstraint = QO.quadraticObjective(D, E, F)
x0 = np.array([[2], [2]])
alpha = -1
gamma = 10
myAugLag = AO.augmentedLagrangianObjective(myObjective, myConstraint, alpha, gamma)

y1 = myAugLag.objective(x0)
y1e = 247
if np.linalg.norm(y1-y1e) > 1.0e-1:
    raise Exception('Your augmentedLagrangianObjective returns a wrong objective')
else:
    print('Check 01 is okay')

y2 = myAugLag.gradient(x0)
y2e = np.array([[280], [280]])
if np.linalg.norm(y2-y2e) > 1.0e-1:
    raise Exception('Your augmentedLagrangianObjective returns a wrong gradient')
else:
    print('Check 02 is okay')

y3 = myAugLag.hessian(x0)
y3e = np.array([[300, 160], [160, 300]])
if np.linalg.norm(y3-y3e) > 1.0e-1:
    raise Exception('Your augmentedLagrangianObjective returns a wrong hessian')
else:
    print('Check 03 is okay')

A = np.array([[4, 0], [0, 2]], dtype=float)
B = np.array([[0], [0]], dtype=float)
C = 1
myObjective = QO.quadraticObjective(A, B, C)
a = np.array([[0], [0]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
D = np.array([[2, 0], [0, 2]], dtype=float)
E = np.array([[0], [0]], dtype=float)
F = -1
myConstraint = QO.quadraticObjective(D, E, F)
x0 = np.array([[1], [1]], dtype=float)
alpha0 = 0
eps = 1.0e-3
delta = 1.0e-6
[xmin, alphamin] = AD.augmentedLagrangianDescent(myObjective, myBox, myConstraint, x0, alpha0, eps, delta, 1)
xmine = np.array([[0], [1]])
if np.linalg.norm(xmin-xmine) > 1.0e-1:
    raise Exception('Your augmentedLagrangianDescent returns a wrong xmin')
else:
    print('Check 04 is okay')
alphamine = -1
if np.linalg.norm(alphamin-alphamine) > 1.0e-1:
    raise Exception('Your augmentedLagrangianDescent returns a wrong alphamin')
else:
    print('Check 05 is okay')

p = np.array([[ 2.9999039 ], [ 1.99851503], [16.05570494]], dtype=float)
myObjective = BO.benchmarkObjective(p)
a = np.array([[0], [-4], [-1]])
b = np.array([[8], [4], [1]])
myBox = PB.projectionInBox(a, b)
D = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float)
E = np.array([[-8], [0], [0]], dtype=float)
F = 7
myConstraint = QO.quadraticObjective(D, E, F)
x0 = np.array([[4], [-2], [0]], dtype=float)
alpha0 = 0
eps = 1.0e-3
delta = 1.0e-6
[xmin, alphamin] = AD.augmentedLagrangianDescent(myObjective, myBox, myConstraint, x0, alpha0, eps, delta, 1)
xmine = np.array([[5.56], [-2.55], [-0.20]])
if np.linalg.norm(xmin-xmine) > 1.0e-1:
    raise Exception('Your augmentedLagrangianDescent returns a wrong xmin for the benchmark problem')
else:
    print('Check 06 is okay')
alphamine = 16.4
if np.linalg.norm(alphamin-alphamine) > 1.0e-1:
    raise Exception('Your augmentedLagrangianDescent returns a wrong alphamin for the benchmark problem')
else:
    print('Check 07 is okay. You also minimized the benchmark problem without noise!')

if AO.matrnr() == 0:
    raise Exception('Please set your matriculation number in augmentedLagrangianObjective.py!')
elif AD.matrnr() == 0:
    raise Exception('Please set your matriculation number in augmentedLagrangianDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
