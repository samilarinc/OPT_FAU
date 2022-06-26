# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import nonlinearObjective as NO
import bananaValleyObjective as BO
import projectedNewtonDescent as PD
import projectionInBox as PB
import BFGSDescent as BD

p = np.array([[1], [1]])
myObjective = SO.simpleValleyObjective(p)
a = np.array([[1], [1]])
b = np.array([[2], [2]])
myBox = PB.projectionInBox(a, b)
x0 = np.array([[2], [2]], dtype=float)
eps = 1.0e-3
xmin = PD.projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected Newton descent is not working.')
else:
    print('Check 01 is okay')

myObjective = NO.nonlinearObjective()
x0 = np.array([[0.1], [0.1]], dtype=float)
xmin = PD.projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected Newton descent is not working for general objective class.')
else:
    print('Check 02 is okay')

a = np.array([[-2], [-2]])
myBox = PB.projectionInBox(a, b)
x0 = np.array([[1.5], [2]], dtype=float)
xmin = PD.projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[-0.26],[0.21]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected Newton descent is not working, maybe your reduction is done wrongly?')
else:
    print('Check 03 is okay')

myObjective = BO.bananaValleyObjective()
a = np.array([[-10], [-10]])
b = np.array([[10], [10]])
myBox = PB.projectionInBox(a, b)
x0 = np.array([[0], [1]], dtype=float)
eps = 1.0e-6
xmin = PD.projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
xe = np.array([[1], [1]], dtype=float)
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your projected Newton descent is not working.')
else:
    print('Check 04 is okay if the number of iterations is less than 25. Otherwise maybe your hessian is reduced wrongly.')

myObjective = NO.nonlinearObjective()
x0 = np.array([[-0.01], [0.01]])
eps = 1.0e-6
xmin = BD.BFGSDescent(myObjective, x0, eps, 1)
xe = np.array([[0.26], [-0.21]])
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your BFGS descent is not working for general objective class.')
else:
    print('Check 05 is okay, if inverse BFGS matrix is close to [[0.0078, 0.0005], [0.0005, 0.0080]]')

x0 = np.array([[0.6], [-0.6]])
eps = 1.0e-3
xmin = BD.BFGSDescent(myObjective, x0, eps, 1)
xe = np.array([[-0.26], [0.21]])
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your BFGS descent is not working for general objective class.')
else:
    print('Check 06 is okay, if inverse BFGS matrix is close to [[0.0150, 0.0012], [0.0012, 0.0156]]')

import bananaValleyObjective as BO

myObjective = BO.bananaValleyObjective()
x0 = np.array([[0], [1]])
eps = 1.0e-6
xmin = BD.BFGSDescent(myObjective, x0, eps, 1)
xe = np.array([[1], [1]])
if np.linalg.norm(xmin-xe) > 1.0e-3:
    raise Exception('Your BFGSdescent is not working correctly, maybe the update is wrong?')
else:
    print('Check 07 is okay, if iterations are less then 100 and inverse BFGS matrix is close to [[0.4996, 0.9993], [0.9993, 2.0040]]')

if PD.matrnr() == 0:
    raise Exception('Please set your matriculation number in projectedNewtonDescent.py!')
elif BD.matrnr() == 0:
    raise Exception('Please set your matriculation number in BFGSdescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
