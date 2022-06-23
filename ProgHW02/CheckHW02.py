# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import nonlinearObjective as NO
import WolfePowellSearch as WP

p = np.array([[0], [1]])
myObjective = SO.simpleValleyObjective(p)
x = np.array([[-1.01], [1]])
d = np.array([[1], [1]])
sigma = 1.0e-3
rho = 1.0e-2
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 1
if t != te:
    raise Exception('Your Wolfe-Powell search is not recognizing t = 1 as valid starting point.')
else:
    print('Check 01 okay')

x = np.array([[-1.2], [1]])
d = np.array([[0.1], [1]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 16
if t != te:
    raise Exception('Your Wolfe-Powell search is not fronttracking correctly.')
else:
    print('Check 02 okay')

x = np.array([[-0.2], [1]])
d = np.array([[1], [1]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.25
if t != te:
    raise Exception('Your Wolfe-Powell search is not refining correctly.')
else:
    print('Check 03 okay')

myObjective = NO.nonlinearObjective()
x = np.array([[0.53], [-0.29]])
d = np.array([[-3.88], [1.43]])
t = WP.WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
te = 0.0938
if np.abs(t-te) > 1.0e-3:
    raise Exception('Your Wolfe-Powell search is not working for general objective class.')
else:
    print('Check 04 okay')

import globalNewtonDescent as GN

x0 = np.array([[-0.01], [0.01]])
eps = 1.0e-6
xmin = GN.globalNewtonDescent(myObjective, x0, eps, 1)
xe = np.array([[0.26], [-0.21]])
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your global Newton Descent is not working correctly.')
else:
    print('Check 05 okay')

x0 = np.array([[-0.6], [0.6]])
xmin = GN.globalNewtonDescent(myObjective, x0, eps, 1)
xe = np.array([[-0.26], [0.21]])
if np.linalg.norm(xmin-xe) > 1.0e-2:
    raise Exception('Your global Newton Descent walks a wrong path, maybe you switch to steepest descent too often?')
else:
    print('Check 06 okay')

x0 = np.array([[0.6], [-0.6]])
xmin = GN.globalNewtonDescent(myObjective, x0, eps, 1)
xe = np.array([[-0.26], [0.21]])
if np.linalg.norm(xmin - xe) > 1.0e-2:
    raise Exception('Your global Newton Descent walks a wrong path, maybe you make mistakes in choosing the descent directions?')
else:
    print('Check 07 okay')

if WP.matrnr() == 0:
    raise Exception('Please set your matriculation number in WolfePowellSearch.py!')
elif GN.matrnr() == 0:
    raise Exception('Please set your matriculation number in globalNewtonDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
