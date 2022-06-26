# Optimization for Engineers - Dr.Johannes Hild
# Programming Homework Check Script
# Do not change this file

print('Welcome to Optimization for Engineers.\n')
print('If this script fails, then your programming homework is not working correctly.')

import numpy as np
import simpleValleyObjective as SO
import benchmarkObjective as BO
import leastSquaresObjective as LS
import levenbergMarquardtDescent as LM

p0 = np.array([[2], [3]])
myObjective = SO.simpleValleyObjective(p0)
xk = np.array([[0, 0, 1, 2], [1, 2, 3, 4]])
fk = np.array([[2, 3, 2.54, 4.76]])
myErrorVector = LS.leastSquaresObjective(myObjective, xk, fk)
res = myErrorVector.residual(p0)
rese = np.array([[2], [3], [10], [20]])
if np.linalg.norm(res-rese) > 1.0e-2:
    raise Exception('Your leastSquaresObjective returns a wrong residual')
else:
    print('Check 01 is okay')

res = myErrorVector.jacobian(p0)
rese = np.array([[0, 1], [1, 1], [4, 1],  [9, 1]])
if np.linalg.norm(res - rese) > 1.0e-2:
    raise Exception('Your leastSquaresObjective returns a wrong jacobian')
else:
    print('Check 02 is okay')

p0 = np.array([[2],[1],[3]])
myObjective = BO.benchmarkObjective(p0)
xk = np.array([[3, 0, 0], [-1, -1, 0], [-1, -1, -1]])
fk = np.array([[3, 2, 1]])
myErrorVector = LS.leastSquaresObjective(myObjective, xk, fk)

res = myErrorVector.residual(p0)
rese = np.array([[4], [2], [2]])
if np.linalg.norm(res-rese) > 1.0e-2:
    raise Exception('Your leastSquaresObjective returns a wrong residual for general objective')
else:
    print('Check 03 is okay')

res = myErrorVector.jacobian(p0)
rese = np.array([[0, -1, 2], [0, -1, 1], [0, 0, 1]])
if np.linalg.norm(res-rese) > 1.0e-2:
    raise Exception('Your leastSquaresObjective returns a wrong jacobian for general objective')
else:
    print('Check 04 is okay')

p0 = np.array([[180],[0]])
myObjective = SO.simpleValleyObjective(p0)
xk = np.array([[0, 0], [1, 2]])
fk = np.array([[2, 3]])
myErrorVector = LS.leastSquaresObjective(myObjective, xk, fk)
eps = 1.0e-4
alpha0 = 1.0e-3
beta = 100
pmin = LM.levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
pe = np.array([[1], [1]])
if np.linalg.norm(pmin-pe) > 1.0e-2:
    raise Exception('Your levenbergMarquardtDescent returns a wrong result')
else:
    print('Check 05 is okay')

p0 = np.array([[0],[0],[0]])
myObjective = BO.benchmarkObjective(p0)
xk = myObjective.getXData()
fk = myObjective.getFData()
myErrorVector = LS.leastSquaresObjective(myObjective, xk, fk)
eps = 1.0e-4
alpha0 = 1.0e-3
beta = 100
pmin = LM.levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
pe = np.array([[3], [2], [16]])
if np.linalg.norm(pmin-pe) > 1.0e-1:
    raise Exception('Your levenbergMarquardtDescent returns a wrong result for the benchmarkProblem')
else:
    print('The parameters of the benchmark problem are:\n', pmin)
    print('Check 06 is okay. Congratulations, you solved the least squares fitting subproblem of the benchmark problem!')

if LS.matrnr() == 0:
    raise Exception('Please set your matriculation number in leastSquaresObjective.py!')
elif LM.matrnr() == 0:
    raise Exception('Please set your matriculation number in levenbergMarquardtDescent.py!')
else:
    print('Everything seems to be fine, please return your files in StudOn')
