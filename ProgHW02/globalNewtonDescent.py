# Optimization for Engineers - Dr.Johannes Hild
# global Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the Newton direction if it satisfies a descent direction check, otherwise choose d_k to be the steepest descent.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = nonlinearObjective()
# x0 = np.array([[-0.01], [0.01]])
# eps = 1.0e-6
# xmin = globalNewtonDescent(myObjective, x0, eps, 1)
# should return
# xmin close to [[0.26],[-0.21]] (exact xmin depends on choice of delta in PrecCGSolver)

# myObjective = nonlinearObjective()
# x0 = np.array([[-0.6], [0.6]])
# eps = 1.0e-3
# xmin = globalNewtonDescent(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]] (exact xmin depends on choice of delta in PrecCGSolver)

# myObjective = nonlinearObjective()
# x0 = np.array([[0.6], [-0.6]])
# eps = 1.0e-3
# xmin = globalNewtonDescent(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]] (exact xmin depends on choice of delta in PrecCGSolver)


import numpy as np
import WolfePowellSearch as WP
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


def globalNewtonDescent(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start globalNewtonDescent...')

    countIter = 0
    x = MISSING
    

    while MISSING STATEMENT:
        MISSING CODE

        countIter = countIter + 1
    

    if verbose:
        gradx = f.gradient(x)
        print('globalNewtonDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx))

    return x
