# Optimization for Engineers - Dr.Johannes Hild
# projected Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the Newton direction of the reduced hessian if it satisfies a descent direction check, otherwise choose d_k to be the steepest descent.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0.1], [0.1]], dtype=float)
# eps = 1.0e-3
# xmin = projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

# myObjective = nonlinearObjective()
# a = np.array([[-2], [-2]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[1.5], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[-0.26],[0.21]] (if it is close to [[0.26],[-0.21]] then maybe your reduction is done wrongly)

# myObjective = bananaValleyObjective()
# a = np.array([[-10], [-10]])
# b = np.array([[10], [10]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[0], [1]], dtype=float)
# eps = 1.0e-6
# xmin = projectedNewtonDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]] in less than 25 iterations. If you have too much iterations, then maybe the hessian is used wrongly.


import numpy as np
import projectedBacktrackingSearch as PB
import PrecCGSolver as PCG



def matrnr():
    # set your matriculation number here
    matrnr = 23062971
    return matrnr

k = 1
def projectedNewtonDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        # print(x0, eps)
        print('Start projectedNewtonDescent...')

    countIter = 0
    xp = x0
    
    xp = P.project(xp)
    grad_x = f.gradient(xp)
    Bk = f.hessian(xp)

    while np.linalg.norm(xp - P.project(xp - grad_x)) > eps:
        L = P.activeIndexSet(xp)
        Bk_pi = Bk.copy()
        Bk_pi[L, :] = 0
        Bk_pi[:, L] = 0
        Bk_pi[L, L] = 1
        
        dk = PCG.PrecCGSolver(Bk_pi, -grad_x)
        if (grad_x.T @ dk).item((0, 0)) > 0:
            dk = -grad_x
    
        tk = PB.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + tk * dk)
        grad_x = f.gradient(xp)
        Bk = f.hessian(xp)
        countIter = countIter + 1

    if verbose:
        print('globalNewtonDescent terminated after ', countIter, ' steps')

    return xp