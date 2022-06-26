# Optimization for Engineers - Dr.Johannes Hild
# global BFGS descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = nonlinearObjective()
# x0 = np.array([[-0.01], [0.01]])
# eps = 1.0e-6
# xmin = BFGSDescent(myObjective, x0, eps, 1)
# should return
# xmin close to [[0.26],[-0.21]] with the inverse BFGS matrix being close to [[0.0078, 0.0005], [0.0005, 0.0080]]

# myObjective = nonlinearObjective()
# x0 = np.array([[0.6], [-0.6]])
# eps = 1.0e-3
# xmin = BFGSDescent(myObjective, x0, eps, 1)
# should return
# xmin close to [[-0.26],[0.21]] with the inverse BFGS matrix being close to [[0.0150, 0.0012], [0.0012, 0.0156]]

# myObjective = bananaValleyObjective()
# x0 = np.array([[0], [1]])
# eps = 1.0e-6
# xmin = BFGSDescent(myObjective, x0, eps, 1)
# should return
# xmin close to [[1],[1]] in less than 100 iterations with the inverse BFGS matrix being almost singular and close to [[0.4996, 0.9993], [0.9993, 2.0040]]


import numpy as np
import WolfePowellSearch as WP


def matrnr():
    # set your matriculation number here
    matrnr = 23062971
    return matrnr


def BFGSDescent(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start BFGSDescent...')

    countIter = 0

    x = x0
    B = np.eye(x.shape[0])
    grad_x = f.gradient(x)

    while np.linalg.norm(grad_x) > eps:
        dk = -np.dot(B, grad_x)
        if (grad_x.T @ dk).item((0, 0)) > 0:
            dk = -grad_x
            B = np.eye(x.shape[0])
        tk = WP.WolfePowellSearch(f, x, dk)
        
        old_x = x.copy()
        old_grad_x = grad_x.copy()
        x = x + tk * dk
        grad_x = f.gradient(x)

        dg = grad_x - old_grad_x
        dx = x - old_x
        rk = dx - (B @ dg)
        B += ((rk @ dx.T) + (dx @ rk.T)) / (dg.T @ dx)
        B -= (rk.T @ dg) * (dx @ dx.T) / ((dg.T @ dx) @ (dg.T @ dx))
        countIter = countIter + 1

    np.set_printoptions(precision=4)
    if verbose:
        gradx = f.gradient(x)
        print('BFGSDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx), 'and the inverse BFGS matrix is')
        print(B)

    return x
