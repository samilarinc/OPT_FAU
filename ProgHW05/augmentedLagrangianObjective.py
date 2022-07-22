# Optimization for Engineers - Dr.Johannes Hild
# Augmented Lagrangian objective

# Purpose: Provides .objective() and .gradient() and .hessian() of the of the augmented Lagrangian mapping x -> A = f(x) + alpha*h(x)+ 0.5*gamma*h(x)**2

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian(), objective
# h: objective class with methods .objective() and .gradient() and .hessian(), equality constraint
# alpha: real value, current guess for Lagrangian multiplier for h.
# gamma: positive value, penalty parameter.

# Output Definition:
# .objective(): real number, evaluation of augmentedLagrangianObjective at x
# .gradient(): real column vector in R^n, evaluation of the gradient at x
# .hessian(): real matrix in R^nxn, evaluation of the Hessian at x

# Required files:
# <none>

# Test cases:
# A = np.array([[2, 0], [0, 2]], dtype=float)
# B = np.array([[0], [0]], dtype=float)
# C = 1
# myObjective = quadraticObjective(A, B, C)
# D = np.array([[2, 0], [0, 2]], dtype=float)
# E = np.array([[0], [0]], dtype=float)
# F = -1
# myConstraint = quadraticObjective(D, E, F)
# x0 = np.array([[2],[2]])
# alpha = -1
# gamma = 10
# myAugLag = augmentedLagrangianObjective(myObjective, myConstraint, alpha, gamma)
# should return
# myAugLag.objective(x0) close to 247
# myAugLag.gradient(x0) close to [[280], [280]]
# myAugLag.hessian(x0) close to [[300, 160], [160, 300]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23062971
    return matrnr


class augmentedLagrangianObjective:

    def __init__(self, f, h, alpha, gamma):
        if gamma <= 0:
            raise TypeError('range of gamma is wrong!')

        self.f = f
        self.h = h
        self.alpha = alpha
        self.gamma = gamma

    def objective(self, x: np.array):
        f_res = self.f.objective(x)
        h_res = self.h.objective(x)
        myObjective = f_res + self.alpha*h_res + 0.5*self.gamma*(h_res**2)

        return myObjective

    def gradient(self, x: np.array):
        f_grad = self.f.gradient(x)
        h_grad = self.h.gradient(x)
        h_res = self.h.objective(x)
        myGradient = f_grad + self.alpha*h_grad + self.gamma*(h_res*h_grad)

        return myGradient

    def hessian(self, x: np.array):
        f_hess = self.f.hessian(x)
        h_res = self.h.objective(x)
        h_grad = self.h.gradient(x)
        h_hess = self.h.hessian(x)
        myHessian = f_hess + self.alpha*h_hess + self.gamma*(h_res*h_hess + h_grad@h_grad.T)

        return myHessian
