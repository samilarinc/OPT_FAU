# Optimization for Engineers - Dr.Johannes Hild
# Nonlinear test function
# Do not change this file

# Required files:
# < none >


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 0
    return matrnr


class nonlinearObjective:
    # Nonlinear function R**2 -> R
    # 2-dimensional nonlinear function mapping x -> -0.03/((x(0) +0.25)**2 +(x(1) -0.2)**2 +0.03) -0.1/((x(0) -0.25)**2 +(x(1)+ 0.2)**2 +0.04) + 0.1/(x(0)**2 +x(1)**2 +0.05) +1 +x(0)**2 +x(1)**2 + 1;
    # Has a local maximizing point at approx [[-0.0158], [0.0126]], and a local minimizing point at approx [-0.265;0.212] and a global minimizing point at approx [[0.261], [-0.209]]

    # Input Definition:
    # x: vector in R**2 (domain space)

    # Output Definition:
    # objective: real number, evaluation of nonlinearObjective at x
    # gradient: real column vector in R**2, evaluation of the gradient with respect to x at x
    # hessian: real 2x2 matrix, evaluation of the hessian with respect to x at x

    # Test cases:

    # myObjective = nonlinearObjective.objective(np.array([[-0.015793], [0.012647]], dtype=float))
    # should return
    # myObjective = 3.0925

    # myGradient = nonlinearObjective.gradient(np.array([[-0.015793], [0.012647]], dtype=float))
    # should return
    # myGradient close to [[0],[0]]

    # myHessian = nonlinearObjective.hessian(np.array([[-0.015793], [0.012647]], dtype=float))
    # should return
    # myHessian = [[-85.299, 16.795],[16.795, -77.739]]

    @staticmethod
    def objective(x: np.array):
        x1 = x[0,0]
        x2 = x[1,0]
        value = -0.03 / ((x1 + 0.25) ** 2 + (x2 - 0.2) ** 2 + 0.03) - 0.1 / ((x1 - 0.25) ** 2 + (x2 + 0.2) ** 2 + 0.04) + 0.1 / (x1 ** 2 + x2 ** 2 + 0.05) + 1 + x1 ** 2 + x2 ** 2 + 1
        return value

    @staticmethod
    def gradient(x: np.array):
        x1 = x[0,0]
        x2 = x[1,0]
        dx1 = 2 * (x1 + 0.25) * 0.03 / ((x1 + 0.25) ** 2 + (x2 - 0.2) ** 2 + 0.03) ** 2 + 2 * (x1 - 0.25) * 0.1 / ((x1 - 0.25) ** 2 + (x2 + 0.2) ** 2 + 0.04) ** 2 - 2 * x1 * 0.1 / (x1 ** 2 + x2 ** 2 + 0.05) ** 2 + 2 * x1
        dx2 = 2 * (x2 - 0.2) * 0.03 / ((x1 + 0.25) ** 2 + (x2 - 0.2) ** 2 + 0.03) ** 2 + 2 * (x2 + 0.2) * 0.1 / ((x1 - 0.25) ** 2 + (x2 + 0.2) ** 2 + 0.04) ** 2 - 2 * x2 * 0.1 / (x1 ** 2 + x2 ** 2 + 0.05) ** 2 + 2 * x2
        g = np.array([[dx1], [dx2]])
        return g

    @staticmethod
    def hessian(x: np.array):
        x1 = x[0,0]
        x2 = x[1,0]
        f_dx11 = -.6e-1 / ((x1 + .25) ** 2 + (x2 - .2) ** 2 + .3e-1) ** 3 * (2 * x1 + .50) ** 2 + .6e-1 / ((x1 + .25) ** 2 + (x2 - .2) ** 2 + .3e-1) ** 2 - .2 / ((x1 - .25) ** 2 + (x2 + .2) ** 2 + .4e-1) ** 3 * (2 * x1 - .50) ** 2 + .2 / ((x1 - .25) ** 2 + (x2 + .2) ** 2 + .4e-1) ** 2 + .8 / (x1 ** 2 + x2 ** 2 + .5e-1) ** 3 * x1 ** 2 - .2 / (x1 ** 2 + x2 ** 2 + .5e-1) ** 2 + 2
        f_dx12 = -.6e-1 / ((x1 + .25) ** 2 + (x2 - .2) ** 2 + .3e-1) ** 3 * (2 * x1 + .50) * (2 * x2 - .4) - .2 / ((x1 - .25) ** 2 + (x2 + .2) ** 2 + .4e-1) ** 3 * (2 * x1 - .50) * (2 * x2 + .4) + .8 / (x1 ** 2 + x2 ** 2 + .5e-1) ** 3 * x1 * x2
        f_dx22 = -.6e-1 / ((x1 + .25) ** 2 + (x2 - .2) ** 2 + .3e-1) ** 3 * (2 * x2 - .4) ** 2 + .6e-1 / ((x1 + .25) ** 2 + (x2 - .2) ** 2 + .3e-1) ** 2 - .2 / ((x1 - .25) ** 2 + (x2 + .2) ** 2 + .4e-1) ** 3 * (2 * x2 + .4) ** 2 + .2 / ((x1 - .25) ** 2 + (x2 + .2) ** 2 + .4e-1) ** 2 + .8 / (x1 ** 2 + x2 ** 2 + .5e-1) ** 3 * x2 ** 2 - .2 / (x1 ** 2 + x2 ** 2 + .5e-1) ** 2 + 2
        h = np.array([[f_dx11, f_dx12], [f_dx12, f_dx22]])
        return h
