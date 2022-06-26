# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

# myObjective = nonlinearObjective()
# x = np.array([[0.53], [-0.29]])
# d = np.array([[-3.88], [1.43]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.0938


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23062971
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x)
    gradx = f.gradient(x)
    descent = gradx.T @ d

    if descent >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5:
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1:
        raise TypeError('range of rho is wrong!')

    if verbose:
        print('Start WolfePowellSearch...')

    t = 1
    W1, W2, fx_plus_td, gradx_plus_td, decent_plus_td = 0, 0, 0, 0, 0
    def update(t):
        nonlocal x, fx, gradx, descent, W1, W2, fx_plus_td, gradx_plus_td, decent_plus_td
        fx = f.objective(x)
        gradx = f.gradient(x)
        descent = gradx.T @ d

        fx_plus_td = f.objective(x + t * d)
        gradx_plus_td = f.gradient(x + t * d)
        descent_plus_td = gradx_plus_td.T @ d

        W1 = fx_plus_td <= fx + t * sigma * descent
        W2 = descent_plus_td >= rho * descent

    update(t)

    if W1 == False:
        t = t/2
        update(t)
        while W1 == False:
            t = t/2
            update(t)
        t_min = t
        t_plus = 2*t
    elif W2 == True:
        return t
    else:
        t = 2*t
        update(t)
        while W1 == True:
            t = 2*t
            update(t)
        t_min = t/2
        t_plus = t
    t = t_min
    update(t)
    while W2 == False:
        t = (t_plus + t_min)/2
        update(t)
        if W1 == True:
            t_min = t
        else:
            t_plus = t
    return t_min
    
    if verbose:
        print('WolfePowellSearch finished!')
    return t
