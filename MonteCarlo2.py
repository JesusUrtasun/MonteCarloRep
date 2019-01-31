# Monte Carlo Integration (II)
# Multi-dimenstional MonteCarlo integrator
# Input limits for the Monte Carlo will be given in a list of size equal to the number of variables

import numpy as np
import random as rand

# Define the functions to integrate

def Cos(x):
    return np.cos(x)

def Func(x):
    return 3 * pow(x, 2) + 1

def FuncMultiDim(v):
    # FuncMultiDim has an array as input
    x = 0.0
    for i in v:
        x = x+i
    return pow(x, 2)    

# Generate a list of random numbers
def RandomNumber(d, mode = None):
    randoms = []
    for _ in range(0,d):
        if mode == "Gauss":
            r = rand.gauss(0, 1)
        else:
            r = rand.random()
        randoms.append(r)
    return np.array(randoms)

# Monte Carlo integrator. Limits of each variable in the limits
def MonteCarlo(func, n, limits):
    if not limits: # True if limts is an empty list.
        d = 1
    else:
        d = len(limits)
    I = 0.0
    func2 = 0.0
    for _ in range(0, n):
        r = RandomNumber(d)
        x = []
        w = 1.0
        for i in range(0, d):
            xmin = limits[i][0] # First element in the i-th tuple
            xmax = limits[i][1] # Second elements in the i-th tuple
            deltax = xmax - xmin # Individual weight for each transformation
            x.append(xmin + r[i] * deltax)
            w = deltax * w # Multiply each transformation to have a complete weight
        x = np.array(x)
        val = func(x) * w
        I += val
        func2 += pow(val, 2)
    #Normalize
    I = I / n
    func2 = func2 / n
    # Compute varance
    I2 = pow(I, 2)
    sigma = (func2 - I2) / (n-1)
    Error = np.sqrt(sigma)

    return I, Error

# Main

# Call the Monte Carlo and print out the result
print ("1. Monte Carlo integration:")
res1 = MonteCarlo(Cos, 100000, [(0, np.pi/2.0)])
res2 = MonteCarlo(Func, 100000, [(0, 1.0)])
res3 = MonteCarlo(FuncMultiDim, 1000000, [(0, 1), (0, 1)])
print ("I1 = {0}, Error = {1}".format(res1[0][0], res1[1][0]))
print ("I2 = {0}, Error = {1}".format(res2[0], res2[1]))
print ("I3 = {0}, Error = {1}".format(res3[0], res3[1]))