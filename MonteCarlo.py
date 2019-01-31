# Monte Carlo Integration
# Define two different MC functions, performing 1-dimensional integral for a given function

import numpy as np
import random as rand 

# Define the functions to integrate #

def Sin(x):
    return np.sin(x)

def Cos(x):
    return np.cos(x)

def Func(x):
    return 3 * pow(x, 3) - 2 * pow(x, 2) + 1

# Generate a random number
def RandomNumber(mode = None):
    # Generate the number uniform of Gaussian distributed between 0 and 1
    if mode == "Gauss":
        r = rand.gauss(0, 1)
    else:
        r = rand.random()
    return r

# Monte Carlo integrator
def MonteCarlo(func, n, xmin = 0.0, xmax = 2 * np.pi):
    # Integrate a 1-dimensional function
    Integral = 0.0
    func2 = 0.0
    for _ in range(0, n):
        r = RandomNumber()
        x = xmin + r * (xmax - xmin)
        val = func(x) * (xmax - xmin)
        Integral += val
        func2 += pow(val, 2)
    # Normalize
    Integral = Integral / n
    func2 = func2 / n
    # Compute variance
    Integral2 = pow(Integral, 2)
    sigma = (func2 - Integral2) / (n - 1)
    error = np.sqrt(sigma)
    return Integral, error


# Confidence level
# Check how many times the result lies inside the 1-sigma confidence interval
def Confidence():
    nin = 0
    for i in range(100):
        result = MonteCarlo(Sin, 10000)
        if(abs(result[0])-result[1]) < 0:
            nin += 1
    return nin

##### Main #####

# Call the MonteCarlo function and print the result
print ("1. Monte Carlo integration:")
res1 = MonteCarlo(Sin, 100000)
res2 = MonteCarlo(Cos, 100000)
res3 = MonteCarlo(Sin, 100000, xmin = 0.0, xmax = 0.5 * np.pi)
res4 = MonteCarlo(Cos, 100000, xmin = 0.0, xmax = 0.5 * np.pi)
res5 = MonteCarlo(Func, 1000000, xmin = 0.0, xmax = 2.0)
print ("I1 = {}, Error = {}".format(res1[0], res1[1]))
print ("I2 = {}, Error = {}".format(res2[0], res2[1]))
print ("I3 = {}, Error = {}".format(res3[0], res4[1]))
print ("I4 = {}, Error = {}".format(res4[0], res4[1]))
print ("I5 = {}, Error = {}".format(res5[0], res5[1]))

# Call the Confidence function and print the events inside the 1-sigma interval
print("2. Confidence level:\n{} events inside (-sigma, sigma)".format(Confidence()))
