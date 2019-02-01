# Monte Carlo Integration (III)
# Performe a Monte Carlo integration in object oriented paradigm (OOP)
# Create an abstract class Integrator, and implement inherited classes MonteCarlo and Quadrature
# Implement an analyzer class that compares both integrators

import numpy as np
import random as rand
from scipy.integrate import quad

# Define functions to integrate

def Sin(x):
    return np.sin(x)

def Cos(x):
    return np.cos(x)

def Func(x):
    return np.sin(x) + pow(x, 2) + 3 * x

# Integrator abstract class
class Integrator:

    # Initialize method. self -> set the atributes any object will have
    def __init__(self, func, xmin, xmax, threshold = 0.01):
        self.func = func
        self.xmin = xmin
        self.xmax = xmax
        self.threshold = threshold

    # Method performing the integral, still to be implemented
    def Integrate(self):
        print("Method still to be implemented")
    
    # Method allowing allowing to update the error of the integrator
    def SetThreshold(self, threshold):
        print("Method still to be implemented")
    
    # Method computing confidence interval
    def Confidence(self):
        print("Method still to be implemented")

# Inherited Monte Carlo class from the abstract Integrator class
class MonteCarlo(Integrator):

    my_name = "Monte Carlo"

    # Add the number of steps as a positional argument
    def __init__(self, func, xmin, xmax, threshold = 0.01, n = 1000000):
        # Let the MonteCarlo class handle the init method of the abstrac class
        super().__init__(func, xmin, xmax, threshold)
        self.n = n

    # Overwrite the Integrate method of the abstract class, to call the MonteCarlo
    def Integrate(self):
        result, error = self.MCIntegrator()
        return result, error
    
    # Generate a random number Uniformly distributed from 0 to 1
    def RandomNumber(self):
        r = rand.random()
        return r
    
    # Perform the change of variables to an integral from 0 to 1, returning x and the Jacobian w
    def ChangeVariables(self, r):
        x = self.xmin + r * (self.xmax - self.xmin)
        w = self.xmax - self.xmin
        return x, w

    # Compute the integral and the error if a given error is achieved
    def ComputeResult(self, Integral, func2, nsteps):
        # Normalize
        Integral_norm = Integral / nsteps
        func2 = func2 / nsteps
        # Compute variance
        Integral2 = pow(Integral_norm, 2)
        sigma = (func2 - Integral2) / (nsteps - 1)
        error = np.sqrt(sigma)
        return Integral_norm, error

    # Perform the Monte Carlo integration, up to a given error rather than number of steps
    def MCIntegrator(self):
        Integral = 0.0
        func2 = 0.0
        for steps in range(1, self.n + 1):
            r = self.RandomNumber()
            x, w = self.ChangeVariables(r)
            val = self.func(x) * w
            Integral += val
            func2 += pow(val, 2)
        Integral, error = self.ComputeResult(Integral, func2, steps)
        return Integral, error


# Inherited Quadrature Class from the abstract Integrator class
class Quadrature(Integrator):

    my_name = "Quadrature"

    # Overwrite the Integrate method of the abstract class, to call the Quadratue
    def Integrate(self):
        result, error = self.QuadIntegrator()
        return result, error
    
    # Perform the integration by Quadrature
    def QuadIntegrator(self):
        result = quad(self.func, self.xmin, self.xmax, epsrel = self.threshold)
        return result[0], result[1]

class Analyzer:
    
    def __init__ (self, integrator1, integrator2):
        self.integrator1 = integrator1
        self.integrator2 = integrator2
    
    # Call the Integrate() method for each class, MonteCarlo and Quadrature
    def ComputeResult(self):
        result1 = self.integrator1.Integrate()
        result2 = self.integrator2.Integrate()
        return result1[0], result1[1], result2[0], result2[1]

    # Compute the time needed to achieve a particular error
    def ComputePlot(self, integrator):
        "ComputePlot method still to be implemented"

##### Main #####

# Instance both the MonteCarlo and the Quadrature classes
print("\n1. OOP Integration. Instance the MonteCarlo and Quadrature classes")
resMC1 = MonteCarlo(Sin, 0.0, 0.5 * np.pi)
resMC2 = MonteCarlo(Func, 0.0, np.pi)
resQuad1 = Quadrature(Sin, 0.0, 0.5 * np.pi)
resQuad2 = Quadrature(Func, 0.0, np.pi)

# Print results manually, by calling the MCintegrator() and QuadIntegrator() on the resMC and resQuad
print("\n2. Compute results manually")
resultMC1 = resMC1.MCIntegrator()
resultMC2 = resMC2.MCIntegrator()
resultQuad1 = resQuad1.QuadIntegrator()
resultQuad2 = resQuad2.QuadIntegrator()
print("Monte Carlo:")
print("Integral1 = {}, error = {}".format(resultMC1[0], resultMC1[1]))
print("Integral2 = {}, error = {}".format(resultMC2[0], resultMC2[1]))
print("Quadrature:")
print("Integral1 = {}, error = {}".format(resultQuad1[0], resultQuad1[1]))
print("Integral2 = {}, error = {}".format(resultQuad2[0], resultQuad2[1]))

# Instance the Analyzer class and call the ComputeResult() method
print("\n3. Compute results using the Analyzer class")
result1 = Analyzer(resMC1, resQuad1)
result2 = Analyzer(resMC2, resQuad2)
result1MC1, result1MC2, result1Quad1, result1Quad2 = result1.ComputeResult()
result2MC1, result2MC2, result2Quad1, result2Quad2 = result2.ComputeResult()
print("Monte Carlo:")
print("Integral1 = {}, error = {}".format(result1MC1, result1MC2))
print("Integral2 = {}, error = {}".format(result2MC1, result2MC2))
print("Quadrature:")
print("Integral1 = {}, error = {}".format(result1Quad1, result1Quad2))
print("Integral2 = {}, error = {}".format(result2Quad1, result2Quad2))
