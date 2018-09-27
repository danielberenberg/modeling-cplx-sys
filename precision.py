"""
Compute the error of each integration method
"""
from numerical import Euler, Heun, approximate, deriv_ivp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

if __name__ == "__main__":
    
    alphas = [0.75, 1.2, 1.5] # alphas to iterate over
    stepsizes = np.logspace(0.01, 1., 50)# stepsizes to consider

    # define the interaction matrix
    A = np.array([[0.5, 0.5, 0.1],
                  [-0.5,-0.1,0.1],
                  [alphas[0],0.1,0.1]])
    
    initial_positions = [0.3, 0.2, 0.1] # xi_0 for each species i 

    for alpha in alphas:
        for h in stepsizes:

