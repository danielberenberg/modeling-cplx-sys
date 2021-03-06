"""
Plot the 3-species lotka volterra system
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

n = 3

def species_deriv(X,A):
    """
    compute the derivative of the position of each of the species in
    the LK model

    args:
        :X (list) - positions of each xi

    returns:
        :(np.array) - dxi/dt
    """

    dXdt = np.zeros(3)

    for i, xi in enumerate(X):
        dXdt[i] = xi*(sum([A[i][j]*(1-xj) for j,xj in enumerate(X)]))
    return dXdt

def deriv_ivp(A):
    def deriv(t, X):
        return species_deriv(X,A)

    return deriv

def Euler(X,A,h):
    """
    approximate the value of X at a timestep distance h away from the current tstep
    using Euler's method

    args:
        :X (np.array) - the positions of each xi
        :h (float) - stepsize

    returns:
        :(np.array) - appx location of each at ti+1
    """
    dXdt = species_deriv(X,A)

    #return np.array(X + np.multiply(h,dXdt))
    return np.array([xi + dXdt[i]*h for i,xi in enumerate(X)],dtype=np.float64)


def Heun(X,A,h):
    """
    approximate the value of X at a timestep distance h away from the current tstep
    using Heun's method

    args:
        :X (np.array) - the positions of each xi
        :h (float) - stepsize

    returns:
        :(np.array) - appx location of each at ti+1
    """

    Xp1 = Euler(X,A,h)
    dXp1dt = species_deriv(Xp1,A)
    dXdt = species_deriv(X,A)

    #return np.array(X + np.multiply((h/2),np.multiply(dXdt, dXp1dt)))
    return np.array([xi + (h/2)*(dXdt[i] + dXp1dt[i]) for i,xi in enumerate(X)],dtype=np.float64)

def approximate(A,initial_positions,h,T=200):
    # zero out the approximations
    Xe = np.zeros((T,n),dtype=np.float64)
    Xh = np.zeros((T,n),dtype=np.float64)
    Xe[0] = initial_positions
    Xh[0] = initial_positions
    for i in range(1, T):
        Xe[i] = Euler(Xe[i-1],A,h)
        Xh[i] = Heun(Xh[i-1],A,h)

    return Xe, Xh

if __name__ == "__main__":
    T = 200 # max timestep
    n = 3   # number of species

    alphas = [0.75, 1.2, 1.5] # alphas to iterate over
    stepsizes = [0.1,0.5,1.] # stepsizes to consider

    # define the interaction matrix
    A = np.array([[0.5, 0.5, 0.1],
                  [-0.5,-0.1,0.1],
                  [alphas[0],0.1,0.1]])

    initial_positions = [0.3, 0.2, 0.1] # xi_0 for each species i

    # solve the 3-species system for each h for each alpha
    k=0
    for h in stepsizes:
        t = np.arange(0,T,h)
        for alpha in alphas:
            print("\rSolving 3-species LK system [a={:.02f},h={:.02f}]".format(alpha,h),end="",flush=True)
            A[2][0] = alpha
            X_ = solve_ivp(deriv_ivp(A), (0, T), initial_positions,t_eval=t, method="RK45")
            x1,x2,x3 = X_.y[0], X_.y[1], X_.y[2]
            Xe, Xh = approximate(A,initial_positions,h, T=len(t))
            x1e, x2e, x3e = zip(*Xe)
            x1h, x2h, x3h = zip(*Xh)

            fig, axarr = plt.subplots(3,1)
            for i, ax in enumerate(axarr):
                ax.set_ylabel("$x_{}$".format(i+1),fontsize=14)
                ax.set_xlabel("$t$",fontsize=14)
                #plt.suptitle(r"3-Species approximation for $\alpha={},h={}$".format(alpha,h))


            if (alpha, h) != (1.5,1):
                axarr[0].plot(t,x1e,label="Euler")
                axarr[0].plot(t,x1h,label="Heun")
                axarr[0].plot(X_.t,x1,label="Builtin RK45",color="k")
                axarr[0].legend()

                axarr[1].plot(t,x2e)
                axarr[1].plot(t,x2h)
                axarr[1].plot(X_.t,x2,color="k")

                axarr[2].plot(t,x3e)
                axarr[2].plot(t,x3h)
                axarr[2].plot(X_.t,x3,color="k")

            else:

                # filter out ridiculusly large or small values to make the figure
                # more visually appealing
                x1e_p = np.array(list(filter(lambda x: x == x and x > 1e-3 and x < 1e+2, x1e)))
                t1e = t[:len(x1e_p)]

                x2e_p = np.array(list(filter(lambda x: x == x and x > 1e-3 and x < 10, x2e)))
                t2e = t[:len(x2e_p)]

                x3e_p = np.array(list(filter(lambda x: x == x and x > 1e-3 and x < 10, x3e)))
                t3e = t[:len(x3e_p)]

                axarr[0].plot(t1e,x1e_p, label="Euler")
                axarr[0].plot(t,x1h, label="Heun")
                axarr[0].plot(X_.t,x1, label="Builtin RK45",color="k")
                axarr[1].plot(t2e,x2e_p)
                axarr[1].plot(t,x2h)
                axarr[1].plot(X_.t,x2,color="k")
                axarr[2].plot(t3e,x3e_p)
                axarr[2].plot(t,x3h)
                axarr[2].plot(X_.t,x3,color="k")

            if k==6:
                axarr[0].legend(loc='upper right')
            else:
                axarr[0].legend().set_visible(False)
            plt.savefig("figs/3SpeciesApprox{}_{}.pdf".format(int(alpha*100),int(h*10)),bbox_inches='tight')
            k+=1
