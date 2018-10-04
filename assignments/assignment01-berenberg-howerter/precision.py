"""
Compute the error of each integration method
"""
from numerical import approximate, deriv_ivp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import linregress

def max_err(soln, appx):
    """
    find the maximum error in the approxiation against the solution

    args:
        :soln, appx (np.array) - the solution and approximation of the solution
                                 of the same length
    """

    return max(abs(soln - appx))

def linear_fit(X,y):
    slope, inter, _, _, _ = linregress(X,y)
    Xs = np.linspace(min(X),max(X),100)

    return Xs, Xs*slope + inter, slope, inter

if __name__ == "__main__":
    global A, n
    n = 3
    T = 50
    alphas = [0.75, 1.2, 1.5] # alphas to iterate over
    stepsizes = np.logspace(-2, 0, 50)# stepsizes to consider

    # define the interaction matrix
    A = np.array([[0.5, 0.5, 0.1],
                  [-0.5,-0.1,0.1],
                  [alphas[0],0.1,0.1]])

    initial_positions = [0.3, 0.2, 0.1] # xi_0 for each species i
    fig, axarr = plt.subplots(3,1,sharex=True,figsize=(7,7))
    for i, alpha in enumerate(alphas):
        A[2][0] = alpha
        max_err_e, max_err_h = [], []
        for h in stepsizes:
            print("\rComputing error: [a={:0.2f}, h={:0.2f}]".format(alpha,h),end="")
            t = np.arange(0, T, h)

            # the numpy numerical soln
            sol = solve_ivp(deriv_ivp(A), (0, T), initial_positions,t_eval=t, method="RK45")
            x1 = sol.y[0]
            Xe, Xh = approximate(A,initial_positions,h, T=len(t))
            (x1e, _, _), (x1h, _, _) = zip(*Xe), zip(*Xh)

            max_err_e.append(max_err(x1, x1e))
            max_err_h.append(max_err(x1, x1h))

        Xs_e, ye, slopee, intere = linear_fit(np.log10(stepsizes), np.log10(max_err_e))
        Xs_h, yh, slopeh, interh = linear_fit(np.log10(stepsizes), np.log10(max_err_h))

        axarr[i].plot(Xs_e, ye)
        axarr[i].plot(Xs_h, yh)
        #axarr[i].set_xticklabels([r"$10^{%d}$" % np.log(_) for _ in stepsizes])
        #axarr[i].set_yscale('log'); axarr[i].set_xscale('log')

        axarr[i].scatter(np.log10(stepsizes),np.log10(max_err_e),label=r"$m_{Euler}=%0.3f$" % slopee)
        axarr[i].scatter(np.log10(stepsizes),np.log10(max_err_h),label=r"$m_{Heun}=%0.3f$" % slopeh)
        axarr[i].set_ylabel(r"$\alpha={}$".format(alpha))

    axarr[0].legend()
    axarr[1].legend()
    axarr[2].legend(loc="upper left")
    axarr[2].set_xlabel("$\log_{10}(h)$")
    plt.savefig("figs/err_fn.pdf",bbox_inches='tight')
