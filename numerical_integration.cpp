#include "numerical_integration.hpp"

double euler(double yi, double ti, double h, const Derivative dydt){
    /*
     * Approximate the position of F = integral(dydt) at timestep t_(i+1).
     *
     * args:
     *     :yi (double) - position of F at timestep t_i
     *     :ti (double) - our current timestep
     *     :h  (double  - stepsize
     *     :dydt (Derivative) - function accepting two input arguments (t, y),
     *                          returning f(ti, yi) = dF/dt |_{t=ti,y=yi}
     *
     * returns:
     *     :(double) -- approximation of F(t_{i+1}) 
     */

    return yi + dydt(ti, yi) * h;
}

double heun(double yi, double ti, double h, const Derivative dydt){
    /*
     * Approximate the position of F = integral(dydt) at timestep t_{i+1}
     * by using runge-kutta 2nd order aka heun's method aka modified euler's.
     *
     * Main difference being that we look two timesteps into the future and evaluate
     * dydt at each of those timesteps, find the average between the two slopes.
     *
     * args:
     *     :yi (double) - position of F at timestep t_i 
     *     :ti (double) - current timestep
     *     :h  (double) - stepsize
     *     :dydt (derivative) - function accepting two input args (t,y),
     *                          returning f(ti,yi) = dF/dt |_{t=ti, y=yi}
     *
     * returns:
     *     :(double) -- approximation of F(t_{i+1})
     */

     double yip1 = euler(yi,ti,h,dydt);     
     return yi + (h/2)*(dydt(ti, yi) + dydt(ti + h, yip1));
}

fn approximate(const double y0, const double h, const int T, Derivative dydt, Solver solver){
    /*
     * Steps through an approximation for T steps of F = integral(dydt) using the
     * approximation method given by solver
     *
     * args:
     *     :y0 (double) - initial condition
     *     :h (double)  - stepsize
     *     :T (int)     - max integration steps
     *     :dydt (Derivative) - derivative function 
     *     :solver (Solver) - approximation method that will be used
     *
     * returns:
     *     :fn - vector of arrays (t, Fhat) symbolizing the approximation of F at timestep t = Fhat
     */

     fn Fhat; 
     int i;
     double ti;

     Fhat.push_back({0,y0});
     i = 1;
     for (i=1; i<T; i++){
        ti = i*h;
        Fhat.push_back({ti,solver(Fhat[i-1][0],Fhat[i-1][1],h,dydt)});  
     }
     return Fhat;
}

fn rk1(const double y0, const double h, const int T, Derivative dydt){
    /*
     * wrapper for batch approximating values of y(t) for t=0,...,T (Euler)
     *
     * args:
     *     :y0 (double) - init. cond. for y(t)
     *     :h  (double) - step size
     *     :T  (int)    - max integration steps
     *     :dydt (Derivative) - of f(t,y) = dy/dt
     *
     * returns:                                                             
     *     :(fn) -- length T vector of length 2 vectors symbolizing each ti,yi for each t
     */
    return approximate(y0, h, T, dydt, euler);
}


fn rk2(const double y0, const double h, const int T, Derivative dydt){
    /*
     * wrapper for batch approximating values of y(t) for t=0,...,T (Heun)
     *
     * args:
     *     :y0 (double) - init. cond. for y(t)
     *     :h  (double) - step size
     *     :T  (int)    - max integration steps
     *     :dydt (Derivative) - of f(t,y) = dy/dt
     *
     * returns:                                                             
     *     :(fn) -- length T vector of length 2 vectors symbolizing each ti,yi for each t
     */
    return approximate(y0, h, T, dydt, heun);
}
