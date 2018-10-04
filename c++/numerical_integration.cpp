#include "numerical_integration.hpp"

template <class T>
double euler(double yi, double ti, double h, T dydt){
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

template <class T>
double heun(double yi, double ti, double h, T dydt){
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

template<class G, class H>
fn approximate(double y0, double h, int T, G dydt, H solver_func){
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
     param p = {};
     p.t = 0;
     p.y = y0;
     Fhat.push_back(p);
     i = 1;
     for (i=1; i<T; i++){
        ti = i*h;
        param pi; pi.t = ti; pi.y = solver_func(Fhat[i-1].t, Fhat[i-1].y,h,dydt); 
        Fhat.push_back(pi);  
     }

     return Fhat;
}

template<class G>
fn rk1(double y0, double h, int T,G dydt){
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

template<class G>
fn rk2(double y0, double h, int T, G dydt){
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
