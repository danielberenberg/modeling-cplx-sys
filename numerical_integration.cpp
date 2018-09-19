#include "numerical_integration.hpp"
#include <functional>

double euler(double ytm1, std::function<double (double)> dydt, double t, double h){
    /*
     * compute Euler's method. Given the derivative dy/dt of an unknown function F,
     * approximate F(t) = y_{t-1} + dy/dt (t-h) * h
     *
     * args:
     *     :(double) ytm1 - the value of y at t-1
     *     :(std::function<double (double)> dydt - derivative of F 
     *     :(double) t - timestep of interest
     *     :(double) h - width of a step 
     * 
     * returns:
     *     :(double) approx F(t)
     */
    return ytm1 + dydt(t-h)*h; 
}

