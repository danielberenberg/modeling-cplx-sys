#include "numerical-integration.hpp"
#include <functional>

double euler(double ytm1, std::function<double (double)> dydt, double t, double h){
    /*
     * compute Euler's method. Given the derivative dy/dt of an unknown function F,
     * approximate F(t) = y_{t-1} + dy/dt (t-h) * h
     */
    return ytm1 + dydt(t-h)*h; 
}

