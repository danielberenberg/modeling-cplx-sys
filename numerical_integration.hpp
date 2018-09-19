#ifndef numerical_integration_hpp
#define numerical_integration_hpp 

#include <functional>

double euler(double ytm1, std::function<double (double)> dydt, double t, double h); 
/* computes euler's method: yn = yn-1 + fn * h */

#endif
