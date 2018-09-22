#ifndef numerical_integration_hpp
#define numerical_integration_hpp 

#include <functional>
#include <vector>

typedef std::function<double (double,double)> Derivative;
typedef std::function<double (double, double, double, Derivative)> Solver;
typedef std::vector<std::vector<double>> fn;

double euler(double yi, double ti, double h, Derivative dydt);
/* computes euler's method: yn = yn-1 + fn * h */

double heun(double yi, double ti, double h, Derivative dydt);
/* computes heun's method = rk2 */

#endif
