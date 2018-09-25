#ifndef numerical_integration_hpp
#define numerical_integration_hpp 

#include <functional>
#include <vector>

typedef std::function<double (double,double)> Derivative;
typedef std::function<double (double, double, double, Derivative)> Solver;
typedef std::vector<std::vector<double>> fn; // pointwise function -- vector of len = 2 vectors (t, F(t))

double euler(double yi, double ti, double h, Derivative dydt);
/* computes euler's method: yn = yn-1 + fn * h */

double heun(double yi, double ti, double h, Derivative dydt);
/* computes heun's method = rk2 */

fn approximate(const double y0, const double h, const int T, Derivative dydt, Solver solver);
/* use an integral solver in {euler, heun} to approximate int(dydt) */

fn rk1(const double y0, const double h, const int T, Derivative dydt);
/* approximate int(dydt) from t=0 ... T using euler's */

fn rk2(const double y0, const double h, const int T, Derivative dydt);
/* approximate int(dydt) from t=0 ... T using heun's */

#endif
