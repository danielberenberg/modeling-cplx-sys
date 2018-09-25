#ifndef numerical_integration_hpp
#define numerical_integration_hpp 

#include <functional>
#include <vector>

//typedef std::function<double (double,double)> Derivative;
//typedef std::function<double (double, double, double, Derivative)> Solver;

struct param {
    double t;
    double y;
};

typedef std::vector<param> fn; // pointwise function -- vector of len = 2 vectors (t, F(t))

template<class T>
double euler(double yi, double ti, double h, T dydt);
/* computes euler's method: yn = yn-1 + fn * h */

template<class T>
double heun(double yi, double ti, double h, T dydt);
/* computes heun's method = rk2 */

template<class G, class H>
fn approximate(double y0, double h, int T, G dydt, H solver_func);
/* use an integral solver in {euler, heun} to approximate int(dydt) */

template<class G>
fn rk1(double y0, double h, int T, G dydt);
/* approximate int(dydt) from t=0 ... T using euler's */

template<class G>
fn rk2(double y0, double h, int T, G dydt);
/* approximate int(dydt) from t=0 ... T using heun's */

#endif
