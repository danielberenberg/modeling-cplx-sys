#ifndef Graph_hpp
#define Graph_hpp

#include <functional>

double euler(double ytm1, std::function<double (double)> dydt, double t, double h); 
/* computes euler's method: yn = yn-1 + fn * h */

#endif /* Graph_hpp */
