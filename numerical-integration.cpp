#include "numerical-integration.hpp"

double euler(double y0, double dydt_0, double h){
    return y0 + dydt_0 * h;    
}
