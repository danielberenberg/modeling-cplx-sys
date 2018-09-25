#include "numerical_integration.hpp"
#include <iostream>

/*
 * Numerically solve the 3-species using both integrators (Heun, Euler)
 *
 * args:
 *     :alpha (double) -- tunable parameter inside interaction matrix
 *     :h (double)     -- step size
 */

double A;

int main(int argc, char** argv){
    double alpha, h;
    
    if (argc != 2) { exit(-1);}
    
    // parse in alpha and h values
    alpha = atof(argv[0]);
    h = atof(argv[1]);
    
    // define interaction matrix
    double A[3][3] = 
    {
        { 0.5,  0.5, 0.5},
        {-0.5, -0.1, 0.1},
        {alpha, 0.1, 0.1}
    };

}
