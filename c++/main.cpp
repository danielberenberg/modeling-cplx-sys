#include "numerical_integration.hpp"
#include <iostream>
#include <vector>

std::vector<double> species_deriv(std::vector<double> xi, double A[3][3]){
    /*
     * Output the derivative of each function given the initial positions
     * of each species in the system
     *
     * args:  
     *     :xi (vector<double>) -- initial position of each species for this ti
     */  
    std::vector<double> xs(3);
    
    // definition of derivatives given by:
    // xi * Sum(Aij * (1-xj))
    //       j
    
    xs[0] = xi[0] * (A[0][0]*(1-xi[0]) + A[0][1]*(1-xi[1]) + A[0][2]*(1-xi[2]));
    xs[1] = xi[1] * (A[1][0]*(1-xi[0]) + A[1][1]*(1-xi[1]) + A[1][2]*(1-xi[2])); 
    xs[2] = xi[2] * (A[2][0]*(1-xi[0]) + A[2][1]*(1-xi[1]) + A[2][2]*(1-xi[2]));

    return xs;
}

std::vector<double> euler(double h, std::vector<double> dydt, std::vector<double> xi){
    /*
     * Approximate the position of each of the species using euler's method
     *
     * args:
     *     :h  - stepsize
     *     :dydt - derivatives at this position
     *     :xi - positions of each animal
     *
     * returns:
     *     :vector of doubles denoting the position of each animal at the next timestep
     */ 
    

    std::vector<double> xs(3);
    
    // compute positions
    xs[0] = xi[0] + dydt[0]*h; 
    xs[1] = xi[1] + dydt[1]*h;
    xs[2] = xi[2] + dydt[2]*h;

    return xs;
}

std::vector<double> heun(double h, std::vector<double> dydt, std::vector<double> xi, std::vector<std::vector<double> > A){
    /** Approximate the position of each of the species using heun's method
     *
     * args:
     *     :h  - stepsize
     *     :dydt - derivatives at this position
     *     :xi - positions of each animal
     *
     * returns:
     *     :vector of doubles denoting the position of each animal at the next timestep
     */ 

    std::vector<double> xip1_appx = euler(h, dydt, xi); // approximate next positions 
    std::vector<double> dydtp1 = species_deriv(xip1_appx, A); // find the derivative at ti+1 (tip1)

    std::vector<double> xs(3);
    
    // compute positions
    xs[0] = xi[0] + (h/2)*(dydt[0] + dydtp1[0]); 
    xs[1] = xi[1] + (h/2)*(dydt[1] + dydtp1[1]);
    xs[2] = xi[2] + (h/2)*(dydt[2] + dydtp1[2]);

    return xs;
}


/*
 * Numerically solve the 3-species using both integrators (Heun, Euler)
 *
 * args:
 *     :alpha (double) -- tunable parameter inside interaction matrix
 *     :h (double)     -- step size
 */
int main(int argc, char** argv){
    double alpha, h;

    if (argc != 3) { 
        std::cout << "[!!] Not enough arguments\n --| alpha --> ";
        std::cout << "the tuning parameter\n --| h ------> the step size";
        std::cout << std::endl;
        exit(-1);
    }
    
    // parse in alpha and h values
    alpha = atof(argv[0]);
    h = atof(argv[1]);
    
    // out current ti and max timestep
    int i, T; 
    T = 200;
    
    // define interaction matrix
    double A[3][3] = 
    {
        { 0.5,  0.5, 0.5},
        {-0.5, -0.1, 0.1},
        {alpha, 0.1, 0.1}
    };
    
    std::vector<std::vector<double> > appx_h(T);
    std::vector<std::vector<double> > appx_e(T);
    
    std::vector<double> initial_pos(3); initial_pos[0] = 0.3; initial_pos[1] = 0.2; initial_pos[2] = 0.1;
    appx_h.push_back(initial_pos);
    appx_e.push_back(initial_pos);
    
    double ti;
    std::vector<double> dydt_e, dydt_h;
    // perform the appx
    for (i=0; i<T; i++){     
        ti = i*h;
        dydt_e = species_deriv(appx_e[i], A); 
        appx_e.push_back(euler(
    }

    //std::cout << alpha << std::endl;
    //std::cout << h << std::endl;

}
