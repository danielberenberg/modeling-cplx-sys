/**
 * @file   tevol_source.hpp
 * @brief  ODE system for 2 species predator-prey system
 *
 * WARNING: This code requires the GNU Scientific Library
 *
 * compile with:
 * g++ -std=c++11 -O3 -o tevol_source ./tevol_source.cpp $(gsl-config --cflags) $(gsl-config --libs)
 *
 * run with:
 * ./tevol_source 0.15 0.25 0.4 50.0 0.1
 *
 * @author  LHD
 * @since   2018-09-21
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>

#include <boost/multi_array.hpp>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv.h>

#include "dyn.hpp"

using namespace std;

int main(int argc, const char *argv[]) {
		 
	//Model parameters	
	double alpha = atof(argv[1]); //interaction parameter
    double x1 = atof(argv[2]); //initial conditions for x_1
    double x2 = atof(argv[3]); //initial conditions for x_2
    double T = atof(argv[4]); //maximum integration time
    double h = atof(argv[5]); //integrator step
    const int dim = 2;

    // Interaction matrix from Flake 12.3
    vector< vector< double > > A(dim, vector< double > (dim,0.0));
    A[0][0] = 0.5; A[0][1] = 0.1;
    A[1][0] = alpha; A[1][1] = 0.1;
    Sparam param = {A,dim};

    // Integrator parameters
    double t = 0;
    double dt = 1e-8;
    double t_step = h;
    const double eps_abs = 1e-10;
    const double eps_rel = 1e-10;

    // Setting initial conditions
    typedef boost::multi_array<double,2> mat_type;
    typedef mat_type::index index;
    mat_type y(boost::extents[1][dim]);
    fill(y.data(),y.data()+y.num_elements(),0.0);
    // Initial conditions
	y[0][0] = x1; //x_1
	y[0][1] = x2; //x_2

    // Define GSL odeiv parameters
    const gsl_odeiv_step_type * step_type = gsl_odeiv_step_rkf45;
    gsl_odeiv_step * step = gsl_odeiv_step_alloc (step_type, dim);
    gsl_odeiv_control * control = gsl_odeiv_control_y_new (eps_abs,eps_rel);
    gsl_odeiv_evolve * evolve = gsl_odeiv_evolve_alloc (dim);
    gsl_odeiv_system sys = {dydt, NULL, dim, &param};
	
	//Integration
    int status(GSL_SUCCESS);
    for (double t_target = t+t_step; t_target < T; t_target += t_step ) { //stop by time
        while (t < t_target) {
            status = gsl_odeiv_evolve_apply (evolve,control,step,&sys,&t,t_target,&dt,y.data());
            if (status != GSL_SUCCESS) {
				cout << "SNAFU" << endl;
                break;
			}
        } // end while
		cout << t << " " << y[0][0] << " " << y[0][1] << "\n";
	} //end for loop

    // Free memory
    gsl_odeiv_evolve_free(evolve);
    gsl_odeiv_control_free(control);
    gsl_odeiv_step_free(step);
    
    return 0;
}
