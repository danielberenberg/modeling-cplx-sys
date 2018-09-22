#ifndef DYN_HPP_INCLUDED
#define DYN_HPP_INCLUDED

//#include <boost/multi_array.hpp>

#include <vector>
/**
 * @file   dyn.hpp
 * @brief  ODE system for 2 species predator-prey system
 *
 * @author  LHD
 * @since   2018-09-21
 */

struct Sparam {
    const std::vector< std::vector< double > > &A;
    const int dim;
}; // parameter structure

//********** function dydt definition **************************************************************
int dydt(double t, const double y[], double f[], void * param) {
// ODE system for 2 species predator-prey system

    // Cast parameters
    Sparam& p = *static_cast<Sparam* >(param);

    // Create multi_array reference to y and f
    //typedef boost::multi_array_ref<const double,2> CSTmatref_type;
    //typedef boost::multi_array_ref<double,2> matref_type;
    //typedef CSTmatref_type::index indexref;
    //CSTmatref_type yref(y,boost::extents[1][p.dim]);
    matref_type fref(f,boost::extents[1][p.dim]);

	
    // Compute derivatives
    // Order: x_1, x_2
    fref[0][0] = yref[0][0]*(p.A[0][0]*(1.0-yref[0][0])+p.A[0][1]*(1.0-yref[0][1]));
    fref[0][1] = yref[0][1]*(p.A[1][0]*(1.0-yref[0][0])+p.A[1][1]*(1.0-yref[0][1]));

    return GSL_SUCCESS;

} //********** end function dydt definition ********************************************************

#endif // DYN_HPP_INCLUDED
