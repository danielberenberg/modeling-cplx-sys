#include "numerical_integration.hpp"
#include <iostream>

double dydt(double t){
    return 2*t;
}

int main(int argc, char** argv){
    std::cout << euler(0,dydt,1,1);       

    return 0;
}
