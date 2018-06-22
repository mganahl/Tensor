#include "typedefs.hpp"
#include <iostream>
#include "lapackroutines.hpp"
#include <chrono>
#include <assert.h>
using namespace std;
using namespace lapackroutines;
using  std::cout;
using  std::endl;


int main(int argc, char** argv){
  double*matrix=new double [100*100];
  double*U=new double [100*100];
  double*VT=new double [100*100];
  double* sing_vals= new double [100];
  svd(matrix, U,VT,sing_vals, 100,100,'S');
  return 0;
}
