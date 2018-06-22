#include "typedefs.hpp"
#include <iostream>
#include <chrono>
#include "blasroutines.hpp"
#include <chrono>
#include <thread>

using  std::cout;
using  std::endl;

using namespace std::chrono;
using namespace blasroutines;
int main(int argc, char** argv){
  size_type size=4000;
  Real *v1=new Real [size];
  Real *v2=new Real [size];
  dot_prod(size,v1,v2);
  Complex *vc1=new Complex [size];
  Complex *vc2=new Complex [size];
  
  dot_prod(size,vc1,vc2);


  Real *m1=new Real [size*size];
  Real *m2=new Real [size*size];
  Real *m3=new Real [size*size];  

  Complex *mc1=new Complex [size*size];
  Complex *mc2=new Complex [size*size];
  Complex *mc3=new Complex [size*size];  
  

  auto start = high_resolution_clock::now();
  mat_mat_prod(0.1,m1 ,size,size, m2, size,size,m3,'n','n');
  auto stop = high_resolution_clock::now();
  std::chrono::duration<double> diff = stop - start;
  cout << "Time taken by function: "<< diff.count() << " seconds" << endl;
  start = high_resolution_clock::now();
  mat_mat_prod(0.1,mc1 ,size,size, mc2, size,size,mc3,'n','n');
  stop = high_resolution_clock::now();
  diff = stop - start;
  cout << "Time taken by function: "<< diff.count() << " seconds" << endl;
   
  return 0;
}
