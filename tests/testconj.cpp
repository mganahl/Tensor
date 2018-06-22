#include <array>
#include "typedefs.hpp"
#include <iostream>
#include <string>
#include <cstdarg>
#include <sstream>
#include <vector>
#include <list>
#include <complex>
#include <tuple>
#include"lib/array/tensoroperations.hpp"
#include"lib/array/tensor.hpp"
//#include "../lib/utils/print.hpp"
//using namespace printfunctions;
using std::cout;
using std::conj;
using std::endl;
using std::vector;
using namespace tensor;
using std::tuple;
//using namespace utilities;
//using namespace tensor;


int main(int argc, char** argv){
  Tensor<Complex> t(4);
  t.randomize(Complex(-1,-1),Complex(1,1));
  cout<<"tensor t"<<endl;
  t.print();
  auto tconj=t.conj();
  cout<<"t.conj()"<<endl;  
  tconj.print();
  auto t_=conjugate(t);
  cout<<"conj(t.conj())"<<endl;    
  t_.print();
  cout<<endl;
  Tensor<Real> t2(4);
  t2.randomize(-1,1);
  cout<<"tensor t (real)"<<endl;  
  t2.print();
  auto t2conj=t2.conj();
  cout<<"t.conj()"<<endl;    
  t2conj.print();
  auto t2_=conjugate(t2);  
  cout<<"conj(t.conj())"<<endl;    
  t2_.print();  
  Tensor<Complex> T(2,2);
  
  T.print();
  herm(T).print();
  
  return 0;  
}
