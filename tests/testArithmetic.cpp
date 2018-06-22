#include "typedefs.hpp"
#include <iostream>
#include <iostream>
#include"lib/array/tensoroperations.hpp"
#include"lib/array/tensor.hpp"
#include"lib/utils/utilities.hpp"
#include"lib/utils/print.hpp"
#include <assert.h>
using namespace std;
using namespace tensor;
using namespace printfunctions;
using  std::cout;
using  std::endl;
using  std::vector;
int main(int argc, char** argv){

  //test random generation of tensor
  auto t1=Tensor<Complex>(2,10,3,4,5);
  t1.randomize(Complex(-1,-1),Complex(1,1));
  auto t2=t1;
  auto t3=t1+t2;

  auto t4=t3-(t1+t2);
  for(uint n=0;n<t1.size();n++){
    assert(t4.data(n)==Complex(0,0));
    assert(t3.data(n)==t1.data(n)+t2.data(n));
    assert(t1.data(n)==t2.data(n));
  }

  t3+=t1;


  int N=6;
  auto t=Tensor<Complex >(N,N);
  auto mat=Tensor<Real >(N,N);
  mat.randomize();
  t.randomize(Complex(-1,-1),Complex(1,1));

  Real v=2.0;
  Complex v2(2.0,2.0);

  auto sum1=mat+t;
  auto sum2=v2*mat+t*v;
  sum1.print();
  sum2.print();  

  auto out1=t*v;
  auto out2=v*t;
  auto out3=t*v2;
  auto out4=v2*t;  
  
  out3.print();
  (out1-out2).print();
  (out3-out4).print();

  auto bla1=mat*v2;
  auto bla2=v2*mat;  
  auto bla3=mat*v;
  auto bla4=v*mat;  

  (bla1-bla2).print();
  (bla3-bla4).print();



  auto outo1=t+v;
  auto outo2=v+t;
  auto outo3=t+v2;
  auto outo4=v2+t;  
  t.print();
  outo3.print();
  (outo1-outo2).print();
  (outo3-outo4).print();

  auto blab1=mat+v2;
  auto blab2=v2+mat;  
  auto blab3=mat+v;
  auto blab4=v+mat;  

  (blab1-blab2).print();
  (blab3-blab4).print();  

  cout<<"test OK"<<endl;
  
}
