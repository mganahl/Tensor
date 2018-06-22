#include "typedefs.hpp"
#include <iostream>
#include"../lib/array/tensor.hpp"
#include "../lib/utils/print.hpp"
#include"lib/array/tensoroperations.hpp"
using namespace printfunctions;

using namespace std;
using namespace tensor;
using  std::cout;
using  std::endl;
using  std::vector;
int main(int argc, char** argv){
  auto t1=Tensor<Complex>(2,3,4);
  Tensor<Complex> t2;
  auto t3=&t2;
  cout<<"testing resize"<<endl;
  t3->resize(10,20,20);
  print(t3->shape());

  cout<<"testing randomize"<<endl;  
  t1.randomize(Complex(-1,-1),Complex(1,1));
  print(t1.shape());
  t1.resize(4,5,6);
  print(t1.shape());


  cout<<"testing zeros"<<endl;
  auto t4=Tensor<Real>().zeros(1,2,3);
  t4.print();
  auto t5=Tensor<Complex>().zeros(1,2,3);
  t5.print();
  cout<<"testing ones"<<endl;
  auto t6=Tensor<Real>().ones(1,2,3);
  t6.print();
  auto t7=Tensor<Complex>().ones(1,2,3);
  t7.print();
  cout<<"testing random"<<endl;  
  auto t8=Tensor<Real>().random(1,2,3);
  t8.print();
  auto t9=Tensor<Complex>().random(1,2,3);
  t9.print();
  cout<<"testing after randomization"<<endl;    
  t9.randomize(Complex(-1,-1),Complex(1,1));
  t9.print();


  cout<<"testing diag"<<endl;
  cout<<"creating random 4 by 4 tensor T:"<<endl;
  auto o1=Tensor<Complex>().random(4,4);
  o1.print();

  cout<<"diag(T):"<<endl;
  auto do1=diag(o1);
  do1.print();
  cout<<"diag(diag(T)):"<<endl;
  auto do2=diag(do1);
  do2.print();
  return 0;

  
  cout<<"testing transpose"<<endl;
  
}
