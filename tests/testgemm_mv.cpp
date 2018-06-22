#include <iostream>
#include"lib/array/tensoroperations.hpp"
#include <chrono>
#include <algorithm>
#include <stdlib.h>
#include <tblis.h>
#include <complex>
//#include <util/basic_types.h>
#include <vector>
#include <set>
#include<time.h>
#include <chrono>
#include "../lib/array/tensor.hpp"
#include "../lib/utils/print.hpp"
#include "typedefs.hpp"
#include "lib/linalg/tblisroutines.hpp"
using std::set;
using std::min;
using tensor::Tensor;
using namespace printfunctions;

using tblis::tblis_init_tensor_z;
using tblis::tblis_init_tensor_d;
using tblis::len_type;
using tblis::tblis_tensor;
using tblis::stride_type;
using tblis::label_type;

using namespace tensor;

using std::vector;
using namespace std;
int main(int argc, char** argv){
  auto t1=Tensor<Complex >(2,2);
  auto t2=Tensor<Complex>(2);
  Tensor<Complex> r1(2),r2(2),r3(2);
  bool OK=true;
  r1(0)=Complex(-1.,5.);
  r1(1)=Complex(-1.,3.4);
  
  r2(0)=Complex(-0.2,2.6);
  r2(1)=Complex(-1.,5);
  
  r3(0)=Complex(2.6,0.2);
  r3(1)=Complex(5,1);

  t1(0,0)=Complex(1.0,1.0);
  t1(0,1)=Complex(1.0,1.0);
  t1(1,0)=Complex(0.2,0.2);
  t1(1,1)=Complex(1.0,1.0);

  t2(0)=Complex(1.0,1.0);
  t2(1)=Complex(1.0,2.0);
  Tensor<Complex> t3;
  cout<<"t1="<<endl;
  t1.print();
  cout<<"t2="<<endl;  
  t2.print();  
  cout<<"starting gemm:"<<endl;  
  auto time0 = std::chrono::high_resolution_clock::now();
  gemm_mv_dot(t1,t2,t3);
  auto time1 = std::chrono::high_resolution_clock::now();    
  cout<< "GEMM: time per contraction: "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";
  t1.print();

  if (norm(t3-r1)>1E-10){
    cout<<"t1.dot(t2) was wrong for Complex numbers"<<endl;
    OK=false;
  }
  gemm_mv_dot(t1,t2,t3,'T');
  if (norm(t3-r2)>1E-10){
    cout<<"t1.transpose().dot(t2) was wrong for Complex numbers"<<endl;
    OK=false;
  }
  gemm_mv_dot(t1,t2,t3,'C');  
  if (norm(t3-r3)>1E-10){
    cout<<"t1.transpose().dot(t2) was wrong for Complex numbers"<<endl;
    OK=false;
  }

  auto t4=Tensor<Real >(2,2);
  auto t5=Tensor<Real>(2);
  t4(0,0)=1.0;
  t4(0,1)=1.0;
  t4(1,0)=0.2;
  t4(1,1)=1.0;

  t5(0)=1.0;
  t5(1)=1.0;
  Tensor<Real> t6,r6(2),t7,r7(2);
  r6(0)=2.0;
  r6(1)=1.2;
  r7(0)=1.2;
  r7(1)=2.0;

  
  cout<<"starting gemm:"<<endl;  
  time0 = std::chrono::high_resolution_clock::now();
  gemm_mv_dot(t4,t5,t6);
 
  time1 = std::chrono::high_resolution_clock::now();    
  cout<< "GEMM: time per contraction: "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";
  if (norm(t6-r6)>1E-10){
    cout<<"t1.dot(t2) was wrong for Real numbers"<<endl;
    OK=false;
  }
  gemm_mv_dot(t4,t5,t7,'T');  
  if (norm(t7-r7)>1E-10){
    cout<<"t1.dot(t2) was wrong for Real numbers"<<endl;
    OK=false;
  }
  if (OK=true){
    cout<<"gemm_mv_dot passed all tests"<<endl;
  }
  
  return 0;
}
