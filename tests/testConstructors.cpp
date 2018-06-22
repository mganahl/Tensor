#include "typedefs.hpp"
#include <chrono>
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
Tensor <Complex>fun(){
  Tensor<Complex> c(2,2);
  c.randomize();
  return c;
}
int main(int argc, char** argv){

  int N=1000;
  // auto bla=Tensor<Complex>().random(4,4);
  // bla.print();  
  // bla=random<Complex>(ShapeType{2,2});
  // bla.print();
  // bla=random<Complex>(3,3);
  // bla.print();

  // bla=Tensor<Real>().random(4,4);
  // bla.print();  
  
  // bla=random<Real>(ShapeType{2,2});
  // bla.print();

  // bla=random<Real>(3,3);
  // bla.print();
  // bla.real().print();
  // bla.imag().print();
  
  // auto bla=random<Real>(4,4)*10;

  // auto bla1=random<Real>(4,4);
  // bla.print();
  // bla1.print();  
  auto bla2=random<Complex>(4,4);   
  // auto bla3=random<Real>(4,4);
  // auto bla4=random<Real>(4,4);  
  // int k1=1;
  // Real k2=1;
  // Complex k3=Complex(1);
  // Complex k4=Complex(1,1);    
  // auto out=bla2*k2;
  // auto out2=bla3*k4;



  // auto diff=bla3-bla4;
  // bla3.print();
  // out.print();
  // out2.print();
  auto t1=random<Complex>(N,N);
  auto t2=random<Complex>(N,N);
  auto t3=random<Complex>(N,N);
  auto t4=random<Complex>(N,N);
  auto t5=random<Complex>(N,N);
  auto t6=random<Complex>(N,N);
  
  auto time0 = std::chrono::high_resolution_clock::now();      
  bla2=random<Complex>(N,N)+random<Complex>(N,N)+random<Complex>(N,N)+random<Complex>(N,N)+random<Complex>(N,N)+random<Complex>(N,N);
  auto time1 = std::chrono::high_resolution_clock::now();
  std::cout << "sum took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";
  time0 = std::chrono::high_resolution_clock::now();      
  bla2=t1+t2+t3+t4+t5+t6; 
  time1 = std::chrono::high_resolution_clock::now();
  std::cout << "sum took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n"; 

  // cout<<endl;
  // bla2.print();
  // bla2+=bla4;
  // bla2.print();
  // bla2+=k2;
  // bla2.print();
  // bla2.print();
  // bla2+=k2;
  // bla2.print();    
  // bla2+=k3;
  //  bla2.print();
  //  bla2+=k4;
  //  bla2.print();  

  // cout<<endl;
  // bla2.print();
  // bla2*=k1;
  // bla2.print();
  // bla2*=k2;
  // bla2.print();    
  //bla2*=k3;
  //bla2.print();
  //bla2*=k4;
  //bla2.print();  

  return 0;
  

}
