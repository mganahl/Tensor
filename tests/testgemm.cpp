#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include"lib/array/tensoroperations.hpp"
#include "../lib/array/tensor.hpp"
#include "../lib/utils/print.hpp"
#include "typedefs.hpp"
#include "lib/linalg/tblisroutines.hpp"

#include <chrono>
using std::min;
using namespace printfunctions;
using namespace tensor;

int main(int argc, char** argv){
  std::istringstream ss1(argv[1]);
  uint Nmax;  
  ss1>>Nmax;
  
  ShapeType s1({200,37,41,100});
  ShapeType s2({84,200,100});  
  LabelType L1{1,-2,-3,2};
  LabelType L2{-1,1,2};
  
    
  auto t1=Tensor<Real >(s1);
  auto t2=Tensor<Real >(s2);
  Tensor<Real> t3;      
  cout<<"contracting tensors of following shapes"<<endl;
  cout<<"tensor A: ";
  print(s1);
  cout<<"tensor B: ";
  print(s2);

  auto time0 = std::chrono::high_resolution_clock::now();  
  t1.randomize();
  t2.randomize();
  auto time1 = std::chrono::high_resolution_clock::now();
  std::cout << "initialization took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";      
  gemmtensordot(t1,t2,L1,L2,t3);

  vector<double> time(Nmax);
  for(uint n=0;n<Nmax;n++){
    time0 = std::chrono::high_resolution_clock::now();
    gemmtensordot(t1,t2,L1,L2,t3);
    time1 = std::chrono::high_resolution_clock::now();
    time[n]=std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count()*1.0;
  }
  auto t4=t1.gemmdot(t2,LabelType(L1),LabelType(L2));
  if ((t4-t3).norm()>1E-10){
    cout<<"tblisdot gave wrong result!!!"<<endl;
  }

  cout<< "GEMM: fastest of "<<Nmax<<" contractions: "<<*std::min_element(time.begin(),time.end()) << " milliseconds\n";      
  cout<< "GEMM: average time per contraction: "<<std::accumulate(time.begin(),time.end(),0.0)*1.0/Nmax << " milliseconds\n";      
  return 0;
}
