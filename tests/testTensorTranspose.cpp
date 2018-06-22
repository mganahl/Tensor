#include "typedefs.hpp"
#include <iostream>
#include"lib/array/tensoroperations.hpp"
#include"lib/array/tensor.hpp"
#include"lib/utils/utilities.hpp"
#include"lib/utils/print.hpp"
#include <chrono>

#include "hptt.h"
using namespace std;
using  namespace tensor;
using printfunctions::print;
using  std::cout;
using  std::endl;
using  std::vector;


int main(int argc, char** argv){
  //test tensor transpose
  //auto t1=Tensor<complex<double> >(500,600,400);
  Tensor<complex<double> > t1(100,200,300);

  t1.randomize();
  t1.norm();    
  auto t2=t1;
  cout<<"== before my transpose ==============="<<endl;
  cout<<"t1(89,10,22)="<<t1(89,10,22)<<endl;
  auto time1 = std::chrono::high_resolution_clock::now();
  t1.naive_transpose(2,0,1);
  auto time2 = std::chrono::high_resolution_clock::now();
  std::cout << "transpose took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count() << " milliseconds\n";
  cout<<"== after tranposeing t1 using my transpose ==============="<<endl;
  cout<<"t1(22,89,10)="<<t1(22,89,10)<<endl;  
  cout<<"========before pauls transpose ========="<<endl;
  cout<<"t2(89,10,22)="<<t2(89,10,22)<<endl;
  auto time3 = std::chrono::high_resolution_clock::now();  
  t2.hptt(2,0,1);
  auto time4 = std::chrono::high_resolution_clock::now();  
  cout<<"========after pauls transpose ========="<<endl;  
  cout<<"t2(22,89,10)="<<t2(22,89,10)<<endl;
  std::cout << "hptt took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count() << " milliseconds\n";
  double d=norm(t1-t2);
  
  cout<<"difference between hptt and naive_transpose: "<<norm(t1-t2)<<endl;
  cout<<"testing transpose functions"<<endl;
  cout<<"t1 before t1.transpose()"<<endl;  
  cout<<"shape: ";print(t1.shape());
  cout<<"stride: ";print(t1.stride());  
  t1.transpose();
  cout<<"t1 after t1.transpose()"<<endl;  
  cout<<"shape: ";print(t1.shape());
  cout<<"stride: ";print(t1.stride());    
  
  return 0;
}
