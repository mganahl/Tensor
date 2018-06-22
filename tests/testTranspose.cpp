#include "lib/typedefs.hpp"
#include <iostream>
#include"lib/array/tensor.hpp"
#include"lib/utils/utilities.hpp"
#include <chrono>

using namespace std;
using  std::cout;
using  std::endl;
using  std::vector;
int main(int argc, char** argv){
  cout<<"Testing constructors"<<endl;
  auto t1r=Tensor<double >(100,200,100);
  auto t1r_=Tensor<double >(std::vector<uint>{100,200,100});  
  auto t1c=Tensor<complex<double> >(100,200,100);
  auto t1c_=Tensor<complex<double> >(std::vector<uint>{100,200,100});
  cout<<"Testing done"<<endl;  
  cout<<"Testing tranpose operation for double"<<endl;

  t1r.randomize();
  auto t2r=t1r;
  cout<<"== before my transpose ==============="<<endl;
  cout<<"t1r(89,10,22)="<<t1r(89,10,22)<<endl;
  auto time1 = std::chrono::high_resolution_clock::now();
  t1r.naive_transpose(2,0,1);
  auto time2 = std::chrono::high_resolution_clock::now();
  std::cout << "transpose took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count() << " milliseconds\n";
  cout<<"== after tranposeing t1r using my transpose ==============="<<endl;
  cout<<"t1r(22,89,10)="<<t1r(22,89,10)<<endl;  
  cout<<"========before pauls transpose ========="<<endl;
  cout<<"t2r(89,10,22)="<<t2r(89,10,22)<<endl;
  time1 = std::chrono::high_resolution_clock::now();  
  t2r.hptt(2,0,1);
  time2 = std::chrono::high_resolution_clock::now();  
  cout<<"========after pauls transpose ========="<<endl;  
  cout<<"t2r(22,89,10)="<<t2r(22,89,10)<<endl;
  std::cout << "hptt took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count() << " milliseconds\n";

  cout<<"Testing tranpose operation for complex<double>"<<endl;
  t1c.randomize();
  auto t2c=t1c;
  cout<<"== before my transpose ==============="<<endl;
  cout<<"t1c(89,10,22)="<<t1c(89,10,22)<<endl;
  time1 = std::chrono::high_resolution_clock::now();
  t1c.naive_transpose(2,0,1);
  
  time2 = std::chrono::high_resolution_clock::now();
  std::cout << "transpose took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count() << " milliseconds\n";
  cout<<"== after tranposeing t1c using my transpose ==============="<<endl;
  cout<<"t1c(22,89,10)="<<t1c(22,89,10)<<endl;  
  cout<<"========before pauls transpose ========="<<endl;
  cout<<"t2c(89,10,22)="<<t2c(89,10,22)<<endl;
  time1 = std::chrono::high_resolution_clock::now();  
  t2c.hptt(2,0,1);
  time2 = std::chrono::high_resolution_clock::now();  
  cout<<"========after pauls transpose ========="<<endl;  
  cout<<"t2c(22,89,10)="<<t2c(22,89,10)<<endl;
  std::cout << "hptt took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count() << " milliseconds\n";  

  return 0;
}
