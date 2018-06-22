#include "typedefs.hpp"
#include <iostream>
#include"lib/array/tensoroperations.hpp"
#include"lib/array/tensor.hpp"

#include"lib/utils/utilities.hpp"
#include"lib/utils/print.hpp"
#include"lib/linalg/lapackroutines.hpp"
#include <chrono>

using namespace std;
using namespace tensor;
using namespace printfunctions;
using  std::cout;
using  std::endl;
using  std::vector;

using namespace std;
int main(int argc, char** argv){
  ShapeType s1({200,37,41,100});
  ShapeType s2({84,200,100});
  auto t1=Tensor<Complex >(s1);
  auto t2=Tensor<Complex>(s2);
  Tensor<Complex> c1,c2,c3,diff;  
  LabelType L1{1,-2,-3,2};
  LabelType L2{-1,1,2};
  bool OK=true;

  
  // ShapeType s3({200,37,100});
  // ShapeType s4({37,200,100});
  // LabelType L3{1,2,3};
  // LabelType L4{2,1,3};
  
  ShapeType s3({2,2});
  ShapeType s4({2,2});  
  LabelType L3{1,2};
  LabelType L4{1,2};
  
  auto t3=Tensor<Complex >(s3);
  auto t4=Tensor<Complex>(s4);
  Tensor<Complex> fc1,fc2,fc3;
  

   
    
  srand (time(NULL));     


  auto time0 = std::chrono::high_resolution_clock::now();
  t1.randomize();
  t2.randomize();
  //t3.randomize();
  //t4.randomize();
  t3.reset(1);
  t4.reset(1);  
  auto time1 = std::chrono::high_resolution_clock::now();
  std::cout << "initialization took "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";

  cout<<"starting tblis:"<<endl;
  time0 = std::chrono::high_resolution_clock::now();
  tblistensordot(t1,t2,L1,L2,c1);
  time1 = std::chrono::high_resolution_clock::now();
  cout<< "TBLIS: time per contraction: "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";

  cout<<"starting gemm:"<<endl;  
  time0 = std::chrono::high_resolution_clock::now();
  gemmtensordot(t1,t2,L1,L2,c2);
  time1 = std::chrono::high_resolution_clock::now();    
  cout<< "GEMM: time per contraction: "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";
  diff=c2-c1;
  if (diff.norm()>1E-10){
    cout<<"ERROR: tblis and gemm gave different results!"<<endl;
    OK=false;
  }
  cout<<"starting tcl:"<<endl;
  time0 = std::chrono::high_resolution_clock::now();
  tcltensordot(t1,t2,L1,L2,c3);
  time1 = std::chrono::high_resolution_clock::now();
  cout<< "TCL: time per contraction: "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";
  diff=c3-c1;
  if (diff.norm()>1E-10){
    cout<<"ERROR: tblis and tcl gave different results!"<<endl;
    OK=false;
  }
  diff=c3-c2;
  if (diff.norm()>1E-10){
    cout<<"ERROR: tcl and gemm gave different results!"<<endl;
    OK=false;
  }


  tblistensordot(t3,t4,L3,L4,fc1);
  gemmtensordot(t3,t4,L3,L4,fc2);
  tcltensordot(t3,t4,L3,L4,fc3);
  diff=fc3-fc2;
  if (diff.norm()>1E-10){
    cout<<diff.norm()<<endl;
    cout<<"ERROR: tcl and gemm gave different results for full contraction!"<<endl;
    OK=false;
  }
  diff=fc2-fc1;
  if (diff.norm()>1E-10){
    cout<<diff.norm()<<endl   ;
    cout<<"ERROR: tblis and gemm gave different results for full contraction!"<<endl;
    OK=false;
  }
  diff=fc3-fc1;
  if (diff.norm()>1E-10){
    cout<<diff.norm()<<endl;
    cout<<"ERROR: tblis and tcl gave different results for full contraction!"<<endl;
    OK=false;
  }
  
  if (OK==true)
    cout<<"all contractions gave identical results"<<endl;
  
  return 0;
}
