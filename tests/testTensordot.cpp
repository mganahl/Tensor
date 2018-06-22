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
  ShapeType s3({84,37,41});
  auto t1=random<Real>(s1);
  auto t2=random<Real>(s2);
  auto t3=random<Real>(s3);  
  LabelType L1{1,-2,-3,2};
  LabelType L2{-1,1,2};
  bool OK=true;
  
  auto time0 = std::chrono::high_resolution_clock::now();
  auto a=tensordot(t1,t2,L1,L2);
  auto d=tensordot(a,t3,LabelType{-1,1,2},LabelType{-2,1,2});
  auto c=tensordot(tensordot(t1,t2,L1,L2),t3,LabelType{-1,1,2},LabelType{-2,1,2});
  auto time1 = std::chrono::high_resolution_clock::now();
  cout<< "tensordot: time per contraction: "<< std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " milliseconds\n";
  return 0;
}
