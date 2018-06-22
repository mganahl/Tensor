#include "typedefs.hpp"
#include <iostream>
#include"lib/array/tensor.hpp"
#include"lib/utils/utilities.hpp"
#include"lib/utils/print.hpp"
#include"lib/linalg/lapackroutines.hpp"
#include <chrono>
#include"lib/array/tensoroperations.hpp"

using namespace std;
using namespace tensor;
using namespace printfunctions;
using  std::cout;
using  std::endl;
using  std::vector;
int main(int argc, char** argv){
  int N=4;
  Tensor<Real >mat(N,N),S(N);
  Tensor<Complex >U,VH;

  auto matc=Tensor<Complex>(N,N);
  matc.randomize(Complex(-1,-1),Complex(1,1));

  auto bla=Tensor<Complex>().ones(ShapeType{1,2,3});
  
  bla.print();
  auto matcH=matc-matc.herm();
  svd(matcH,U,S,VH);
  S.print();
  diag(S).print();
  // (U.dot(diag(S)).dot(VH)-matcH).print();
  // mat.randomize(-1,1);
  // matc.randomize(Complex(-1,1),Complex(-1,1));
  // mat.dot(matc);
  // matc.dot(mat);  
  return 0;
}
