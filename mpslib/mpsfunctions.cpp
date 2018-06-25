#ifndef MPSFUNS_HPP_
#define MPSFUNS_HPP_
#include "typedefs.hpp"
#include "../utils/utilities.hpp"
#include "../utils/print.hpp"
#include "tensor.hpp"

using namespace std;
using namespace utilities;
using namespace printfunctions;
using std::cout;
using printfunctions::print;
using std::endl;
using std::vector;
using tensor::Tensor
namespace mpsfunctions{
  template<typename T>
  Tensor<T> prepareTensor(const Tensor<T> &A,int dir){
    if (dir>0){
      A.reshape(ShapeType{A.shape(0)*A.shape(1),A.shape(2)});
    }
  }
}
#endif 
