#ifndef MPSFUNS_HPP_
#define MPSFUNS_HPP_
#include "typedefs.hpp"
#include "../lib/utils/utilities.hpp"
#include "../lib/utils/print.hpp"
#include "../lib/array/tensoroperations.hpp"
#include "../lib/array/tensor.hpp"

using namespace std;
using namespace utilities;
using namespace tensor;
using namespace printfunctions;
using std::cout;
using printfunctions::print;
using std::endl;
using std::vector;
namespace mpsfunctions{
  template<typename T>
  std::pair<Tensor<T> ,Tensor<T> >prepareTensor(const Tensor<T> &A,int dir,bool normalize=true){
    if (dir>0){
      auto M=A;
      auto[D1,d,D2]=shape<3>(M);
      M.reshape(D1*d,D2);
      auto [Q,R]=qr(M);
      if (normalize)
	R/=trace(R);
      Q.reshape(D1,d,D2);
      return {Q,R};
    }
    if (dir<0){
      auto M=A;
      auto[D1,d,D2]=shape<3>(M);
      M.reshape(D1,d*D2);
      auto [Q,R]=qr(M.herm());
      R.hermitian();
      Q.hermitian();
      if (normalize)
	R/=trace(R);
      Q.reshape(D1,d,D2);
      return {Q,R};
    }
  }
}
#endif 
