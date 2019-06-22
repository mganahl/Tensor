
#include "typedefs.hpp"
#include <array>
#include <iostream>
#include <string>
#include <cstdarg>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <list>
#include <complex>
#include <tuple>
#include <algorithm>
#include <tblis.h>
#include"../lib/utils/utilities.hpp"
#include"../lib/array/tensoroperations.hpp"
#include"../lib/array/tensor.hpp"
#include "lib/linalg/tblisroutines.hpp"
#include "lib/linalg/lapackroutines.hpp"
#include "lib/linalg/blasroutines.hpp"
#include "lib/utils/exceptions.hpp"
#include "lib/array/tensor.hpp"
#include "../lib/utils/print.hpp"
using namespace printfunctions;
using std::cout;
using std::conj;
using std::endl;
using std::vector;
using namespace std;
using namespace tensor;
using namespace utilities;
using std::tuple;


template <typename  E >
class VecExpression {
public:
  auto operator[](size_t i) const { return static_cast<E const&>(*this)[i];     }
  size_t size()             const { return static_cast<E const&>(*this).size(); }
};

template<typename ValueType>
class Vec : public VecExpression<Vec<ValueType> > {
  std::vector<ValueType> elems;

public:
  ValueType operator[](size_t i) const { return elems[i]; }
  ValueType &operator[](size_t i)      { return elems[i]; }
  size_t size() const               { return elems.size(); }

  Vec(size_t n) : elems(n) {}

  // construct vector using initializer list
  Vec(std::initializer_list<ValueType>init){
    for(auto i:init)
      elems.push_back(i);
  }

  // A Vec can be constructed from any VecExpression, forcing its evaluation.
  template <typename E>
  Vec(VecExpression<E> const& vec) : elems(vec.size()) {
    for (size_t i = 0; i != vec.size(); ++i) {
      elems[i] = vec[i];
    }
  }
};


template <typename E1, typename E2>
class VecSum : public VecExpression<VecSum<E1, E2> > {
  E1 const& _u;
  E2 const& _v;
   
public:
  VecSum(E1 const& u, E2 const& v) : _u(u), _v(v) {
    assert(u.size() == v.size());
  }
  
  auto  operator[](size_t i) const { return _u[i] + _v[i]; }
  size_t size()               const { return _v.size(); }
};

template <typename E1, typename E2>
class VecMultScalar : public VecExpression<VecMultScalar<E1, E2> > {
  E1 const& _u;
  E2 _v;
   
public:
  VecMultScalar(E1 const& u, E2 v) : _u(u), _v(v) {}
  
  auto  operator[](size_t i) const { return _u[i]*_v; }
  size_t size()               const { return _u.size(); }
};


template <typename E1, typename E2>
VecMultScalar<E1,E2>
operator*(E1 const& u, E2 v) {
  return VecMultScalar<E1, E2>(u, v);
}

template <typename E1, typename E2>
VecSum<E1,E2>
operator+(E1 const& u, E2 const& v) {
  return VecSum<E1, E2>(u, v);
}


int main(int argc, char** argv){
  Vec<Complex> v0 = {Complex(1,1),Complex(2,2)};
  Vec<double> v1 = {1,2};
  Vec<Complex> v2 = {Complex(1,1),Complex(2,2)};
  Real n=4.0;
  auto sum1=(v1*n+v2)+v0;
  cout<<sum1.size()<<endl;
  cout<<sum1[0]<<endl;
  //auto sum = v0+(v1*4)+v2;
  //cout<<sum[0]<<" "<<sum[1]<<endl;


  // vector<int> v{1,2,3};
  // const size_type s=v.size();
  // constexpr size_type  s2=s;
  // vectorToTuple<s2>(v);
  // return 0;
}
