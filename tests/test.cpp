
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
// #include"../lib/array/tensor.hpp"
#include"../lib/array/tensoroperations.hpp"
#include "lib/linalg/tblisroutines.hpp"
#include "lib/linalg/lapackroutines.hpp"
#include "lib/linalg/blasroutines.hpp"
#include "lib/utils/exceptions.hpp"
#include "lib/array/tensor.hpp"

// #include "../lib/utils/print.hpp"
//using namespace printfunctions;
using std::cout;
using std::conj;
using std::endl;
using std::vector;
using namespace std;
using namespace tensor;
using namespace utilities;
using std::tuple;
//using namespace utilities;
//using namespace tensor;
// Real addhelper(Real a, Real b){
//   return a+b;
// }
// Complex addhelper(Complex a, Real b){
//   return a+b;
// }

// Complex addhelper(Real a, Complex b){
//   return a+b;
// }

// Complex addhelper(Complex a,Complex b){
//   return a+b;
// }

template<typename T>
 class Cl{
 public:
   Cl():val_(Real(0.0)){}
   Cl(Real v):val_(v){}  
   void print(){
     cout<<val_<<endl;
   }
 private:
   Real val_;
 };
template<typename T>
void dosomething(std::vector<T>v){
  cout<<"bla"<<endl;
}

template<typename T,typename A>
T* factory(A&& a){
  return new T(std::forward<A>(a));
}

template<typename T>
void fun2(const T&& c){
  cout<<"hello"<<endl;
  c=5.0;
}
template<typename T>
void fun(const T&& c){
  fun2(std::forward<Real>(c));
  //fun2(c);   
}
void fun3(const Real& c){}

template<typename T>
void funbla(const Cl<T>& c){}

Cl<Real> fun4(){
  return Cl<Real>();
}


std::tuple<int> v2t_1(vector<int>vec){
  std::make_tuple(vec[0]);
}
std::tuple<int,int,int> v2t_2(vector<int>vec){
  std::make_tuple(vec[0],vec[1]);
}
auto v2t_3(vector<int>vec){
  std::make_tuple(vec[0],vec[1],vec[2]);
}
std::tuple<int,int,int,int> v2t_4(vector<int>vec){
  std::make_tuple(vec[0],vec[1],vec[2],vec[3]);
}

Complex determineReturnType(Complex&, Complex&);
Complex determineReturnType(Complex&, Real&);
Complex determineReturnType(Real&, Complex&);
Complex determineReturnType(Real&, Real&);  

template<typename T1,typename T2>
auto add(T1 a, T2 b)->decltype(determineReturnType(a,b)){
  return "h";
}
int main(int argc, char** argv){
  Real a=1;
  Complex b=2;
  auto d=add(a,b);
  cout<<d<<endl;
}
