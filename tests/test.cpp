
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
// #include"../lib/utils/utilities.hpp"
// #include"../lib/array/tensor.hpp"
//#include"../lib/array/tensoroperations.hpp"
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
int main(int argc, char** argv){
  const char *p=std::getenv("BLA");
  if(p!=NULL){
    std::string s(p);
    cout<<s<<endl;
    if(s=="hello"){
      cout<<"hello!"<<endl;
    }
  }else{
    cout<<"BLA not found"<<endl;
  }

  float a=1.00000000002340923409203589;
  int b=(int) a;
  return 0;
}
