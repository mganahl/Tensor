#ifndef ARRAY_H
#define ARRAY_H
//#ifndef NDEBUG
//#define NDEBUG
#include <cstdlib>
#include <time.h>
#include <assert.h>
#include <type_traits>
#include <typeinfo>
#include <algorithm>
#include <valarray>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <tuple>

#include "hptt.h"
#include "typedefs.hpp"
#include "../utils/utilities.hpp"
#include "../utils/print.hpp"
#include "lib/linalg/tblisroutines.hpp"
#include "lib/linalg/lapackroutines.hpp"
#include "lib/linalg/blasroutines.hpp"
#include "lib/utils/exceptions.hpp"
//NDEBUG can be set so that no assertions are checked
#ifndef NDEBUG
#   define _assert_(Expr,Msg)  _assert_(Expr,Msg,__FILE__,__LINE__)
#else
#   define _assert_(Expr,Msg) ;
#endif

using namespace std;
using namespace utilities;
using namespace exceptions;
using namespace blasroutines;
using namespace tblisroutines;
using namespace printfunctions;
using  std::cout;
using  std::setprecision;
using  std::endl;
using  std::vector;

namespace tensor{

  //################################################3      some forward declarations from tensoroperations.hpp        #############################################################
  template<typename T>
  class Tensor;
  template<typename T,typename... Shapes>
  Tensor<T> ones(const Shapes& ... shape);
  template<typename T>
  Tensor<T> ones(const ShapeType &shape);
  template<typename T,typename... Shapes>
  Tensor<T> zeros(const Shapes& ... shape);
  template<typename T>
  Tensor<T> zeros(const ShapeType &shape);
  template<typename T>
  Tensor<T> eye(int shape);
  template<typename T>
  Tensor<T> eye(size_type shape);
  
  template<typename... Shapes>
  Tensor<Complex> random_complex(const Shapes& ... shape);
  template<typename... Shapes>
  Tensor<Real> random_real(const Shapes& ... shape);
  
  Tensor<Complex> random_complex(const ShapeType& shape);
  Tensor<Real> random_real(const ShapeType& shape) ;

  template<typename T>
  Tensor<T> diag(const Tensor<T>&tensor);
  
  template<typename T>
  void eigh(const Tensor<T>&A,Tensor<Real>&EV,Tensor<T>&U);

  template<typename T>  
  std::tuple<Tensor<Real>,Tensor<T> > eigh(const Tensor<T>&A);

  template<typename T>
  std::tuple<Tensor<Complex>,Tensor<Complex>,Tensor<Complex> > eig(const Tensor<T>&A);
  
  template<typename T>
  void eig(const Tensor<T>&A,Tensor<Complex>&VL,Tensor<Complex>&EV,Tensor<Complex>&VR);
  
  template<typename T>
  int svd(Tensor<T>A,Tensor<T>&U,Tensor<Real> &S,Tensor<T> &VH);

  template<typename T>  
  std::pair<Tensor<T>,Tensor<T> > qr(const Tensor<T>A,bool full=false);
  
  template<typename T>  
  std::tuple<Tensor<T>,Tensor<Real>,Tensor<T> > svd(const Tensor<T>&A);
  
  template<typename T>
  void tblistensordot(Tensor<T>&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C);

  template<typename T>
  void tblistensordot(Tensor<T>&&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C);
  
  template<typename T>
  void tblistensordot(Tensor<T>&A,Tensor<T>&&B,LabelType labela,LabelType labelb,Tensor<T> &C);
  
  template<typename T>
  void tblistensordot(Tensor<T>&&A,Tensor<T>&&B,LabelType labela,LabelType labelb,Tensor<T> &C);
  
  template<typename T>
  void tcltensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);

  template<typename T>
  void tcltensordot(Tensor<T>&&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void tcltensordot(Tensor<T>&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void tcltensordot(Tensor<T>&&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void gemmtensordot(Tensor<T>&&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void gemmtensordot(Tensor<T>&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void gemmtensordot(Tensor<T>&&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void gemmtensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C);
  
  template<typename T>
  void gemm_mv_dot(const Tensor<T>&A,const Tensor<T>&B,Tensor<T> &C,char TR='N');
  
  template<typename T>
  Real gemm_vv_dot(const Tensor<T> &A,const Tensor<T>&B);
  
  template<typename T>
  Real norm(const Tensor<T> &t);
  
  Tensor<Complex> dot(Tensor<Complex>&A,Tensor<Complex>&B);
  
  Tensor<Complex> dot(Tensor<Real>&A,Tensor<Complex>&B);
  
  Tensor<Complex> dot(Tensor<Complex>&A,Tensor<Real>&B);
  
  Tensor<Real> dot(Tensor<Real>&A,Tensor<Real>&B);
  
  template<typename T>
  Tensor<T> conjugate(const Tensor<T> &A);
  
  Real c_rand(Real min,Real max);
  
  Complex c_rand(Complex min,Complex max);
  
  Complex random_unit(Complex t);
  
  Real random_unit(Real t);
  
  Real random_unit(int t);
  
  //##################################################################################################################################################################

  template<typename T>
  class Tensor{
  public:
    /* Constructor Definitions*/
    /*construct an empty object*/
    Tensor();
    
    template<typename... Dimensions>
    Tensor(const Dimensions&... dims);
    /*constructs a tensor of rank given by length of shape; data is stored in a std::vector<T>; the data is not initialized
      signature of the constructor: Tensor<T> t(std::vector<unsigned int>{s1,s2,s3,...,sN}) constructs a tensor of type T and size (s1,s2,...,sN)
    */
    
    Tensor(const ShapeType& shape);
    /*
      constructs an empty tensor from a vector of Slice objects
    */
    Tensor(const vector<Slice>& slices);

    //copy constructor
    Tensor(const Tensor<T>& other);

    //move constructor
    Tensor(Tensor<T>&& other) noexcept;
    
    /* cast Tensor */
    operator Tensor<Real>()const;
    operator Tensor<float>()const;
    operator Tensor<int>()const;
    operator Tensor<lint>()const;            
    operator Tensor<Complex>()const;

    /* Access Operations*/
    /*raw() returns a pointer to the data-container*/
    T* raw();
    const T* raw()const;

    /*data() returns a reference to the data-container*/
    std::vector<T>& data();
    const std::vector<T>& data()const;
    /*returns n-th element in the data-container*/
    T data(lint n)const;
    /*returns a reference to the n-th element in the data-container*/
    T& data(lint n);

    /*returns a new copy of the tensor*/
    Tensor<T> copy()const;
    
    /*access elementes with the operator()
      call signature: T value=A(k,m,...,n) returns the value of tensor A at index (k,m,...,n)
      A(k,m,...,n) =value changes the value of tensor A at index (k,m,...,n) to value
    */
    template<typename... Inds>
    T& operator()(Inds... inds);
    template<typename... Inds>    
    T operator()(Inds... inds)const;
    
    /*
      check_integral takes a parameter pack and recursively checks if all types are integral
      signature: check_integral(inds...)
    */
    template<typename U,typename... Args>
    void check_integral(U first,Args... args)const;
    
    /*
      check_integral takes a parameter checks if the type is integral
      signature: check_integral(var)
      this routine is the termination function for the recursively called check_integral (see above)
    */
    template<typename U>
    void check_integral(U var)const;

    //returns a slice of Tensor<T>, according to the specified slices
    template<typename... Slices>
    Tensor<T> slice(const Slices&... slices)const;
    
    
    //returns a slice of Tensor<T>, according to the the specified slices; manually runs through the whole tensor and copies it element by element; not very fast
    //the argument is passed as copy because slice modifies slcs's content if the Slice object's range is out of bound ot
    //match the dimension of the tensor leg
    Tensor<T> slice(std::vector<Slice>slcs)const;
    
    //takes a Tensor<T>& tensor and a number of slices and inserts tensor into this at the position given by the slices
    template<typename... Slices>
    void insert_slice(Tensor<T>& tensor,Slices&... slices);
    
    //takes a Tensor<T>& tensor and a number of slices and inserts tensor into this at the position given by the slices
    void insert_slice(Tensor<T>& tensor,const std::vector<Slice>& slcs);
    //randomize an existing tensor with values between min and max
    void randomize(T min=T(-1.0),T max=T(1.0));
    void randomize_int();

    //returns a random tensor with values between 0 and +1 for Real and Complex(0,0) and Complex (1,0) for Complex
    template<typename... Shapes>    
    Tensor<T> random(const Shapes&... shape);
    Tensor<T> random(const ShapeType& shape);

    //returns a random tensor filled with 0
    template<typename... Shapes>    
    Tensor<T> zeros(const Shapes&... shape);
    Tensor<T> zeros(const ShapeType& shape);

    //returns a random tensor filled with 1    
    template<typename... Shapes>    
    Tensor<T> ones(const Shapes&... shape);
    Tensor<T> ones(const ShapeType& shape);

    //compute the norm of the tensor
    Real norm ()const;

    //reset all values to val
    void reset(T val=T(0));

    //arithmetic operations
    //copy the data of tensor to this    
    Tensor<T> & operator=(const Tensor<T> &tensor);
    Tensor<T> & operator=(Tensor<T> &&tensor);
    
    Tensor<T> & operator+=(const Tensor<T> &tensor);
    Tensor<T> & operator-=(const Tensor<T> &tensor);
    //+= for scalars has to be overloaded due to ambiguity
    Tensor<T> & operator+=(int val);
    Tensor<T> & operator+=(uint val);
    Tensor<T> & operator+=(lint val);
    Tensor<T> & operator+=(luint val);
    Tensor<T> & operator+=(float val);    
    Tensor<T> & operator+=(Real val);
    Tensor<T> & operator+=(Complex val);    
    Tensor<T> & operator-=(T val);
    Tensor<T> & operator*=(T val);
    Tensor<T> & operator/=(T val);

    Tensor<Real> real()const;
    Tensor<Real> imag()const;    
    //in-place conjugation
    void conjugate();
    //return conjugated tensor
    Tensor<T> conj()const;
    //return hermitian conjugated tensor    
    Tensor<T> herm()const;
    void hermitian();    
    //inplace transpose (reverse all labels)
    void transpose();
    //return transposed tensor (reverse all labels)
    Tensor<T> transp()const;
    
    //transpose labels of the tensor
    //tranpose methods return a new tensor; can be memory critical for large tensors due to internal memory allocation
    //uses hptt method for transposition
    template<typename... Labels>
    void transpose(const Labels&... labs);
    void transpose(const LabelType&labels);

    //transpose labels of the tensor
    //tranpose methods return a new tensor; can be memory critical for large tensors due to internal memory allocation
    //naive transposition, can be slow
    template<typename... Labels>    
    void naive_transpose(const Labels&... labs);
    void naive_transpose(const LabelType&labels);
    
    template<typename... Labs>
    void hptt(const Labs&... labs);
    void hptt(const LabelType&labels,lint numThreads=4);
    
    //dot-methods returns a new tensor
    Tensor<T> tblisdot(Tensor<T>& other,const LabelType& labelself,const LabelType& labelother);
    // Tensor<T> tblisdot(Tensor<T>& other,LabelType&& labelself,LabelType&& labelother);
    // Tensor<T> tblisdot(Tensor<T>&& other,LabelType&& labelself,LabelType&& labelother);
    
    Tensor<T> gemmdot(Tensor<T>& other,const LabelType& labelself,const LabelType& labelother);
    // Tensor<T> gemmdot(Tensor<T>& other,LabelType&& labelself,LabelType&& labelother);
    // Tensor<T> gemmdot(Tensor<T>&& other,LabelType&& labelself,LabelType&& labelother);    

    Tensor<T> tcldot(Tensor<T>& other,const LabelType& labelself,const LabelType& labelother);
    // Tensor<T> tcldot(Tensor<T>& other,LabelType&& labelself,LabelType&& labelother);
    // Tensor<T> tcldot(Tensor<T>&& other,LabelType&& labelself,LabelType&& labelother);

    //compute a-b dot products, where a and b can be either matrix of vector like 
    Tensor<T> dot(Tensor<T>& other);
    Tensor<T> dot(Tensor<T>&& other);
    
    // template<typename T1,typename T2>
    // GemmProxy<T1,T2> dot_2(Tensor<T>& other);
    // template<typename T1,typename T2>    
    // GemmProxy<T1,T2> dot_2(Tensor<T>&& other);
    
    //resize the tensor
    template<typename... Shapes>
    void resize(Shapes... newsize);
    void resize(ShapeType newsize);

    //reshape the tensor    
    template<typename... Shapes>
    void reshape(const Shapes&...newshape);
    void reshape(ShapeType shape);

    //return the shape of the tensor
    size_type shape(const lint n) const;
    const ShapeType& shape() const;
    template<std::size_t>
    //returns the shape of the tensor; call signature is
    //A.shape<r>() for a tensor with rank r (has to be known at compile time)
    auto shape()const;

    //returns a reference to the ShapeType shape_ member of Tensor; ShapeType is std::vector<std::size_t>
    ShapeType& shape();
    
    //return the size (=total number of elements) of the tensor    
    size_type size() const;

    //return the rank of the tensor        
    lint rank() const;

    //return the stride of the tensor            
    const StrideType& stride () const;
    StrideType& stride ();
    //print the tensor; for vectors and matrices, print does some editing;
    void print(int L=4,int prec=5)const;

  private:
    size_type size_;  //the total number of elements in the tensor      
    ShapeType shape_; //shape of the tensor
    StrideType stride_;//the stride of the tensor
    std::vector<T> data_;//the data container

    lint rank_;   //the rank of the tensor
    //initialization function; allocates the memory for the tensor
    void init();
    void check_range(IndexType inds)const;
  };

    
  // ============================================     operations on tensors ===========================================

  Tensor<Complex> toComplex(const Tensor<Real> &A){
    Tensor<Complex> out(A.shape());
    for(luint n=0;n<A.size();n++){
      out.raw()[n]=Complex(A.raw()[n],0.0);
    }
    return out;
  }
  
  Tensor<Complex> toComplex(const Tensor<Real> &&A){
    Tensor<Complex> out(A.shape());
    for(luint n=0;n<A.size();n++){
      out.raw()[n]=Complex(A.raw()[n],0.0);
    }
    return out;
  }

  const Tensor<Complex>& toComplex(const Tensor<Complex> &A){
    return A;
  }

  
  // ++++++++ operator *+++++++++++++++++++++++++++++++++++++++
  Tensor<Real> operator*(Real val,const Tensor<Real>& A){
    Tensor<Real>  out=A;
    out*=val;
    return out;
  }

  Tensor<Real> operator*(const Tensor<Real>& A,Real val){
    Tensor<Real>  out=A;
    out*=val;
    return out;
  }

  Tensor<Complex> operator*(Complex val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    out*=val;
    return out;
  }

  Tensor<Complex> operator*(const Tensor<Complex>& A,Complex val){
    Tensor<Complex>  out=A;
    out*=val;
    return out;
  }
  
  Tensor<Complex> operator*(Real val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    out*=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator*(const Tensor<Complex>& A,Real val){
    Tensor<Complex>  out=A;
    out*=Complex(val,0.0);
    return out;
  }
  
  Tensor<Complex> operator*(const Tensor<Real>& A,Complex val){
    Tensor<Complex>  out=A;
    out*=val;
    return out;
  }
  
  Tensor<Complex> operator*(Complex val,const Tensor<Real>& A){
    Tensor<Complex>  out=A;
    out*=val;
    return out;
  }


  //============================   rvalue
  Tensor<Real> operator*(Real val,Tensor<Real>&& A){
    Tensor<Real>  out=std::move(A);
    out*=val;
    return out;
  }

  Tensor<Real> operator*(Tensor<Real>&& A,Real val){
    Tensor<Real>  out=std::move(A);
    out*=val;
    return out;
  }

  Tensor<Complex> operator*(Complex val,Tensor<Complex>&& A){
    Tensor<Complex>  out=std::move(A);
    out*=val;
    return out;
  }

  Tensor<Complex> operator*(Tensor<Complex>&& A,Complex val){
    Tensor<Complex>  out=std::move(A);
    out*=val;
    return out;
  }
  
  Tensor<Complex> operator*(Real val,Tensor<Complex>&& A){
    Tensor<Complex>  out=std::move(A);
    out*=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator*(Tensor<Complex>&& A,Real val){
    Tensor<Complex>  out=std::move(A);
    out*=Complex(val,0.0);
    return out;
  }
  


  // ++++++++ operator / +++++++++++++++++++++++++++++++++++++++

  Tensor<Real> operator/(Real val,const Tensor<Real>& A){
    Tensor<Real>  out(ShapeType(A.shape()));
    for (lint n=0;n<A.size();n++){
      out.raw()[n]=val/A.raw()[n];
    }
    return out;
  }

  Tensor<Real> operator/(const Tensor<Real>& A,Real val){
    Tensor<Real>  out=A;
    out/=val;
    return out;
  }

  Tensor<Complex> operator/(Complex val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    for (lint n=0;n<A.size();n++){
      out.raw()[n]=val/A.raw()[n];
    }
    return out;
  }

  Tensor<Complex> operator/(const Tensor<Complex>& A,Complex val){
    Tensor<Complex>  out=A;
    out/=val;
    return out;
  }

  
  Tensor<Complex> operator/(Real val,const Tensor<Complex>& A){
    Tensor<Complex>  out(ShapeType(A.shape()));
    for (lint n=0;n<A.size();n++){
      out.raw()[n]=val/A.raw()[n];
    }
    return out;
  }

  Tensor<Complex> operator/(const Tensor<Complex>& A,Real val){
    Tensor<Complex>  out=A;
    out/=Complex(val,0.0);
    return out;
  }
  
  Tensor<Complex> operator/(const Tensor<Real>& A,Complex val){
    Tensor<Complex>  out=A;
    out/=val;
    return out;
  }
  
  Tensor<Complex> operator/(Complex val,const Tensor<Real>& A){
    Tensor<Complex> out(ShapeType(A.shape()));
    for (lint n=0;n<A.size();n++){
      out.raw()[n]=val/A.raw()[n];
    }
    return out;
  }


  //rvalyue

  Tensor<Real> operator/(Tensor<Real>&& A,Real val){
    Tensor<Real>  out=std::move(A);
    out/=val;
    return out;
  }

  Tensor<Complex> operator/(Tensor<Complex>&& A,Complex val){
    Tensor<Complex>  out=std::move(A);
    out/=val;
    return out;
  }
  
  Tensor<Complex> operator/(Tensor<Complex>&& A,Real val){
    Tensor<Complex>  out=std::move(A);
    out/=Complex(val,0.0);
    return out;
  }


  // ++++++++ operator + (scalar) +++++++++++++++++++++++++++++++++++++++
  Tensor<Real> operator+(Real val,const Tensor<Real>& A){
    Tensor<Real>  out=A;
    out+=val;
    return out;
  }
  
  Tensor<Real> operator+(const Tensor<Real>& A,Real val){
    Tensor<Real>  out=A;
    out+=val;
    return out;
  }

  Tensor<Complex> operator+(Complex val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    out+=val;
    return out;
  }

  Tensor<Complex> operator+(const Tensor<Complex>& A,Complex val){
    Tensor<Complex>  out=A;
    out+=val;
    return out;
  }

  Tensor<Complex> operator+(Real val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    out+=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator+(const Tensor<Complex>& A,Real val){
    Tensor<Complex>  out=A;
    out+=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator+(const Tensor<Real>& A,Complex val){
    Tensor<Complex>  out=toComplex(A);
    out+=val;
    return out;
  }
  
  Tensor<Complex> operator+(Complex val,const Tensor<Real>& A){
    Tensor<Complex>  out=toComplex(A);
    out+=val;
    return out;
  }

  //rvalue
  Tensor<Real> operator+(Real val,Tensor<Real>&& A){
    Tensor<Real>  out=std::move(A);
    out+=val;
    return out;
  }

  Tensor<Real> operator+(Tensor<Real>&& A,Real val){
    Tensor<Real>  out=std::move(A);
    out+=val;
    return out;
  }

  Tensor<Complex> operator+(Complex val,Tensor<Complex>&& A){
    Tensor<Complex>  out=std::move(A);
    out+=val;
    return out;
  }

  Tensor<Complex> operator+(Tensor<Complex>&& A,Complex val){
    Tensor<Complex>  out=std::move(A);
    out+=val;
    return out;
  }
  
  Tensor<Complex> operator+(Real val,Tensor<Complex>&& A){
    Tensor<Complex>  out=std::move(A);
    out+=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator+(Tensor<Complex>&& A,Real val){
    Tensor<Complex>  out=std::move(A);
    out+=Complex(val,0.0);
    return out;
  }


  
  // ++++++++ operator - (scalar) -++++++++++++++++++++++++++++++++++++++  
  Tensor<Real> operator-(Real val,const Tensor<Real>& A){
    Tensor<Real>  out=A;
    out-=val;
    return out;
  }

  Tensor<Real> operator-(const Tensor<Real>& A,Real val){
    Tensor<Real>  out=A;
    out-=val;
    return out;
  }

  Tensor<Complex> operator-(Complex val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    out-=val;
    return out;
  }

  Tensor<Complex> operator-(const Tensor<Complex>& A,Complex val){
    Tensor<Complex>  out=A;
    out-=val;
    return out;
  }

  Tensor<Complex> operator-(Real val,const Tensor<Complex>& A){
    Tensor<Complex>  out=A;
    out-=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator-(const Tensor<Complex>& A,Real val){
    Tensor<Complex>  out=A;
    out-=Complex(val,0.0);
    return out;
  }
  
  Tensor<Complex> operator-(const Tensor<Real>& A,Complex val){
    Tensor<Complex>  out=toComplex(A);
    out-=val;
    return out;
  }
  
  Tensor<Complex> operator-(Complex val,const Tensor<Real>& A){
    Tensor<Complex>  out=toComplex(A);
    out-=val;
    return out;
  }
  

  Tensor<Real> operator-(Real val,Tensor<Real>&& A){
    Tensor<Real>  out=std::move(A);
    out-=val;
    return out;
  }

  Tensor<Real> operator-(Tensor<Real>&& A,Real val){
    Tensor<Real>  out=std::move(A);
    out-=val;
    return out;
  }

  Tensor<Complex> operator-(Complex val,Tensor<Complex>&& A){
    Tensor<Complex>  out=std::move(A);
    out-=val;
    return out;
  }

  Tensor<Complex> operator-(Tensor<Complex>&& A,Complex val){
    Tensor<Complex>  out=std::move(A);
    out-=val;
    return out;
  }
  
  Tensor<Complex> operator-(Real val,Tensor<Complex>&& A){
    Tensor<Complex>  out=std::move(A);
    out-=Complex(val,0.0);
    return out;
  }

  Tensor<Complex> operator-(Tensor<Complex>&& A,Real val){
    Tensor<Complex>  out=std::move(A);
    out-=Complex(val,0.0);
    return out;
  }


  
  Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B){
    Tensor<Complex> result=A;
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B){
    Tensor<Complex> result=A;
    try{
      result+=toComplex(B);
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(const Tensor<Real> &A,const Tensor<Complex> &B){
    Tensor<Complex> result=B;
    try{
      result+=toComplex(A);
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B){
    Tensor<Real> result=A;
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }



  //++++++++++++++++++++++++++++++++++++++++++=
  
  Tensor<Complex> operator+(Tensor<Complex>&&A,const Tensor<Complex> &B){
    Tensor<Complex> result=std::move(A);
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(Tensor<Complex> &&A,const Tensor<Real> &B){
    Tensor<Complex> result=std::move(A);
    try{
      result+=toComplex(B);
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(Tensor<Real> &&A,const Tensor<Complex> &B){
    Tensor<Complex> result=B;
    try{
      result+=toComplex(A);
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator+(Tensor<Real> &&A,const Tensor<Real> &B){
    Tensor<Real> result=std::move(A);
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  
  Tensor<Complex> operator+(const Tensor<Complex> &A,Tensor<Complex> &&B){
    Tensor<Complex> result=std::move(B);
    try{
      result+=A;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(const Tensor<Complex> &A,const Tensor<Complex> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(const Tensor<Real> &A,Tensor<Complex> &&B){
    Tensor<Complex> result=std::move(B);
    try{
      result+=A;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }
  
  Tensor<Complex> operator+(const Tensor<Complex> &A,Tensor<Real> &&B){
    Tensor<Complex> result=A;
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator+(const Tensor<Real> &A,Tensor<Real> &&B){
    Tensor<Real> result=std::move(B);
    try{
      result+=A;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(const Tensor<Real> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }



  Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Complex> &&B){
    Tensor<Complex> result=std::move(A);
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Complex> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Complex> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Complex> &&B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Real> &&B){
    Tensor<Complex> result=std::move(A);
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Real> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Real> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator+(Tensor<Complex> &&A,Tensor<Real> &&B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator+(Tensor<Real> &&A,Tensor<Complex> &&B){
    Tensor<Complex> result=std::move(B);
    try{
      result+=A;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(Tensor<Complex> &&A,Tensor<Real> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(Tensor<Complex> &&A,Tensor<Real> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(Tensor<Complex> &&A,Tensor<Real> &&B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator+(Tensor<Real> &&A,Tensor<Real> &&B){
    Tensor<Real> result=std::move(A);
    try{
      result+=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator+(Tensor<Real> &&A,Tensor<Real> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(Tensor<Real> &&A,Tensor<Real> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator+(Tensor<Real> &&A,Tensor<Real> &&B): sizes of tensors are not matching");      
    }
  }

  
  //=============================================

  Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B){
    Tensor<Complex> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B){
    Tensor<Complex> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(const Tensor<Real> &A,const Tensor<Complex> &B){
    Tensor<Complex> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B){
    Tensor<Real> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  
  //++++++++++++++++++++++++++++++++++++++++++=
  
  Tensor<Complex> operator-(Tensor<Complex>&&A,const Tensor<Complex> &B){
    Tensor<Complex> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(Tensor<Complex> &&A,const Tensor<Real> &B){
    Tensor<Complex> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(Tensor<Real> &&A,const Tensor<Complex> &B){
    Tensor<Complex> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator-(Tensor<Real> &&A,const Tensor<Real> &B){
    Tensor<Real> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Complex> operator-(const Tensor<Complex>&A,Tensor<Complex> &&B){
    Tensor<Complex> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Complex> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(const Tensor<Complex> &A,Tensor<Real> &&B){
    Tensor<Complex> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(const Tensor<Real> &A,Tensor<Complex> &&B){
    Tensor<Complex> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Complex> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator-(const Tensor<Real> &A,Tensor<Real> &&B){
    Tensor<Real> result=A;
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(const Tensor<Real> &A,const Tensor<Real> &B): sizes of tensors are not matching");      
    }
  }

  
  //---

  Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Complex> &&B){
    Tensor<Complex> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Complex> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Complex> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Complex> &&B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Real> &&B){
    Tensor<Complex> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Real> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Real> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Complex> operator-(Tensor<Complex> &&A,Tensor<Real> &&B): sizes of tensors are not matching");      
    }
  }

  Tensor<Complex> operator-(Tensor<Real> &&A,Tensor<Complex> &&B){
    Tensor<Complex> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(Tensor<Complex> &&A,Tensor<Real> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(Tensor<Complex> &&A,Tensor<Real> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(Tensor<Complex> &&A,Tensor<Real> &&B): sizes of tensors are not matching");      
    }
  }


  Tensor<Real> operator-(Tensor<Real> &&A,Tensor<Real> &&B){
    Tensor<Real> result=std::move(A);
    try{
      result-=B;
      return result;      
    }
    catch(RankMismatchError){
      throw RankMismatchError("Tensor<Real> operator-(Tensor<Real> &&A,Tensor<Real> &&B): ranks of tensors are not matching");
    }
    catch(ShapeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(Tensor<Real> &&A,Tensor<Real> &&B): shapes of tensors are not matching");
    }
    catch(SizeMismatchError){
      throw ShapeMismatchError("Tensor<Real> operator-(Tensor<Real> &&A,Tensor<Real> &&B): sizes of tensors are not matching");      
    }
  }

  
  //helper function for conjugation; does nothing for real tensor
  void conj_inplace(Tensor<Real> &A){}

  //helper function for conjugation; changes A.data() to its conjugate for Complex tensors
  void conj_inplace(Tensor<Complex> &A){
    auto ptr=A.raw();
#pragma omp parallel for        
    for(uint n=0;n<A.data().size();n++){
      ptr[n]=std::conj(ptr[n]);
    }
  }
  

  // ######################################             implementation of Tensor members form here onward           #####################################################

  
  template<typename T>
  //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Tensor<T>::Tensor(){data_.resize(0);size_=0;};
  /*constructs a tensor of rank dims; data is stored in a std::vector<T>; the data is not initialized
    signature of the constructor: Tensor<T> t(s1,s2,s3,...,sN) constructs a tensor of type T and size (s1,s2,...,sN)
  */
  //template<typename T,typename... Dimensions>  
  template<typename T>  
  template<typename... Dimensions>
  Tensor<T>::Tensor(const Dimensions&... dims) {
    try{
      check_integral(dims...);
      vector<int>i={dims...};
      ShapeType s(i.begin(),i.end());
      shape_=s;
      init();
    }
    catch(NotAnIntegerError){
      cout<<"Tensor(const Dimensions&... dims): Tensor constructor got a non-integer type argument"<<endl;
      throw;
    }
  }

  /*constructs a tensor of rank given by length of shape; data is stored in a std::vector<T>; the data is not initialized
    signature of the constructor: Tensor<T> t(std::vector<unsigned int>{s1,s2,s3,...,sN}) constructs a tensor of type T and size (s1,s2,...,sN)
  */
  template<typename T>  
  Tensor<T>::Tensor(const ShapeType& shape) :shape_(shape) {
    init();
  }
  
  /*
    constructs an empty tensor from a vector of Slice objects
  */
  template<typename T>  
  Tensor<T>::Tensor(const vector<Slice>& slices) {
    shape_.resize(slices.size());
    for (lint n=0;n<slices.size();n++){
      shape_[n]=slices[n].size();
    }
    init();
  }

  //copy  constructor
  template<typename T>    
  Tensor<T>::Tensor(const Tensor<T>& other):data_(other.data_), size_(other.size_), shape_(other.shape_),stride_(other.stride_),rank_(other.rank_){}
  
  //move constructor
  template<typename T>      
  Tensor<T>::Tensor(Tensor<T>&& other) noexcept:data_(std::move(other.data_)), size_(std::move(other.size_)), shape_(std::move(other.shape_)),stride_(std::move(other.stride_)),rank_(std::move(other.rank_)){
  }

  template<typename T>    
  Tensor<T>::operator Tensor<float>()const{
    Tensor<float> out(ShapeType(this->shape()));
    bool warn=false;
    for (luint n=0;n<this->size();n++){
      if ((warn==false) and (std::abs(std::imag(this->raw()[n]))>1E-12))
	warn=true;
      out.raw()[n]=(float)std::real(this->raw()[n]);
    }
    if (warn==true)
      Warning("casting Complex to Real discards the imaginary part");
    return out;
  }

  
  template<typename T>    
  Tensor<T>::operator Tensor<Real>()const{
    Tensor<Real> out(ShapeType(this->shape()));
    bool warn=false;
    for (luint n=0;n<this->size();n++){
      if ((warn==false) and (std::abs(std::imag(this->raw()[n]))>1E-12))
	warn=true;
      out.raw()[n]=std::real(this->raw()[n]);
    }
    if (warn==true)
      Warning("casting Complex to Real discards the imaginary part");
    return out;
  }

  template<typename T>    
  Tensor<T>::operator Tensor<int>()const{
    Tensor<int> out(ShapeType(this->shape()));
    for (luint n=0;n<this->size();n++){
      out.raw()[n]=(int)this->raw()[n];
    }
    return out;    
  }
  

  template<typename T>      
  Tensor<T>::operator Tensor<Complex>()const{
    Tensor<Complex> out(ShapeType(this->shape()));
    for (luint n=0;n<this->size();n++){
      out.raw()[n]=Complex(std::real(this->raw()[n]),std::imag(this->raw()[n]));
    }
    return out;
  }

  /*this has the same effect as Tensor<Real>, i.e. casting the Tensor to complex<double> type; it is merely for convenience
    to be able to write (Complex)tensor instead of (Tensor<Complex>)tensor*/


  
  /* Access Operations*/
  /*raw() returns the pointer to the data-container*/
  template<typename T>  
  T* Tensor<T>::raw(){return data_.data();}
  template<typename T>  
  const T* Tensor<T>::raw()const{return data_.data();}  
  
  /*data() returns a reference to the data-container*/
  template<typename T>  
  std::vector<T>& Tensor<T>::data(){return data_;}
  
  template<typename T>  
  const std::vector<T>& Tensor<T>::data()const{return data_;}


  //returns a copy of the tensor
  template<typename T>  
  Tensor<T> Tensor<T>::copy()const {
    return Tensor<T>(*this);
  }
  
  template<typename T>  
  T Tensor<T>::data(lint n)const{return data_[n];}
  
  template<typename T>
  T& Tensor<T>::data(lint n){return data_[n];}




  
  /*access elementes with the operator()
    call signature: T value=A(k,m,...,n) returns the value of tensor A at index (k,m,...,n)
    A(k,m,...,n) =value changes the value of tensor A at index (k,m,...,n) to value
  */
  //template<typename T,typename... Inds>  
  template<typename T>
  template<typename... Inds>  
  T&Tensor<T>:: operator()(Inds... inds){
    try{

      check_integral(inds...);//check if inds are of intgral type
      IndexType indices={inds...};
      // cout<<"calling T& with"<<endl;
      // printfunctions::print(indices);
      assert(indices.size()==rank_);
      try{
	check_range(indices);   //check if the inds[n] is smaller than shape_[n]
	return data_[to_integer(indices,stride_)];
      }
      catch(OutOfBoundError){
	throw OutOfBoundError("Tensor<T>.operator(): cannot access element; index out of bounds");
      }
    }
    catch(NotAnIntegerError){
      cout<<"Tensor<T> operator()(Inds... inds): operator () got a non-integer type argument"<<endl;
      throw;
    }
  }

  template<typename T>
  template<typename... Inds>  
  T Tensor<T>:: operator()(Inds... inds)const{
    try{
      check_integral(inds...);//check if inds are of intgral type
      IndexType indices={inds...};
      // cout<<"calling T with"<<endl;
      // printfunctions::print(indices);
      assert(indices.size()==rank_);
      try{
	check_range(indices);   //check if the inds[n] is smaller than shape_[n]
	return data_[to_integer(indices,stride_)];
      }
      catch(OutOfBoundError){
	throw OutOfBoundError("Tensor<T>.operator(): cannot access element; index out of bounds");
      }
    }
    catch(NotAnIntegerError){
      cout<<"Tensor<T> operator()(Inds... inds): operator () got a non-integer type argument"<<endl;
      throw;
    }
  }

  
  /*
    check_integral takes a parameter pack and recursively checks if all types are integral
    signature: check_integral(inds...)
    
  */
  template<typename T>  
  template<typename U,typename... Args>
  void Tensor<T>::check_integral(U first,Args... args)const{
    if (not std::is_integral<U>::value)
      throw NotAnIntegerError("check_integral: not an integer");
    check_integral(args...);
  }
  
  /*
    check_integral takes a parameter checks if the type is integral
    signature: check_integral(var)
    this routine is the termination function for the recursively called check_integral (see above)
  */
  template<typename T>    
  template<typename U>
  void Tensor<T>::check_integral(U var)const{
    if (not std::is_integral<U>::value)
      throw NotAnIntegerError("check_integral: not an integer");      
    //_assert_(std::is_integral<U>::value,"in Tensor<T>::operator(): index must be of integer type");
  }

  //returns a slice of Tensor<T>, according to the specified slices
  template<typename T>    
  template<typename... Slices>
  Tensor<T> Tensor<T>::slice(const Slices&... slices)const{    
    std::vector<Slice> slcs={slices...};
    return this->slice(slcs);
  }
  
  
  //returns a slice of Tensor<T>, according to the the specified slices; manually runs through the whole tensor and copies it element by element; not very fast
  //slcs is passed by value because slice might change it, and resuing slcs on another tensor might be screwed up
  template<typename T>  
  Tensor<T> Tensor<T>::slice(std::vector<Slice>slcs)const{
    IndexType index(rank_,0);
    if (slcs.size()>rank_){
      throw std::runtime_error("in Tensor<T>::slice():rank of tensor is smaller than number of provided given slices ");
    }
    for(lint n=0; n<slcs.size();n++){
      index[n]=slcs[n].start();
      if (slcs[n].stop()>=shape_[n]){
	if (shape_[n]>0){
	  slcs[n].stop()=shape_[n]-1;
	  slcs[n].reset();
	}else{
	  throw SliceError("cannot slice tensor leg of dimension 0)");
	}
      }
    }
    //add missing slices, each one selects all
    for(lint n=slcs.size();n<rank_;n++){
      if (shape_[n]>0){
	slcs.push_back(Slice(0,shape_[n]-1,1));
      }else{
	throw SliceError("cannot slice tensor leg of dimension 0)");
      }
    }

    Tensor<T> sliced_tensor(slcs);
    lint ind1,ind2=0;
    do{
      ind1=to_integer(index,stride_);
      sliced_tensor.raw()[ind2]=this->raw()[ind1];
      ind2++;
    }
    while(next_multiindex(index,slcs));
    return sliced_tensor;
  }


  //takes a Tensor<T>& tensor and a number of slices and inserts tensor into this at the position given by the slices
  template<typename T>    
  template<typename... Slices>
  void Tensor<T>::insert_slice(Tensor<T>& tensor,Slices&... slices){
    std::vector<Slice> slcs={slices...};
    IndexType index(slcs.size(),0);
    if (tensor.rank()!=slcs.size()){
      throw std::runtime_error("in Tensor<T>::insert_slice(): tensor._rank!=slcs.size()");
    }

    for(lint n=0; n<slcs.size();n++){
      index[n]=slcs[n].start();
      if (slcs[n].size()!=tensor.shape(n)){
	throw std::runtime_error("in Tensor<T>::insert_slice(): tensor.shape is incompatible with slice");
      }
      
    }
    
    lint ind1,ind2=0;
    do{
      ind1=to_integer(index,stride_);
      this->raw()[ind1]=tensor.raw()[ind2];
      ind2++;
    }
    while(next_multiindex(index,slcs));
  }
  //takes a Tensor<T>& tensor and a number of slices and inserts tensor into this at the position given by the slices
  template<typename T>  
  void Tensor<T>::insert_slice(Tensor<T>& tensor,const std::vector<Slice>& slcs){
    IndexType index(slcs.size(),0);
    if (tensor.rank()!=slcs.size()){
      throw std::runtime_error("in Tensor<T>::insert_slice(): tensor._rank!=slcs.size()");
    }

    for(lint n=0; n<slcs.size();n++){
      index[n]=slcs[n].start();
      if (slcs[n].size()!=tensor.shape(n)){
	throw std::runtime_error("in Tensor<T>::insert_slice(): tensor.shape is incompatible with slice");
      }
    }
    
    lint ind1,ind2=0;
    do{
      ind1=to_integer(index,stride_);
      this->raw()[ind1]=tensor.raw()[ind2];
      ind2++;
    }
    while(next_multiindex(index,slcs));
  
  }  
  //initialize the tensor with r,andom numbers
  template<typename T>  
  void Tensor<T>::randomize(T min,T max){
    for (lint n=0;n<this->size();n++){
      this->data()[n]=c_rand(min,max);      
    }
    //randomize(*this,range);
  }
  
  template<typename T>
  void Tensor<T>::randomize_int(){
    for (lint n=0;n<data_.size();n++){
      data_[n]=rand();
    }
  }

  //returns a tensor with shape shape..., initialized with random values of type T with real part between -1 and 1 and imaginary part 0
  template<typename T>
  template<typename... Shapes>  
  Tensor<T> Tensor<T>::random(const Shapes&... shape){
    auto out=Tensor<T>(shape...);
    for (lint n=0;n<out.size();n++){
      out.data()[n]=random_unit(T(0.0));      
    }
    return out;
  }
  
  //returns a tensor with shape shape..., initialized with random values of type T with real part between -1 and 1 and imaginary part 0  
  template<typename T>
  Tensor<T> Tensor<T>::random(const ShapeType& shape){
    auto out=Tensor<T>(shape);
    for (lint n=0;n<out.size();n++){
      out.data()[n]=random_unit(T(0.0));      
    }
    return out;
  }

  template<typename T>
  template<typename... Shapes>  
  Tensor<T> Tensor<T>::zeros(const Shapes&... shape){
    auto out=Tensor<T>(shape...);
    out.reset(T(0));
    return out;
  }
  
  template<typename T>
  Tensor<T> Tensor<T>::zeros(const ShapeType& shape){
    auto out=Tensor<T>(shape);
    out.reset(T(0));
    return out;
  }

  template<typename T>
  template<typename... Shapes>  
  Tensor<T> Tensor<T>::ones(const Shapes&... shape){
    auto out=Tensor<T>(shape...);
    out.reset(T(1));
    return out;
  }
  
  template<typename T>
  Tensor<T> Tensor<T>::ones(const ShapeType& shape){
    auto out=Tensor<T>(shape);
    out.reset(T(1));
    return out;
  }
  
  template<typename T>  
  Real Tensor<T>::norm()const{
    Real Z=0;
    T d;
    for (uint n=0;n<this->data_.size();n++){
      d=this->data_[n];
      Z+=(d*std::conj(d)).real();
    }
    return sqrt(Z);
  }

  template<typename T>  
  void Tensor<T>::reset(T val){
    data_.assign(data_.size(),val);
  }

  //copy the data of tensor to this
  template<typename T>  
  Tensor<T> & Tensor<T>::operator=(const Tensor<T> &tensor){
    //check for self-assignment
    if (this==&tensor){
      return *this;
    }
    //resize the container to the size of tensor
    this->data_.resize(tensor.size());
    //copy the data from tensor.raw() to this->raw()
    std::memcpy(this->raw(),tensor.raw(),tensor.size()*sizeof(T));
    this->shape_=tensor.shape();
    this->rank_=tensor.rank();
    this->size_=tensor.size();
    this->stride_=tensor.stride();       
    return *this;
  }
  
  //copy the data of tensor to this
  template<typename T>  
  Tensor<T>& Tensor<T>::operator=(Tensor<T> &&other){
    //check for self-assignment
    if (this==&other){
      return *this;
    }
    //resize the container to the size of tensor
    data_=std::move(other.data_);
    shape_=std::move(other.shape_);
    rank_=std::move(other.rank_);
    size_=std::move(other.size_);
    stride_=std::move(other.stride_);
    return *this;
  }

  
  //arithmetic operations
  template<typename T>  
  Tensor<T> & Tensor<T>::operator+=(const Tensor<T> &tensor){
    if (tensor.rank()!=this->rank())
      throw RankMismatchError("Tensor<T> operator+=(const Tensor<T> &tensor): ranks of tensors are not matching");
    if (tensor.shape()!=this->shape())
      throw ShapeMismatchError("Tensor<T> operator+=(const Tensor<T> &tensor): shapes of tensors are not matching");
    if (tensor.size()!=this->size())
      throw SizeMismatchError("Tensor<T> operator+=(const Tensor<T> &tensor): sizes of tensors are not matching");      
    
    T* thisptr=this->raw();
    const T* tensorptr=tensor.raw();
#pragma omp parallel for    
    for(lint n=0;n<tensor.data().size();n++){
      thisptr[n]+=tensorptr[n];
    }
    return *this;
  }

  template<typename T>  
  Tensor<T> & Tensor<T>::operator-=(const Tensor<T> &tensor){
    if (tensor.rank()!=this->rank())
      throw RankMismatchError("Tensor<T> operator-=(const Tensor<T> &tensor): ranks of tensors are not matching");
    if (tensor.shape()!=this->shape())
      throw ShapeMismatchError("Tensor<T> operator-=(const Tensor<T> &tensor): shapes of tensors are not matching");
    if (tensor.size()!=this->size())
      throw SizeMismatchError("Tensor<T> operator-=(const Tensor<T> &tensor): sizes of tensors are not matching");      
    
    T* thisptr=this->raw();
    const T* tensorptr=tensor.raw();
#pragma omp parallel for        
    for(lint n=0;n<tensor.data().size();n++){
      thisptr[n]-=tensorptr[n];
    }
    return *this;
  }

   template<typename T>
   //template<typename T2>
   Tensor<T> & Tensor<T>::operator*=(T val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]*=val;
     }
     return *this;
   }

  
   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(int val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }
  
   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(lint val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }
  
   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(luint val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }
   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(uint val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }


   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(float val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }

   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(Real val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }


   template<typename T>  
   Tensor<T> & Tensor<T>::operator+=(Complex val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]+=val;
     }
     return *this;
   }

  
   template<typename T>  
   Tensor<T> & Tensor<T>::operator-=(T val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]-=val;
     }
     return *this;
   }

   template<typename T>
   Tensor<T> & Tensor<T>::operator/=(T val){
     T* thisptr=this->raw();
 #pragma omp parallel for            
     for(lint n=0;n<this->data().size();n++){
       thisptr[n]/=val;
     }
     return *this;
   }

  
  //returns a new tensor; can be memory critical for large tensors due to internal memory allocation
  template<typename T>    
  template<typename... Labels>
  void Tensor<T>::naive_transpose(const Labels&... labs){
    LabelType labels={labs...};
    naive_transpose(labels);
  }
  
  //returns a new tensor; can be memory critical for large tensors due to internal memory allocation
  template<typename T>  
  void Tensor<T>::naive_transpose(const LabelType&labels){
    assert(labels.size()==rank_);
    //std::vector<Slice> slices(rank_);
    bool iden=true;
    LabelType check(rank_,0);    
    for (int n=0;n<rank_;n++){
      if (not (labels[n]<rank_)){
	throw std::runtime_error("in Tensor<T>::transpose: invalid axis for this array");
      }
      
      check[labels[n]]+=1;
      if (labels[n]!=n){
	iden=false;
      }
      //slices[n]=Slice(0,shape_[n]-1,1);      
    }
    for (auto n:check){
      if (n!=1){
	throw std::runtime_error("in Tensor<T>::transpose: repeated axis in transpose");
      }
    }
    if (iden==true){
      return;
    }
    ShapeType newshape(rank_);
    IndexType oldmultiindex(rank_,0),newmultiindex(rank_,0);
    StrideType newstride(rank_);    
    std::vector<T> newdata(size_);
    for(int n=0;n<rank_;n++){
      newshape[n]=shape_[labels[n]];
    }
    newstride[0]=1;
    for(int n=1; n<rank_; n++){
      newstride[n]=newstride[n-1]*newshape[n-1];
    }
    for(llint s=0;s<size_;s++){
      to_multiindex(oldmultiindex,s,stride_);
      for(int n=0;n<oldmultiindex.size();n++){
	newmultiindex[n]=oldmultiindex[labels[n]];
      }
      newdata[to_integer(newmultiindex,newstride)]=data_[s];
    }
    data_=newdata;
    shape_=newshape;
    stride_=newstride;
  }
  
  //returns copy of the real part of the tensor
  template<typename T>    
  Tensor<Real> Tensor<T>::real()const{
    Tensor<Real> out(ShapeType(this->shape()));
    for (luint n=0;n<this->size();n++){
      out.raw()[n]=std::real(this->raw()[n]);
    }
    return out;
  }

  //returns copy of the real part of the tensor
  template<typename T>    
  Tensor<Real> Tensor<T>::imag()const{
    Tensor<Real> out(ShapeType(this->shape()));
    for (luint n=0;n<this->size();n++){
      out.raw()[n]=std::imag(this->raw()[n]);
    }
    return out;
  }

  
  //returns conjugate copy of the tensor
  template<typename T>    
  Tensor<T> Tensor<T>::conj()const{
    Tensor<T> out=*this;
    out.conjugate();    
    return out;
  }
  
  //returns hermitian conjugate copy of this  
  template<typename T>    
  Tensor<T> Tensor<T>::herm()const{
    Tensor<T> out=*this;
    out.conjugate();
    out.transpose();
    return out;
  }

  template<typename T>    
  void Tensor<T>::hermitian(){
    this->conjugate();
    this->transpose();
  }
  
  //in-place complex conjugation
  template<typename T>
  void Tensor<T>::conjugate(){
    conj_inplace(*this);
  }
  
  //in-place; reverses the order the labels
  template<typename T>    
  void Tensor<T>::transpose(){
    LabelType label;
    for (int n=rank()-1;n>=0;n--)
      label.push_back(n);
    hptt(label);
  }
  //returns a new tensor with reversed order of the labels
  template<typename T>    
  Tensor<T> Tensor<T>::transp()const{
    LabelType label;
    for (int n=rank()-1;n>=0;n--)
      label.push_back(n);
    Tensor<T> out=*this;
    out.hptt(label);
    return out;
  }
  
  template<typename T>    
  template<typename... Labels>
  void Tensor<T>::transpose(const Labels&... labs){
    LabelType labels={labs...};
    hptt(labels);
  }
  
  template<typename T>
  void Tensor<T>::transpose(const LabelType&labels){
    hptt(labels);
  }
  
  template<typename T>    
  template<typename... Labels>
  void Tensor<T>::hptt(const Labels&... labs){
    LabelType labels={labs...};
    hptt(labels);
  }

  //returns a new tensor; can be memory critical for large tensors due to internal memory allocation
  template<typename T>  
  void Tensor<T>::hptt(const LabelType&labels,lint numThreads){
    assert(labels.size()==rank_);
    bool iden=true;
    LabelType check(rank_,0);    
    for (int n=0;n<rank_;n++){
      if (not (labels[n]<rank_)){
	throw std::runtime_error("in Tensor<T>::transpose: invalid axis for this array");
      }
      
      check[labels[n]]+=1;
      if (labels[n]!=n){
	iden=false;
      }
    }
    for (auto n:check){
      if (n!=1){
	throw std::runtime_error("in Tensor<T>::transpose: repeated axis in transpose");
      }
    }
    if (iden==true){
      return;
    }
    ShapeType newshape(rank_);
    StrideType newstride(rank_);    
    std::vector<T> newdata(size_);
    for(int n=0;n<rank_;n++){
      newshape[n]=shape_[labels[n]];
    }
    newstride[0]=1;
    for(int n=1; n<rank_; n++){
      newstride[n]=newstride[n-1]*newshape[n-1];
    }
    int perm[rank_];
    int size[rank_];
    for (int n =0;n<rank_;n++){
      perm[n]=labels[n];
      size[n]=(int)shape_[n];
    }
    T alpha=1.0,beta=0.0;

    auto plan = hptt::create_plan(perm, rank_,alpha,data_.data(),size, NULL,beta,newdata.data(), NULL,hptt::ESTIMATE, numThreads);
    plan->execute();
    std::memcpy(this->raw(),newdata.data(),newdata.size()*sizeof(T));
    shape_=newshape;
    stride_=newstride;
  }

  
  template<typename T>  
  Tensor<T> Tensor<T>::tblisdot(Tensor<T> &other,const LabelType& labelself,const LabelType& labelother){
    Tensor<T> result;    
    tblistensordot((*this),other,labelself,labelother,result);
    return result;
  }
  
  template<typename T>
  Tensor<T> Tensor<T>::gemmdot(Tensor<T>& other,const LabelType &labelself,const LabelType &labelother){
    Tensor<T> result;
    gemmtensordot(*this,other,labelself,labelother,result);
    return result;
  }

  template<typename T>
  Tensor<T> Tensor<T>::tcldot(Tensor<T>& other,const LabelType& labelself,const LabelType& labelother){
    Tensor<T> result;
    tcltensordot(*this,other,labelself,labelother,result);
    return result;
  }

  template<typename T>
  Tensor<T> Tensor<T>::dot(Tensor<T> &other){
    Tensor<T> C;
    if((this->rank()==1) and (other.rank()==1)){
      C=gemmdot(other,LabelType{1},LabelType{1});
    }else if((this->rank()==2) and (other.rank()==2)){
      C=gemmdot(other,LabelType{-1,1},LabelType{1,-2});
    }else if((this->rank()==1) and (other.rank()==2)){
      try{
	gemm_mv_dot(other,*this,C,'T');      }
      catch(std::runtime_error){
	throw std::runtime_error("couldn't do the vector-matrix multiplicataino");
      }
    }else if((this->rank()==2) and (other.rank()==1)){
      gemm_mv_dot(*this,other,C);
    }else{
      throw std::runtime_error("Tensor<T>.dot(other): other is neither vector nor matrix; use gemmdot, tblisdot or tcldot instead");
    }
    return C;
  }
  
  template<typename T>
  Tensor<T> Tensor<T>::dot(Tensor<T> &&other){
    return dot(other);
  }


  template<typename T>      
  template<typename... Shapes>
  void Tensor<T>::resize(Shapes... newshape){
    try{
      check_integral(newshape...);
      vector<int>i={newshape...};
      ShapeType s(i.begin(),i.end());
      shape_=s;
      init();
    }
    catch(NotAnIntegerError){
      cout<<"Tensor<T>.resize()(Shapes... newsize) got a non-integer type argument"<<endl;
      throw;
    }
    
  }
  template<typename T>  
  void Tensor<T>::resize(ShapeType newshape){
    shape_=newshape;
    init();
  }
  
  template<typename T>  
  template<typename... Shapes>
  void Tensor<T>::reshape(const Shapes&...newshape){
    try{
      check_integral(newshape...);
      vector<int>i={newshape...};
      ShapeType shape(i.begin(),i.end());
      reshape(shape);
    }
    catch(NotAnIntegerError){
      cout<<"Tensor<T>.reshape()(Shapes... newshape) got a non-integer type argument"<<endl;
      throw;
    }
  }
  
  template<typename T>    
  void Tensor<T>::reshape(ShapeType shape){
    lint D=1;
    for (auto d:shape){
      D*=d;
    }
    if(D!=size_){
      throw std::runtime_error("Tensor<T>::reshape: cannot change number of elements");
    }
    shape_=shape;
    stride_.resize(shape.size());
    rank_=shape_.size();    
    stride_[0]=1;
    for(int n=1; n<rank_; n++){
      stride_[n]=stride_[n-1]*shape_[n-1];
    }
  }

  template<typename T>
  size_type Tensor<T>::shape(const lint n) const {assert(n<rank_);return shape_[n];}
  
  template<typename T>  
  const ShapeType& Tensor<T>::shape () const {return shape_;}
  
  template<typename T>  
  ShapeType& Tensor<T>::shape () {return shape_;}

  template<typename T>
  template<std::size_t rank>
  auto Tensor<T>::shape()const {return vectorToTuple<rank>(shape_);}
  
  template<typename T>  
  size_type Tensor<T>::size() const {return this->data_.size();}
  template<typename T>  
  lint Tensor<T>::rank() const {return rank_;}
  template<typename T>  
  const StrideType& Tensor<T>::stride () const {return stride_;}
  template<typename T>  
  StrideType& Tensor<T>::stride () {return stride_;}

  template<typename T>  
  void Tensor<T>::print(int L,int prec)const{
    cout<<"Tensor:"<<endl;
    cout<<"shape: ";
    printfunctions::print(shape_);
    cout<<"tensor data"<<endl;
    if (rank_>2 or rank_<2)
      printfunctions::print(data_);
    else if (rank_==2){
      bool printdotsn,printdotsm=true;
      for(uint m=0;m<shape_[0];m++){
	if ((m<L) or m>((int)shape_[0]-L-1)){	    	
	  printdotsn=true;
	  for(uint n=0;n<shape_[1];n++){
	    if ((n<L) or n>((int)shape_[1]-L-1)){
	      cout<<std::showpos<<std::fixed<<setprecision(prec)<<this->operator()(m,n)<<"  ";
	    }
	    else if (printdotsn==true){
	      cout<<"... ";
	      printdotsn=false;
	    }
	  }
	  cout<<endl;
	}else if (printdotsm==true){
	    cout<<" . "<<endl;
	    cout<<" . "<<endl;
	    cout<<" . "<<endl;
	    printdotsm=false;
	}
      }
    }
  }
  
  template<typename T>  
  void Tensor<T>::init(){
    size_=1;
    rank_=shape_.size();
    stride_.resize(rank_);
    stride_[0]=1;
    for(int n=0; n<rank_; n++){
      size_*=shape_[n];
    }
    for(int n=1; n<rank_; n++){
      stride_[n]=stride_[n-1]*shape_[n-1];
    }
    data_.resize(size_);
  }
  template<typename T>
  void Tensor<T>::check_range(IndexType inds)const{
    for (int n=0;n<inds.size();n++){
      if (inds[n]>=shape_[n]){
	throw OutOfBoundError();
      }
    }
  }
  
  
}  
#endif
