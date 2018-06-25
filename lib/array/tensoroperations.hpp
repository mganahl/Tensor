#ifndef TENSOROPS_HPP_
#define TENSOROPS_HPP_
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
#include <tuple>
#include <stdexcept>

#include <tblis.h>
#include <tcl.h>
#include "hptt.h"
#include "typedefs.hpp"
#include "../utils/utilities.hpp"
#include "../utils/print.hpp"
#include "../linalg/tblisroutines.hpp"
#include "../linalg/lapackroutines.hpp"
#include "../linalg/blasroutines.hpp"
#include "../utils/exceptions.hpp"
#include "tensor.hpp"


using namespace std;
using namespace utilities;
using namespace exceptions;
using namespace blasroutines;
using namespace tblisroutines;
using namespace printfunctions;
using std::cout;
using std::setprecision;
using std::endl;
using std::vector;

namespace tensor{
  Tensor<Complex> determineReturnType(const Tensor<Complex>&,const  Tensor<Complex>&){};
  Tensor<Complex> determineReturnType(const Tensor<Complex>&,const  Tensor<Real>&){};
  Tensor<Complex> determineReturnType(const Tensor<Real>&,const  Tensor<Complex>&){};
  Tensor<Real> determineReturnType(Tensor<Real>&, Tensor<Real>&){};
  
  bool isComplex(const Tensor<Complex>&){return true;}
  bool isComplex(const Tensor<Real>&){return false;}
  Real c_rand(Real min,Real max){
    assert(min<=max);
    return min +fabs(max-min)*rand()/RAND_MAX;
  }
  
  Complex c_rand(Complex min,Complex max){
    return Complex(c_rand(min.real(),max.real()),c_rand(min.imag(),max.imag()));
  }

  Complex random_unit(Complex t){
    return Complex(c_rand(-1,1),c_rand(-1,1));
  }
  Real random_unit(Real t){
    return c_rand(-1,1);
  }
  Real random_unit(int t){
    return c_rand(-1000,1000);
  }

  /*
    returns a new tensor of shape shape... filled with ones;
    auto t=ones<T>(d1,d2,...,dn) returns a rank n tensor with dimensions d1,d2,..,dn filled with ones
  */
  template<typename T,typename... Shapes>
  Tensor<T> ones(const Shapes& ... shape){
    return Tensor<T>().ones(shape...);
  }
  
  /*
    returns a new tensor of shape "shape" filled with ones;
    auto t=ones<T>(ShapType{d1,d2,...,dn}) returns a rank n tensor with dimensions d1, d2,..., dn filled with ones
  */
  
  template<typename T>
  Tensor<T> ones(const ShapeType &shape){
    return Tensor<T>().ones(shape);
  }
  /*
    returns a new tensor of shape shape... filled with zeors;
    auto t=ones<T>(d1,d2,...,dn) returns a rank n tensor with dimensions d1,d2,..,dn filled with zeros
  */

  template<typename T,typename... Shapes>
  Tensor<T> zeros(const Shapes& ... shape){
    return Tensor<T>().zeros(shape...);
  }
  
  /*
    returns a new tensor of shape "shape" filled with zeros;
    auto t=ones<T>(ShapType{d1,d2,...,dn}) returns a rank n tensor with dimensions d1, d2,..., dn filled with zeros
  */

  template<typename T>
  Tensor<T> zeros(const ShapeType &shape){
    return Tensor<T>().zeros(shape);
  }
  /*
    returns an identity of dimension "shape": auto eye<T>(d) is a d by d identity of type T
   */
  template<typename T>
  Tensor<T> eye(int shape){
    return diag(ones<T>(shape));
  }
  /*
    returns an identity of dimension "shape": auto eye<T>(d) is a d by d identity of type T
  */
  
  template<typename T>
  Tensor<T> eye(size_type shape){
    return diag(ones<T>((int)shape));
  }


  /*
    returns a random tensor of type T and dimension shape...;  values are initialized with values uniformly [-1,1] or in the unit-square
    auto t= random<T>(d1,d2,d...,dn) 
  */
  template<typename T,typename... Shapes>
  Tensor<T> random(const Shapes& ... shape){
    return Tensor<T>().random(shape...);
  }
  /*
    returns a random tensor of type T and dimension shape;  values are initialized with values uniformly in [-1,1] or in the unit-square
    auto t= random_real(ShapeType{d1,...,dn}) 
  */
  template<typename T>
  Tensor<T> random(const ShapeType& shape){
    return Tensor<T>().random(shape);
  }

  template<typename T>
  T sum(const Tensor<T>& A){
    return std::accumulate(A.data().begin(),A.data().end(),T(0.0));
  }

  template<typename T>
  T trace(const Tensor<T>& A){
    if (A.rank()!=2){
      throw RankMismatchError("in trace(const Tensor<T>& A: expected a rank 2 tensor");
    }
    return sum(diag(A));
  }
  
  /*
    create a diagonal matrix from a rank 1 tensor or return the diagonal of a rank 2 tensor
    if tensor is rank 1, returns a diagonal tensor of rank 2 with elements of tensor on the diagonal
    if tensor is rank 2, returns the diagonal tensor in a rank 1 tensor
    if rank != 1 or 2, throws a std::runtime_error
  */
  template<typename T>
  Tensor<T> diag(const Tensor<T>&tensor){
    if ((tensor.rank()!=1) and (tensor.rank()!=2)){
      throw std::runtime_error("in diag: tensor rank is != 1 and !=2");
    }
    if (tensor.rank()==1){
      Tensor<T> out(ShapeType{tensor.shape(0),tensor.shape(0)});
      for(uint n=0;n<out.shape(0);n++)
	out(n,n)=tensor(n);
      return out;      
    }else if(tensor.rank()==2){
      //auto size=min(tensor.shape(0),tensor.shape(1));
      auto size=(*min_element(tensor.shape().begin(),tensor.shape().end()));
      Tensor<T> out(ShapeType{size});
      out.reset(T(0));
      for (uint n=0;n<size;n++){
	out(n)=tensor(n,n);
      }
      return out;
    } 
  }


  /*
    calculates eigenvectors and eigenvalues (EV) of a square hermitian real or complex matrix A;
    and write the result into EV (eigenvalues) and U (eigenvectors)
    A=U.dot(diag(EV)).dot(herm(U));
  */
  
  template<typename T>
  void eigh(const Tensor<T>&A,Tensor<Real>&EV,Tensor<T>&U){
    //do some checks:
    if (A.rank()!=2)
      throw RankMismatchError("eig: rank of tensor !=2!");
    if (A.shape(0)!=A.shape(1))
      throw SizeMismatchError("eig: matrix is not square!");
    U=A.copy();
    EV.resize(ShapeType{U.shape(0)});
    //eig destroys the contenst of the first argument! pass a copy to preserve A
    lapackroutines::eigh(U.raw(),U.shape(0),EV.raw(),'V');        
  }


  /*
    calculates eigenvectors  (VR) and eigenvalues (EV) of a square hermitian real or complex matrix A;
    returns a pair containin EV and VR; 
    
  */
  template<typename T>
  std::pair<Tensor<Real>,Tensor<T> > eigh(const Tensor<T>&A){
    //do some checks:
    if (A.rank()!=2)
      throw RankMismatchError("eig: rank of tensor !=2!");
    if (A.shape(0)!=A.shape(1))
      throw SizeMismatchError("eig: matrix is not square!");
    
    Tensor<T>U=A;
    Tensor<Real>EV(ShapeType{A.shape(0)});
    //eig destroys the contenst of the first argument! pass a copy to preserve A
    lapackroutines::eigh(U.raw(),A.shape(0),EV.raw(),'V');
    return {EV,U};    
  }

  /*
    calculates left (VL) and right (VR) eigenvectors and eigenvalues (EV) of a square matrix A;
    returns a tuple containin VL,EV and VR; 
    note that lapack::eig destroys the contenst of the first argument! pass a copy to preserve A
  */
  template<typename T>
  std::tuple<Tensor<Complex>,Tensor<Complex>,Tensor<Complex> > eig(Tensor<T>A){
    //do some checks:
    if (A.rank()!=2)
      throw RankMismatchError("eig: rank of tensor !=2!");
    if (A.shape(0)!=A.shape(1))
      throw SizeMismatchError("eig: matrix is not square!");
    
    Tensor<Complex> VL(A.shape()),EV(ShapeType{A.shape(0)}),VR(A.shape());

    lapackroutines::eig(A.raw(),A.shape(0),VL.raw(),EV.raw(),VR.raw(),'V','V');        
    return {VL,EV,VR};    
  }

  
  template<typename T>
  std::tuple<Tensor<Complex>,Tensor<Complex>,Tensor<Complex> > eig(Tensor<T>&&A){
    //do some checks:
    if (A.rank()!=2)
      throw RankMismatchError("eig: rank of tensor !=2!");
    if (A.shape(0)!=A.shape(1))
      throw SizeMismatchError("eig: matrix is not square!");
    
    Tensor<Complex> VL(A.shape()),EV(ShapeType{A.shape(0)}),VR(A.shape());

    lapackroutines::eig(A.raw(),A.shape(0),VL.raw(),EV.raw(),VR.raw(),'V','V');        
    return {VL,EV,VR};    
  }
  
  /*
    calculates left (VL) and right (VR) eigenvectors and eigenvalues (EV) of a square matrix A;
    VL,VR and EV are internally resized to match the dimension of A
    note that below, lapackroutines::eig destroys the contents of A! A is thus passed as a copy to eig
  */
  
  template<typename T>
  void eig(Tensor<T>A,Tensor<Complex>&VL,Tensor<Complex>&EV,Tensor<Complex>&VR){
    //do some checks:
    if (A.rank()!=2)
      throw RankMismatchError("eig: rank of tensor !=2!");
    if (A.shape(0)!=A.shape(1))
      throw SizeMismatchError("eig: matrix is not square!");
    
    VL.resize(A.shape()),EV.resize(ShapeType{A.shape(0)}),VR.resize(A.shape());

    lapackroutines::eig(A.raw(),A.shape(0),VL.raw(),EV.raw(),VR.raw(),'V','V');        
  }
  
  template<typename T>
  void eig(Tensor<T>&&A,Tensor<Complex>&VL,Tensor<Complex>&EV,Tensor<Complex>&VR){
    //do some checks:
    if (A.rank()!=2)
      throw RankMismatchError("eig: rank of tensor !=2!");
    if (A.shape(0)!=A.shape(1))
      throw SizeMismatchError("eig: matrix is not square!");
    
    VL.resize(A.shape()),EV.resize(ShapeType{A.shape(0)}),VR.resize(A.shape());

    lapackroutines::eig(A.raw(),A.shape(0),VL.raw(),EV.raw(),VR.raw(),'V','V');        
  }

  /*
    QR decomposition of Tensor A with shape M,N; A has to be of A.rank()==2!;
    if M>N and full=true, returns Q.shape()=(M,M), and R.shape()=(M,N) matrix, with Q a unitary and R an upper triangular matrix
    if M>N and full=false, returns Q with Q.shape()=(M,N) and R with R.shape()= (N,N), with Q an isometry of the form herm(Q).dot(Q)=11 and R an upper triangular matrix; 
    note that Q.dot(herm(Q))!=11!
    if M<=N, the value of full has no effect on the result; in this case, Q.shape()=(M,M), R.shape()=(M,N) for any value of full, and Q will be a unitary matrix
  */
  
  template<typename T>
  std::pair<Tensor<T>,Tensor<T> > qr(Tensor<T>A,bool full){
    if (A.rank()>2)
      throw std::runtime_error("in QR: rank(A)>2; qr can currently only handle 2-d arrays");
    if (A.rank()<2)
      throw std::runtime_error("in QR: rank(A)<2; qr can currently only handle 2-d arrays");
    
    if (A.rank()==2){
      Tensor<T> R,Q;
      size_type dA1=A.shape(0),dA2=A.shape(1);
      //Q has to be first resized to dA1,dA1
      Q.resize(ShapeType{dA1,dA1});
      R.resize(ShapeType{dA1,dA2});
      int out=lapackroutines::qr(A.raw(),dA1,dA2,Q.raw(),R.raw());
      if (full==false){
	if (dA1>dA2){
	  Q.resize(ShapeType{dA1,dA2});
	  R.transpose();
	  R.resize(ShapeType{dA2,dA2});
	  R.transpose();	
	}
      }

      // for(lint n=0;n<Q.size();n++){
      // 	Q.raw()[n]=A.raw()[n];
      // }
      return {Q,R};          
    }
  }

  //svd of Tensor A, A has to be A.rank()==2; U,S,VH are resized within the routine
  //A has to be passed by value because lapack svd is not preserving it
  //this svd is always of "economic" type
  template<typename T>
  int svd(Tensor<T>A,Tensor<T>&U,Tensor<Real> &S,Tensor<T> &VH){
    if (A.rank()>2)
      throw std::runtime_error("in SVD: rank(A)>2; svd can currently only handle 2-d arrays");
    if (A.rank()<2)
      throw std::runtime_error("in SVD: rank(A)<2; svd can currently only handle 2-d arrays");
    
    if (A.rank()==2){
      size_type dA1=A.shape(0),dA2=A.shape(1);
      if (dA1>dA2){
	U.resize(ShapeType{dA1,dA2});
	S.resize(ShapeType{dA2});
	VH.resize(ShapeType{dA2,dA2});
      }
      else if(dA1<=dA2){
	U.resize(ShapeType{dA1,dA1});
	S.resize(ShapeType{dA1});
	VH.resize(ShapeType{dA1,dA2});
      }
      int out=lapackroutines::svd(A.raw(), U.raw(),VH.raw(),S.raw(),dA1,dA2,'S');
    }
  }

  //svd of Tensor A, A has to be A.rank()==2;
  //returns a tuple containing U,S,VH, such that A=U.dot(diag(S)).dot(VH);
  template<typename T>  
  std::tuple<Tensor<T>,Tensor<Real>,Tensor<T> > svd(const Tensor<T>&A){
    Tensor<T> U,VH;
    Tensor<Real> S;    
    int out=svd(A,U,S,VH);
    return {U,S,VH};    
  }


  /* 
     see void tblistensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
  */
  template<typename T>
  void tblistensordot(Tensor<T>&&A,Tensor<T>&&B,LabelType labela,LabelType labelb,Tensor<T> &C){
    tblistensordot(A,B,labela,labelb,C);
  }
  /* 
     see void tblistensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
  */
  
  template<typename T>
  void tblistensordot(Tensor<T>&A,Tensor<T>&&B,LabelType labela,LabelType labelb,Tensor<T> &C){
    tblistensordot(A,B,labela,labelb,C);
  }
  /* 
     see void tblistensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
  */
  
  template<typename T>
  void tblistensordot(Tensor<T>&&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C){
    tblistensordot(A,B,labela,labelb,C);
  }


  /*
    void tblistensordot(Tensor<T>&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C):
    
    calculates the contraction of tensors A and B over the indices labela and labelb, using the tblis library (see devin matthews on github)
    labela and labelb are integer labels; positive labels are used to denote common contractions, negative labels
    are used to denote uncontracted indices. The uncontracted indices of the resulting tensor are ordered
    according with the largest indices coming first, e.g.
    labela=(1,2,-2,,3,-3), labelb=(2,-1,1,3) -> labelout=(-1,-2,-3), contraction over labels (1,2,3)
  */
  template<typename T>
  void tblistensordot(Tensor<T>&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C){
    //for the tblis tensordor, the resulting tensor has to be initialized with 0.0!
    //=================== do some checks ========================================
    for (auto m:labela)
      if (m==0)
       	throw std::runtime_error("in tblistensordot: one of the labels of Tensor A is 0; use positive or negative values only");
    for (auto m:labelb)
      if (m==0)
       	throw std::runtime_error("in tblistensordot: one of the labels of Tensor B is 0; use positive or negative values only");
    if (A.rank()!=labela.size())
      throw std::runtime_error("in tblistensordot: number of labels given for tensor A is different from rank(A)");
    if (B.rank()!=labelb.size())
      throw std::runtime_error("in tblistensordot: number of labels given for tensor B is different from rank(B)");

    auto common=intersection(labela,labelb);
    LabelType labelout=resultinglabel(labela,labelb);
    for(auto n:common){
      if(n<0)
       	throw std::runtime_error("in tblistensordot: got an outgoing negative label more than once; use positive labels to denote common contraction");
      int l,m;
      for (m=0;m<labela.size();m++)
       	if (labela[m]==n)
       	  break;
      for (l=0;l<labelb.size();l++)
       	if (labelb[l]==n)
       	  break;
      if (A.shape(m)!=B.shape(l))
       	throw std::runtime_error("in tblistensordot: some dimensions of the tensorlegs are not matching");
    }
    for(auto n: labelout){
      if(n>0)
       	throw std::runtime_error("in tblistensordot: got an outgoing label>0; positive labels are used to denote uncontracted indices");
    }

    //=================== finished checks ========================================
    //if labelout.size()==0, the two tensors have to be fully contracted (i.e. all labels of A appear in B and vice veres).
    if (labelout.size()==0){
      C.resize(1);
      C.reset(T(0.0));
      C(0)=tblis_dot(&A.shape(), &A.stride(), A.raw(),&B.shape(),&B.stride(),B.raw(), &labela,&labelb);
    }      
    if (labelout.size()>0){    
      ShapeType dims;
      for (auto n:labelout){
	for (int m=0;m<labela.size();m++)
	  if (n==labela[m]){
	    dims.push_back(A.shape(m));
	    break;
	  }
	for (int m=0;m<labelb.size();m++)
	  if (n==labelb[m]){
	    dims.push_back(B.shape(m));
	    break;
	  }
      }
      C.resize(dims);
      C.reset(T(0.0));
      std::vector<lint> sA(A.shape().begin(),A.shape().end()),sB(B.shape().begin(),B.shape().end()),sC(C.shape().begin(),C.shape().end());
      tblis::tensor <T> tblisA(sA);
      tblis::tensor <T> tblisB(sB);
      tblis::tensor <T> tblisC(sC);       
      std::copy(A.data().begin(),A.data().end(),tblisA.data());
      std::copy(B.data().begin(),B.data().end(),tblisB.data());       
      tblis::mult(T(1.0),tblisA,labela.data(),tblisB,labelb.data(),T(1.0),tblisC,labelout.data());
      std::copy(tblisC.data(),tblisC.data()+C.size(),C.raw());
    }
  }



  /* 
     see void tcltensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
  */
  template<typename T>
  void tcltensordot(Tensor<T>&&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
    tcltensordot(A,B,labela,labelb,C);
  }
  /* 
     see void tcltensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
  */
  
  template<typename T>
  void tcltensordot(Tensor<T>&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
    tcltensordot(A,B,labela,labelb,C);
  }
  /* 
     see void tcltensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
  */
  
  template<typename T>
  void tcltensordot(Tensor<T>&&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
    tcltensordot(A,B,labela,labelb,C);
  }

  
  /*
    void tcltensordot(Tensor<T>&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C):
    calculates the contraction of tensors A and B over the indices labela and labelb, using the tcl library (see springer13 on github).
    for full contractions routines switches to ddot_ from blas;
    labela and labelb are integer labels; positive labels are used to denote common contractions, negative labels
    are used to denote uncontracted indices. The uncontracted indices of the resulting tensor are ordered
    according with the largest indices coming first, e.g.
    labela=(1,2,-2,,3,-3), labelb=(2,-1,1,3) -> labelout=(-1,-2,-3), contraction over labels (1,2,3)
  */
  
  template<typename T>
  void tcltensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){

    for (auto m:labela)
      if (m==0)
       	throw std::runtime_error("in tcltensordot: one of the labels of Tensor A is 0; use positive or negative values only");
    for (auto m:labelb)
      if (m==0)
       	throw std::runtime_error("in tcltensordot: one of the labels of Tensor B is 0; use positive or negative values only");
    if (A.rank()!=labela.size())
      throw std::runtime_error("in tcltensordot: number of labels given for tensor A is different from rank(A)");
    if (B.rank()!=labelb.size())
      throw std::runtime_error("in tcltensordot: number of labels given for tensor B is different from rank(B)");

    auto common=intersection(labela,labelb);
    LabelType labelout=resultinglabel(labela,labelb);
    for(auto n:common){
      if(n<0)
       	throw std::runtime_error("in tcltensordot: got an outgoing negative label more than once; use positive labels to denote common contraction");
      int l,m;
      for (m=0;m<labela.size();m++)
       	if (labela[m]==n)
       	  break;
      for (l=0;l<labelb.size();l++)
       	if (labelb[l]==n)
       	  break;
      if (A.shape(m)!=B.shape(l))
       	throw std::runtime_error("in tcltensordot: some dimensions of the tensorlegs are not matching");
    }
    for(auto n: labelout){
      if(n>0)
       	throw std::runtime_error("in tcltensordot: got an outgoing label>0; positive labels are used to denote uncontracted indices");
    }

    if (labelout.size()==0){
      /*
	do a full contraction; bring labels of tensor two B the same order as tensor one; this uses ddot_ for full contraction.
      */
      LabelType transp_pos_label_b,final_pos_label_b;
      for (auto la: labela){
	for(uint n=0;n<labelb.size();n++){
	  if (la==labelb[n]){
	    transp_pos_label_b.push_back(n);
	    break;
	  }
	}
      }
      //transpose b into the same order as a
      B.hptt(transp_pos_label_b);
      C.resize(1);
      //contract the two tensors into C    
      C(0)=dot_prod(A.size(),A.raw(),B.raw());
      //restore the old order of b
      for(auto lb:labelb){
	for(int n=0;n<labela.size();n++){
	  if(lb==labela[n]){
	    final_pos_label_b.push_back(n);
	    break;
	  }
	}
      }
      B.hptt(final_pos_label_b);
    }
    else if (labelout.size()!=0){
      ShapeType dims;
      for (auto n:labelout){
	for (int m=0;m<labela.size();m++)
	  if (n==labela[m]){
	    dims.push_back(A.shape(m));
	    break;
	  }
	for (int m=0;m<labelb.size();m++)
	  if (n==labelb[m]){
	    dims.push_back(B.shape(m));
	    break;
	  }
      }
      C.resize(dims);
      vector<tcl::sizeType> lA,lB,lC;
      for(auto n:A.shape())
	lA.push_back((int)n);
      for(auto n:B.shape())
	lB.push_back((int)n);
    
      for(auto n:C.shape())
	lC.push_back((int)n);

      tcl::Tensor<T> tclA(lA, A.raw());
      tcl::Tensor<T> tclB(lB, B.raw());
      tcl::Tensor<T> tclC(lC, C.raw());
      T alpha = T(1);
      T beta = T(0);
      auto sla=LabelType_to_string(labela);
      auto slb=LabelType_to_string(labelb);
      auto slc=LabelType_to_string(labelout);
      auto err = tcl::tensorMult<T>( alpha, tclA[sla], tclB[slb], beta, tclC[slc] );
      if( err != tcl::SUCCESS ){
	printf("ERROR: %s\n", tcl::getErrorString(err));
	exit(-1);
      }
    }
  }

  /* 
     see void gemmtensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
   */
  template<typename T>
  void gemmtensordot(Tensor<T>&&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
    gemmtensordot(A,B,labela,labelb,C);
  }
  /* 
     see void gemmtensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
   */
  
  template<typename T>
  void gemmtensordot(Tensor<T>&A,Tensor<T>&&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
    gemmtensordot(A,B,labela,labelb,C);
  }
  /* 
     see void gemmtensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
   */
  
  template<typename T>
  void gemmtensordot(Tensor<T>&&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){
    gemmtensordot(A,B,labela,labelb,C);
  }
  
  /* 
     void gemmtensordot(Tensor<T>&A,Tensor<T>&B,LabelType labela,LabelType labelb,Tensor<T> &C):
     
     this routine uses zgemm and dgemm to do a tensor contractoin; A and B are unchanged upon finishing of the routine; note
     however that the routine temporarily reshapes transposes A and B in place; this will cause trouble if A and B are 
     are simultaneously used during parallel computations, using e.g. openmp threading. Note that for most cases it might be 
     faster to copy A and B instead of passing by ref, so you don't have to transpose them back afterwards (can be slow);
     A small test confirmed a slight difference in speed in favour of copying;
     labela and labelb are integer labels; positive labels are used to denote common contractions, negative labels
     are used to denote uncontracted indices. The uncontracted indices of the resulting tensor are ordered
     according with the largest indices coming first, e.g.
     labela=(1,2,-2,,3,-3), labelb=(2,-1,1,3) -> labelout=(-1,-2,-3), contraction over labels (1,2,3)
  */

  template<typename T>
  void gemmtensordot(Tensor<T>&A,Tensor<T>&B,const LabelType& labela,const LabelType& labelb,Tensor<T> &C){

    for (auto m:labela)
      if (m==0)
       	throw std::runtime_error("in gemmtensordot: one of the labels of Tensor A is 0; use positive or negative values only");
    for (auto m:labelb)
      if (m==0)
       	throw std::runtime_error("in gemmtensordot: one of the labels of Tensor B is 0; use positive or negative values only");
    if (A.rank()!=labela.size())
      throw std::runtime_error("in gemmtensordot: number of labels given for tensor A is different from rank(A)");
    if (B.rank()!=labelb.size())
      throw std::runtime_error("in gemmtensordot: number of labels given for tensor B is different from rank(B)");

    auto common=intersection(labela,labelb);
    LabelType labelout=resultinglabel(labela,labelb);
    for(auto n:common){
      if(n<0)
       	throw std::runtime_error("in gemmtensordot: got an outgoing negative label more than once; use positive labels to denote common contraction");
      int l,m;
      for (m=0;m<labela.size();m++)
       	if (labela[m]==n)
       	  break;
      for (l=0;l<labelb.size();l++)
       	if (labelb[l]==n)
       	  break;
      if (A.shape(m)!=B.shape(l))
       	throw std::runtime_error("in gemmtensordot: some dimensions of the tensorlegs are not matching");
    }
    for(auto n: labelout){
      if(n>0)
       	throw std::runtime_error("in gemmtensordot: got an outgoing label>0; positive labels are used to denote uncontracted indices");
    }
    //the resulting label can have length 0, which means that the tensors are fully contracted;
    if (labelout.size()==0){
      /*do a full contraction; bring labels of tensor two into the same order as tensor one */
      LabelType transp_pos_label_b,final_pos_label_b;
      for (auto la: labela){
	for(uint n=0;n<labelb.size();n++){
	  if (la==labelb[n]){
	    transp_pos_label_b.push_back(n);
	    break;
	  }
	}
      }
      //transpose b into the same order as a
      B.hptt(transp_pos_label_b);
      C.resize(1);
      //contraction
      C(0)=dot_prod(A.size(),A.raw(),B.raw());
      //restore the old order of b
      for(auto lb:labelb){
	for(int n=0;n<labela.size();n++){
	  if(lb==labela[n]){
	    final_pos_label_b.push_back(n);
	    break;
	  }
	}
      }
      B.hptt(final_pos_label_b);
    }
    //if the resulting label has length larger than 0,
    if (labelout.size()>0){
      ShapeType dims;
      for (auto n:labelout){
	for (int m=0;m<labela.size();m++)
	  if (n==labela[m]){
	    dims.push_back(A.shape(m));
	    break;
	  }
	for (int m=0;m<labelb.size();m++)
	  if (n==labelb[m]){
	    dims.push_back(B.shape(m));
	    break;
	  }
      }

      //CONTRACTED indices of A are transposed to the end of A, the contracted indices of B are transposed to the beginning of B:
      //aminuscommon has all negative labels of A
      auto aminuscommon=difference(labela,common);
      auto bminuscommon=difference(labelb,common);
      LabelType transp_pos_label_a,transp_pos_label_b,transplabela,transplabelb,finallabel,final_pos_label_a,final_pos_label_b;
      ShapeType newshape;
      size_type dA1=1,dAB=1,dB2=1;
      for(auto i:aminuscommon){
	transplabela.push_back(i);
	finallabel.push_back(i);
	for(int n=0;n<labela.size();n++)
	  if (labela[n]==i){
	    dA1*=A.shape(n);
	    newshape.push_back(A.shape(n));
	    transp_pos_label_a.push_back(n);
	    break;
	  }
      }
      for(auto c:common){
	transplabela.push_back(c);
	transplabelb.push_back(c);      
	for(int n=0;n<labela.size();n++)
	  if (labela[n]==c){
	    dAB*=A.shape(n);
	    transp_pos_label_a.push_back(n);
	    break;
	  }
	for(int n=0;n<labelb.size();n++)
	  if (labelb[n]==c){	
	    transp_pos_label_b.push_back(n);
	  }
      }
    
      for(auto i:bminuscommon){
	transplabelb.push_back(i);
	finallabel.push_back(i);            
	for(int n=0;n<labelb.size();n++)
	  if (labelb[n]==i){
	    dB2*=B.shape(n);
	    newshape.push_back(B.shape(n));	  
	    transp_pos_label_b.push_back(n);
	    break;
	  }
      }
      A.hptt(transp_pos_label_a);
      B.hptt(transp_pos_label_b);

      ShapeType origshapeA(A.shape());
      ShapeType origshapeB(B.shape());
      A.reshape(ShapeType{dA1,dAB});
      B.reshape(ShapeType{dAB,dB2});
      C.resize(ShapeType({dA1,dB2}));

      mat_mat_prod(T(1.0),A.raw(),dA1,dAB,B.raw(),dAB,dB2,C.raw(),'n','n' );
      C.reshape(newshape);
      LabelType finallabelsorted(finallabel),final_pos_label;;
      std::sort(finallabelsorted.begin(),finallabelsorted.end(),largerthan);
      for (auto fp:finallabelsorted)
	for(int n=0;n<finallabel.size();n++)
	  if (finallabel[n]==fp){
	    final_pos_label.push_back(n);
	  }
      C.hptt(final_pos_label);

      //restore the original shapes of A and B
      for(auto la:labela){
	for(int n=0;n<transplabela.size();n++){
	  if(la==transplabela[n]){
	    final_pos_label_a.push_back(n);
	    break;
	  }
	}
      }
      for(auto lb:labelb){
	for(int n=0;n<transplabelb.size();n++){
	  if(lb==transplabelb[n]){
	    final_pos_label_b.push_back(n);
	    break;
	  }
	}
      }
      A.reshape(origshapeA);
      B.reshape(origshapeB);

      A.hptt(final_pos_label_a);
      B.hptt(final_pos_label_b);
    }
  }

  template<typename T>
  Tensor<T> tensordot(Tensor<T> &A, Tensor<T>&B,const LabelType& labela, const LabelType& labelb){
    const char* type=std::getenv("TDOTMETHOD");
    Tensor<T> C;    
    if (type!=NULL){
      std::string method(type);
      if((method==string("TBLIS")) or (method==string("tblis"))){
	tblistensordot(A,B,labela,labelb,C);
      }else if((method==string("GEMM")) or (method==string("gemm"))){
	gemmtensordot(A,B,labela,labelb,C);
      }else{
	tcltensordot(A,B,labela,labelb,C);
      }
    }else{
      tcltensordot(A,B,labela,labelb,C);
    }
    return C;
  }
  template<typename T>
  Tensor<T> tensordot(Tensor<T> &&A, Tensor<T>&B,const LabelType& labela, const LabelType& labelb){
    const char* type=std::getenv("TDOTMETHOD");
    Tensor<T> C;    
    if (type!=NULL){
      std::string method(type);
      if((method==string("TBLIS")) or (method==string("tblis"))){
	tblistensordot(A,B,labela,labelb,C);
      }else if((method==string("GEMM")) or (method==string("gemm"))){
	gemmtensordot(A,B,labela,labelb,C);
      }else{
	tcltensordot(A,B,labela,labelb,C);
      }
    }else{
      tcltensordot(A,B,labela,labelb,C);
    }
    return C;
  }
  template<typename T>
  Tensor<T> tensordot(Tensor<T> &A, Tensor<T>&&B,const LabelType& labela, const LabelType& labelb){
    const char* type=std::getenv("TDOTMETHOD");
    Tensor<T> C;    
    if (type!=NULL){
      std::string method(type);
      if((method==string("TBLIS")) or (method==string("tblis"))){
	tblistensordot(A,B,labela,labelb,C);
      }else if((method==string("GEMM")) or (method==string("gemm"))){
	gemmtensordot(A,B,labela,labelb,C);
      }else{
	tcltensordot(A,B,labela,labelb,C);
      }
    }else{
      tcltensordot(A,B,labela,labelb,C);
    }
    return C;
  }
  template<typename T>
  Tensor<T> tensordot(Tensor<T> &&A, Tensor<T>&&B,const LabelType& labela, const LabelType& labelb){
    const char* type=std::getenv("TDOTMETHOD");
    Tensor<T> C;    
    if (type!=NULL){
      std::string method(type);
      if((method==string("TBLIS")) or (method==string("tblis"))){
	tblistensordot(A,B,labela,labelb,C);
      }else if((method==string("GEMM")) or (method==string("gemm"))){
	gemmtensordot(A,B,labela,labelb,C);
      }else{
	tcltensordot(A,B,labela,labelb,C);
      }
    }else{
      tcltensordot(A,B,labela,labelb,C);
    }
    return C;
  }


  /*
    computes A.dot(B), where A is a matrix and B is a vector,
    char='N' computes A*B, char ='T' computes A.transpose()*B, and char ='C' computes A.transpose().conjugate()*B
  */
  template<typename T>
  void gemm_mv_dot(const Tensor<T>&A,const Tensor<T>&B,Tensor<T> &C,char TR){
    if(A.rank()!=2)
      throw RankMismatchError("in gemm_mv_dot: rank(A)!=2");
    if(B.rank()!=1)
      throw RankMismatchError("in gemm_mv_dot: rank(B)!=1");
    if (A.shape(1)!=B.shape(0))
      throw ShapeMismatchError("in gemm_mv_dot: matrix-vector dimensions don't match");      
    C.resize(ShapeType{A.shape(0)});
    mat_vec_prod(A.raw(),A.shape(0),A.shape(1),B.raw(),C.raw(),TR);
  }

  /*
    computes the dot product between two tensors A and B; A and B have to have the same shapes
  */
  
  template<typename T>
  Real gemm_vv_dot(const Tensor<T> &A,const Tensor<T>&B){
    if(A.shape()!=B.shape())
      throw ShapeMismatchError("in gemm_vv_dot: A and B have different shapes");
    T result=dot_prod(A.size(),A.raw(),B.raw());
    return abs(result);
  }

  /*
    computes the norm of the tensor A
  */
  
  template<typename T>
  Real norm(const Tensor<T> &A){
    return A.norm();
  }

  //returns the complex conjugate of A
  template<typename T>
  Tensor<T> conjugate(const Tensor<T> &A){
    Tensor<T> out=A;
    return out.conj();
  }

  //returns the complex conjugate of A
  template<typename T>
  Tensor<T> transpose(const Tensor<T> &A,const LabelType &label){
    Tensor<T> out=A;
    out.transpose(label);
    return out;
  }
  //returns the complex conjugate of A
  template<typename T>
  Tensor<T> herm(const Tensor<T> &A){
    return A.herm();
  }

  template<typename T>
  Tensor<T> reshape(const Tensor<T> &A,const ShapeType& shape){
    Tensor<T> out(A);
    out.reshape(shape);
    return out;
  }

  template<typename T,typename ...Shapes>
  Tensor<T> reshape(const Tensor<T> &A,const Shapes& ... shape){
    Tensor<T> out=A;
    out.reshape(shape...);
    return out;
  }
  
  template<typename T>
  Tensor<T> reshape(Tensor<T> &&A,const ShapeType& shape){
    Tensor<T> out=std::move(A);
    out.reshape(shape);
    return out;
  }
  

  template<typename T,typename ...Shapes>
  Tensor<T> reshape(Tensor<T> &&A,const Shapes& ... shape){
    Tensor<T> out=std::move(A);
    out.reshape(shape...);
    return out;
  }

  template<typename T>
  Tensor<T> reshape(const Tensor<T> &&A,const ShapeType& shape){
    Tensor<T> out=std::move(A);
    out.reshape(shape);
    return out;
  }

  /*
    returns the shape of Tensor A in a tuple of type int
    auto[x1,x2,....,xN]=shape<N>(A);
    x1,..,xN contain the dimensions of the legs of A
    N has to be passed as a constant expression, i.e. int N=A.rank() is NOT possible right now
  */

  template <std::size_t rank, typename T>
  auto shape(const Tensor<T>& A) {
    if (A.rank() != rank){
      throw OutOfBoundError("shape<rank>(Tensor<T> A): template parameter 'rank' does not match A.rank()");
    }
    vector<int> shape(A.shape().begin(),A.shape().end());
    return vectorToTupleHelper(shape, std::make_index_sequence<rank>());
  }
  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/
  Tensor<Complex> dot(Tensor<Complex>&A,Tensor<Complex>&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/
  Tensor<Complex> dot(Tensor<Complex>&&A,Tensor<Complex>&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Complex>&A,Tensor<Complex>&&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Complex>&&A,Tensor<Complex>&&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Real>&A,Tensor<Complex>&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,(Tensor<Complex>)A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot((Tensor<Complex>)A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }


  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Real>&&A,Tensor<Complex>&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,(Tensor<Complex>)A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot((Tensor<Complex>)A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Real>&A,Tensor<Complex>&&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,(Tensor<Complex>)A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot((Tensor<Complex>)A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Real>&&A,Tensor<Complex>&&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot((Tensor<Complex>)A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,(Tensor<Complex>)A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot((Tensor<Complex>)A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }


  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Complex>&A,Tensor<Real>&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot((Tensor<Complex>)B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,(Tensor<Complex>)B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  
  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Complex>&&A,Tensor<Real>&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot((Tensor<Complex>)B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,(Tensor<Complex>)B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  
  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Complex>&A,Tensor<Real>&&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot((Tensor<Complex>)B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,(Tensor<Complex>)B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }


  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Complex> dot(Tensor<Complex>&&A,Tensor<Real>&&B){
    Tensor<Complex> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,(Tensor<Complex>)B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot((Tensor<Complex>)B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,(Tensor<Complex>)B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Real> dot(Tensor<Real>&A,Tensor<Real>&B){
    Tensor<Real> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }


  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/
  Tensor<Real> dot(Tensor<Real>&A,Tensor<Real>&&B){
    Tensor<Real> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }
  

  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Real> dot(Tensor<Real>&&A,Tensor<Real>&B){
    Tensor<Real> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  
  /*calculate the dot product between tensors and B; A and B have to be vectors or matrices of matching dimensions*/  
  Tensor<Real> dot(Tensor<Real>&&A,Tensor<Real>&&B){
    Tensor<Real> C;
    if((A.rank()==1) and (B.rank()==1)){
      gemmtensordot(A,B,LabelType{1},LabelType{1},C);      
    }else if((A.rank()==2) and (B.rank()==2)){
      gemmtensordot(A,B,LabelType{-1,1},LabelType{1,-2},C);
    }else if((A.rank()==1) and (B.rank()==2)){
      gemm_mv_dot(B,A,C,'T');      
    }else if((A.rank()==2) and (B.rank()==1)){
      gemm_mv_dot(A,B,C);
    }else{
      throw std::runtime_error("in dot-function:: tensors cannot be dotted due to wrong shapes");
    }
    return C;
  }

  
}
#endif 
