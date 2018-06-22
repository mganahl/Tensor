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
int main(int argc, char** argv){
  int N=4;
  Tensor<Real >mat(N,N);
  auto matc=Tensor<Complex>(N,N);
  mat.randomize(-1,1);
  matc.randomize(Complex(-1,-1),Complex(1,1));
  bool OK=true;
  {
    cout<<"testing c svd: "<<endl;  
    auto [U,S,VH]=svd(matc);
    if(norm(U.dot(diag(S)).dot(VH)-matc)>1E-14){
      cout<<"there was a problem with complex svd"<<endl;
      OK=false;
    }
  }
  {
    cout<<"testing svd: "<<endl;  
    auto [U,S,VH]=svd(mat);
    if (norm(U.dot(diag(S)).dot(VH)-mat)>1E-14){
      cout<<"there was a problem with real svd"<<endl;
      OK=false;
    }
  }

  {
    int N1=4,N2=6;
    Tensor<Real >mat2(N1,N1);
    mat2.randomize(-1,1);    
    cout<<"testing qr: "<<endl;
    auto [Q,R]=qr(mat2);
    if ((eye<Real>(Q.shape(0))-Q.dot(Q.herm())).norm()>1E-14){
      cout<<"there was a problem with real qr: Q is not unitary"<<endl;
      OK=false;
    }
    if (norm(Q.dot(R)-mat2)>1E-14){
      cout<<"there was a problem with real qr"<<endl;
      OK=false;
    }
    mat2.resize(N1,N2);
    mat2.randomize(-1,1);
    auto o=qr(mat2);
    Q=o.first;
    R=o.second;
    if ((eye<Real>(Q.shape(0))-Q.dot(Q.herm())).norm()>1E-14){
      cout<<"there was a problem with real qr: Q is not unitary"<<endl;
      OK=false;
    }
    
    if (norm(Q.dot(R)-mat2)>1E-14){
      cout<<"there was a problem with real qr"<<endl;
      OK=false;
    }

    mat2.resize(N2,N1);
    mat2.randomize(-1,1);
    o=qr(mat2,true);
    Q=o.first;
    R=o.second;
    if ((eye<Real>(Q.shape(0))-Q.dot(Q.herm())).norm()>1E-14){
      cout<<"there was a problem with real qr: Q is not unitary"<<endl;
      OK=false;
    }
    if (norm(Q.dot(R)-mat2)>1E-14){
      cout<<"there was a problem with real qr"<<endl;
      OK=false;
    }
  }
  
  {
    int N1=4,N2=6;
    Tensor<Complex >mat2(N1,N1);
    mat2.randomize(-1,1);    
    cout<<"testing complex qr: "<<endl;
    auto [Q,R]=qr(mat2);
    if ((eye<Complex>(Q.shape(0))-Q.dot(Q.herm())).norm()>1E-14){
      cout<<"there was a problem with complex qr: Q is not unitary"<<endl;
      OK=false;
    }
    if (norm(Q.dot(R)-mat2)>1E-14){
      cout<<"there was a problem with complex qr"<<endl;
      OK=false;
    }
    mat2.resize(N1,N2);
    mat2.randomize(-1,1);
    auto o=qr(mat2);
    Q=o.first;
    R=o.second;
    if ((eye<Real>(Q.shape(0))-Q.dot(Q.herm())).norm()>1E-14){
      cout<<"there was a problem with complex qr: Q is not unitary"<<endl;
      OK=false;
    }
    
    if (norm(Q.dot(R)-mat2)>1E-14){
      cout<<"there was a problem with complex qr"<<endl;
      OK=false;
    }

    mat2.resize(N2,N1);
    mat2.randomize(-1,1);
    o=qr(mat2,true);
    Q=o.first;
    R=o.second;
    if ((eye<Real>(Q.shape(0))-Q.dot(Q.herm())).norm()>1E-14){
      cout<<"there was a problem with real qr: Q is not unitary"<<endl;
      OK=false;
    }
    if (norm(Q.dot(R)-mat2)>1E-14){
      cout<<"there was a problem with real qr"<<endl;
      OK=false;
    }
  }
  
  cout<<"testing eig:"<<endl;
  {
    auto matctemp=toComplex(mat);
    auto [VL,EV,VR]=eig(mat);
    
    auto VLH=VL.transp().conj();
    for (uint n =0;n<N;n++){
      auto g1=matctemp.dot(VR.slice(Slice(),Slice(n,n,1)))-EV(n)*VR.slice(Slice(),Slice(n,n,1));
      auto g2=VLH.slice(Slice(n,n,1),Slice()).dot(matctemp)-EV(n)*VLH.slice(Slice(n,n,1),Slice());
      if (g1.norm()>1E-14){
	cout<<"there was a problem with right eigenvector of real eig"<<endl;
	OK=false;	
      }
      if (g2.norm()>1E-14){
	cout<<"there was a problem with left eigenvector of real eig"<<endl;
	OK=false;	
      }
    }
    
    eig(mat,VL,EV,VR);
    VLH=VL.transp().conj();
    for (uint n =0;n<N;n++){
      auto g1=matctemp.dot(VR.slice(Slice(),Slice(n,n,1)))-EV(n)*VR.slice(Slice(),Slice(n,n,1));
      auto g2=VLH.slice(Slice(n,n,1),Slice()).dot(matctemp)-EV(n)*VLH.slice(Slice(n,n,1),Slice());
      if (g1.norm()>1E-14){
	cout<<"there was a problem with right eigenvector of real eig(A,B,C,D)"<<endl;
	OK=false;	
      }
      if (g2.norm()>1E-14){
	cout<<"there was a problem with left eigenvector of real eig(A,B,C,D)"<<endl;
	OK=false;
      }
    }
  }


  cout<<"testing eig c:"<<endl;
  matc.randomize();
  {
    auto [VL,EV,VR]=eig(matc);
    auto VLH=VL.transp().conj();
    for (uint n =0;n<N;n++){
      auto g1=matc.dot(VR.slice(Slice(),Slice(n,n,1)))-EV(n)*VR.slice(Slice(),Slice(n,n,1));
      auto g2=VLH.slice(Slice(n,n,1),Slice()).dot(matc)-EV(n)*VLH.slice(Slice(n,n,1),Slice());
      if (g1.norm()>1E-14){
	cout<<"there was a problem with right eigenvector of complex eig"<<endl;
	OK=false;	
      }
      if (g2.norm()>1E-14){
	cout<<"there was a problem with left eigenvector of complex eig"<<endl;
	OK=false;
      }

    }
    eig(matc,VL,EV,VR);
    VLH=VL.transp().conj();
    for (uint n =0;n<N;n++){
      auto g1=matc.dot(VR.slice(Slice(),Slice(n,n,1)))-EV(n)*VR.slice(Slice(),Slice(n,n,1));
      auto g2=VLH.slice(Slice(n,n,1),Slice()).dot(matc)-EV(n)*VLH.slice(Slice(n,n,1),Slice());
      if (g1.norm()>1E-14){
	cout<<"there was a problem with right eigenvector of complex eig(A,B,C,D)"<<endl;
	OK=false;	
      }
      if (g2.norm()>1E-14){
	cout<<"there was a problem with left eigenvector of complex eig(A,B,C,D)"<<endl;
	OK=false;
      }
    }
  }

  cout<<"testing eigh :"<<endl;
  mat.randomize(-1,1);
  auto matH=(mat+mat.herm())/2.0;
  {
    auto [EV,V]=eigh(matH);
    if ((V.dot(diag(EV)).dot(V.herm())-matH).norm()>1E-14){
      cout<<"there was a problem with real eigh"<<endl;
      OK=false;      
    }
    Tensor<Real >EVR;
    
    eigh(matH,EV,V);
    eigh(matH,EV,V);
    if ((V.dot(diag(EV)).dot(V.herm())-matH).norm()>1E-14){
      cout<<"there was a problem with real eigh(a,b,c)"<<endl;
      OK=false;      
    }

  }
  cout<<"testing eigh c:"<<endl;
  matc.randomize(Complex(-1,-1),Complex(1,1));
  auto matcH=(matc+matc.herm())/2.0;
  {
    auto [EV,V]=eigh(matcH);
    if ((V.dot(diag(EV)).dot(V.herm())-matcH).norm()>1E-14){
      cout<<"there was a problem with complex eigh"<<endl;
      OK=false;      
    }
    Tensor<Real >EVR;
    eigh(matcH,EVR,V);
    EV=toComplex(EVR);
    if ((V.dot(diag(EVR)).dot(V.herm())-matcH).norm()>1E-14){
      cout<<"there was a problem with complex eigh(a,b,c)"<<endl;
      OK=false;      
    }

  }
  cout<<"all tests OK"<<endl;
  return 0;
}
