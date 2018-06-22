#include "typedefs.hpp"
#include <iostream>
#include"lib/array/tensor.hpp"
#include"lib/utils/utilities.hpp"
#include"lib/utils/print.hpp"
#include"lib/linalg/lapackroutines.hpp"
#include"lib/array/tensoroperations.hpp"
#include <chrono>

using namespace std;
using namespace tensor;
using namespace printfunctions;
using  std::cout;
using  std::endl;
using  std::vector;
int main(int argc, char** argv){

  //test double types
  int N=10;
  bool OK=true;  
  {
    Tensor<double>mat1(N,N),mat2(N,N),vector(N),matmat(N,N),matvec(N),vecmat(N);
    mat1.randomize(-1,1);vector.randomize(-1,1);mat2.randomize(-1,1);
    matmat.reset(0);matvec.reset(0);
    for (int n=0;n<N;n++){
      for (int m=0;m<N;m++){
	matvec(n)+=mat1(n,m)*vector(m);
	vecmat(n)+=mat1(m,n)*vector(m);
	for(int l=0;l<N;l++){
	  matmat(n,m)+=mat1(n,l)*mat2(l,m);
	}
      }
    }

    auto t1=mat1.dot(mat2);
    if ((t1-matmat).norm()>1E-14){
      OK=false;
      cout<<"Tensor<Real>.dot(Tensor<Real>) gave a wrong result"<<endl;
    }
    t1=dot(mat1,mat2);
    if ((t1-matmat).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Real>,Tensor<Real>) gave a wrong result"<<endl;
    }
    t1=mat1.dot(vector);
    if ((t1-matvec).norm()>1E-14){
      OK=false;
      cout<<"Tensor<Real>.dot(Vector<Real>) gave a wrong result"<<endl;
    }
    t1=dot(mat1,vector);
    if ((t1-matvec).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Real>,Vector<Real>) gave a wrong result"<<endl;
    }

    t1=vector.dot(mat1);
    if ((t1-vecmat).norm()>1E-14){
      OK=false;
      cout<<"Vector<Real>.dot(Tensor<Real>) gave a wrong result"<<endl;
    }
    t1=dot(vector,mat1);
    if ((t1-vecmat).norm()>1E-14){
      OK=false;
      cout<<"dot(Vector<Real>,Tensor<Real>) gave a wrong result"<<endl;
    }
  }

  {
    Tensor<Complex>mat1(N,N),mat2(N,N),vector(N),matmat(N,N),matvec(N),vecmat(N);
    mat1.randomize(-1,1);vector.randomize(-1,1);mat2.randomize(-1,1);
    matmat.reset(0);matvec.reset(0);
    for (int n=0;n<N;n++){
      for (int m=0;m<N;m++){
	matvec(n)+=mat1(n,m)*vector(m);
	vecmat(n)+=mat1(m,n)*vector(m);
	for(int l=0;l<N;l++){
	  matmat(n,m)+=mat1(n,l)*mat2(l,m);
	}
      }
    }

    auto t1=mat1.dot(mat2);
    if ((t1-matmat).norm()>1E-14){
      OK=false;
      cout<<"Tensor<Complex>.dot(Tensor<Complex>) gave a wrong result"<<endl;
    }
    t1=dot(mat1,mat2);
    if ((t1-matmat).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Complex>,Tensor<Complex>) gave a wrong result"<<endl;
    }
    t1=mat1.dot(vector);
    if ((t1-matvec).norm()>1E-14){
      OK=false;
      cout<<"Tensor<Complex>.dot(Vector<Complex>) gave a wrong result"<<endl;
    }
    t1=dot(mat1,vector);
    if ((t1-matvec).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Complex>,Vector<Complex>) gave a wrong result"<<endl;
    }

    t1=vector.dot(mat1);
    if ((t1-vecmat).norm()>1E-14){
      OK=false;
      cout<<"Vector<Complex>.dot(Tensor<Complex>) gave a wrong result"<<endl;
    }
    t1=dot(vector,mat1);
    if ((t1-vecmat).norm()>1E-14){
      OK=false;
      cout<<"dot(Vector<Complex>,Tensor<Complex>) gave a wrong result"<<endl;
    }

  }


  {
    Tensor<Complex>mat1(N,N),matvec(N),vecmat(N),matmat(N,N),matmat2(N,N);
    Tensor<Real> mat2(N,N),vector(N);
    mat1.randomize(Complex(-1,-1),Complex(1,1));vector.randomize(-1,1);mat2.randomize(-1,1);
    matmat.reset(0);matvec.reset(0),matmat2.reset(0);
    for (int n=0;n<N;n++){
      for (int m=0;m<N;m++){
	matvec(n)+=mat1(n,m)*vector(m);
	vecmat(n)+=mat1(m,n)*vector(m);
	for(int l=0;l<N;l++){
	  matmat(n,m)+=mat1(n,l)*mat2(l,m);
	  matmat2(n,m)+=mat2(n,l)*mat1(l,m);	  
	}
      }
    }
  
    auto t1=mat1.dot(mat2);
    if ((t1-matmat).norm()>1E-14){
      OK=false;
      cout<<"Tensor<Complex>.dot(Tensor<Real>) gave a wrong result"<<endl;
    }

    t1=dot(mat1,mat2);
    if ((t1-matmat).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Complex>,Tensor<Real>) gave a wrong result"<<endl;
    }

    t1=dot(mat2,mat1);
    if ((t1-matmat2).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Complex>,Tensor<Real>) gave a wrong result"<<endl;
    }
    
    t1=mat1.dot(vector);
    if ((t1-matvec).norm()>1E-14){
      OK=false;
      cout<<"Tensor<Complex>.dot(Vector<Real>) gave a wrong result"<<endl;
    }
    t1=dot(mat1,vector);
    if ((t1-matvec).norm()>1E-14){
      OK=false;
      cout<<"dot(Tensor<Complex>,Vector<Real>) gave a wrong result"<<endl;
    }
    t1=dot(vector,mat1);
    if ((t1-vecmat).norm()>1E-14){
      OK=false;
      cout<<"dot(Vector<Real>,Tensor<Complex>) gave a wrong result"<<endl;
    }

  }

  
  if (OK==true)
    cout<<"passed all tests"<<endl;

  
  return 0;
}
