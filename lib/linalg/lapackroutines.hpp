#ifndef LAPACKROUTINES_H
#define LAPACKROUTINES_H
#include <iostream>
#include <string>
#include "typedefs.hpp"
using std::string;
namespace lapackroutines{
  //int tridiag(const double *diag,int length_diag,const double* sub_diag,double*eigvec,char JOBZ);
  //int tridiag(const Complex *diag,int length_diag,const  Complex* sub_diag,double*eigvec,char JOBZ);

  int qr(Real* A,int M, int N, Real* Q,Real* R);
  int qr(Complex* A,int M, int N, Complex* Q,Complex* R);  
  
  /* 
     computes the complex SVD of "matrix"; U,VT and D have to be allocated outside the routine, and should help to ensure that the matrix 
     objects owning the data are consistent with the memory size
     dim1, dim2 are first and second dimension of "matrix", 
     char JOBU='S' or 'A' for economic and full svd; 
     string WHICH="_GESDD" or "_GESVD"; defaults to "_DGESDD", which is the faster method for real matrices
     Note that matrix is not preserved upon finishing of the routine
   */

  int svd(Complex *matrix, Complex *U,Complex*VT,Real *sing_val, size_type dim1,size_type dim2,char JOBU='S',string WHICH="_GESDD");
  /* 
     computes the real SVD of "matrix"; U,VT and D have to be allocated outside the routine, and should help to ensure that the matrix 
     objects owning the data are consistent with the memory size
     dim1, dim2 are first and second dimension of "matrix", 
     char JOBU='S' or 'A' for economic and full svd; 
     string WHICH="_GESDD" or "_GESVD"; defaults to "_DGESDD", which is the faster method for real matrices
     Note that matrix is not preserved upon finishing of the routine
  */
  int svd(double *matrix, double *U,double*VT,Real *sing_val, size_type dim1,size_type dim2,char JOBU='S',string WHICH="_GESDD");



  /*
    computes the left and right eigenvectors and eigenvalues of a real matrix "mat". "mat" has to be square.
    N: dimension of "mat"
    Complex* VL: upon finishing, VL will contain the left eigenvectors
    Complex* EV: upon finishing, EV will contain the eigenvalues;
    Complex* VR: upon finishing, VR will contain the right eigenvectors
    JOBVL='V' (default): left eigenvectors are computed; 'N': left eigenvectors are not computed
    JOBVR='V' (default): right eigenvectors are computed; 'N': right eigenvectors are not computed
  */
  void eig(Real* mat, size_type  N,Complex* VL,Complex* EV,Complex* VR,char JOBVL='V',char JOBVR='V');
  /*
    computes the left and right eigenvectors and eigenvalues of a complex matrix "mat". "mat" has to be square.
    N: dimension of "mat"
    Complex* VL: upon finishing, VL will contain the left eigenvectors
    Complex* EV: upon finishing, EV will contain the eigenvalues;
    Complex* VR: upon finishing, VR will contain the right eigenvectors
    JOBVL='V' (default): left eigenvectors are computed; 'N': left eigenvectors are not computed
    JOBVR='V' (default): right eigenvectors are computed; 'N': right eigenvectors are not computed
  */
  
  void eig(Complex* mat, size_type  N,Complex* VL,Complex* EV,Complex* VR,char JOBVL='V',char JOBVR='V');

  

  /*
    computes the eigenvectors and eigenvalues of a complex hermitian matrix "mat". "mat" has to be square.
    N: dimension of "mat"
    Complex* EV: upon finishing, EV will contain the eigenvalues;
    JOBZ='V' (default): eigenvectors and eigenvalues are computed; 'N': only eigenvalues are computed
    note that mat is not preserved
  */
  void eigh(Complex* mat, size_type  N,Real* EV,char JOBZ='V');  
  void eigh(Real* mat, size_type  N,Real* EV,char JOBZ='V');
  
}
#endif
