#include"lapackroutines.hpp"
#include <assert.h>
#include <cstdio>
#include <algorithm>
using std::cout;
using std::endl;
using std::min;

namespace lapackroutines{

  
  extern "C"
  {
    /* see http://www.netlib.org/lapack/lapack-3.1.1/html/dgesvd.f.html for more details on the arguments */
    void dgesvd_(char* JOBU,char* JOBVT,int* M,int*  N, double* A,int* LDA,double* S,double* U,int* LDU,double* VT,int* LDVT,double* WORK,int*  LWORK,int*  INFO );
    /*see http://www.netlib.org/lapack/lapack-3.1.1/html/dgesdd.f.html for a detailed description of arguments  
      For real matrices, dgesdd is much faster than dgesvd; numpy for example also uses dgesdd and zgesdd for their svd;*/
    void dgesdd_(char* JOBZ,int* M,int*  N, double* A,int* LDA,double* S,double* U,int* LDU,double* VT,int* LDVT,double* WORK,int*  LWORK,int*IWORK,int*  INFO );
    /*see http://www.netlib.org/lapack/explore-3.1.1-html/zgesdd.f.html for a detailed desription of arguments*/
    void zgesdd_( char* JOBZ, int* M, int *N, Complex *A, int *LDA, double * S, Complex * U, int* LDU, Complex *VT, int *LDVT, Complex *WORK, int *LWORK, double *RWORK,int* IWORK,int * INFO );
    /*see http://www.netlib.org/lapack/explore-3.1.1-html/zgesvd.f.html for more details on the arguments*/
    void zgesvd_( char* JOBU, char* JOBVT, int* M, int *N, double *A, int *LDA, double * S, double * U, int* LDU, double *VT, int *LDVT, double *WORK, int *LWORK, double *RWORK,int * INFO );
    
    /*see http://www.netlib.org/lapack/explore-3.1.1-html/dgeev.f.html for more details on the arguments of the routine */
    void dgeev_( char *JOBVL, char *JOBVR, int* N, double * A, int* LDA, double* WR, double *WI, double *VL, int* LDVL, double *VR,int* LDVR,double * WORK, int* LWORK, int* INFO );
    /*see   http://www.netlib.org/lapack/explore-3.1.1-html/zgeev.f.html for more details on the arguments of the routine */    
    void zgeev_( char *JOBVL, char *JOBVR, int* N, Complex * A, int* LDA, Complex* WR, Complex *VL, int* LDVL, Complex *VR,int* LDVR,Complex * WORK, int* LWORK, double* RWORK, int* INFO );

    /* see http://www.netlib.org/lapack/lapack-3.1.1/html/zheevd.f.html*/
    void dsyevd_( char *JOBZ, char *UPLO, int* N, Real * A, int* LDA, double *W, Real * WORK, \
		  int* LWORK, int* IWORK, int*LIWORK, int* INFO );
    void zheevd_( char *JOBZ, char *UPLO, int* N, Complex * A, int* LDA, double *W, Complex * WORK, \
		  int* LWORK, double* RWORK, int* LRWORK, int* IWORK, int*LIWORK, int* INFO );
    void dgeqrf_(int* M, int* N, double* A, int* LDA, double* TAU, double* WORK, int* LWORK, int* INFO )    ;
    void dorgqr_(int* M, int* N, int* K, double* A, int* LDA, double* TAU, double* WORK, int* LWORK, int* INFO );
    void zgeqrf_(int* M, int* N, Complex* A, int* LDA, Complex* TAU, Complex* WORK, int* LWORK, int* INFO )    ;
    void zungqr_(int* M, int* N, int* K, Complex* A, int* LDA, Complex* TAU, Complex* WORK, int* LWORK, int*INFO);
  }

  //A: on input, its the M by N matrix whose QR decomposition should be computed
  //Q: an M by M Complex array; on exit contains Q
  //R: an M by N Complex array; on exit contains R

  int qr(Complex* A,int M, int N, Complex* Q,Complex* R){
    int lda=M;
    int info,lwork=-1;

    Complex* tau=new Complex[std::min(M,N)];    
    Complex* tmpwork=new Complex[1];
    zgeqrf_(&M,&N,A,&lda,tau,tmpwork,&lwork,&info);
    lwork=static_cast<int>(tmpwork[0].real());
    Complex* work=new Complex[lwork];
    zgeqrf_(&M,&N,A,&lda,tau,work,&lwork,&info);
    delete [] work;
    uint cnt=0;
    for(uint n=0;n<N;n++){
      for(uint m=0;m<M;m++){	
	if(n>=m){
	  R[cnt]=A[cnt];
	}else{
	  Q[cnt]=A[cnt];
	  R[cnt]=Complex(0.0);
	}
	cnt++;
      }
    }
    //fill the rest of Q with 0s
    for(uint n=N;n<M;n++){
      for(uint m=0;m<M;m++){	
	Q[cnt]=Complex(0.0);	  
	cnt++;
      }
    }

    int k=min(M,N);
    lwork=-1;

    zungqr_(&M, &M, &k, Q, &lda, tau,tmpwork, &lwork,&info);
    lwork=static_cast<int>(tmpwork[0].real());    
    delete[] tmpwork;
    work=new Complex[lwork];
    zungqr_(&M, &M, &k, Q, &lda, tau,work, &lwork,&info);

    delete [] work;
    delete [] tau ;   
  }

      
  //A: on input, its the M by N matrix whose QR decomposition should be computed
  //Q: an M by M double array; on exit contains Q
  //R: an M by N double array; on exit contains R
  int qr(Real* A,int M, int N, Real* Q,Real* R){
    int lda=M;
    int info,lwork=-1;

    double* tau=new double[std::min(M,N)];    
    double* tmpwork=new double[1];
    dgeqrf_(&M,&N,A,&lda,tau,tmpwork,&lwork,&info);
    lwork=static_cast<int>(tmpwork[0]);
    double* work=new double[lwork];
    dgeqrf_(&M,&N,A,&lda,tau,work,&lwork,&info);
    delete [] work;
    uint cnt=0;
    for(uint n=0;n<N;n++){
      for(uint m=0;m<M;m++){	
	if(n>=m){
	  R[cnt]=A[cnt];
	}else{
	  Q[cnt]=A[cnt];
	  R[cnt]=0.0;
	}
	cnt++;
      }
    }
    //fill the rest of Q with 0s
    for(uint n=N;n<M;n++){
      for(uint m=0;m<M;m++){	
	Q[cnt]=0.0;	  
	cnt++;
      }
    }

    int k=min(M,N);
    lwork=-1;

    dorgqr_(&M, &M, &k, Q, &lda, tau,tmpwork, &lwork,&info);
    lwork=static_cast<int>(tmpwork[0]);    
    delete[] tmpwork;
    work=new double[lwork];
    dorgqr_(&M, &M, &k, Q, &lda, tau,work, &lwork,&info);

    delete [] work;
    delete [] tau ;   
  }

  /*
    computes the left and right eigenvectors and eigenvalues of a real matrix "mat". "mat" has to be square.
    N: dimension of "mat"
    Complex* VL: upon finishing, VL will contain the left eigenvectors
    Complex* EV: upon finishing, EV will contain the eigenvalues;
    Complex* VR: upon finishing, VR will contain the right eigenvectors
    JOBVL='V' (default): left eigenvectors are computed; 'N': left eigenvectors are not computed
    JOBVR='V' (default): right eigenvectors are computed; 'N': right eigenvectors are not computed
  */
  void eig(Real* mat, size_type  N,Complex* VL,Complex* EV,Complex* VR,char JOBVL,char JOBVR){
    int size=(int)N;
    int lda=(int)N;
    int ldvr=(int)N;
    int ldvl=(int)N;
    int lwork=-1, info;
    double *EV_r=new double[N];
    double *EV_c=new double[N];
    double *VL_=new double[N*N];
    double *VR_=new double[N*N];    
    double *temp=new double[1];    
    dgeev_(&JOBVL, &JOBVR,&size, mat, &lda, EV_r, EV_c, VL_, &ldvl, VR_,&ldvr,temp,&lwork,&info);
    lwork=static_cast<int>(temp[0]);
    delete[] temp;
    double *WORK=new double[lwork];        
    dgeev_(&JOBVL, &JOBVR,&size, mat, &lda, EV_r, EV_c, VL_, &ldvl, VR_,&ldvr,WORK,&lwork,&info);
    delete[] WORK;    
    luint j=0;
    while(j<N){
      if (std::abs(EV_c[j])<1E-14){
    	EV[j]=Complex(EV_r[j],0.0);
    	if (JOBVL=='V'){

    	  for (luint k=0;k<N;k++){
    	    VL[j*N+k]=Complex(VL_[j*N+k],0.0);
    	    VR[j*N+k]=Complex(VR_[j*N+k],0.0);	    
    	  }
    	}
    	j++;
      }else{
    	EV[j]=Complex(EV_r[j],EV_c[j]);
    	EV[j+1]=Complex(EV_r[j+1],EV_c[j+1]);
    	if (JOBVR=='V'){
    	  for (luint k=0;k<N;k++){
    	    VL[j*N+k]=Complex(VL_[j*N+k],VL_[(j+1)*N+k]);
    	    VL[(j+1)*N+k]=Complex(VL_[j*N+k],-VL_[(j+1)*N+k]);
	    
    	    VR[j*N+k]=Complex(VR_[j*N+k],VR_[(j+1)*N+k]);
    	    VR[(j+1)*N+k]=Complex(VR_[j*N+k],-VR_[(j+1)*N+k]);	  
	    
    	  }
    	}
    	j+=2;
      }
    }
    delete[] VR_;
    delete[] VL_;
    delete[] EV_c;
    delete[] EV_r;    
  }
  /*
    computes the left and right eigenvectors and eigenvalues of a complex matrix "mat". "mat" has to be square.
    N: dimension of "mat"
    Complex* VL: upon finishing, VL will contain the left eigenvectors
    Complex* EV: upon finishing, EV will contain the eigenvalues;
    Complex* VR: upon finishing, VR will contain the right eigenvectors
    JOBVL='V' (default): left eigenvectors are computed; 'N': left eigenvectors are not computed
    JOBVR='V' (default): right eigenvectors are computed; 'N': right eigenvectors are not computed
  */

  void eig(Complex* mat, size_type  N,Complex* VL,Complex* EV,Complex* VR,char JOBVL,char JOBVR){
    int size=(int)N;
    int lda=(int)N;
    int ldvr=(int)N;
    int ldvl=(int)N;
    int lwork=-1, info;
    double *rwork=new double[2*N];

    //compute optimal lwork
    Complex *temp=new Complex[1];    
    zgeev_(&JOBVL, &JOBVR,&size, mat, &lda, EV,VL, &ldvl,VR,&ldvr,temp,&lwork,rwork,&info);
    lwork=static_cast<int>(temp[0].real());
    delete[] temp;
    
    Complex *work=new Complex[lwork];        
    zgeev_(&JOBVL, &JOBVR,&size, mat, &lda, EV,VL, &ldvl,VR,&ldvr,work,&lwork,rwork,&info);    
    delete[] work;
  }

  /*
    computes the eigenvectors and eigenvalues of a complex hermitian matrix "mat". "mat" has to be square.
    N: dimension of "mat"
    Complex* EV: upon finishing, EV will contain the eigenvalues;
    JOBZ='V' (default): eigenvectors and eigenvalues are computed; 'N': only eigenvalues are computed
    note that mat is not preserved
  */
  
  void eigh(Real* mat, size_type  N,Real* EV,char JOBZ){
    char UPLO='U';
    int size=(int)N;
    int lda=(int)N;
    int lwork=-1,liwork=-1,info;
    Real *tmpwork=new Real[1];    
    int *tmpiwork=new int[1];
    //compute optimal lwork
    dsyevd_(&JOBZ, &UPLO,&size, mat, &lda, EV,tmpwork,&lwork,tmpiwork,&liwork,&info);
    lwork=static_cast<int>(tmpwork[0]);
    liwork=tmpiwork[0];
    delete[] tmpiwork;        
    delete[] tmpwork;

    Real *work=new Real[lwork];    
    int *iwork=new int[liwork];
    dsyevd_(&JOBZ, &UPLO,&size, mat, &lda, EV,work,&lwork,iwork,&liwork,&info);    
    delete[] work;
    delete[] iwork;        
  }
  
  void eigh(Complex* mat, size_type  N,Real* EV,char JOBZ){
    char UPLO='U';
    int size=(int)N;
    int lda=(int)N;
    int lwork=-1,lrwork=-1,liwork=-1,info;
    Complex *tmpwork=new Complex[1];    
    double *tmprwork=new double[1];
    int *tmpiwork=new int[1];
    //compute optimal lwork
    zheevd_(&JOBZ, &UPLO,&size, mat, &lda, EV,tmpwork,&lwork,tmprwork,&lrwork,tmpiwork,&liwork,&info);
    lwork=static_cast<int>(tmpwork[0].real());
    lrwork=static_cast<int>(tmprwork[0]);
    liwork=tmpiwork[0];
    delete[] tmpiwork;        
    delete[] tmprwork;
    delete[] tmpwork;



    Complex *work=new Complex[lwork];    
    double *rwork=new double[lrwork];
    int *iwork=new int[liwork];
    zheevd_(&JOBZ, &UPLO,&size, mat, &lda, EV,work,&lwork,rwork,&lrwork,iwork,&liwork,&info);
    delete[] work;
    delete[] rwork;
    delete[] iwork;        
  }

  

  /* 
     computes the complex SVD of "matrix"; U,VT and D have to be allocated outside the routine, and should help to ensure that the matrix 
     objects owning the data are consistent with the memory size
     dim1, dim2 are first and second dimension of "matrix", 
     char JOBU='S' or 'A' for economic and full svd; 
     string WHICH="_GESDD" or "_GESVD"; defaults to "_DGESDD", which is the faster method for real matrices
   */

  
  int svd(Complex *matrix, Complex *U,Complex*VT,Real *D, size_type dim1,size_type  dim2,char JOBU,string WHICH)
  {

    char JOBVT = JOBU;
    if ((JOBU!='S') and  (JOBU!='A')){
      throw std::runtime_error("lapack::svd; invalid value fo JOBU; use 'A' or 'S'");
    }
  
    int lda = (int)dim1,ldu = (int)dim1;
    int ldvt = (int)(dim1<=dim2?dim1:dim2);
    int info;
    int lwork = -1;
    int d1=(int)dim1;
    int d2=(int)dim2;
    

    if(WHICH==string("_GESVD")){
      //double *Dreal = new double[min(dim1,dim2)];            
      double*mat=reinterpret_cast <double *>(matrix);
      double* umat=reinterpret_cast <double *>(U);
      double* vtmatt=reinterpret_cast <double *>(VT);  

      double *rwork = new double [5 * (dim1<=dim2?dim1:dim2)];
      //compute optimal lwork      
      double *temp = new double[1];
      zgesvd_(&JOBU,&JOBVT, &d1, &d2, mat, &lda, D,umat, &ldu, vtmatt, &ldvt, temp, &lwork, rwork,&info );
      lwork =static_cast<int>(temp[0]);
      delete[] temp;
      double *work = new double[lwork];
      zgesvd_( &JOBU,&JOBVT, &d1, &d2,mat,&lda,D,umat, &ldu, vtmatt, &ldvt, work, &lwork, rwork,&info );
      delete[] work;
      delete[] rwork;
      // for(uint n=0;n<min(dim1,dim2);n++){
      // 	D[n]=Complex(Dreal[n],0.0);
      // }
      // delete[] Dreal;
    }

    if(WHICH==string("_GESDD")){
      //double *Dreal = new double[min(dim1,dim2)];                  
      int lrwork=5*min(dim1,dim2)*min(dim1,dim2)+7*min(dim1,dim2);
      double *rwork = new double [lrwork];
      int *iwork = new int [8*min(dim1,dim2)];

      //compute optimal lwork
      Complex *temp = new Complex[1];
      zgesdd_(&JOBU, &d1, &d2, matrix, &lda, D,U,&ldu,VT, &ldvt, temp, &lwork, rwork,iwork,&info);
      lwork =static_cast<int>(temp[0].real());
      delete[] temp;
      Complex *work = new Complex[lwork];
      zgesdd_(&JOBU, &d1, &d2, matrix, &lda, D,U,&ldu,VT, &ldvt, work, &lwork, rwork,iwork,&info);
      delete[] work;
      delete[] iwork;
      delete[] rwork;
      // for(uint n=0;n<min(dim1,dim2);n++){
      // 	D[n]=Complex(Dreal[n],0.0);
      // }
      // delete[] Dreal;
    }
    return info;
  }
  /* 
     computes the real SVD of "matrix"; U,VT and D have to be allocated outside the routine, and should help to ensure that the matrix 
     objects owning the data are consistent with the memory size
     dim1, dim2 are first and second dimension of "matrix", 
     char JOBU='S' or 'A' for economic and full svd; 
     string WHICH="_GESDD" or "_GESVD"; defaults to "_DGESDD", which is the faster method for real matrices
  */
  int svd(double *matrix, double *U,double*VT,double *sing_val, size_type dim1,size_type dim2,char JOBU,string WHICH)
  {
    //char JOBU = 'S';
    char JOBVT = JOBU;
    if ((JOBU!='S') and  (JOBU!='A')){
      throw std::runtime_error("lapack::rsvd; invalid value fo JOBU; use 'A' or 'S'");
    }
    int lda = (int)dim1;
    int ldvt = (int)(dim1>dim2?dim2:dim1);
    int info;
    int lwork = -1;
    int d1=(int)dim1;
    int d2=(int)dim2;

    if (WHICH==string("_GESDD")){
      int *iwork=new int[min(d1,d2)*8];      
      double *temp = new double[1];      
      dgesdd_( &JOBU,&d1, &d2, matrix, &lda, sing_val,U, &lda, VT, &ldvt, temp, &lwork,iwork,&info );
      lwork =static_cast<int>(temp[0]);
      delete[] temp;
      double *work = new double[lwork];
      dgesdd_( &JOBU,&d1, &d2, matrix, &lda, sing_val,U, &lda, VT, &ldvt,work,&lwork,iwork,&info);    
      delete[] work;
      delete[] iwork;
    }else if (WHICH==string("_GESVD")){
      double *temp = new double[1];      
      dgesvd_( &JOBU,&JOBVT,&d1, &d2, matrix, &lda, sing_val,U, &lda, VT, &ldvt, temp, &lwork,&info );
      lwork =static_cast<int>(temp[0]);
      delete[] temp;
      double *work = new double[lwork];
      dgesvd_( &JOBU,&JOBVT,&d1, &d2, matrix, &lda, sing_val,U, &lda, VT, &ldvt,work,&lwork,&info);
      delete[] work;
    }      
    return info;
  }
} 

