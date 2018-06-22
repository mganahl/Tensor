/*
 * blasroutines.cpp
 *
 *  Created on: 07.09.2010
 *      Author: martinjg
 */


#include "blasroutines.hpp"

namespace blasroutines{
  extern "C"
  {
    /*
      SUBROUTINE ZGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
      *     .. Scalar Arguments ..
      DOUBLE COMPLEX ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
      *     ..
      *     .. Array Arguments ..
      DOUBLE COMPLEX A(LDA,*),B(LDB,*),C(LDC,*)
      *     ..
      *
      *  Purpose
      *  =======
      *
      *  ZGEMM  performs one of the matrix-matrix operations
      *
      *     C := alpha*op( A )*op( B ) + beta*C,
      *
      *  where  op( X ) is one of
      *
      *     op( X ) = X   or   op( X ) = X'   or   op( X ) = conjg( X' ),
      *
      *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
      *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
      *
      *  Arguments
      *  ==========
      *
      *  TRANSA - CHARACTER*1.
      *           On entry, TRANSA specifies the form of op( A ) to be used in
      *           the matrix multiplication as follows:
      *
      *              TRANSA = 'N' or 'n',  op( A ) = A.
      *
      *              TRANSA = 'T' or 't',  op( A ) = A'.
      *
      *              TRANSA = 'C' or 'c',  op( A ) = conjg( A' ).
      *
      *           Unchanged on exit.
      *
      *  TRANSB - CHARACTER*1.
      *           On entry, TRANSB specifies the form of op( B ) to be used in
      *           the matrix multiplication as follows:
      *
      *              TRANSB = 'N' or 'n',  op( B ) = B.
      *
      *              TRANSB = 'T' or 't',  op( B ) = B'.
      *
      *              TRANSB = 'C' or 'c',  op( B ) = conjg( B' ).
      *
      *           Unchanged on exit.
      *
      *  M      - INTEGER.
      *           On entry,  M  specifies  the number  of rows  of the  matrix
      *           op( A )  and of the  matrix  C.  M  must  be at least  zero.
      *           Unchanged on exit.
      *
      *  N      - INTEGER.
      *           On entry,  N  specifies the number  of columns of the matrix
      *           op( B ) and the number of columns of the matrix C. N must be
      *           at least zero.
      *           Unchanged on exit.
      *
      *  K      - INTEGER.
      *           On entry,  K  specifies  the number of columns of the matrix
      *           op( A ) and the number of rows of the matrix op( B ). K must
      *           be at least  zero.
      *           Unchanged on exit.
      *
      *  ALPHA  - COMPLEX*16      .
      *           On entry, ALPHA specifies the scalar alpha.
      *           Unchanged on exit.
      *
      *  A      - COMPLEX*16       array of DIMENSION ( LDA, ka ), where ka is
      *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
      *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
      *           part of the array  A  must contain the matrix  A,  otherwise
      *           the leading  k by m  part of the array  A  must contain  the
      *           matrix A.
      *           Unchanged on exit.
      *
      *  LDA    - INTEGER.
      *           On entry, LDA specifies the first dimension of A as declared
      *           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
      *           LDA must be at least  max( 1, m ), otherwise  LDA must be at
      *           least  max( 1, k ).
      *           Unchanged on exit.
      *
      *  B      - COMPLEX*16       array of DIMENSION ( LDB, kb ), where kb is
      *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
      *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
      *           part of the array  B  must contain the matrix  B,  otherwise
      *           the leading  n by k  part of the array  B  must contain  the
      *           matrix B.
      *           Unchanged on exit.
      *
      *  LDB    - INTEGER.
      *           On entry, LDB specifies the first dimension of B as declared
      *           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
      *           LDB must be at least  max( 1, k ), otherwise  LDB must be at
      *           least  max( 1, n ).
      *           Unchanged on exit.
      *
      *  BETA   - COMPLEX*16      .
      *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
      *           supplied as zero then C need not be set on input.
      *           Unchanged on exit.
      *
      *  C      - COMPLEX*16       array of DIMENSION ( LDC, n ).
      *           Before entry, the leading  m by n  part of the array  C must
      *           contain the matrix  C,  except when  beta  is zero, in which
      *           case C need not be set on entry.
      *           On exit, the array  C  is overwritten by the  m by n  matrix
      *           ( alpha*op( A )*op( B ) + beta*C ).
      *
      *  LDC    - INTEGER.
      *           On entry, LDC specifies the first dimension of C as declared
      *           in  the  calling  (sub)  program.   LDC  must  be  at  least
      *           max( 1, m ).
      *           Unchanged on exit.
      */


    void zgemm_(char* TRANSA,char *TRANSB,int* M,int* N,int * K,Complex* ALPHA,const Complex *A,int* LDA,const Complex *B,int *LDB,Complex* BETA,Complex *C,int* LDC);

  }
  //no check if dimensions of matrices match!!!! mat_out has to be allocated before it is passed to cmat_mat_prod! note that you have to allocate space for
  //real and imaginary parts of the matrices
  void mat_mat_prod(Complex alpha,const Complex*matA ,size_type dim1A,size_type dim2A, const Complex* matB,size_type dim1B, size_type dim2B,Complex* mat_out,char tra,char trb)
  {
    char Transa = tra,Transb=trb;
    int M,N,Ka,Kb;
    int lda = (int)dim1A,ldb=(int)dim1B,ldc;
	
    if(tra=='n'||tra=='N')
      {
	M = (int)dim1A;
	Ka = (int)dim2A;
      }
    else
      {
	M = (int)dim2A;
	Ka = (int)dim1A;
      }
    if(trb =='n'||trb=='N') {
      N = (int)dim2B;
      Kb = (int)dim1B;
    }
    else{
      N = (int)dim1B;
      Kb = (int)dim2B;
    }
    ldc = M;
    assert(Ka==Kb);
	

    Complex beta=Complex(0.0,0.0);

    zgemm_(&Transa,&Transb,&M,&N,&Ka,&alpha,matA,&lda,matB,&ldb,&beta,mat_out,&ldc);

  }


  extern "C"
  {
    extern double ddot_(int*,const double*,int*,const double*,const int*);
  }

  double dot_prod(size_type size,const double* v1,const double*v2)
  {
    int inc1 = 1,inc2 = 1;
    int s=(int) size;
    double out= ddot_(&s,v1,&inc1,v2,&inc2);
    return out;
  }

  Complex dot_prod(size_type size,  const Complex* v1, const Complex*v2)
  {
    double *v1real = new double[size];
    double *v1imag = new double[size];
    double *v2real = new double[size];
    double *v2imag = new double[size];
    double out;
    for(int i=0;i<size;i++)
      {
	v1real[i] = v1[i].real();
	v1imag[i] = -v1[i].imag();
	v2real[i] = v2[i].real();
	v2imag[i] = v2[i].imag();		

      }
    out=dot_prod(size,v1real,v2real)+dot_prod(size,v1imag,v2imag);
    delete [] v1real;
    delete [] v1imag;
    delete [] v2real;
    delete [] v2imag;	

    return out;
  }



  extern "C"
  {
    /*
      SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
      *     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER INCX,INCY,LDA,M,N
      CHARACTER TRANS
      *     ..
      *     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
      *     ..
      *
      *  Purpose
      *  =======
      *
      *  DGEMV  performs one of the matrix-vector operations
      *
      *     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
      *
      *  where alpha and beta are scalars, x and y are vectors and A is an
      *  m by n matrix.
      *
      *  Arguments
      *  ==========
      *
      *  TRANS  - CHARACTER*1.
      *           On entry, TRANS specifies the operation to be performed as
      *           follows:
      *
      *              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
      *
      *              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
      *
      *              TRANS = 'C' or 'c'   y := alpha*A'*x + beta*y.
      *
      *           Unchanged on exit.
      *
      *  M      - INTEGER.
      *           On entry, M specifies the number of rows of the matrix A.
      *           M must be at least zero.
      *           Unchanged on exit.
      *
      *  N      - INTEGER.
      *           On entry, N specifies the number of columns of the matrix A.
      *           N must be at least zero.
      *           Unchanged on exit.
      *
      *  ALPHA  - DOUBLE PRECISION.
      *           On entry, ALPHA specifies the scalar alpha.
      *           Unchanged on exit.
      *
      *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
      *           Before entry, the leading m by n part of the array A must
      *           contain the matrix of coefficients.
      *           Unchanged on exit.
      *
      *  LDA    - INTEGER.
      *           On entry, LDA specifies the first dimension of A as declared
      *           in the calling (sub) program. LDA must be at least
      *           max( 1, m ).
      *           Unchanged on exit.
      *
      *  X      - DOUBLE PRECISION array of DIMENSION at least
      *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
      *           and at least
      *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
      *           Before entry, the incremented array X must contain the
      *           vector x.
      *           Unchanged on exit.
      *
      *  INCX   - INTEGER.
      *           On entry, INCX specifies the increment for the elements of
      *           X. INCX must not be zero.
      *           Unchanged on exit.
      *
      *  BETA   - DOUBLE PRECISION.
      *           On entry, BETA specifies the scalar beta. When BETA is
      *           supplied as zero then Y need not be set on input.
      *           Unchanged on exit.
      *
      *  Y      - DOUBLE PRECISION array of DIMENSION at least
      *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
      *           and at least
      *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
      *           Before entry with BETA non-zero, the incremented array Y
      *           must contain the vector y. On exit, Y is overwritten by the
      *           updated vector y.
      *
      *  INCY   - INTEGER.
      *           On entry, INCY specifies the increment for the elements of
      *           Y. INCY must not be zero.
      *           Unchanged on exit.
      *
      */

    void dgemv_(char* TRANS,int* M,int* N,double* ALPHA,const double *A,int* LDA,const double *X,int* INCX,double* BETA,double *Y,int* INCY);

  }

  //vec_out is not checked for dimension consistency!! dimension mismatch may cause unpredictable output!
  void mat_vec_prod(const double*mat,size_type M,size_type N,const double*vec_in,double*vec_out,char tr)
  {
    char Trans = tr;
    double alpha = 1.0;
    int lda = (int)M;
    int incx = 1;
    double beta = 0;
    //double y[M];
    int incy =1;
    int dM=(int)M;
    int dN=(int)N;

    dgemv_(&Trans,&dM,&dN,&alpha,mat,&lda,vec_in,&incx,&beta,vec_out,&incy);
  }




  extern "C"
  {/*
     SUBROUTINE ZGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
     *     .. Scalar Arguments ..
     DOUBLE COMPLEX ALPHA,BETA
     INTEGER INCX,INCY,LDA,M,N
     CHARACTER TRANS
     *     ..
     *     .. Array Arguments ..
     DOUBLE COMPLEX A(LDA,*),X(*),Y(*)
     *     ..
     *
     *  Purpose
     *  =======
     *
     *  ZGEMV  performs one of the matrix-vector operations
     *
     *     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,   or
     *
     *     y := alpha*conjg( A' )*x + beta*y,
     *
     *  where alpha and beta are scalars, x and y are vectors and A is an
     *  m by n matrix.
     *
     *  Arguments
     *  ==========
     *
     *  TRANS  - CHARACTER*1.
     *           On entry, TRANS specifies the operation to be performed as
     *           follows:
     *
     *              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
     *
     *              TRANS = 'T' or 't'   y := alpha*A'*x + beta*y.
     *
     *              TRANS = 'C' or 'c'   y := alpha*conjg( A' )*x + beta*y.
     *
     *           Unchanged on exit.
     *
     *  M      - INTEGER.
     *           On entry, M specifies the number of rows of the matrix A.
     *           M must be at least zero.
     *           Unchanged on exit.
     *
     *  N      - INTEGER.
     *           On entry, N specifies the number of columns of the matrix A.
     *           N must be at least zero.
     *           Unchanged on exit.
     *
     *  ALPHA  - COMPLEX*16      .
     *           On entry, ALPHA specifies the scalar alpha.
     *           Unchanged on exit.
     *
     *  A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
     *           Before entry, the leading m by n part of the array A must
     *           contain the matrix of coefficients.
     *           Unchanged on exit.
     *
     *  LDA    - INTEGER.
     *           On entry, LDA specifies the first dimension of A as declared
     *           in the calling (sub) program. LDA must be at least
     *           max( 1, m ).
     *           Unchanged on exit.
     *
     *  X      - COMPLEX*16       array of DIMENSION at least
     *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
     *           Before entry, the incremented array X must contain the
     *           vector x.
     *           Unchanged on exit.
     *
     *  INCX   - INTEGER.
     *           On entry, INCX specifies the increment for the elements of
     *           X. INCX must not be zero.
     *           Unchanged on exit.
     *
     *  BETA   - COMPLEX*16      .
     *           On entry, BETA specifies the scalar beta. When BETA is
     *           supplied as zero then Y need not be set on input.
     *           Unchanged on exit.
     *
     *  Y      - COMPLEX*16       array of DIMENSION at least
     *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
     *           and at least
     *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
     *           Before entry with BETA non-zero, the incremented array Y
     *           must contain the vector y. On exit, Y is overwritten by the
     *           updated vector y.
     *
     *  INCY   - INTEGER.
     *           On entry, INCY specifies the increment for the elements of
     *           Y. INCY must not be zero.
     *           Unchanged on exit.
     *
     */


    void zgemv_(char* TRANS,int* M,int* N,Complex* ALPHA,const Complex *A,int* LDA,const Complex *X,int* INCX,Complex* BETA,Complex *Y,int* INCY);

  }

  //vec_out is not checked for dimension consistency!! dimension mismatch may cause unpredictable output!
  void mat_vec_prod(const Complex*mat,size_type M,size_type N,const Complex*vec_in,Complex*vec_out,char tr)
  {

    char Trans = tr;
    Complex alpha = Complex(1);
    Complex beta = Complex(0);	
    int lda = (int)M;
    int incx = 1;
    int incy =1;
    int dM=(int)M;
    int dN=(int)N;    
    zgemv_(&Trans,&dM,&dN,&alpha,mat,&lda,vec_in,&incx,&beta,vec_out,&incy);
  }






  extern "C"
  {
    /*
      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
      *     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
      *     ..
      *     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
      *     ..
      *
      *  Purpose
      *  =======
      *
      *  DGEMM  performs one of the matrix-matrix operations
      *
      *     C := alpha*op( A )*op( B ) + beta*C,
      *
      *  where  op( X ) is one of
      *
      *     op( X ) = X   or   op( X ) = X',
      *
      *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
      *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
      *
      *  Arguments
      *  ==========
      *
      *  TRANSA - CHARACTER*1.
      *           On entry, TRANSA specifies the form of op( A ) to be used in
      *           the matrix multiplication as follows:
      *
      *              TRANSA = 'N' or 'n',  op( A ) = A.
      *
      *              TRANSA = 'T' or 't',  op( A ) = A'.
      *
      *              TRANSA = 'C' or 'c',  op( A ) = A'.
      *
      *           Unchanged on exit.
      *
      *  TRANSB - CHARACTER*1.
      *           On entry, TRANSB specifies the form of op( B ) to be used in
      *           the matrix multiplication as follows:
      *
      *              TRANSB = 'N' or 'n',  op( B ) = B.
      *
      *              TRANSB = 'T' or 't',  op( B ) = B'.
      *
      *              TRANSB = 'C' or 'c',  op( B ) = B'.
      *
      *           Unchanged on exit.
      *
      *  M      - INTEGER.
      *           On entry,  M  specifies  the number  of rows  of the  matrix
      *           op( A )  and of the  matrix  C.  M  must  be at least  zero.
      *           Unchanged on exit.
      *
      *  N      - INTEGER.
      *           On entry,  N  specifies the number  of columns of the matrix
      *           op( B ) and the number of columns of the matrix C. N must be
      *           at least zero.
      *           Unchanged on exit.
      *
      *  K      - INTEGER.
      *           On entry,  K  specifies  the number of columns of the matrix
      *           op( A ) and the number of rows of the matrix op( B ). K must
      *           be at least  zero.
      *           Unchanged on exit.
      *
      *  ALPHA  - DOUBLE PRECISION.
      *           On entry, ALPHA specifies the scalar alpha.
      *           Unchanged on exit.
      *
      *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
      *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
      *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
      *           part of the array  A  must contain the matrix  A,  otherwise
      *           the leading  k by m  part of the array  A  must contain  the
      *           matrix A.
      *           Unchanged on exit.
      *
      *  LDA    - INTEGER.
      *           On entry, LDA specifies the first dimension of A as declared
      *           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
      *           LDA must be at least  max( 1, m ), otherwise  LDA must be at
      *           least  max( 1, k ).
      *           Unchanged on exit.
      *
      *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
      *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
      *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
      *           part of the array  B  must contain the matrix  B,  otherwise
      *           the leading  n by k  part of the array  B  must contain  the
      *           matrix B.
      *           Unchanged on exit.
      *
      *  LDB    - INTEGER.
      *           On entry, LDB specifies the first dimension of B as declared
      *           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
      *           LDB must be at least  max( 1, k ), otherwise  LDB must be at
      *           least  max( 1, n ).
      *           Unchanged on exit.
      *
      *  BETA   - DOUBLE PRECISION.
      *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
      *           supplied as zero then C need not be set on input.
      *           Unchanged on exit.
      *
      *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
      *           Before entry, the leading  m by n  part of the array  C must
      *           contain the matrix  C,  except when  beta  is zero, in which
      *           case C need not be set on entry.
      *           On exit, the array  C  is overwritten by the  m by n  matrix
      *           ( alpha*op( A )*op( B ) + beta*C ).
      *
      *  LDC    - INTEGER.
      *           On entry, LDC specifies the first dimension of C as declared
      *           in  the  calling  (sub)  program.   LDC  must  be  at  least
      *           max( 1, m ).
      *           Unchanged on exit.
      */
    void dgemm_(char* TRANSA,char *TRANSB,int* M,int* N,int * K,double* ALPHA,const double *A,int* LDA,const double *B,int *LDB,double* BETA,double *C,int* LDC);

  }
  //no check if dimensions of matrices match!!!! mat_out has to be allocated before it is passed to cmat_mat_prod!
  void mat_mat_prod(double alpha,const  double*matA ,size_type dim1A,size_type dim2A,const  double* matB, size_type dim1B, size_type dim2B,double* mat_out,char tra,char trb)
  {
    char Transa = tra,Transb=trb;
	
    int M,N,Ka,Kb;
    int lda = dim1A,ldb=dim1B,ldc;
    if(tra=='n'||tra=='N')
      {
	M = (int)dim1A;
	Ka = (int)dim2A;
      }
    else
      {
	M = (int)dim2A;
	Ka =(int)dim1A;
      }
    if(trb =='n'||trb=='N') {
      N = (int)dim2B;
      Kb = (int)dim1B;
    }
    else{
      N = (int)dim1B;
      Kb = (int)dim2B;
    }
    ldc = M;
    assert(Ka==Kb);
    double beta = 0;

    dgemm_(&Transa,&Transb,&M,&N,&Ka,&alpha,matA,&lda,matB,&ldb,&beta,mat_out,&ldc);
  }


}
