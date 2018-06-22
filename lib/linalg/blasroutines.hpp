/*
 * blasroutines.hpp
 *
 *  Created on: 07.09.2010
 *      Author: martinjg
 */

#ifndef BLASROUTINES_HPP_
#define BLASROUTINES_HPP_
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "typedefs.hpp"
namespace blasroutines{
  void mat_mat_prod(Complex alpha,const Complex *matA ,size_type dim1A,size_type dim2A,const  Complex* matB,size_type dim1B, size_type dim2B,Complex* mat_out,char tra,char trb);
  void mat_mat_prod(double alpha, const double*matA ,size_type dim1A,size_type dim2A,const double* matB, size_type dim1B, size_type dim2B,double* mat_out,char tra,char trb);
  void mat_vec_prod(const Complex *mat,size_type M,size_type N,const Complex *vec_in,Complex*vec_out,char tr);
  void mat_vec_prod(const Real*mat,size_type M,size_type N,const Real*vec_in,Real*vec_out,char tr);
  Real dot_prod(size_type size,const Real* v1,const Real*v2);
  Complex dot_prod(size_type size,const Complex* v1,const Complex*v2);
  //int tridiag(double *diag,int length_diag, double* sub_diag,double*eigvec,char JOBZ = 'N');
}
#endif /* BLASROUTINES_HPP_ */
