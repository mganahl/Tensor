#ifndef TBLISROUTINES_HPP_
#define TBLISROUTINES_HPP_

#include "typedefs.hpp"
#include <vector>
using std::vector;
namespace tblisroutines{
  void tblis_contract_tensors( ShapeType* shapeA,  StrideType* strideA, Complex * A, ShapeType* shapeB,  StrideType* strideB, Complex * B,\
			       ShapeType* shapeResult, StrideType* strideResult,Complex * Result, \
			       LabelType* labelsA, LabelType* labelsB, LabelType* labelsResult);


  void tblis_contract_tensors( ShapeType* shapeA,  StrideType* strideA, Real * A, ShapeType* shapeB, StrideType* strideB, Real * B,\
			       ShapeType* shapeResult, StrideType* strideResult,Real * Result, \
			       LabelType* labelsA, LabelType* labelsB, LabelType* labelsResult);

  Real tblis_dot( ShapeType* shapeA,  StrideType* strideA, Real * A, ShapeType* shapeB, StrideType* strideB, Real * B, \
		  LabelType* labelsA, LabelType* labelsB);
  
  Complex tblis_dot( ShapeType* shapeA,  StrideType* strideA, Complex * A, ShapeType* shapeB, StrideType* strideB, Complex * B, \
		     LabelType* labelsA, LabelType* labelsB);
  
}
#endif 
