#include <iostream>
#include "typedefs.hpp"
#include <tblis.h>
#include <vector>
using std::vector;
using tblis::tblis_init_tensor_z;
using tblis::tblis_init_tensor_d;
using tblis::tblis_init_scalar_z;
using tblis::tblis_init_scalar_d;
using tblis::tblis_scalar;
using tblis::len_type;
using tblis::tblis_tensor;
using tblis::stride_type;
using tblis::label_type;
using std::cout;
using std::endl;
/*basic tensor contraction routine; data is passed as pointer
  no checks are done, the Result data has to be allocated already; 
  shape and stride of Result have to be set already
*/
namespace tblisroutines{


  
  void tblis_contract_tensors( ShapeType* shapeA,  StrideType* strideA, Complex * A, ShapeType* shapeB, StrideType* strideB, Complex * B,\
			       ShapeType* shapeResult, StrideType* strideResult,Complex * Result, \
			       LabelType* labelsA, LabelType* labelsB, LabelType* labelsResult){
  

    tblis_tensor tblisA;
    uint ndimA=shapeA->size();
    std::vector<lint> shapeA_cast(shapeA->begin(),shapeA->end());
    len_type *la = shapeA_cast.data();//reinterpret_cast<lint *>(shapeA->data());
    stride_type *sa= strideA->data();
    tblis_init_tensor_z(&tblisA, ndimA, la,A,sa);
  
    tblis_tensor tblisB;
    uint ndimB=shapeB->size();
    std::vector<lint> shapeB_cast(shapeB->begin(),shapeB->end());      
    len_type *lb = shapeB_cast.data();//reinterpret_cast<lint *>(shapeB->data());
    stride_type *sb= strideB->data();
    tblis_init_tensor_z(&tblisB, ndimB, lb,B,sb);
  
    tblis_tensor tblisR;
    uint ndimR=shapeResult->size();
    std::vector<lint> shapeR_cast(shapeResult->begin(),shapeResult->end());    
    len_type *lR = shapeR_cast.data();//reinterpret_cast<lint *>(shapeResult->data());
    stride_type *sR= strideResult->data();
    tblis_init_tensor_z(&tblisR, ndimR, lR,Result,sR);
  
    label_type *labelsr=labelsResult->data();
    label_type* labelsa=labelsA->data();
    label_type* labelsb=labelsB->data();
  
    tblis_tensor_mult(NULL, NULL, &tblisA, labelsa, &tblisB, labelsb, &tblisR,labelsr);
  }


  void tblis_contract_tensors( ShapeType* shapeA,  StrideType* strideA, Real * A, ShapeType* shapeB, StrideType* strideB, Real * B,\
			       ShapeType* shapeResult, StrideType* strideResult,Real * Result, \
			       LabelType* labelsA, LabelType* labelsB, LabelType* labelsResult){

    tblis_tensor tblisA;
    uint ndimA=shapeA->size();
    std::vector<lint> shapeA_cast(shapeA->begin(),shapeA->end());
    len_type *la = shapeA_cast.data();//reinterpret_cast<lint *>(shapeA->data());
    stride_type *sa= strideA->data();
    tblis_init_tensor_d(&tblisA, ndimA, la,A,sa);

    tblis_tensor tblisB;
    uint ndimB=shapeB->size();
    std::vector<lint> shapeB_cast(shapeB->begin(),shapeB->end());  
    len_type *lb = shapeB_cast.data();//reinterpret_cast<lint *>(shapeB->data());
    stride_type *sb= strideB->data();
    tblis_init_tensor_d(&tblisB, ndimB, lb,B,sb);
  
    tblis_tensor tblisR;
    uint ndimR=shapeResult->size();
    std::vector<lint> shapeR_cast(shapeResult->begin(),shapeResult->end());    
    len_type *lR = shapeR_cast.data();//reinterpret_cast<lint *>(shapeResult->data());
    stride_type *sR= strideResult->data();
    tblis_init_tensor_d(&tblisR, ndimR, lR,Result,sR);

    label_type *labelsr=labelsResult->data();
    label_type* labelsa=labelsA->data();
    label_type* labelsb=labelsB->data();

    tblis_tensor_mult(NULL, NULL, &tblisA, labelsa, &tblisB, labelsb, &tblisR,labelsr);
  }


  Complex tblis_dot( ShapeType* shapeA,  StrideType* strideA, Complex * A, ShapeType* shapeB, StrideType* strideB, Complex * B, \
		     LabelType* labelsA, LabelType* labelsB){
    
    tblis_tensor tblisA;
    uint ndimA=shapeA->size();
    std::vector<lint> shapeA_cast(shapeA->begin(),shapeA->end());
    len_type *la = shapeA_cast.data();
    stride_type *sa= strideA->data();
    tblis_init_tensor_z(&tblisA, ndimA, la,A,sa);

    tblis_tensor tblisB;
    uint ndimB=shapeB->size();
    std::vector<lint> shapeB_cast(shapeB->begin(),shapeB->end());  
    len_type *lb = shapeB_cast.data();
    stride_type *sb= strideB->data();
    tblis_init_tensor_z(&tblisB, ndimB, lb,B,sb);
  
    tblis_scalar tbliss;
    tblis_init_scalar_z(&tbliss, Complex(0.0,0.0));

    label_type* labelsa=labelsA->data();
    label_type* labelsb=labelsB->data();

    tblis_tensor_dot(NULL, NULL, &tblisA, labelsa, &tblisB, labelsb, &tbliss);
    return tbliss.get<Complex>();
  }
  

  
  Real tblis_dot( ShapeType* shapeA,  StrideType* strideA, Real * A, ShapeType* shapeB, StrideType* strideB, Real * B, \
		  LabelType* labelsA, LabelType* labelsB){
    
    tblis_tensor tblisA;
    uint ndimA=shapeA->size();
    std::vector<lint> shapeA_cast(shapeA->begin(),shapeA->end());
    len_type *la = shapeA_cast.data();
    stride_type *sa= strideA->data();
    tblis_init_tensor_d(&tblisA, ndimA, la,A,sa);
    
    tblis_tensor tblisB;
    uint ndimB=shapeB->size();
    std::vector<lint> shapeB_cast(shapeB->begin(),shapeB->end());  
    len_type *lb = shapeB_cast.data();
    stride_type *sb= strideB->data();
    tblis_init_tensor_d(&tblisB, ndimB, lb,B,sb);
    
    tblis_scalar tbliss;
    tblis_init_scalar_d(&tbliss, Real(0.0));

    label_type* labelsa=labelsA->data();
    label_type* labelsb=labelsB->data();

    tblis_tensor_dot(NULL, NULL, &tblisA, labelsa, &tblisB, labelsb, &tbliss);
    return tbliss.get<Real>();
  }
  
}
