#ifndef UTILITIES_H
#define UTILITIES_H

#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <assert.h>
#include <type_traits>
#include <typeinfo>
#include <algorithm>
#include<iostream>
#include <map>
#include <numeric>
#include <vector>
#include <iterator>
#include "../../typedefs.hpp"
#include "exceptions.hpp"
using namespace std;
using namespace exceptions;
using std::size_t;
using std::vector;
namespace utilities {
  void msg(string s){
    cout<<s<<endl;
  }
  
  string LabelType_to_string(LabelType label){
    std::string sl="";
    for(int n=0;n<label.size()-1;n++){
      sl+=std::to_string(label[n])+",";
    }
    sl+=std::to_string((*(label.end()-1)));
    return sl;
  }
  
  bool largerthan (int i,int j) { return (i>j);}
  
  void _assert_(bool expr,const char* msg,const char* file,int line){
    if (expr==false){
      std::cerr<<"assertion failed in file "<<file<<" at line "<<line<<": "<<msg<<std::endl;
      abort();
    }

  }

  //cimulative product of a vector
  template<typename T>
  T cumprod(vector<T>&v){
    T i=T(1.0);
    for (auto n:v)
      i=i*n;
    return i;
  }

  //intersection of elements of std vector a and b
  LabelType intersection(LabelType a, LabelType b){
    std::sort(a.begin(),a.end());
    std::sort(b.begin(),b.end());
    LabelType out; out.resize(min(a.size(),b.size()));
    auto it=std::set_intersection (a.begin(),a.end(),b.begin(),b.end(),out.begin());
    out.resize(it-out.begin());
    return out;
  }
  //set_difference of elements of std vector a and b (a-b)
  LabelType difference(LabelType a, LabelType b){
    std::sort(a.begin(),a.end(),largerthan);
    std::sort(b.begin(),b.end(),largerthan);
    LabelType out;
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
			std::inserter(out, out.begin()));
    return out;
  }

  //returns the resulting label of std::vectors a and b
  LabelType  resultinglabel(LabelType a, LabelType b){
    auto I1=difference(a,b);
    auto I2=difference(b,a);
    LabelType  out;
    for (auto n:I1) out.push_back(n);
    for (auto n:I2) out.push_back(n);
    std::sort(out.begin(),out.end(),largerthan);  
    return out;
  }


  
  /*Generic class for accumulating numbers of template type T in a simple fashion
    Useage: 
    accumulator=Accumulator<T>;
    T a,b,c,d,e;
    accumulator<<a,b,c,d,e;
    cout<<accumulator[3]; //prints the value of d
    cout<<accumulator[4]; //prints the value of e
  */

  template<typename T>
  class  Accumulator
  {
  private:
    std::vector<T> v_;  
  public:
    Accumulator(){v_.resize(0);}
    Accumulator<T>& operator<<(T val){
      this->operator,(val);return *this;
    }
    Accumulator<T>& operator,(T val){
      v_.push_back(val);
      return *this;
    }
    T operator[](const lint i) const {
      return v_[i];
    }
    lint size()const {return v_.size();}
    T* data(){return v_.data();}

    typename std::vector<T>::const_iterator begin()const{return v_.begin();}
    typename std::vector<T>::const_iterator end()const{return v_.end();}
    typename std::vector<T>::iterator begin(){return v_.begin();}
    typename std::vector<T>::iterator end(){return v_.end();}
  };


  /*
    a simple Slice class: 
    memebers: 
    long int start_: start of the slice
    long int stop_: stop of the slice
    long int stride_: stride of the slice
    used to produce slices of tensors

    member functions start(), stop(), stride() can be used to access or change start_, stop_ and stride_;
    use reset() after any change done to Slice!
    The empty constructor constructs an All-like slice which selects all indices
  */
  
  class Slice{
  public:
    Slice():start_(0),stop_(((long int)1)<<62),stride_(1){}
    Slice(lint start,lint stop,lint stride): start_(start),stop_(stop),stride_(stride){
      if (stride_<1)
	throw StrideError("in Slice: stride has to be >!");	
      if (start_<0)
	throw NonsenseError("in Slice: start<0; this is nonsense, you idiot!");	
      if (stop_<0)
	throw NonsenseError("in Slice: stop<0; this is nonsense, you idiot!");	
      if (stride_<0)
	throw NonsenseError("in Slice: stride<0; this is nonsense, you idiot!");		
      
      if(start_<stop_){
	size_=((stop_-start_)-(stop_-start_)%stride_)/stride_+1;
      }else if(start_==stop_){
	size_=1;
      }else if(start_>stop_){
	throw NonsenseError("in Slice(lint,lint,lint): start of slice > stop of slice; that is nonsense, you idiot!");
      }
    }
    void reset(){
      if(start_<stop_){
	size_=((stop_-start_)-(stop_-start_)%stride_)/stride_+1;
      }else if(start_==stop_){
	size_=1;
      }else if(start_>stop_){
	throw NonsenseError("in Slice.reset(): start of slice > stop of slice; that is nonsense, you idiot!");
      }
    }
    
    lint size() const{
      return size_;
    }
    lint start() const{
      return start_;
    }
    lint& start(){
      return start_;
    }
    lint stop() const{
      return stop_;
    }
    lint& stop(){
      return stop_;
    }
    
    lint stride() const{
      return stride_;
    }
    lint& stride(){
      return stride_;
    }
    
    virtual void print(){
      cout<<"Slice"<<endl;
      cout<<"start: "<<start_<<"; stop: "<<stop_<<"; stride: "<<stride_<<"; size: "<<size_<<endl;
    }
  private:
    lint start_;
    lint stop_;
    lint stride_;
    lint size_;
  };


  /*
    subclasses Slice; can be used in Tensor<T>.slice to select the complete leg; this works only for 64 bit architectures, and only 
    if long int is 64 bits!
  */
  class All:public Slice{
  public:
    All():Slice(0,((long int)1)<<62,1){}
    virtual void print(){
      cout<<"All-slice"<<endl;
      cout<<"start: "<<Slice::start()<<"; stop: "<<Slice::stop()<<"; stride: "<<Slice::stride()<<"; size: "<<Slice::size()<<endl;
    }
   
  };

  
  /*maps a std::vector of indices to a single integer; assumes column-major ordering of the tensor
    stride is the accumulated dimension of the tensor: stride[n]=prod_{i<n} shape[i]
  */
  lint to_integer(const IndexType& indices,const StrideType& stride){
    lint index=0;
    for(lint d=0;d<indices.size();d++){
      index+=stride[d]*indices[d];
    }
    return index;
  }

  //maps a std::vector of indices to a single integer; assumes column-major ordering of the tensor
  //stride is the cumulated dimension of the tensor: stride[n]=prod_{i<n} shape[i]
  //n1 + d1*n2 +d1*d2*n3 + d1*d2*d3*n4 ...
  void to_multiindex(IndexType& indices,const lint& index,const StrideType& stride){
    //std::vector<int> indices(stride.size());
    lint residual;
    lint temp=index;
    for(lint d=(indices.size()-1);d>=0;d--){
      residual=temp%stride[d];
      indices[d]=(temp-residual)/stride[d];
      temp=residual;
    }
  }
  

  //takes an initial multi-index index and a vector<Slice> slices of Slice-types and generates
  //the next multiindex by increasing the frist increasable index at position d by slices[d].stride()
  bool next_multiindex(IndexType& index,const std::vector<Slice>&slices){
    lint d=0;
    while(index[d]>(slices[d].stop()-slices[d].stride())){
      d++;
      if (d==slices.size()){
	return false;      
      }
    }
    index[d]+=slices[d].stride();
    for(lint d2=0;d2<d;d2++){
      index[d2]=slices[d2].start();
    }
    assert(index[d]<=slices[d].stop());
    return true;
  }

  //helper function of linearize_slices
  void linearize(IndexType& inds,lint& num,lint pos,const std::vector<Slice>& slices,std::vector<IndexType >&list)
  {
    assert(inds.size()==slices.size());
    for(lint n=slices[pos].start();n<=slices[pos].stop();n+=slices[pos].stride()){
      if (pos<(slices.size()-1)){
	inds[pos]=n;
	linearize(inds,num,pos+1,slices,list);
      }
      if(pos==(slices.size()-1)){
	for(lint m=0;m<pos;m++){
	  list[num][m]=inds[m];
	}
	list[num][pos]=n;
	num++;      
      }
    }
  }

  //takes a std::vector<Slice> containing slices; computes all tensor indices that can be formed by the slices and
  //maps it to an integer "index", the position in the linear storage-space of the tensor elements.
  //returns a std::map<int,std::vector<int> > that maps each "index" to its tuple-valued position.
  std::map<lint,IndexType > linearize_slices(const std::vector<Slice> slices,const StrideType& stride)
  {
    lint N=1;
    for (lint n=0;n<slices.size();n++){
      N*=slices[n].size();
    }
  
    std::vector<IndexType >indexlist(N,IndexType(slices.size(),0));
    IndexType indices(slices.size());
    lint num=0,pos=0;
    linearize(indices,num,pos,slices,indexlist);
    std::map<lint,IndexType> indexmap;
    for(lint i=0;i<indexlist.size();++i){
      indexmap[to_integer(indexlist[i],stride)]=indexlist[i];
    }
    return indexmap;
  }



}
#endif
