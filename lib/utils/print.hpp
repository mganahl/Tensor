#ifndef PRINT_H
#define PRINT_H

#include <cstdlib>
#include <time.h>
#include <assert.h>
#include <type_traits>
#include <typeinfo>
#include <algorithm>
#include "../../typedefs.hpp"
#include "utilities.hpp"
#include<iostream>
#include <map>
using namespace std;
using namespace utilities;
using std::size_t;

namespace printfunctions {


  template<class T>
  void print(std::vector<T> &v){
    std::cout<<"[";
    if (v.size()>0){
      auto i=v.begin();
      while(i!=(--v.end())){
	std::cout<<*i<<",";
	i++;
      }
    std::cout<<*i;
    }    
    std::cout<<"]"<<std::endl;
  }

  template<class T>
  void print(const std::vector<T> &v){
    std::cout<<"[";
    if (v.size()>0){
      auto i=v.begin();
      while(i!=(--v.end())){
	std::cout<<*i<<",";
	i++;
      }
    std::cout<<*i;
    }    
    std::cout<<"]"<<std::endl;
  }

  
  void print(uint i){
    std::cout<<i<<std::endl;
  }
  void print(int i){
    std::cout<<i<<std::endl;
  }

  void print(Real i){
    std::cout<<i<<std::endl;
  }
  
  void print(Complex i){
    std::cout<<i<<std::endl;
  }
  void print(std::size_t i){
    std::cout<<i<<std::endl;
  }

  template<typename T>
  void print(Accumulator<T> &acc){
    cout<<"[";
    for (uint i=0;i!=(acc.size()-1);i++){
      cout<<acc[i]<<", ";
    }
    cout<<acc[acc.size()-1]<<"]"<<endl;
  }

}
#endif
