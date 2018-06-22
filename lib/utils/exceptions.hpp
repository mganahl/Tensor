#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <stdlib.h>
#include <string>
#include <assert.h>
#include "typedefs.hpp"
#include<iostream>
using std::cout;
using std::endl;
namespace exceptions{
  class NotAnIntegerError: public std::exception{
  public:
    NotAnIntegerError(){
      msg_="not an integer";
    };
    NotAnIntegerError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };

  class OutOfBoundError: public std::exception{
  public:
    OutOfBoundError(){
      msg_="index out of bounds";
    };
    OutOfBoundError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };
  
  class DimensionMismatchError: public std::exception{
  public:
    DimensionMismatchError(){
      msg_="dimensional mismatch";
    };
    DimensionMismatchError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };

  class RankMismatchError: public std::exception{
  public:
    RankMismatchError(){
      msg_="rank mismatch";
    };
    RankMismatchError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };
  
  class SizeMismatchError: public std::exception{
  public:
    SizeMismatchError(){
      msg_="size mismatch";
    };
    SizeMismatchError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };
  
  class ShapeMismatchError: public std::exception{
  public:
    ShapeMismatchError(){
      msg_="size mismatch";
    };
    ShapeMismatchError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };


  class NonsenseError: public std::exception{
  public:
    NonsenseError(){
      msg_="does not make sense, you idiot!";
    };
    NonsenseError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };
  
  class SliceError: public std::exception{
  public:
    SliceError(){
      msg_="cannot slice tensor";
    };
    SliceError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };
  
  class StrideError: public std::exception{
  public:
    StrideError(){
      msg_="stride error";
    };
    StrideError(const char * msg):msg_(msg){
    }
    virtual const char *what() const throw(){
      return msg_;
    }
  private:
    const char* msg_;
  };
  
  
  class Warning{
  public:
    Warning(std::string warning){cout<<"Warning: "<<warning<<endl;}
  };
  
}


#endif
