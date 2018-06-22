#include <array>
#include <iostream>
#include <cstdarg>
#include <sstream>
#include "typedefs.hpp"
#include "lib/array/tensoroperations.hpp"
#include "lib/array/tensor.hpp"
#include "lib/utils/print.hpp"
using namespace std;
using namespace printfunctions;
using namespace tensor;
template<typename... Args>
void testfun(Args... params){
  std::vector<int> indices={params...};
  for (uint n =0;n<indices.size();++n){
    cout<<indices[n]<<" ";
  }
}

int main(int argc, char** argv){
  // std::istringstream ss1(argv[1]);
  // std::istringstream ss2(argv[2]);
  // std::istringstream ss3(argv[3]);    
  // int start,stop,stride;
  // ss1>>start;
  // ss2>>stop;
  // ss3>>stride;
  int start=0,stop=3,stride=1;
  Slice s1(0,7,1);
  Slice s2(0,0,1);
  All s3;
  Tensor<Complex> a(8,8);

  a.randomize(-0.5,0.5);
  auto sliceb=a.slice(s2,s1);
  sliceb.print();
  auto fg=a.dot(a.slice(Slice(0,0,1),Slice()).transp());
  fg.print();


  //slicea.print();

  // auto b=a.slice(s1,s2);
  // cout<<"tensor a"<<endl;
  // for(uint m=0;m<a.shape(0);m++){
  //   for(uint n=0;n<a.shape(1);n++){
  //     cout<<a(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }

  // cout<<endl;
  // cout<<"tensor b"<<endl;  
  // for(uint m=0;m<b.shape(0);m++){
  //   for(uint n=0;n<b.shape(1);n++){
  //     cout<<b(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }
  // a.insert_slice(c,s1,s2);
  // cout<<endl;
  // cout<<"tensor c"<<endl;
  // for(uint m=0;m<c.shape(0);m++){
  //   for(uint n=0;n<c.shape(1);n++){
  //     cout<<c(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }
  // cout<<endl;
  // cout<<"tensor a"<<endl;
  
  // for(uint m=0;m<a.shape(0);m++){
  //   for(uint n=0;n<a.shape(1);n++){
  //     cout<<a(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }
  
  // cout<<endl;
  // cout<<"tensor a"<<endl;
  // for(uint m=0;m<a.shape(0);m++){
  //   for(uint n=0;n<a.shape(1);n++){
  //     cout<<a(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }
  
  // a.transpose(1,0);
  // cout<<endl;
  // cout<<"tensor a"<<endl;
  // for(uint m=0;m<a.shape(0);m++){
  //   for(uint n=0;n<a.shape(1);n++){
  //     cout<<a(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }
  // Tensor<Real> e(2,3,4,2);
  // e.randomize(1.0);    
  // cout<<e.size()<<endl;
  
  // e.transpose(2,0,3,1);
  // e.reshape(6,8);
  // cout<<endl;
  // cout<<"tensor e"<<endl;
  // for(uint m=0;m<e.shape(0);m++){
  //   for(uint n=0;n<e.shape(1);n++){
  //     cout<<e(m,n)<<"  ";
  //   }
  //   cout<<endl;
  // }
  
  //a(4,7,8)=Complex(10.0,1.);

  // cout<<endl;
  // cout<<"a(4,7,8)="<<a(4,7,8)<<endl;
  // cout<<"a.size()="<<a.size()<<endl;  
  // cout<<"a.shape("<<index<<")="<<a.shape(index)<<endl;
  // cout<<"a.rank()="<<a.rank()<<endl;
  // cout<<"( ";  
  // for(uint n=0;n<a.rank();n++){
  //   cout<<a.shape(n)<<" ";
  // }
  // cout<<')'<<endl;


  // Tensor<Complex> b(std::vector<int>{10,10,12});    
  // b.random_complex(2.0);  

  // cout<<endl;
  // cout<<"b(4,7,8)="<<b(4,7,8)<<endl;
  // cout<<"b.size()="<<b.size()<<endl;  
  // cout<<"b.shape("<<index<<")="<<b.shape(index)<<endl;
  // cout<<"b.rank()="<<b.rank()<<endl;
  // cout<<"( ";  
  // for(uint n=0;n<b.rank();n++){
  //   cout<<b.shape(n)<<" ";
  // }
  // cout<<')'<<endl;

  return 0;  
}
