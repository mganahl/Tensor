#include <chrono>
#include <tblis.h>
#include "lib/linalg/blasroutines.hpp"
typedef double Real;
typedef complex<double> Complex;

using std::cout;
using std::endl;
using std::vector;

int main(int argc, char** argv){

  
  int d1=200,d2=200,d3=200;
  vector < tblis::label_type > la={1,2};
  vector < tblis::label_type > lb={2,3};
  vector < tblis::label_type > lc={1,3};
  tblis::tensor < Complex> A({d1,d2});
  tblis::tensor < Complex> B({d2,d3});
  tblis::tensor < Complex > C({d1,d3});
  Complex beta=0.0;
  Complex alpha=1.0;
  auto mA=new Complex [d1*d2];
  auto mB=new Complex [d2*d3];
  auto mC=new Complex [d1*d3];
  int Nmax=1000;
  vector<double> time(Nmax);
  vector<double> time2(Nmax);
  cout<<"doing TBLIS contraction"<<endl;
  for(uint n=0;n<Nmax;n++){
    auto time0 = std::chrono::high_resolution_clock::now();  
    tblis::mult(alpha,A,la.data(),B,lb.data(),beta,C,lc.data());
    auto time1 = std::chrono::high_resolution_clock::now();    
    time[n]=std::chrono::duration_cast<std::chrono::microseconds>(time1-time0).count();
  }
  cout<<"doing ZGEMM contraction"<<endl;  
  for(uint n=0;n<Nmax;n++){
    auto time0 = std::chrono::high_resolution_clock::now();  
    blasroutines::mat_mat_prod(alpha,mA ,d1,d2,mB,d2,d3,mC,'N','N');
    auto time1 = std::chrono::high_resolution_clock::now();
    time2[n]=std::chrono::duration_cast<std::chrono::microseconds>(time1-time0).count();
  }
  cout<< "TBLIS: fastest of "<<Nmax<<" contractions: "<<*std::min_element(time.begin(),time.end()) << " microseconds\n";        
  cout<< "TBLIS: average time per contraction: "<<std::accumulate(time.begin(),time.end(),0.0)/Nmax << " microseconds\n";
  cout<< "ZGEMM: fastest of "<<Nmax<<" contractions: "<<*std::min_element(time2.begin(),time2.end()) << " microseconds\n";        
  cout<< "ZGEMM: average time per contraction: "<<std::accumulate(time2.begin(),time2.end(),0.0)/Nmax << " microseconds\n";

  
  delete[] mA;
  delete[] mB;
  delete[] mC;  
  return 0;
}
