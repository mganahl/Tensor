#include <array>
#include <vector>
#include <iostream>
#include <cstdarg>
#include <sstream>
#include "lib/utilities.hpp"
using namespace utilities;
using namespace std;
int main(int argc, char** argv){
  auto v=vector<int>{1,2,3};

  auto a = Accumulator<int>();
  auto b = Accumulator<double>();  
  b<<1.1,2.2,3.5,4.01,5,6,7;
  for (int n=0;n<b.size();n++){
    std::cout<<b[n]<<" ";
  }
  std::cout<<std::endl;
  auto c=b.data();
  return 0;

}
