#include "numcpp.h"
namespace nc{


mat<int>eye(int k){
  mat<int>e;
  for (int i=0;i<k;i++){
      vector<int> row;  
      row.resize(k,0);  
      row[i] = 1;  
      e.push_back(row);
  } 
  return e;
}
vector<int> arange(int start, int end, int step) {
  std::vector<int> result;
  for (int i = start; i < end; i += step) {
      result.push_back(i);
  }
  return result;
}
}