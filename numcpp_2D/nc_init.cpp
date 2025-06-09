#include "numcpp.h"
namespace nc{
vector<int> zeros_vec(int size){
  vector<int> zeros; 
  zeros.resize(size, 0); 
  return zeros;
}
mat zeros_mat(int row, int col){
  mat zeros; 
  for (int i=0;i<row;i++){
    vector<int> row;  
    row.resize(col,0);
    zeros.push_back(row);  
  }
  return zeros;
}
mat eye(int k){
  mat e;
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