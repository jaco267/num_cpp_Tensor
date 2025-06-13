#pragma once

namespace nc{
template <typename T>
vector<T> zeros_vec(int size){
  vector<T> zeros; 
  zeros.resize(size, 0); 
  return zeros;
}
template <typename T>
vector<T> ones_vec(int size){
  vector<T> zeros; 
  zeros.resize(size, 1); 
  return zeros;
}
template <typename T>
std::vector<std::vector<T>> zeros_mat(int row, int col) {
    std::vector<std::vector<T>> zeros;
    for (int i = 0; i < row; ++i) {
        std::vector<T> r(col, static_cast<T>(0));
        zeros.push_back(r);
    }
    return zeros;
}
template<typename T>
vector<T> arange_vec(T start, T end, T step=1) {
  if (start > end || step <= 0){
     throw std::invalid_argument( "arange_vec currently only support end > start and step > 0" );
  }
  vector<T> result;
  for (T i = start; i < end; i += step) {
      result.push_back((T)i);
  }
  return result;
}
template<typename T>
mat<T> eye(int k){
  mat<T>e;
  for (int i=0;i<k;i++){
      vector<T> row;  
      row.resize(k,0);  
      row[i] = 1;  
      e.push_back(row);
  } 
  return e;
}

}