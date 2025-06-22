#pragma once
#include "numcpp.h"
#include "tensor.h"
namespace nc{
template <typename T>
Tensor<T> zeros (vector<int> shape){
  int val ; 
  val = mul_vec<int>(shape);
  vector<T> data_vec = zeros_vec<T>(val);

  Tensor<T> data {data_vec,shape};
  return data;
}
template <typename T>
Tensor<T> zeros (std::initializer_list<int> shape){
  return zeros<T>(vector<int>(shape.begin(),shape.end()));
}

template <typename T>
Tensor<T> ones (vector<int> shape){
  int val ; 
  val = mul_vec<int>(shape);
  vector<T> data_vec = ones_vec<T>(val);

  Tensor<T> data {data_vec,shape};
  return data;
}
template <typename T>
Tensor<T> ones (std::initializer_list<int> shape){
  return ones<T>(vector<int>(shape.begin(),shape.end()));
}
template <typename T>
Tensor<T> arange (T start, T end, T step = 1){
  vector<T> data_vec = arange_vec<T>(start,end,step); 
  vector<int> shape = {int(data_vec.size())};
  Tensor<T> data {data_vec,shape};
  return data;
}

}