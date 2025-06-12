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
}