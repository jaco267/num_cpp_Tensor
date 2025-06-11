#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "nc_def.h"
using std::string;
using std::vector;
using std::cout;
using std::endl;
#pragma once
namespace nc{
template <typename T>
class Tensor{
public: 
  Tensor(){}
  Tensor(vector<T>& data,const vector<int>&shape);
  Tensor(vector<T>& data,std::initializer_list<int> shape
  ): Tensor(data, vector<int>(shape.begin(), shape.end())){
    // cout<<"with initializer"<<endl;
    /*
    vector<int> zz = zeros_vec<int>(16);
    Tensor v1 {zz,{4,4}};
    */
  };
  Tensor<T> reshape(vector<int> new_shape);
  Tensor<T> reshape(std::initializer_list<int> new_shape){
    return reshape(vector<int>(new_shape));
  };
  T index(vector<int> indices);
  T index(std::initializer_list<int> indices){
      return index(vector<int>(indices)); // Convert and forward
  };
  Tensor<T> slice(int dim, int start, int end);
  void slice_put(int dim, int start, int end, Tensor<T> in_data);
  void info();
  void print();
  string print_recur(int depth, vector<int> index);
  vector<T> toVec();
  mat<T> toMat();
private:
  vector<T> data_;  
  vector<int> strides_;  
  vector<int> shape_;  
  int ndim_; 
  int size_;
};
}

#include "tensor_init.h"