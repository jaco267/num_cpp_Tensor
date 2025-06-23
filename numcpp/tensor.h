#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "nc_def.h"

#include "tensor_index.h"
using std::string;
using std::vector;
using std::cout;
using std::endl;
#pragma once
namespace nc{
using nc_Slice_Index = std::variant<
  nc::indexing::Slice, 
  int,std::monostate
>;

template <typename T>
class Tensor{
public: 
  Tensor(){}
  Tensor(const vector<T>& data,const vector<int>&shape);
  Tensor(std::initializer_list<T> data,std::initializer_list<int> shape
  ): Tensor(vector<T>(data.begin(), data.end()), 
            vector<int>(shape.begin(), shape.end())){};
  Tensor(vector<T>& data,std::initializer_list<int> shape
  ): Tensor(data, vector<int>(shape.begin(), shape.end())){};
  void init_tensor(const vector<T>& data, const vector<int>&shape);
  
  Tensor<T> reshape(vector<int> new_shape);
  Tensor<T> reshape(std::initializer_list<int> new_shape){
    return reshape(vector<int>(new_shape));
  };
  void info();
  void print();
  string print_recur(int depth, vector<int> index);
  vector<T> toVec();
  mat<T> toMat();
  void fromMat(const mat<T>& m);
  //*-----tensor_arith.cpp----
  Tensor<T> add(const Tensor<T>& a);
  Tensor<T> minus();
  //*-----tensor index_utils.cpp-----
  T index(vector<int> indices);
  T index(std::initializer_list<int> indices){
      return index(vector<int>(indices)); // Convert and forward
  };
  void create_new_shape_from_slice(
    const mat<int>& new_slice, vector<int>& new_shape);
  void extract_slice_index(const vector<nc_Slice_Index>& slice_indices,
    mat<int>& new_slice);
  Tensor<T> index(const vector<nc_Slice_Index>& indices);
  void index_put(const vector<nc_Slice_Index>& indices,
     const Tensor<T> & in_data);
  Tensor<T> slice(int dim, int start, int end);
  void slice_put(int dim, int start, int end, Tensor<T> in_data);
  //*-----------------------
public:
  vector<int> shape_;  

  vector<T> data_; 
private: 
  vector<int> strides_;  
  int ndim_; 
  int size_;
};
}

#include "ten_init.h"