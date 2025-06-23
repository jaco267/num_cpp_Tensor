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
template <typename T> 
Tensor<T> hstack(const std::initializer_list<Tensor<T>> tensors) {
  return hstack(vector<Tensor<T>>(tensors.begin(), tensors.end()));
}
template <typename T>
Tensor<T> hstack(const vector<Tensor<T>>& tensors) {
  if (tensors.empty()) {return Tensor<T>();}// Check if there are tensors to stack
  size_t ndim = tensors[0].shape_.size();
  for (const auto& tensor : tensors) {// Check all tensors have same num of dim
      ASSERT_THROW(tensor.shape_.size() == ndim, "All tensors must have the same number of dimensions");
  }
  if (ndim==1){
    vector<T> out_data;
    int tot_len = 0;
    for (const auto& tensor : tensors) {
        tot_len += (int) tensor.data_.size();
        out_data.insert(out_data.end(), tensor.data_.begin(), tensor.data_.end());
    }
    return Tensor<T>(out_data, {tot_len});
  }else if (ndim == 2){
    // Check all dim except the 2nd dim match
    std::vector<int> ref_shape = tensors[0].shape_;
    int total_cols = tensors[0].shape_[1];
    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& shape_ = tensors[i].shape_;
        for (int dim = 0; dim < static_cast<int>(ndim); ++dim) {
            if (dim == 1) continue; // Skip second dimension
            ASSERT_THROW(shape_[dim] == ref_shape[dim], 
                        "All tensors must have the same shape except in dimension 1");
        }
        total_cols += shape_[1];
    }
    // Create output shape
    vector<int> out_shape = ref_shape;
    out_shape[1] = total_cols;
    // Concatenate data
    vector<T> out_data;
    // For 2D tensors
    size_t outer_dim_size = 1;
    for (int dim = 0; dim < static_cast<int>(ndim); ++dim) {
        if (dim == 1) continue;
        outer_dim_size *= ref_shape[dim];
    }
    for (size_t outer = 0; outer < outer_dim_size; ++outer) {
        for (const auto& tensor : tensors) {
            const auto& t_shape = tensor.shape_;
            size_t t_inner_size = t_shape[1];
            size_t offset = outer * t_inner_size; 
            for (size_t inner = 0; inner < t_inner_size; ++inner) {
                out_data.push_back(tensor.data_[offset + inner]);
            }
        }
    }
    return Tensor<T>(out_data, out_shape);
  }else{
     throw std::invalid_argument( "currently only support 1d and 2d hstack" );
  }
  

}
}