#pragma once
#include "numcpp.h"
#include "tensor.h"
namespace nc{
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
  }else if (ndim == 2){ // For 2D tensors
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
  }else{throw std::invalid_argument( "currently only support 1d and 2d hstack" );}
}
template <typename T> 
Tensor<T> vstack(const std::initializer_list<Tensor<T>> tensors) {
  return vstack(vector<Tensor<T>>(tensors.begin(), tensors.end()));
}
template <typename T>
Tensor<T> vstack(const vector<Tensor<T>>& tensors) {
  if (tensors.empty()) {return Tensor<T>();}
  // Check if all tensors have the same number of dimensions
  size_t ndim = tensors[0].shape_.size();
  for (const auto& tensor : tensors) {
      ASSERT_THROW(tensor.shape_.size() == ndim,  "All tensors must have the same number of dimensions");
  }
  if (ndim == 1) {// For 1D tensors, vstack them to 2D (row vectors) and stacks
    vector<T> out_data;
    size_t total_rows = tensors.size();
    size_t cols = tensors[0].data_.size();
    for (const auto& tensor : tensors) {// Check all 1D tensors have same length
        ASSERT_THROW(tensor.data_.size() == cols,"All 1D tensors must have same length for vstack");
        out_data.insert(out_data.end(), tensor.data_.begin(), tensor.data_.end());
    }
    return Tensor<T>(out_data, {static_cast<int>(total_rows), static_cast<int>(cols)});
  }else if (ndim == 2) {
    // For 2D tensors, check all dimensions except the first match
    std::vector<int> ref_shape = tensors[0].shape_;
    int total_rows = tensors[0].shape_[0];
    for (size_t i = 1; i < tensors.size(); ++i) {
      const auto& shape_ = tensors[i].shape_;
      for (int dim = 0; dim < static_cast<int>(ndim); ++dim) {
        if (dim == 0) continue; // Skip first dimension
        ASSERT_THROW(shape_[dim] == ref_shape[dim],
                    "All tensors must have same shape except in dim 0");
      }
      total_rows += shape_[0];  
    }
    // Create output shape
    vector<int> out_shape = ref_shape;
    out_shape[0] = total_rows;
    // Concatenate data
    vector<T> out_data;
    // For vstack, we can simply concat all data sequentially
    // since row-major order means rows are contiguous in memory
    for (const auto& tensor : tensors) {
        out_data.insert(out_data.end(), tensor.data_.begin(), tensor.data_.end());
    }
    return Tensor<T>(out_data, out_shape);
  }
  else {throw std::invalid_argument("Currently only support 1D and 2D vstack");}
}


}