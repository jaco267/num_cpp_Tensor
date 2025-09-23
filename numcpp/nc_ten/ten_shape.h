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
        if (tensor.device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda hstack yet..." );
        }else{ASSERT_THROW(tensor.device_ == "cpu","unknown device\n");}
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
        if (tensor.device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda hstack yet..." );
        }else{ASSERT_THROW(tensor.device_ == "cpu","unknown device\n");}
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
        if (tensor.device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda vstack yet..." );
        }else{ASSERT_THROW(tensor.device_ == "cpu","unknown device\n");}
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
        if (tensor.device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda vstack yet..." );
        }else{ASSERT_THROW(tensor.device_ == "cpu","unknown device\n");}
        out_data.insert(out_data.end(), tensor.data_.begin(), tensor.data_.end());
    }
    return Tensor<T>(out_data, out_shape);
  }
  else {throw std::invalid_argument("Currently only support 1D and 2D vstack");}
}

template <typename T>
Tensor<T> max(const Tensor<T>& input, int dim, bool keepdim = false) {
    if (input.device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda max yet..." );
    }else{ASSERT_THROW(input.device_ == "cpu","unknown device\n");}
    // Validate input dimension
    if (dim < 0 || dim >= static_cast<int>(input.shape_.size())) {
        throw std::invalid_argument("Dimension out of range");
    }
    const auto& shape = input.shape_;
    const auto& data = input.data_;
    // Calculate output shape
    std::vector<int> out_shape;
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    // Special case: if reducing all dimensions (like numpy's max())
    if (shape.size() == 1 && !keepdim) {
        T max_val = *std::max_element(data.begin(), data.end());
        return Tensor<T>({max_val}, {});
    }

    // Calculate strides and sizes for iteration
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }

    int inner_size = 1;
    for (int i = dim + 1; i < static_cast<int>(shape.size()); ++i) {
        inner_size *= shape[i];
    }

    int dim_size = shape[dim];
    int step = inner_size * dim_size;

    // Compute max values
    std::vector<T> out_data;
    out_data.reserve(outer_size * inner_size);

    for (int outer = 0; outer < outer_size; ++outer) {
        for (int inner = 0; inner < inner_size; ++inner) {
            T current_max = data[outer * step + inner];
            
            for (int d = 1; d < dim_size; ++d) {
                T val = data[outer * step + d * inner_size + inner];
                if (val > current_max) {
                    current_max = val;
                }
            }
            
            out_data.push_back(current_max);
        }
    }

    return Tensor<T>(out_data, out_shape);
}
}