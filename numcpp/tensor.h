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
  void fromMat(const mat<T>& m){
    if (m.size()==0){
       throw std::invalid_argument( "mat_size should > 0" );
    }
    int row_size =(int) m.size();
    int col_size =(int) m[0].size(); 
    vector<int> shape = {row_size,col_size}; 
    vector<T> v = mat2vec<T>(m); 
    init_tensor(v, shape);

  }
  Tensor<T> add(const Tensor<T>& a){
    
    ASSERT_THROW(vec_equal(a.shape_, shape_),"add:currently a need to have same shape");
    ASSERT_THROW(a.data_.size()==data_.size(),"add: data should have same size");
    vector<T> out_v = sum_vec<T>(a.data_, data_);  
    Tensor<T> out (out_v, shape_); 
    return out;
  }
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