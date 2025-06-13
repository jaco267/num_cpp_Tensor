#pragma once
#include <vector>
#include <iostream>
#include <string>
#include "nc_def.h"

#include "ten_index.h"
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
  void create_new_shape_from_slice(
    const mat<int>& new_slice,
    vector<int>& new_shape){
    for (int i=0; i<new_slice.size(); i++){
      ASSERT_THROW(new_slice[i].size()==2,"slice len should be 2");
      int start = new_slice[i][0];
      int end = new_slice[i][1];
      ASSERT_THROW(end-start>0 && end >=0 && start >=0, "end should > start, and shouldb both >=0");
      new_shape.push_back(end-start);
    }
  }
  void extract_slice_index(
    const vector<nc_Slice_Index>& slice_indices,
    mat<int>& new_slice){
    std::vector<nc_Slice_Index> processed_indices;
    for (const auto& idx : slice_indices) {
        processed_indices.push_back(
            std::visit([](auto&& arg) -> nc_Slice_Index {return arg;
            }, idx)
        );
    }
    // Now print the processed_indices
    int count = 0;
    for (const auto& val : processed_indices) {
        std::visit([&count, &new_slice, this](const auto& v) {
          using T2 = std::decay_t<decltype(v)>;  // Get the actual type
          if constexpr (std::is_same_v<T2, std::monostate>) {
            vector<int> slice_idx = {0,this->shape_[count]};
            new_slice.push_back(slice_idx);
          }else if constexpr (std::is_same_v<T2, int>) {
            vector<int> slice_idx = {v,v+1};
            new_slice.push_back(slice_idx);
          }else if constexpr (std::is_same_v<T2, nc::indexing::Slice>){
            vector<int> start_end_vec=v.start_end_step_vec; 
            vector<int> None_vec=v.None_vec_;  
            ASSERT_THROW(start_end_vec.size() == 3 && None_vec.size()==3 , "size ==3 start end step");
            if (start_end_vec[2] != 1){
               throw std::invalid_argument( "currently Slice only support step==1" );
            }
            vector<int> slice_idx = {0,0};
            if (None_vec[0]==1 && None_vec[1]==1 ){
              slice_idx[1] = this->shape_[count]; //None:None 
            }else if (None_vec[0]==0 && None_vec[1]==1 ){//start:None
              slice_idx[0] = start_end_vec[0]; //start  
              slice_idx[1] = this->shape_[count];  
            }else if (None_vec[0]==1 && None_vec[1]==0){ //[:end]
              slice_idx[1] = start_end_vec[1];
            }else{
              ASSERT_THROW(None_vec[0]==0 && None_vec[1]==0, "should not be None idx");
              slice_idx[0] = start_end_vec[0];
              slice_idx[1] = start_end_vec[1];
            }
            new_slice.push_back(slice_idx);
          } 
          else{
             throw std::invalid_argument( "should be one of (int,Slice,None)" );
          }
          count+=1;
        }, val);
    }
  };
  Tensor<T> index(const vector<nc_Slice_Index>& indices);
  void index_put(const vector<nc_Slice_Index>& indices,
     const Tensor<T> & in_data);
  Tensor<T> slice(int dim, int start, int end);
  void slice_put(int dim, int start, int end, Tensor<T> in_data);
  void info();
  void print();
  string print_recur(int depth, vector<int> index);
  vector<T> toVec();
  mat<T> toMat();
public:
  vector<int> shape_;  
private:
  vector<T> data_;  
  vector<int> strides_;  
  int ndim_; 
  int size_;
};
}

#include "tensor_init.h"