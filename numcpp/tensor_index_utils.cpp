#include "tensor.h"
#include "numcpp.h"

namespace nc{
//*Explicit template
template <typename T>
void Tensor<T>::create_new_shape_from_slice(
  const mat<int>& new_slice,
  vector<int>& new_shape){
  for (int i=0; i<(int)new_slice.size(); i++){
    ASSERT_THROW(new_slice[i].size()==2,"slice len should be 2");
    int start = new_slice[i][0];
    int end = new_slice[i][1];
    ASSERT_THROW(end-start>0 && end >=0 && start >=0, "end should > start, and shouldb both >=0");
    new_shape.push_back(end-start);
  }
}
template <typename T>
void Tensor<T>::extract_slice_index(
  const vector<nc_Slice_Index>& slice_indices,
  mat<int>& new_slice){
  std::vector<nc_Slice_Index> processed_indices;
  for (const auto& idx : slice_indices) {
      processed_indices.push_back(
          std::visit([](auto&& arg) -> nc_Slice_Index {return arg;
          }, idx)
      );
  }
  // cout<<"exitract"<<endl;
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
}
template <typename T>
T Tensor<T>::index(vector<int> indices){
  int index = 0;  
  for (int i=0; i<ndim_; i++){
      index+= indices[i]*strides_[i];  
  }
  T result;  
  result = data_[index];  
  return result;
}
template <typename T>
Tensor<T> Tensor<T>::index(const vector<nc_Slice_Index>& slice_indices){
  ASSERT_THROW(slice_indices.size()==shape_.size(),"mult size should be equal to shape size");
  mat<int> new_slice;   //store the [start:end,start:end..]
  extract_slice_index(slice_indices,new_slice);
  // cout<<"---new_slice---"<<endl;print_mat(new_slice);
  ASSERT_THROW(new_slice.size()==shape_.size(),"currently new shape size must ==shape size");
  vector<int> new_shape;  
  create_new_shape_from_slice(new_slice,new_shape);
  // cout<<"new shape:"; print_vec(new_shape);
  int new_size = mul_vec(new_shape); 
  vector<T> new_data;
  new_data.reserve(new_size);
  vector<int> indices(shape_.size(), 0);//* indices for current iteration's row and col 
  ASSERT_THROW(shape_.size()==new_slice.size(),"currently shape must equal otherwise original idx will have error");
  for (size_t i = 0; i <  (size_t) new_size; ++i){// Iterate through all elements and copy the ones in the slice
    // Calculate the original tensor's index
    size_t original_idx = 0;
    for (int d = 0; d < (int) shape_.size(); ++d) {
      int start = new_slice[d][0];
      original_idx += (indices[d] + start) * strides_[d];
    }
    // Add the element to new storage
    new_data.push_back(data_[original_idx]);
    // Update indices (row-major order)  indices 000 -> 001  -> 010 -> ...
    for (int d = shape_.size() - 1; d >= 0; --d) {
        indices[d]++;
        if (indices[d] < new_shape[d]) {break;}
        indices[d] = 0;
    }
  }
  ASSERT_THROW((int)new_data.size()==new_size,"new_data.size()==new_size");
  Tensor<T> v0 {new_data, new_shape};
  return v0;
}
template <typename T>
void Tensor<T>::index_put(
  const vector<nc_Slice_Index>& slice_indices,
  T  in_data
){
  ASSERT_THROW(slice_indices.size()==shape_.size(),"mult size should be equal to shape size");
  mat<int> new_slice;   //store the [start:end,start:end..]
  extract_slice_index(slice_indices,new_slice);
  ASSERT_THROW(new_slice.size()==shape_.size(),"currently new shape size must ==shape size");
  vector<int> new_shape;  
  create_new_shape_from_slice(new_slice,new_shape);
  int new_size = mul_vec(new_shape); //* 2*3 = 6
  vector<int> indices(shape_.size(), 0);//* indices for current iteration's row and col 
  ASSERT_THROW(shape_.size()==new_slice.size(),"currently shape must equal otherwise original idx will have error");
  for (size_t i = 0; i < (size_t) new_size; ++i){
     // Calculate the original tensor's index
    size_t original_idx = 0;
    for (int d = 0; d < (int) shape_.size(); ++d) {
      int start = new_slice[d][0];
      original_idx += (indices[d] + start) * strides_[d];
    }
    // update storage with input Tensor's value
    data_[original_idx] = in_data;
    // Update indices (row-major order)  indices 000 -> 001  -> 010 -> ...
    for (int d = shape_.size() - 1; d >= 0; --d) {
        indices[d]++;
        if (indices[d] < new_shape[d]) {break;}
        indices[d] = 0;
    }
  }
}
template <typename T>
void Tensor<T>::index_put(
  const vector<nc_Slice_Index>& slice_indices,
  const Tensor<T> & in_data
){
  ASSERT_THROW(slice_indices.size()==shape_.size(),"mult size should be equal to shape size");
  mat<int> new_slice;   //store the [start:end,start:end..]
  extract_slice_index(slice_indices,new_slice);
  ASSERT_THROW(new_slice.size()==shape_.size(),"currently new shape size must ==shape size");
  vector<int> new_shape;  
  create_new_shape_from_slice(new_slice,new_shape);
  ASSERT_THROW(vec_equal(new_shape, in_data.shape_),"put tensor should have same shape as new_shape(currently no broadcast)");
  int new_size = mul_vec(new_shape); //* 2*3 = 6
  vector<int> indices(shape_.size(), 0);//* indices for current iteration's row and col 
  ASSERT_THROW(shape_.size()==new_slice.size(),"currently shape must equal otherwise original idx will have error");
  vector<int> in_indices(new_shape.size(),0);
  for (size_t i = 0; i < (size_t) new_size; ++i){
     // Calculate the original tensor's index
    size_t original_idx = 0;
    for (int d = 0; d < (int) shape_.size(); ++d) {
      int start = new_slice[d][0];
      original_idx += (indices[d] + start) * strides_[d];
    }
    size_t in_idx = 0;
    for (int d = 0; d < (int) new_shape.size(); ++d){
      in_idx += in_indices[d]*in_data.strides_[d];
    }
    // update storage with input Tensor's value
    data_[original_idx] = in_data.data_[in_idx];
    // Update indices (row-major order)  indices 000 -> 001  -> 010 -> ...
    for (int d = shape_.size() - 1; d >= 0; --d) {
        indices[d]++;
        if (indices[d] < new_shape[d]) {break;}
        indices[d] = 0;
    }
    for (int d = new_shape.size() - 1; d >= 0; --d) {
        in_indices[d]++;
        if (in_indices[d] < new_shape[d]) {break;}
        in_indices[d] = 0;
    }
  }
}
template <typename T>
Tensor<T> Tensor<T>::slice(int dim, int start, int end){
  // check for valid dimension 
  if (dim<0 || dim >= (int) shape_.size()){throw std::out_of_range( "Dim out of range" );}
  // check for valid start/end indices 
  if (start<0 ||end < 0|| end > shape_[dim]||start>=end){throw std::out_of_range("Invalid start or end index");}
  // Create a copy of the current shape and modify the specified dimension
  vector<int> new_shape = shape_;
  new_shape[dim] = end - start;
  int new_size = mul_vec(new_shape);// total elements in new tensor
  vector<T> new_data; new_data.reserve(new_size);   // Create new storage
  // Create index vector for iteration
  vector<int> indices(shape_.size(), 0);  //* indices for current iteration's row and col 
  //*ex. shape(2,2) dim 1 start 0 end 1   stride  = 2,1  -> original idx = 0 : d=0   0* 
  for (size_t i = 0; i < (size_t) new_size; ++i) {// Iterate through all elements and copy the ones in the slice
      // Calculate the original tensor's index
      size_t original_idx = 0;
      for (int d = 0; d < (int) shape_.size(); ++d) {
          if (d == dim) { original_idx += (indices[d] + start) * strides_[d];
          } else {        original_idx += indices[d] * strides_[d];}
      }
      // Add the element to new storage
      new_data.push_back(data_[original_idx]);
      // Update indices (row-major order)  indices 000 -> 001  -> 010 -> ...
      for (int d = shape_.size() - 1; d >= 0; --d) {
          indices[d]++;
          if (indices[d] < new_shape[d]) {break;}
          indices[d] = 0;
      }
  }
  ASSERT_THROW((int)new_data.size()==new_size,"new_data.size()==new_size");
  Tensor<T> v0 {new_data, new_shape};
  return v0;
}  
template <typename T>
void Tensor<T>::slice_put(int dim, int start, int end, Tensor<T> in_data){
  // check for valid dimension 
  if (dim<0 || dim >= (int)shape_.size()){throw std::out_of_range( "Dim out of range" );}
  // check for valid start/end indices 
  if (start<0 ||end < 0|| end > shape_[dim]||start>=end){throw std::out_of_range("Invalid start or end index");}
  // Create a copy of the current shape and modify the specified dimension
  vector<int> new_shape = shape_;
  new_shape[dim] = end - start;
  // Calculate total elements in the new tensor
  size_t new_size = 1;
  for (int s : new_shape) { new_size *= s;}
  ASSERT_THROW (vec_equal(in_data.shape_, new_shape),"new shape should be same shape as data");  
  // Create index vector for iteration
  vector<int> indices(shape_.size(), 0);  //* indices for current iteration's row and col 
  //*ex. shape(2,2) dim 1 start 0 end 1   stride  = 2,1  -> original idx = 0 : d=0   0* 
  vector<int> in_indices(new_shape.size(),0);
  for (size_t i = 0; i < new_size; ++i) {// Iterate through all elements and copy the ones in the slice
    // Calculate the original tensor's index
    size_t original_idx = 0;
    for (int d = 0; d < (int) shape_.size(); ++d) {
        if (d == dim) { original_idx += (indices[d] + start) * strides_[d];
        } else {        original_idx += indices[d] * strides_[d];}
    }
    size_t in_idx = 0;
    for (int d = 0; d < (int) new_shape.size(); ++d){
      in_idx += in_indices[d]*in_data.strides_[d];
    }
    // update storage with input Tensor's value
    data_[original_idx] = in_data.data_[in_idx];
    // Update indices (row-major order)  indices 000 -> 001  -> 010 -> ...
    for (int d = shape_.size() - 1; d >= 0; --d) {
        indices[d]++;
        if (indices[d] < new_shape[d]) {break;}
        indices[d] = 0;
    }
    for (int d = new_shape.size() - 1; d >= 0; --d) {
        in_indices[d]++;
        if (in_indices[d] < new_shape[d]) {break;}
        in_indices[d] = 0;
    }
  }
}
}
#include "explicit_init.cpp"