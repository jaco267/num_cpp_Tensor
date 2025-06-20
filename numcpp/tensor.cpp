#include "tensor.h"
#include "numcpp.h"
#include <sstream>
namespace nc{
//todo  see practice/C/c_libraries/libtorch/PyNorch/b_my$ 
template <typename T>
Tensor<T>::Tensor(vector<T>& data,const vector<int>&shape
  ): shape_(shape),data_(data) {
  ndim_ = shape_.size();
  
  size_ = 1;  
  for (int i =0 ; i<ndim_; i++){
    size_*=shape_[i];
  }
  
  ASSERT_THROW(size_==(int)data_.size(),"data size should equal to multiplication of shape ");
  int stride = 1;  
  strides_.resize(ndim_,0);
  for(int i = ndim_ - 1; i >= 0; i--){
    strides_[i] = stride; stride *= shape_[i];
  }
}
template <typename T>
Tensor<T> Tensor<T>::reshape(vector<int> new_shape){
  int total_elements = size_;
  int minus_count=0;
  int inferred_dim = -100;
  int known_dims_product = 1;
  
  for(int i=0;i<(int)new_shape.size();i++){
    int dim = new_shape[i];
    if (dim==-1){
      minus_count++;
      inferred_dim = dim;
    }else{
      known_dims_product *= dim;
    }
  }
  if (minus_count>1){
     throw std::invalid_argument( "Only one dimension can be inferred (set to -1)" );
  }
  //* Calculate the inferred dimension if -1 is present
  if (inferred_dim == -1){
    int inferred_dim_size= total_elements / known_dims_product; 
    for(int i=0;i<(int)new_shape.size();i++){
      int dim = new_shape[i];
      if (dim == -1){
        new_shape[i] = inferred_dim_size;
      }
    }
  }
  // int new_ndim = (int)new_shape.size(); 

  int size = 1;
  for(int i=0;i<(int)new_shape.size();i++){
    size*=new_shape[i];
  }
  ASSERT_THROW(size==size_,"Cannot reshape tensor. Total number of elements in new shape does not match the current size of the tensor");
  Tensor<T> return_ten {data_,new_shape};
  return return_ten;
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
  const Tensor<T> & in_data){
  ASSERT_THROW(slice_indices.size()==shape_.size(),"mult size should be equal to shape size");
  mat<int> new_slice;   //store the [start:end,start:end..]
  extract_slice_index(slice_indices,new_slice);
  // cout<<"---new_slice---"<<endl;print_mat(new_slice);
  ASSERT_THROW(new_slice.size()==shape_.size(),"currently new shape size must ==shape size");
  vector<int> new_shape;  
  create_new_shape_from_slice(new_slice,new_shape);
  //* self.shape (4,2,3)
  //* new_shape == in_data.shape_ == (2,1,3)
  // cout<<"new_shape:";
  // print_vec(new_shape);print_vec(in_data.shape_);
  ASSERT_THROW(vec_equal(new_shape, in_data.shape_),"put tensor should have same shape as new_shape(currently no broadcast)");

  int new_size = mul_vec(new_shape); //* 2*3 = 6
  vector<T> new_data;
  new_data.reserve(new_size);
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


template <typename T>
void Tensor<T>::info(){
  print();
  cout<<" n_dim:"<<ndim_<<", size:"<<size_<<", strides:";
  print_vec(strides_,0);
  cout<<", shape:"; print_vec(shape_);
}
template <typename T>
void Tensor<T>::print(){
  vector<int> index = zeros_vec<int>(ndim_);
  string result = "tensor([\n";
  result+=print_recur(0,index);
  result+="\n])";
  cout<<result;
}
template <typename T>
string Tensor<T>::print_recur(int depth, vector<int> ind){
  std::ostringstream oss;  // Create a stringstream
  ASSERT_THROW((int)shape_.size()==ndim_,"len(shape)==ndim");
  ASSERT_THROW(shape_.size()==ind.size(),"len(shape)==len(ind)");
  if (depth==ndim_-1){
    for(int i =0; i<shape_[ndim_-1]; i++){
      ind[ndim_-1] = i; 
      oss<<index(ind)<<", ";
    }
    return oss.str(); 
  }else{
    if (depth>0){oss<<"";}
    for(int i=0; i< shape_[depth]; i++){
      ind[depth] = i; 
      if (depth==0 ){
        oss<<"  [";
      }else if(depth==1&&i==0){ //&&(ind[0]==2)
        oss<<"[";
      }else{
        oss<<"   [";
      }
      oss<<print_recur(depth+1,ind)<<"],";
      if (i< shape_[depth]-1){// not the end of this dim   row1[] \n  row2[] \n   
        oss<<"\n";
      }
    }
    return oss.str(); 
  }
  
   throw std::invalid_argument( "should not come here" );
}
template <typename T>
vector<T> Tensor<T>::toVec(){
  ASSERT_THROW(shape_.size()==1," we can only convert 1d tensor to vector");  
  return data_;
}
template <typename T>
mat<T> Tensor<T>::toMat(){
  mat<T> m; 
  ASSERT_THROW(shape_.size()==2," we can only convert 2d tensor to matrix");
  
  const size_t rows = shape_[0];
  const size_t cols = shape_[1]; 
  for (size_t i =0; i< rows; i++){
    vector<T> row;
    row.resize(cols,0); 
    m.push_back(row);
    for (size_t j = 0; j < cols; ++j) {
        m[i][j] = data_[i * strides_[0] + j * strides_[1]];
    }
  }
  return m;
}
//*Explicit template
template class Tensor<int>;
template class Tensor<float>;
template class Tensor<long>;

template class Tensor<unsigned long>;
}