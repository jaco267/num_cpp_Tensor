#include "tensor.h"
#include "numcpp.h"
#include <sstream>
namespace nc{
//*Explicit template
//todo  see practice/C/c_libraries/libtorch/PyNorch/b_my$ 
template <typename T>
Tensor<T>::Tensor(vector<T>& data, const vector<int>&shape
  ){
  init_tensor(data,shape);
}
template <typename T>
void Tensor<T>::init_tensor(vector<T>& data, const vector<int>&shape){
  shape_ = shape;
  data_ = data;
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

}

#include "explicit_init.cpp"