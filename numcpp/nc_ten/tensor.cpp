#include "tensor.h"
#include "numcpp.h"
#include <cuda_runtime.h>
#include <sstream>
namespace nc{
//*Explicit template
//todo  see practice/C/c_libraries/libtorch/PyNorch/b_my$ 
template <typename T>
Tensor<T>::Tensor(const vector<T>& data, const vector<int>&shape,string device
  ){
  init_tensor(data,shape,device);
}
template <typename T>
void Tensor<T>::init_tensor(const vector<T>& data, const vector<int>&shape,
  string device
){
  device_ = device;  
  
  shape_ = shape;
  if(device == "cpu"){
    data_ = data;
  }else if (device == "cuda"){
    cudaMalloc((void**)&data_cu_, data.size() * sizeof(T));
    // Copy data from host vector to device buffer
    cudaMemcpy(data_cu_,data.data(),data.size()*sizeof(T), cudaMemcpyHostToDevice);
  }else{throw std::invalid_argument( "unknown device" );}
  ndim_ = shape_.size();
  size_ = 1;  
  for (int i =0 ; i<ndim_; i++){size_*=shape_[i];}
  
  ASSERT_THROW(size_==(int)data.size(),"data size should equal to multiplication of shape ");
  int stride = 1;  
  strides_.resize(ndim_,0);
  for(int i = ndim_ - 1; i >= 0; i--){  strides_[i] = stride; stride *= shape_[i];}
}
template <typename T>
void Tensor<T>::clear_cu(){
  ASSERT_THROW(device_=="cuda","clear cu dev should be cuda\n"); 
  if(data_cu_){
    cudaFree(data_cu_); 
    data_cu_ = nullptr;  
    size_ = 0; 
    shape_.clear(); 
    ASSERT_THROW(data_.size()==0, "data size should be 0\n"); 
    strides_.clear(); 
    ndim_=0; 
  }
}
template <typename T>
void Tensor<T>::to(string device){
  ASSERT_THROW((device=="cpu")||(device=="cuda"),"dev should be cpu or cuda\n");
  ASSERT_THROW((device_=="cpu")||(device_=="cuda"),"dev should be cpu or cuda\n");
  if( ((device_=="cpu")&&(device=="cpu"))||
      ((device_=="cuda")&&(device=="cuda"))
    ){return;}
  if((device_=="cpu")&&(device=="cuda")){
    cpu_to_cuda(this);
    return;  
  }
  if((device_=="cuda")&&(device=="cpu")){
    cuda_to_cpu(this);
    return;  
  }
   throw std::invalid_argument( "shouldn't be in herer" );
}
//*------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::reshape(vector<int> new_shape){
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda reshape yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
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
void Tensor<T>::info() {
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda info yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  print();
  cout<<" n_dim:"<<ndim_<<", size:"<<size_<<", strides:";
  print_vec(strides_,0);
  cout<<", shape:"; print_vec(shape_);
}
template <typename T>
void Tensor<T>::print() {
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda print yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  vector<int> index = zeros_vec<int>(ndim_);
  string result = "tensor([\n";
  result+=print_recur(0,index);
  result+="\n])";
  cout<<result;
}
template <typename T>
string Tensor<T>::print_recur(
  int depth, vector<int> ind) {
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda print_recur yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
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
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda toVec yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  ASSERT_THROW(shape_.size()==1," we can only convert 1d tensor to vector");  
  return data_;
}
template <typename T>
mat<T> Tensor<T>::toMat(){
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda toMat yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
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
template <typename T>
void Tensor<T>::fromMat(const mat<T>& m){
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda fromMat yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  if (m.size()==0){
     throw std::invalid_argument( "mat_size should > 0" );
  }
  int row_size =(int) m.size();
  int col_size =(int) m[0].size(); 
  vector<int> shape = {row_size,col_size}; 
  vector<T> v = mat2vec<T>(m); 
  init_tensor(v, shape);
}
}

#include "explicit_init.cpp"