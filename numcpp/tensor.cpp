#include "tensor.h"
#include "numcpp.h"
#include <sstream>
namespace nc{
//todo  see practice/C/c_libraries/libtorch/PyNorch/b_my$ 
template <typename T>
Tensor<T>::Tensor(vector<T>& data,const vector<int>&shape
  ): data_(data), shape_(shape){
  ndim_ = shape_.size();
  
  size_ = 1;  
  for (int i =0 ; i<ndim_; i++){
    size_*=shape_[i];
  }
  
  ASSERT_THROW(size_==data_.size(),"data size should equal to multiplication of shape ");
  int stride = 1;  
  strides_.resize(ndim_,0);
  for(int i = ndim_ - 1; i >= 0; i--){
    strides_[i] = stride; stride *= shape_[i];
  }
  info();
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
void Tensor<T>::info(){
  cout<<"shape:";
  print_vec(shape_);
  cout<<"n_dim:"<<ndim_<<endl;
  cout<<"size:"<<size_<<endl;
  cout<<"strides:";
  print_vec(strides_);
  cout<<"data:"<<endl;
  print();
}
template <typename T>
void Tensor<T>::print(){
  vector<int> index = zeros_vec(ndim_);
  cout<<"index:"; 
  print_vec(index);
  string result = "tensor([\n";
  result+=print_recur(0,index);
  result+="\n])";
  cout<<result<<endl;

}
template <typename T>
string Tensor<T>::print_recur(int depth, vector<int> ind){
  std::ostringstream oss;  // Create a stringstream
  ASSERT_THROW(shape_.size()==ndim_,"len(shape)==ndim");
  ASSERT_THROW(shape_.size()==ind.size(),"len(shape)==len(ind)");
  if (depth==ndim_-1){
    for(int i =0; i<shape_[ndim_-1]; i++){
      ind[ndim_-1] = i; 
      oss<<index(ind)<<", ";
    }
    return oss.str(); 
  }else{
    if (depth>0){oss<<"\n";}
    for(int i=0; i< shape_[depth]; i++){
      ind[depth] = i; 
      oss<<"[";
      oss<<print_recur(depth+1,ind)<<"],";
      if (i< shape_[depth]-1){// not the end of this dim   row1[] \n  row2[] \n   
        oss<<"\n";
      }
    }
    return oss.str(); 
  }
  
   throw std::invalid_argument( "should not come here" );
}

// Explicit instantiations (only these types will work)
template class Tensor<int>;
template class Tensor<float>;
}