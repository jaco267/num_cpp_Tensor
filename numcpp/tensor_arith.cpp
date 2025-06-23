#include "tensor.h"
#include "numcpp.h"

namespace nc{
template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& a){
  ASSERT_THROW(vec_equal(a.shape_, shape_),"add:currently a need to have same shape");
  ASSERT_THROW(a.data_.size()==data_.size(),"add: data should have same size");
  vector<T> out_v = sum_vec<T>(a.data_, data_);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::add(T a){
  vector<T> out_v = sum_vec<T>(data_,a);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::minus() const{
  vector<T> out_v;
  for(int i =0; i<(int)data_.size(); i++){
    out_v.push_back(-data_[i]);
  }
  Tensor<T> out (out_v, shape_);  
  return out; 
}
template <typename T>
Tensor<T> Tensor<T>::mul(const Tensor<T>& a){
  ASSERT_THROW(vec_equal(a.shape_, shape_),"add:currently a need to have same shape");
  ASSERT_THROW(a.data_.size()==data_.size(),"add: data should have same size");
  vector<T> out_v = mul_vec<T>(a.data_, data_);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::mul(T a){
  vector<T> out_v = mul_vec<T>(data_,a);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::div1() const{ // dev1(a) = return 1/a 
  vector<T> out_v;
  for(int i =0; i<(int)data_.size(); i++){
    if (data_[i] == 0){
       throw std::invalid_argument( "shouldn't div by zero" );
    }
    out_v.push_back(1/data_[i]);
  }
  Tensor<T> out (out_v, shape_);  
  return out; 
}
}

#include "explicit_init.cpp"