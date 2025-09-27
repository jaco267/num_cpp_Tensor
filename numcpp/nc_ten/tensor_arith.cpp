#include "tensor.h"
#include "numcpp.h"
//*defined  in header.h  
#include <cuda_runtime.h>
#include "kern_func.cuh" 
namespace nc{
template <typename T>
Tensor<T> Tensor<T>::add(const Tensor<T>& a){
  ASSERT_THROW(vec_equal(a.shape_, shape_),"add:currently a need to have same shape");
  if (device_ == "cuda"){
    ASSERT_THROW(a.device_ == "cuda","add: tensor should be on same device\n");
    Tensor<T> out;
    cudaMalloc((void**)&out.data_cu_, size_*sizeof(T));
    ///todo weird when change <T> to <int> we got error 
    launch_add_kernel<T>(data_cu_, a.data_cu_, out.data_cu_, size_);
    cudaDeviceSynchronize();
    
    out.shape_ = shape_; 
    out.device_ = "cuda";
    out.strides_ = strides_;
    out.ndim_ = ndim_;
    out.size_ = size_; 
    return out; 
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  ASSERT_THROW(a.data_.size()==data_.size(),"add: data should have same size");
  vector<T> out_v = sum_vec<T>(a.data_, data_);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::add(T a){
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda add yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  vector<T> out_v = sum_vec<T>(data_,a);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::minus() const{
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement minus reshape yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  vector<T> out_v;
  for(int i =0; i<(int)data_.size(); i++){
    out_v.push_back(-data_[i]);
  }
  Tensor<T> out (out_v, shape_);  
  return out; 
}
template <typename T>
Tensor<T> Tensor<T>::mul(const Tensor<T>& a){
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda mul yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  ASSERT_THROW(vec_equal(a.shape_, shape_),"add:currently a need to have same shape");
  ASSERT_THROW(a.data_.size()==data_.size(),"add: data should have same size");
  vector<T> out_v = mul_vec<T>(a.data_, data_);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::mul(T a){
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda mul yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
  vector<T> out_v = mul_vec<T>(data_,a);  
  Tensor<T> out (out_v, shape_); 
  return out;
}
template <typename T>
Tensor<T> Tensor<T>::div1() const{ // dev1(a) = return 1/a 
  if (device_ == "cuda"){ throw std::invalid_argument( "haven't implement cuda div yet..." );
  }else{ASSERT_THROW(device_ == "cpu","unknown device\n");}
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