#pragma once
#include "numcpp.h"
#include "tensor.h"
namespace nc{
template <typename T>
void cpu_to_cuda(Tensor<T>* tensor) { //* will free cpu memery and allocate to new cuda memeory 
    ASSERT_THROW(tensor->device_ == "cpu", "cpu_to_cuda, dev should == cpu\n");
    T* data_tmp;
    cudaMalloc((void **)&data_tmp, tensor->size_ * sizeof(T));
    cudaMemcpy(data_tmp, tensor->data_.data(), tensor->size_ * sizeof(T), cudaMemcpyHostToDevice);
    

    int size_tmp = 1;  
    for (int i =0 ; i<tensor->ndim_; i++){size_tmp*=tensor->shape_[i];}
    ASSERT_THROW(size_tmp == tensor->size_, "size shoould be mul to shape\n");  
    ASSERT_THROW(tensor->data_cu_==nullptr,"data_cu_ should be null ptr\n");
    tensor->data_cu_ = data_tmp;
    tensor->data_.clear();
    tensor->device_ = "cuda";

    cout<<"Successfully sent tensor to: "<< tensor->device_<<endl;
}
template <typename T>
void cuda_to_cpu(Tensor<T>* tensor) { //* will free cuda memery and allocate to new cpu memeory 
    ASSERT_THROW(tensor->device_ == "cuda", "cuda_to_cpu, dev should == cuda\n");
    int size_tmp = 1;  
    for (int i =0 ; i<tensor->ndim_; i++){size_tmp*=tensor->shape_[i];}
    ASSERT_THROW(size_tmp == tensor->size_, "size shoould be mul to shape\n");

    vector<T> data_tmp(tensor->size_);
    cudaMemcpy(data_tmp.data(), tensor->data_cu_, tensor->size_ * sizeof(T), cudaMemcpyDeviceToHost);
    ASSERT_THROW(tensor->data_.size()==0, "data size shoudl be 0\n");
    tensor->data_ = data_tmp;
    cudaFree(tensor->data_cu_);
    tensor->data_cu_ = nullptr;
    tensor->device_ = "cpu";

    cout<<"Successfully sent tensor to: "<< tensor->device_<<endl;
}
}