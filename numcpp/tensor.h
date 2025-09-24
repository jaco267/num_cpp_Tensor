#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "nc_def.h"
#include "nc_ten/tensor_index.h"
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
  Tensor():data_cu_(nullptr),device_("cpu"),ndim_(0),size_(0){}
  Tensor(const vector<T>& data,const vector<int>&shape, string device = "cpu");//*Tensor(vec,vec) important
  Tensor(std::initializer_list<T> data,std::initializer_list<int> shape, string device = "cpu"
  ): Tensor(vector<T>(data.begin(), data.end()), 
            vector<int>(shape.begin(), shape.end()),
            device
           ){}; //* just call Tensor(vec,vec)
  Tensor(vector<T>& data,std::initializer_list<int> shape, string device = "cpu"
  ): Tensor(data, vector<int>(shape.begin(), shape.end()),
            device
           ){};//*just call Tensor(vec,vec)
  void init_tensor(const vector<T>& data, const vector<int>&shape, string device="cpu");
  //*-------------------------------------------------------------------------
  ~Tensor(){  //* destructor  clear cuda memory if device is cuda  //* so it works like RAII wrapper class, (ex vector class)
    if(device_ == "cuda"){clear_cu();
      cout<<"Free cuda memory in tensor destructor"<<endl;
    }// cout<<"todo cudaFree"<<endl; 
  }
  void clear_cu();
  void to(string device);
  //*------------------------------
  Tensor(const Tensor<T>& other) : shape_(other.shape_),data_(other.data_), device_(other.device_),
    strides_(other.strides_),ndim_(other.ndim_), size_(other.size_)
  { // copy CPU data //*copy constructor  ex. Tensor<float> a = b; 
      // cout<<"copy construct  haven't test yet, will vector<T> work?"<<endl;
      if (other.device_ == "cuda" && other.data_cu_ != nullptr) {// allocate GPU memory and copy
          cudaMalloc((void**)&data_cu_, size_*sizeof(T));
          cudaMemcpy(data_cu_, other.data_cu_, size_*sizeof(T), cudaMemcpyDeviceToDevice);
      } else { data_cu_ = nullptr;} // if CPU, or no data
  }
  
  Tensor<T>& operator=(const Tensor<T>& other) {// Copy assignment //*ex Tensor<float> c; c= a;
    cout<<"carefull haven't test yet, will vector<T> work?"<<endl;
    if (this == &other) return *this; // self-assignment check
    // Free old CUDA memory if needed
    if (device_ == "cuda" && data_cu_ != nullptr) { cudaFree(data_cu_);}
    // Copy metadata and CPU data
    shape_ = other.shape_; strides_ = other.strides_; ndim_ = other.ndim_;   
    size_ = other.size_;   device_ = other.device_;   data_ = other.data_;
    // Copy CUDA data if needed
    if (other.device_ == "cuda" && other.data_cu_ != nullptr) {
        cudaMalloc((void**)&data_cu_, size_ * sizeof(T));
        cudaMemcpy(data_cu_, other.data_cu_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {   data_cu_ = nullptr;}
    return *this;
  }
  
  Tensor<T>& operator=(Tensor<T>&& other) noexcept {//? Move assignment
    //?ex Tensor<float> d = std::move(a);
    cout<<"carefull haven't test yet, will vector<T> work?"<<endl;
    if (this != &other) {
        if (device_ == "cuda" && data_cu_ != nullptr) {
            cudaFree(data_cu_);
        }
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        ndim_ = other.ndim_;
        size_ = other.size_;
        device_ = std::move(other.device_);
        data_ = std::move(other.data_);
        data_cu_ = other.data_cu_;
        other.data_cu_ = nullptr;
    }
    return *this;
  }
  //*----------------------------
  Tensor<T> reshape(vector<int> new_shape);
  Tensor<T> reshape(std::initializer_list<int> new_shape){
    return reshape(vector<int>(new_shape));
  };
  void info(bool verbose = true) ;
  friend std::ostream& operator<<(std::ostream& os, Tensor& ten) {ten.info(); return os;}
  void print() ;
  string print_recur(int depth, vector<int> index) ;
  vector<T> toVec();
  mat<T> toMat();
  void fromMat(const mat<T>& m);
  //*-----tensor_arith.cpp----
  Tensor<T> add(const Tensor<T>& a);
  Tensor<T> add(T a);
  Tensor<T> minus() const;
  Tensor<T> minus(const Tensor<T>& a){return add(a.minus());}
  Tensor<T> minus(T a){return add(-a);}
  Tensor<T> mul(const Tensor<T>& a);
  Tensor<T> mul(T a);
  Tensor<T> div1() const;
  Tensor<T> div(const Tensor<T>& a){return mul(a.div1()); }
  Tensor<T> div(T a){if (a==0){ throw std::invalid_argument( "div by zero error" );}
    return mul(1/a); 
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
     T in_data);
  void index_put(const vector<nc_Slice_Index>& indices,
     const Tensor<T> & in_data);
  Tensor<T> slice(int dim, int start, int end);
  void slice_put(int dim, int start, int end, Tensor<T> in_data);
  //*-----------------------
public:
  vector<int> shape_;  

  vector<T> data_; 
  T*  data_cu_; 
  string device_; //todo
// private: 
  vector<int> strides_;  
  int ndim_; 
  int size_;
};
}

#include "nc_ten/ten_init.h"
#include "nc_ten/ten_cu.h"
#include "nc_ten/ten_shape.h"
#include "nc_ten/ten_print.h"