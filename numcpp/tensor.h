#pragma once
#include <vector>
#include <iostream>
#include <string>
using std::string;
using std::vector;
using std::cout;
using std::endl;
namespace nc{
template <typename T>
class Tensor{
  public: 
  Tensor(){}
  Tensor(vector<T>& data,const vector<int>&shape);
  Tensor(vector<T>& data,std::initializer_list<int> shape
  ): Tensor(data, vector<int>(shape.begin(), shape.end())){
    // cout<<"with initializer"<<endl;
    /*
    vector<int> zz = zeros_vec(16);
    Tensor v1 {zz,{4,4}};
    */
  };
  T index(vector<int> indices);
  T index(std::initializer_list<int> indices){
      return index(vector<int>(indices)); // Convert and forward
  };
  void info();
  void print();
  string print_recur(int depth, vector<int> index);
  vector<T> data_;  //todo template 
  vector<int> strides_;  
  vector<int> shape_;  
  int ndim_; 
  int size_;
};
}