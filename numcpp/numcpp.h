#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <bitset>
#include <cstdint>  // For uint32_t, int32_t, etc.
#include <random>
#include <algorithm> //sort
// using namespace std;  //todo wait this is weird if I uncomment this some nasty error occurs
//todo avoid using namespace std, otherwise it will cause conflict between 
using std::vector;
using std::cout;
using std::endl;
#include "nc_def.h"
#include "nc_init.h"
namespace nc{
  template <typename T>
  bool vec_equal(const vector<T>& a, const vector<T>& b){
    ASSERT_THROW(a.size()==b.size(), "two vec should have same size");  
    bool flag = 1;
    for(int i=0; i< a.size(); i++){
      if (a[i]!=b[i]){
        flag = 0;
        break;
      }
    }
    return flag;
  }
  //* https://gist.github.com/HViktorTsoi/58eabb4f7c5a303ced400bcfa816f6f5
  template<typename T>
  std::vector<size_t> argsort(const std::vector<T> &array) {
      std::vector<size_t> indices(array.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(),
                [&array](int left, int right) -> bool {
                    // sort indices according to corresponding array element
                    return array[left] < array[right];
                });
  
      return indices;
  }

  mat<int> kron(const mat<int>& A, const mat<int>& B);
  mat<int> hstack3(const mat<int> & A, const mat<int> & B, const mat<int>& C);
  vector<int> setdiff1d(const vector<int> &input, const vector<int> &diff_list);
  void setdiff1d_rev(vector<int> &ones_rows, vector<int> &fixed_rows_list 
       ,vector<int> &del_rows);
  template <typename T>
  T np_sum(vector<T> v){
    T sum = 0;
    for (unsigned i =0; i< v.size(); i++){
      sum += v[i];
    }
    return sum;
  }
  template <typename T>
  T mul_vec(const vector<T>& v){
    T val = 1; 
    for (unsigned i =0; i< v.size(); i++){
      val *= v[i];
    }
    return val;
  }
  template <typename T>
  void print_mat(vector<vector<T>> G){
      for (int row =0 ; row<G.size(); row++){
          if (row==0){cout<<"[[";
          }else{      cout<<" [";}
          for (int col=0; col<G[0].size(); col++){cout<<G[row][col]<<",";} 
          if (row==G.size()-1){cout<<"]]";
          }else{ cout<<"],";}
          cout<<endl;
      }
  }
  
  template <typename T>
  void print_vec(vector<T> v, int newline=1){
  //   for (int row =0 ; row<v.size(); row++){cout<<v[row]<<" "; } cout<<endl;
    for (const T& elem : v) {cout << elem << " ";}
    if (newline>0){ cout << endl;}
  } 
  int bin_mat_sum(const mat<int> &A);
  mat<int> bin_mat_mul(const mat<int> &A, const mat<int> & B);
  //* -----basic---------
  int bin_to_dec(const vector<int>& bin_list);
  vector<int> dec_to_bin(int n, int x);
  vector<int> mat2comp_vec(const mat<int>& g);
  mat<int> comp_vec2mat(const vector<int> & v,int size_w);
  
  void swap_row(mat<int> &a, int i, int j);
  void add_row(mat<int> &a, int i, int j);
}


#include "nc_rand.h"
#include "tensor.h"