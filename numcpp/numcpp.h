#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <bitset>
#include <cstdint>  // For uint32_t, int32_t, etc.
#include <random>
#include "tensor.h"
#include "nc_def.h"
// using namespace std;  //todo wait this is weird if I uncomment this some nasty error occurs
//todo avoid using namespace std, otherwise it will cause conflict between 
//todo std::nullopt (from <optional>) and c10::nullopt 
using std::vector;
using std::cout;
using std::endl;


#define ASSERT_THROW(cond, msg) if (!(cond)) throw std::runtime_error(msg);
// typedef std::vector<std::vector<int>> mat;
// typedef std::vector<std::vector<float>> matf;

namespace nc{
  extern const uint32_t E2MAX;
  vector<int> zeros_vec(int size);
  template <typename T>
  std::vector<std::vector<T>> zeros_mat(int row, int col) {
      std::vector<std::vector<T>> zeros;
      for (int i = 0; i < row; ++i) {
          std::vector<T> r(col, static_cast<T>(0));
          zeros.push_back(r);
      }
      return zeros;
  }
  template <typename T>
  bool vec_equal(const vector<T>& a, const vector<T>& b);


  mat<int> eye(int k);
  vector<int> arange(int start, int end, int step = 1);
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
  //* ----nc_rand----
  double generate_gaussian_noise(
    std::uniform_int_distribution<uint32_t> & dist,
    std::mt19937& rng, double mean, double stddev);
  vector<double> randn(  std::uniform_int_distribution<uint32_t> & dist,
      std::mt19937& rng, int len_noise, double mean, double stddev);
}

