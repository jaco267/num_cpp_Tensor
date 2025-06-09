#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <bitset>
#include <cstdint>  // For uint32_t, int32_t, etc.
#include <random>

using std::vector;
using std::cout;
using std::endl;


#define ASSERT_THROW(cond, msg) if (!(cond)) throw std::runtime_error(msg);
typedef std::vector<std::vector<int>> mat;
namespace nc{
  extern const uint32_t E2MAX;
  vector<int> zeros_vec(int size);
  mat zeros_mat(int row, int col);
  mat eye(int k);
  vector<int> arange(int start, int end, int step = 1);
  mat kron(const mat& A, const mat& B);
  mat hstack3(const mat & A, const mat & B, const mat& C);
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
  void print_vec(vector<T> v){
  //   for (int row =0 ; row<v.size(); row++){cout<<v[row]<<" "; } cout<<endl;
    for (const T& elem : v) {cout << elem << " ";}cout << endl;
  } 
  int bin_mat_sum(const mat &A);
  mat bin_mat_mul(const mat &A, const mat & B);
  //* -----basic---------
  int bin_to_dec(const vector<int>& bin_list);
  vector<int> dec_to_bin(int n, int x);
  vector<int> mat2comp_vec(const mat& g);
  mat comp_vec2mat(const vector<int> & v,int size_w);
  
  void swap_row(mat &a, int i, int j);
  void add_row(mat &a, int i, int j);
  //* ----nc_rand----
  double generate_gaussian_noise(
    std::uniform_int_distribution<uint32_t> & dist,
    std::mt19937& rng, double mean, double stddev);
}

