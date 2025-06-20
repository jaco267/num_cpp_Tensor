#pragma once
#include <vector>
#include <iostream>
#include <cmath>
#include <bitset>
#include <cstdint>  // For uint32_t, int32_t, etc.
#include <random>
#include <algorithm> //sort
#include <map>
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
    for(int i=0; i< (int) a.size(); i++){
      if (a[i]!=b[i]){
        flag = 0;
        break;
      }
    }
    return flag;
  }
  template <typename T>
  vector<T> mat2vec(const mat<T>& m){
    vector<T> v;
    if (m.size()==0){return v;}
    for(int r=0;r<m.size(); r++){
      for(int c=0; c<m[0].size();c++){
        v.push_back(m[r][c]);
      }
    }
    return v;
  }
  template <typename T>
  mat<T> vec2mat(const vector<T>& v, int row, int col){
    mat<T> m;
    if (v.size()==0){return m;}
    for(int r=0;r<row; r++){
      vector<T> row_vector;
      for(int c=0; c<col;c++){
        row_vector.push_back(v[r*col+c]);
      }
      m.push_back(row_vector);
    }
    return m;
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
      for (int row =0 ; row<(int)G.size(); row++){
          if (row==0){cout<<"[[";
          }else{      cout<<" [";}
          for (int col=0; col<(int)G[0].size(); col++){cout<<G[row][col]<<",";} 
          if (row==(int)G.size()-1){cout<<"]]";
          }else{ cout<<"],";}
          cout<<endl;
      }
  }
  
  template <typename T>
  void print_vec(vector<T> v, int newline=1){
  //   for (int row =0 ; row<v.size(); row++){cout<<v[row]<<" "; } cout<<endl;
    cout<<"[";
    for (const T& elem : v) {cout << elem << ",";}
    cout<<"]";
    if (newline>0){ cout << endl;}
  } 
  int bin_mat_sum(const mat<int> &A);
  mat<int> bin_mat_mul(const mat<int> &A, const mat<int> & B);
  //* -----basic---------
  int bin_to_dec(const vector<int>& bin_list);
  vector<int> dec_to_bin(int n, int x);
  vector<int> mat2comp_vec(const mat<int>& g);
  
  template <typename T>
  vector<T> mat2comp_vec_t(const mat<T>& g){
    vector<T> g_list;  
    for (auto row : g){
      T val = 0;  
      for (unsigned i =0; i< row.size(); i++){
          if (row[i] == 1){
              val += std::pow(2,i);
          }
      }
      g_list.push_back(val);
    }
    return g_list;
  }

  mat<int> comp_vec2mat(const vector<int> & v,int size_w);
  
  void swap_row(mat<int> &a, int i, int j);
  void add_row(mat<int> &a, int i, int j);
//*----others 
//*https://stackoverflow.com/questions/70868307/c-equivalent-of-numpy-unique-on-stdvector-with-return-index-and-return-inver
template <typename Container>
auto unique_map( const Container & xs )
-> std::map <typename Container::value_type, std::vector <std::size_t> >{
  decltype(unique_map(xs)) S;
  std::size_t n = 0;
  for (const auto & x : xs){
    S[ x ].push_back( n++ );
  }
  return S;
}
template <typename Container>
Container unique_(const Container & in ){
  Container out;
  auto ys = unique_map(in);
  for (auto y : ys) {
    out.push_back(y.first);
  }
  return out;
}
}


#include "nc_rand.h"
#include "tensor.h"