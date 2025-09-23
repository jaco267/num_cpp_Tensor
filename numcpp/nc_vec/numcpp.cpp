#include "numcpp.h"
namespace nc{
const uint32_t E2MAX = (1ULL << 32) - 1;
mat<int>kron(const mat<int>& A, const mat<int>& B) {
    size_t a_rows = A.size();
    size_t a_cols = A[0].size();
    size_t b_rows = B.size();
    size_t b_cols = B[0].size();
    mat<int>result(a_rows * b_rows, std::vector<int>(a_cols * b_cols));
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < a_cols; ++j) {
            for (size_t bi = 0; bi < b_rows; ++bi) {
                for (size_t bj = 0; bj < b_cols; ++bj) {
                    result[i * b_rows + bi][j * b_cols + bj] = A[i][j] * B[bi][bj];
                }
            }
        }
    }
    return result;
}

mat<int>hstack3(const mat<int>& A, const mat<int>& B, const mat<int>& C) {
  mat<int>result;
  for (size_t i = 0; i < A.size(); ++i) {
      vector<int> row = A[i];
      row.insert(row.end(), B[i].begin(), B[i].end());
      row.insert(row.end(), C[i].begin(), C[i].end());
      result.push_back(row);
  }
  return result;
}
vector<int> setdiff1d(const vector<int> &input, const vector<int> &diff_list){
  vector<int> ret_rows;
  bool flag; 
  for(unsigned i = 0; i<input.size(); i++){
      flag = true; 
      for (unsigned j = 0;j<diff_list.size(); j++){
        if(input[i] == diff_list[j]){flag = false;}
      }
      if (flag){
        ret_rows.push_back(input[i]);
      }
  }
  return ret_rows;
}
void setdiff1d_rev(vector<int> &ones_rows, vector<int> &fixed_rows_list 
     ,vector<int> &del_rows){
  bool flag; 
  for(int i = (int) ones_rows.size()-1; i>=0; i--){
      flag = true; 
      for (unsigned j = 0;j<fixed_rows_list.size(); j++){
        if(ones_rows[i] == fixed_rows_list[j]){flag = false;}
      }
      if (flag){
        del_rows.push_back(ones_rows[i]);
      }
  }
}

int sum_mat(const mat<int>&A){
  int sum_val = 0;  
  for (auto row : A){
    for (int val : row){
      sum_val += val; 
    }
  } 
  return sum_val;
}
mat<int>bin_mat_mul(const mat<int>&A, const mat<int>& B){
  mat<int>out;  
  ASSERT_THROW (A.size()>0 && B.size()>0 , " mat<int>should have row num > 0");
  int m = A.size();  
  int n = A[0].size();  
  ASSERT_THROW ((int) B.size()==n , "2nd dim of A should be same as 1st dim in B in bin mat_mul");
  // (m,n)@(n,k) --> out m,k
  int k = (int) B[0].size();  
  for(int r = 0; r<m; r++){
    vector<int> row;  
    for(int c = 0; c<k; c++){
      int val = 0;  
      for(int j = 0; j<n; j++){
        val += (A[r][j]* B[j][c]);
      }
      row.push_back(val%2);  
    }
    out.push_back(row);
  }
  return out;
}
//* -------basic-----------
int bin_to_dec(const std::vector<int>& bin_list) {
  int dec = 0;
  int n = (int) bin_list.size();
  for (int i = 0; i < n; ++i) {
      dec += bin_list[i] * std::pow(2, n - 1 - i);
  }
  return dec;
}
vector<int> dec_to_bin(int n, int x){
  std::string binary = std::bitset<32>(n).to_string(); // Convert to 32-bit binary string
  binary = binary.substr(32 - x); // Keep only the last x bits
  std::vector<int> result;
  for (char c : binary) {
      result.push_back(c - '0'); // Convert char to int
  }
  return result;
}
mat<int>comp_vec2mat(const vector<int> & v,int size_w){
  mat<int>g;  
  for(unsigned i=0; i < v.size(); i ++){
    vector<int> row; 
    for(int j=0; j < size_w; j++ ){
      row.push_back((v[i]>>j)&1);
    }
    g.push_back(row);
  }
  return g;
}
void swap_row(mat<int>&a, int i, int j){
  vector<int> tmp = a[i];  
  a[i] = a[j];  a[j] = tmp;
}
void add_row(mat<int>&a, int i, int j){
  for (unsigned col_i =0; col_i < a[0].size(); col_i ++){
    a[i][col_i] ^= a[j][col_i];
  } 
}
  

}

