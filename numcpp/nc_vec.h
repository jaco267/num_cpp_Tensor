#pragma once
#include "nc_def.h"
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
T sum_vec(vector<T> v){
  T sum = 0;
  for (unsigned i =0; i< v.size(); i++){
    sum += v[i];
  }
  return sum;
}
template <typename T>
vector<T> sum_vec(vector<T> a, vector<T> b){
  vector<T> out;
  ASSERT_THROW(a.size()==b.size(), "sum_vec(a,b):two vec should have same size");  
  
  for (unsigned i =0; i< a.size(); i++){
    out.push_back(a[i]+b[i]);
  }
  return out;
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
void print_vec(vector<T> v, int newline=1){
//   for (int row =0 ; row<v.size(); row++){cout<<v[row]<<" "; } cout<<endl;
  cout<<"[";
  for (const T& elem : v) {cout << elem << ",";}
  cout<<"]";
  if (newline>0){ cout << endl;}
} 
}