#include "examples.h"
using std::stoi;
using std::vector;
using std::queue;
using std::cout;
using std::cin;
using std::endl;
using std::string; 
using namespace nc;
void ex0(){ //* random vector
    int seed = 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, nc::E2MAX);
    std::cout << rng() << std::endl;
    std::cout << rng() << std::endl;
    cout<<"e2 max"<<nc::E2MAX<<endl;

    cout<<generate_gaussian_noise(dist, rng, 0, 0.5)<<endl;
    cout<<generate_gaussian_noise(dist, rng, 0, 0.5)<<endl;
}

void ex1(){ //* init vector
  cout<<"zeros vector (3)\n"; 
  vector<int> z_v = zeros_vec<int>(3);  print_vec(z_v);  
  cout<<"ones vector (4)\n"; 
  vector<float> o_fv = ones_vec<float>(4); print_vec(o_fv);
  cout<<"zeros matrix (3,3)"<<endl; // mat<int>is just vector<vector<int>>
  mat<int> z_m = zeros_mat<int>(3,3);
  print_mat(z_m);
  cout<<"arange_vec(1,5,1)"<<endl;
  z_v = arange_vec<int>(1,5,1);
  print_vec(z_v); 
  vector<float> z_f = arange_vec<float> (0,5,1);
  z_f[0] = 0.2;
  print_vec(z_f);
  cout<<"eye(3)"<<endl; 
  z_m = eye<int>(3); 
  print_mat(z_m);
}

void ex2(){
  vector<int> v0 = {0,1,2}; 
  vector<int> v1 = {0,1,2};
  cout<<"vec_equal: "<<vec_equal(v0,v1)<<endl;
  cout<<"zeros matrix (3,3)"<<endl; // mat<int>is just vector<vector<int>>
  mat<int> z_m = zeros_mat<int>(3,3);
  z_m[0][1] = 3; print_mat(z_m); 
  cout<<"---mat2vec---"<<endl; 
  vector<int> z_m_v = mat2vec<int>(z_m);
  print_vec(z_m_v);
  cout<<"---vec2mat---"<<endl;
  print_mat(vec2mat<int>(z_m_v,3,3));
}

void ex3(){
  vector<int> v1 = {3,0,-1, 2,9}; 
  vector<size_t> ind = argsort(v1); 
  cout<<"----argsort---"<<endl;
  print_vec(ind);
  mat<int>F2 = {
    {1,0},
    {1,1}
  };  
  cout<<"----kron----"<<endl;
  mat<int>kern = F2;  
  for (int i =0; i<2-1; i++){
    kern = kron(F2,kern);
  }
  mat<int>F4 = kern; 
  cout<<"F4"<<endl;
  print_mat(F4);
  mat<int>e1 = eye<int>(4);  
  cout<<"---h stack(A,B,C)----"<<endl;  
  mat<int>s0 = hstack3(F4,e1,F4);
  print_mat(s0);
  cout<<"setdiff1d (arange(0,5), [1,3])"<<endl; 
  v1 = {1,3}; 
  vector<int> v2 = arange_vec<int>(0, 5); 
  print_vec(setdiff1d(v2,v1));
  vector<int> out; 
  setdiff1d_rev(v2,v1,out);
  print_vec(out);


}
void ex4(){
  vector<float> a = arange_vec<float>(0,4); 
  vector<float> b = {-0.1, 3.6,2.1, 0.5};
  cout<<"----sum_vec(b)---"<<endl<<sum_vec(b)<<endl;
  cout<<"----sum_vec(a,b)---"<<endl;
  print_vec(sum_vec(a,b));
  cout<<"----mul_vec(b)---"<<endl<<mul_vec(b)<<endl; //-0.1*3.6*2.1*0.5
  cout<<"---sum_mat(e1)---"<<endl;
  mat<int>e1 = eye<int>(4); 
  cout<<sum_mat(e1)<<endl;
  mat<int>res = bin_mat_mul(e1,e1);  
  cout<<"---(e1@e1)%2="<<endl;
  print_mat(res);
}

void ex5(){
  mat<int>e1 = eye<int>(4); 
  cout<<"swap e1 row0 row1"<<endl; 
  swap_row(e1 , 0,1); 
  print_mat(e1);
  cout<<"bin add e1 row 1 += row2"<<endl; 
  add_row(e1, 1,2);
  print_mat(e1);
  //                  8 4 2 1
  vector<int> l0 = {0,1,1,0,1};  
  cout<<"---bin_to_dec"<<endl; 
  print_vec(l0);
  int l0_int = bin_to_dec(l0);
  cout<<"--->"<<l0_int<<endl; 
  vector<int> l2 = dec_to_bin(l0_int, l0.size());
  cout<<"back to vec"<<endl;
  print_vec(l2);
  cout<<"e1 to compressed int vector"<<endl;  
  vector<int> e1_i = mat2comp_vec<int>(e1); 
  print_vec(e1_i);
  cout<<"back to mat"<<endl; 
  print_mat(comp_vec2mat(e1_i, e1.size()));
}

void ex6(){
  cout<<"---unique_(v0)---"<<endl;
  vector<int> v0 = {1,1,0,2,3,3,3,-1,5,2,-1};
  vector<int> v_unique = unique_(v0); 
  
  print_vec(v_unique);
}