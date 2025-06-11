#include <iostream>
#include <vector>
#include <queue>    //* queue<int> q; //FIFO
#include <algorithm>  //*generate,sort,begin,end, reverse, max,min 
#include <functional>  //*std::function,placeholders::_1
#include <memory>   //* shared_ptr
#include "numcpp.h"
//* ---pyrallis--- 
#include <stdio.h> 
#include <unistd.h> 
#include <bits/getopt_core.h>
#include <filesystem>
#include <string>
#include <iostream>
#include <random>
#include <iomanip>
#include "numcpp.h"

#define GET_VARIABLE_NAME(Variable) (#Variable)
using std::stoi;
using std::vector;
using std::queue;
using std::cout;
using std::cin;
using std::endl;
using std::string; 
using namespace nc;
const uint32_t E2MAX = (1ULL << 32) - 1;
//*./main -i -o file.txt -lr -x 'hero'
int main(int argc, char *argv[]){
  int opt; 
  int run_opt = 0;
  int mc_iter; 
  bool perf = true;
  while((opt = getopt(argc, argv, ":i:o:p:r:k:x")) != -1) { 
    switch(opt) { 
      case 'o':  run_opt = stoi(optarg);  break; 
      case ':': printf("option needs a value\n"); break; 
      case '?': printf("unknown option: %c\n", optopt); break; 
    } 
  } 
  for(; optind < argc; optind++){	 
      printf("extra arguments: %s\n", argv[optind]); 
  } 
  //*-------------------------
  if (run_opt==0){
    int seed = 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, nc::E2MAX);
    std::cout << rng() << std::endl;
    std::cout << rng() << std::endl;
    cout<<"e2 max"<<nc::E2MAX<<endl;

    cout<<generate_gaussian_noise(dist, rng, 0, 0.5)<<endl;
    cout<<generate_gaussian_noise(dist, rng, 0, 0.5)<<endl;
  }else if (run_opt==1){ //numcpp_2D nc_init.cpp
    cout<<"zeros vector (3)\n";
    vector<int> z_v = zeros_vec(3);  
    print_vec(z_v);  
    cout<<"zeros matrix (3,3)"<<endl;
    // mat<int>is just vector<vector<int>>
    mat<int> z_m = zeros_mat<int>(3,3);
    print_mat(z_m); 
    cout<<"eye"<<endl; 
    z_m = eye(3); 
    print_mat(z_m);
    cout<<"arange(1,5,1)"<<endl;
    z_v = arange(1,5,1);
    print_vec(z_v); 
  }else if (run_opt==2){
    mat<int>F2 = {
      {1,0},
      {1,1}
    };  
    mat<int>kern = F2;  
    for (int i =0; i<2-1; i++){
      kern = kron(F2,kern);
    }
    mat<int>F4 = kern; 
    cout<<"F4"<<endl;
    print_mat(F4);
    mat<int>e1 = eye(4);  
    cout<<"sum(e1)="<<bin_mat_sum(e1)<<endl;
    mat<int>res = bin_mat_mul(e1,F4);  
    cout<<"(F4@e1)%2="<<endl;
    print_mat(res);

    cout<<"h stack(A,B,C)"<<endl;  
    mat<int>s0 = hstack3(F4,e1,F4);
    print_mat(s0);

    vector<int> v1 = {1,3}; 
    vector<int> v2 = arange(0, 5); 
    cout<<"setdiff1d (arange(0,5), [1,3])"<<endl; 
    print_vec(setdiff1d(v2,v1));
    
    cout<<"e1"<<endl; 
    print_mat(e1);
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
    cout<<"F4 to compressed int vector"<<endl;  
    vector<int> F4_i = mat2comp_vec(F4); 
    print_vec(F4_i);
    cout<<"back to mat"<<endl; 
    print_mat(comp_vec2mat(F4_i, F4.size()));
  } else{
    cout<<"-----nc::tensor---"<<endl;
    vector<int> zz = zeros_vec(16);
    vector<int> shape = {4,4};
    Tensor<int> v0 {zz,shape};
    vector<float> zz2 = {0.1,0.3,0.2,-1.1,2.3,6.1};
    Tensor<float> v1 {zz2,{2,3}};
    cout<<"v1\n";
    v1.info();
    cout<<"v1[0,0]:"<<v1.index({0,0})<<endl;
    vector<int> id0 = {0,1};
    cout<<"v1[0,1]:"<<v1.index(id0)<<endl;
    cout<<"----slice---"<<endl;
    Tensor<float> vout = v1.slice(/*dim*/1,1,2);
    vout.info();
    cout<<"----slice_put---"<<endl;
    vector<float> p_val_vec = {1,2,3,4}; 
    Tensor<float> p_val {p_val_vec,{2,2}};
    v1.slice_put(/*dim*/1,1,3, p_val);
    v1.info();
    cout<<"---reshape----"<<endl;
    vector<int> new_shape = {2,-1};
    Tensor<float> newv1 = v1.reshape(new_shape);
    v1.info();
    newv1.info();
    mat<float> mm = newv1.toMat();
    cout<<"matrix version:\n";
    print_mat(mm);
    Tensor<float> newv2 = v1.reshape({-1});
    newv2.info();
    vector<float> vv = newv2.toVec();
    cout<<"vector version:";
    print_vec(vv);
  }
    return 0;
}

