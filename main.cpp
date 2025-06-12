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

#include "examples.h"

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
    ex0();
  }else if (run_opt==1){ //numcpp_2D nc_init.cpp
    ex1();
  }else if (run_opt==2){
    ex2();
  }else if(run_opt==3){
    //** init Tensor with vector */
    vector<int> zz = zeros_vec<int>(16);
    zz[0]=3; zz[5]=9;
    vector<int> shape = {4,4};
    Tensor<int> v0 {zz,shape};
    v0.info();
    //** init Tensor directly */
    cout<<"----T0----"<<endl;
    shape = {4,3};
    Tensor<float> T0 = zeros<float> (shape); 
    T0.info();
    cout<<"---reshape T0----"<<endl;
    vector<int> new_shape = {2,-1};
    Tensor<float> T0_r = T0.reshape(new_shape);
    T0_r.info();
    mat<float> mm = T0_r.toMat();
    cout<<"matrix version:\n";
    print_mat(mm);

    Tensor<float> newv2 = T0.reshape({-1});
    newv2.info();
    vector<float> vv = newv2.toVec();
    cout<<"vector version:";
    print_vec(vv);
  }else{
    cout<<"-----nc::tensor---"<<endl;
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
  }
    return 0;
}

