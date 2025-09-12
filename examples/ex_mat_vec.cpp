#include "examples.h"
using std::stoi;
using std::vector;
using std::queue;
using std::cout;
using std::cin;
using std::endl;
using std::string; 
using namespace nc;
void ex0(bool verbose){ //* random vector
    int seed = 42;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist(0, nc::E2MAX);
    
    std::cout <<"ex0:"<< rng() << std::endl;
    if (verbose){
      std::cout << rng() << std::endl;
      cout<<"e2 max"<<nc::E2MAX<<endl;
      cout<<generate_gaussian_noise(dist, rng, 0, 0.5)<<endl;
      cout<<generate_gaussian_noise(dist, rng, 0, 0.5)<<endl;
    }
}

void ex1(bool verbose){ //* init vector
  vector<int> z_v = zeros_vec<int>(3);  
  vector<float> o_fv = ones_vec<float>(4); 
  mat<int> z_m = zeros_mat<int>(3,3); // mat<int>is just vector<vector<int>>
  vector<float> z_f = arange_vec<float> (0,5,1);
  z_f[0] = 0.2;
  cout<<"ex2: zeros vector (3):"; print_vec(z_v);
  if (verbose){
    cout<<"ones vector (4)\n"; print_vec(o_fv);
    cout<<"zeros matrix (3,3)"<<endl;print_mat(z_m);
    cout<<"arange_vec(0,5,1), v[0]=0.2: "<<endl; print_vec(z_f);
  }  
  z_v = arange_vec<int>(1,5,1);
  z_m = eye<int>(3); 
  if (verbose){
    cout<<"arange_vec(1,5,1)"<<endl; print_vec(z_v); 
    cout<<"eye(3)"<<endl; print_mat(z_m);
  }
}

void ex2(bool verbose){//* mat2vec, vec2mat
  vector<int> v0 = {0,1,2}; 
  vector<int> v1 = {0,1,2};
  cout<<"ex2: vec_equal: "<<vec_equal(v0,v1)<<endl;
  mat<int> z_m = zeros_mat<int>(3,3);
  z_m[0][1] = 3; 
  vector<int> z_m_v = mat2vec<int>(z_m);
  if (verbose){
    cout<<"zeros matrix (3,3)"<<endl; // mat<int>is just vector<vector<int>>
    print_mat(z_m); 
    cout<<"---mat2vec---"<<endl; 
    print_vec(z_m_v);
    cout<<"---vec2mat---"<<endl;
    print_mat(vec2mat<int>(z_m_v,3,3));
  }
}

void ex3(bool verbose){//* argsort/kron/hstack/setdiff1d
  vector<int> v1 = {3,0,-1, 2,9}; 
  vector<size_t> ind = argsort(v1); 
  cout<<"ex3:----argsort: "; print_vec(ind);
  mat<int>F2 = {
    {1,0},
    {1,1}
  };  
  //---------kron-----------------
  mat<int>kern = F2;  
  for (int i =0; i<2-1; i++){
    kern = kron(F2,kern);
  }
  mat<int>F4 = kern; 
  
  mat<int>e1 = eye<int>(4);  
  //--------hstack3----- 
  mat<int>s0 = hstack3(F4,e1,F4);
  v1 = {1,3}; 
  vector<int> v2 = arange_vec<int>(0, 5); 
  vector<int> out; 
  setdiff1d_rev(v2,v1,out);
  if (verbose){
    cout<<"----kron----"<<endl;
    cout<<"F4"<<endl;
    print_mat(F4);
    cout<<"---h stack(A,B,C)----"<<endl; 
    print_mat(s0);
    cout<<"setdiff1d (arange(0,5), [1,3])"<<endl; 
    print_vec(setdiff1d(v2,v1));
    print_vec(out);
  }

}
void ex4(bool verbose){
  vector<float> a = arange_vec<float>(1,5); 
  vector<float> b = {-0.1, 3.6,2.1, 0.5};
  cout<<"ex4: a:"; print_vec(a);
  if (verbose){
    cout<<"-----b----"<<endl; 
    print_vec(b);
    cout<<"----sum_vec(b)---"<<endl<<sum_vec(b)<<endl;
    cout<<"----sum_vec(a,b)---"<<endl;
    print_vec(sum_vec(a,b));
    cout<<"----sum_vec(b, scalar=1)"<<endl; 
    print_vec(sum_vec(b,1.0f));
    cout<<"----mul_vec(a)---"<<endl<<mul_vec(a)<<endl; //-0.1*3.6*2.1*0.5
    cout<<"-----mul_vec(a,b)----"<<endl; 
    print_vec(mul_vec(a,b));
    cout<<"-----mul_vec(a,scalar=2)"<<endl; 
    print_vec(mul_vec(a,2.0f));
  }
}
void ex5(bool verbose){
  mat<int>e1 = eye<int>(4); 
  mat<int>res = bin_mat_mul(e1,e1);
  cout<<"ex5: sum_mat(e1)---:"<<sum_mat(e1)<<endl;
  if (verbose){
    cout<<"---(e1@e1)%2="<<endl;
    print_mat(res);
  }  
}

void ex6(bool verbose){
  mat<int>e1 = eye<int>(4); 
  cout<<"ex6"<<endl;
  swap_row(e1 , 0,1); 
  if (verbose){
    cout<<"swap e1 row0 row1"<<endl; 
    print_mat(e1);
  }
  add_row(e1, 1,2);
  if (verbose){
    cout<<"bin add e1 row 1 += row2"<<endl; 
    print_mat(e1);
  }
  //                  8 4 2 1
  vector<int> l0 = {0,1,1,0,1};  
  int l0_int = bin_to_dec(l0);
  vector<int> l2 = dec_to_bin(l0_int, l0.size());
  vector<int> e1_i = mat2comp_vec<int>(e1); 
  if(verbose){
    cout<<"---bin_to_dec"<<endl; 
    print_vec(l0);
    cout<<"--->"<<l0_int<<endl; 
    cout<<"back to vec"<<endl;
    print_vec(l2);
    cout<<"e1 to compressed int vector"<<endl;  
    print_vec(e1_i);
    cout<<"back to mat"<<endl; 
    print_mat(comp_vec2mat(e1_i, e1.size()));
  }

}
void ex7(bool verbose){
  
  vector<int> v0 = {1,1,0,2,3,3,3,-1,5,2,-1};
  vector<int> v_unique = unique_(v0); 
  cout<<"ex7:---unique_(v0)- :";
  print_vec(v_unique);
}
void ex8(){

}