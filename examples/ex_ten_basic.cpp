#include "examples.h"
void ex3(){
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
}
void ex4(){
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
void ex5(){
    // nc::indexing::Slice(0,1,std::monostate{});

  nc::indexing::Slice(1);
  nc::indexing::Slice(1,0, nc::indexing::None);
  vector<float> zz2 = {0.1,0.3,0.2,
                         1, 2 , 3, 

                      -1.1,2.3,6.1,
                         4, 5 , 6, 

                      0.9, 0.8,-0.5,
                        7, 8 ,  8 , 

                      -3, -2,-9,
                      10, 11, 12 };
  Tensor<float> v1 {zz2,{4,2,3}};
  cout<<"v1\n";
  v1.info();
  cout<<"----v1[2:4,1,:]---"<<endl;
  Tensor <float> out_v1 = v1.index({nc::indexing::Slice(2,4),1,nc::indexing::None});
  out_v1.info();
  cout<<"----v1[2,:,1:]=0---"<<endl;
  Tensor<float> put0 = zeros<float>({1,2,2});
//   put0.info();
  v1.index_put({2,nc::indexing::Slice(nc::indexing::None),nc::indexing::Slice(1)},put0);
  v1.info();
  cout<<"----indexing[2:4]:";
  cout<<nc::indexing::Slice(2,4)<<endl;
  cout<<"----indexing[:4]: ";
  cout<<nc::indexing::Slice(nc::indexing::None,4)<<endl;
  cout<<"----indexing[:]:  ";
  cout<<nc::indexing::Slice(nc::indexing::None,nc::indexing::None)<<endl;
  
}