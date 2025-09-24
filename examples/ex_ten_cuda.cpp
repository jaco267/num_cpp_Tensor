#include "examples.h"

void ex20(bool verbose){
  cout<<"ex20:"<<endl;
  vector<int> zf = arange_vec<int>(0,6);
  
  print_vec(zf);
  Tensor<int> t1 (zf, {2,3}, "cuda"); 
  Tensor<int> t2 = t1;  //* copy constructor  
  t2.to("cpu"); //* t2 data_cu_ is freed
  t2.info();
  // t2.index({0,0}) = 100; //todo assignement  
  t2.index_put({0,0},100); 
  // //todo gpu device    info()
  t2.info();
  t1.to("cpu"); //* t1 data_cu_ is freed
  t1.info();
  t1.to("cuda");
  t1.info();

  cout<<t1.data_.size()<<endl;
  t1.info();
//   t1.info();
}

void ex21(bool verbose){
  cout<<"ex20:"<<endl;
  Tensor<int> a =  arange<int>(100,106);
  a.to("cuda");
  Tensor<int> b =  arange<int>(0,6);
  b.to("cuda");
  
  cout<<a;
  cout<<b;
  Tensor<int> out = a.add(b);
  cout<<"out:\n"<<out;
} 