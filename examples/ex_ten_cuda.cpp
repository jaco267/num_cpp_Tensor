#include "examples.h"

void ex20(bool verbose){
  cout<<"ex20:"<<endl;
  vector<int> zf = arange_vec<int>(0,6);
  
  print_vec(zf);
  Tensor<int> t1 (zf, {2,3}, "cuda"); 
  Tensor<int> t2 = t1; 
  t2.to("cpu");
  t2.info();
  // t2.index({0,0}) = 100; //todo assignement  
  t2.index_put({0,0},100); 
  //todo gpu device    info()
  t2.info();
  t1.to("cpu");
  t1.info();
  t1.to("cuda");
//   t1.info();
  //todo  also copy constructor  
}