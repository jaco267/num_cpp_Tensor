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
    ex3();
  }else if(run_opt==4){
    ex4();
  }else{
    ex5();
  }
    return 0;
}

