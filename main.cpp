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
  if (run_opt==0){ ex0();
  }else if (run_opt==1){ex1(); //numcpp_2D nc_init.cpp
  }else if (run_opt==2){ex2();
  }else if (run_opt==3){ex3();
  }else if (run_opt==4){ex4();
  }else if (run_opt==5){ex5();
  }else if (run_opt==6){ex6();
  }else if (run_opt==7){ex7();
  //*-----tensor-----
  }else if(run_opt==10){ex10();
  }else if(run_opt==11){ex11();
  }else if(run_opt==12){ex12();
  }else if(run_opt==13){ex13();
  }else if(run_opt==14){ex14();
  }else if(run_opt==15){ex15();
  }else if(run_opt==16){ex16();
  }else{
     throw std::invalid_argument( 
      "--opt should be 0~7 or 10~16" );
  }
    return 0;
}

