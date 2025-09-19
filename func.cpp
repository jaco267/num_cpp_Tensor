#include "func.h"
#include "func.cuh" 
void ex18(bool verbose){
    cout<<"ex18: test cu\n";

    const int N = 5;
    int a[N] = {1, 2, 3, 4, 5};
    int b[N] = {10, 20, 30, 40, 50};
    int c[N] = {0};

    // Call CUDA function
    addArrays(a, b, c, N);
    if (verbose){
      // Print results
      for (int i = 0; i < N; i++) {
          std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
      }
    }
}