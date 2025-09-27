// func.cu
#include <cuda_runtime.h>
#include "func.cuh"
#include <iostream>
using std::cout; 
using std::endl;
// CUDA kernel
__global__ void addKernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// wrapper implementation
extern "C" void addArrays(const int* a, const int* b, int* c, int n) {
    int *d_a, *d_b, *d_c;
    // size_t size = n * sizeof(int);
    cout<<"add array....!!!"<<endl;
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
