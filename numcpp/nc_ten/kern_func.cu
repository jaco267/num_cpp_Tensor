#include <cuda_runtime.h>
#include "kern_func.cuh"

template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

template <typename T>
void launch_add_kernel(const T* a, const T* b, T* out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, out, n);
}
//? I dont know why I need unsigned long  
// Explicit instantiations (important if you only want float/double etc.)
template void launch_add_kernel<int>(const int*, const int*, int*, int);
template void launch_add_kernel<long>(const long*, const long*, long*, int);
template void launch_add_kernel<unsigned long>(const unsigned long*, const unsigned long*, unsigned long*, int);
template void launch_add_kernel<float>(const float*, const float*, float*, int);
template void launch_add_kernel<double>(const double*, const double*, double*, int);
