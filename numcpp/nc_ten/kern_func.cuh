#pragma once

// Forward declaration of the kernel
template <typename T>
__global__ void add_kernel(const T* a, const T* b, T* out, int n);

// Host launcher
template <typename T>
void launch_add_kernel(const T* a, const T* b, T* out, int n);

// template <typename T>
// __global__ void add_kernel(const T* a, const T* b, T* out, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) out[i] = a[i] + b[i];
// }

// template <typename T>
// void launch_add_kernel(const T* a, const T* b, T* out, int n) {
//     int threads = 256;
//     int blocks = (n + threads - 1) / threads;
//     add_kernel<T><<<blocks, threads>>>(a, b, out, n);
// }