#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>

#define KERNEL_LOOP 65536

__constant__ static const int const_data_01 = 0x55555555;
__constant__ static const int const_data_02 = 0x77777777;
__constant__ static const int const_data_03 = 0x33333333;
__constant__ static const int const_data_04 = 0x11111111;

__device__ static  int data_01 = 0x55555555;
__device__ static  int data_02 = 0x77777777;
__device__ static  int data_03 = 0x33333333;
__device__ static  int data_04 = 0x11111111;


__global__ void warmup(int* const data, const int num_elements){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements){
        int d = 0x55555555;
        for (int i = 0; i < KERNEL_LOOP; i++){
            d ^= 0x55555555;
            d |= 0x77777777;
            d &= 0x33333333;
            d |= 0x11111111;
        }
        data[tid] = d;
    }
}

__global__ void const_test_gpu_literal(int* const data, const int num_elements){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements){
        int d = 0x55555555;
        for (int i = 0; i < KERNEL_LOOP; i++){
            d ^= 0x55555555;
            d |= 0x77777777;
            d &= 0x33333333;
            d |= 0x11111111;
        }
        data[tid] = d;
    }
}

__global__ void const_test_gpu_const(int* const data, const int num_elements){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements){
        int d = 0x55555555;
        for (int i = 0; i < KERNEL_LOOP; i++){
            d ^= const_data_01;
            d |= const_data_02;
            d &= const_data_03;
            d |= const_data_04;
        }
        data[tid] = d;
    }
}

__global__ void const_test_gpu_gmem(int* const data, const int num_elements){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements){
        int d = 0x55555555;
        for (int i = 0; i < KERNEL_LOOP; i++){
            d ^= data_01;
            d |= data_02;
            d &= data_03;
            d |= data_04;
        }
        data[tid] = d;
    }
}


int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);

    const int num_elements = 128 * 1024;
    const int num_threads = 256;
    const int num_blocks = (num_elements + num_threads - 1)/num_threads;
    const int num_bytes = num_elements * sizeof(int);

    int *data;
    cudaMalloc((int**)&data, num_bytes);

    dim3 block(num_threads, 1);
    dim3 grid(num_blocks,1);

    Timer timer;
    timer.start();
    warmup<<<grid,block>>>(data, num_elements);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    printf("warmup <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    timer.start();
    const_test_gpu_literal<<<grid,block>>>(data, num_elements);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("const_test_gpu_literal <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);
    
    timer.start();
    const_test_gpu_const<<<grid,block>>>(data, num_elements);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("const_test_gpu_const <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);
    

    timer.start();
    const_test_gpu_gmem<<<grid,block>>>(data, num_elements);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("const_test_gpu_gmem <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    cudaFree(data);
    cudaDeviceReset();
    return 0;

    
}