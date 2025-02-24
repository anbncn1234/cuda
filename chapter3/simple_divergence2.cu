#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>
#include "../common/common.h"

typedef unsigned long DWORD;



__global__ void mathKernel1( float *C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (tid % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    C[tid] = a + b;
}

__global__ void mathKernel2( float *C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    C[tid] = a + b;
}

__global__ void mathKernel3( float *C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    bool ipred = (tid % 2 == 0);
    if (ipred){
        a = 100.0f;
    }
    if  (!ipred){
        b = 200.0f;
    }
    C[tid] = a + b;
}

__global__ void warmingup( float *C){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if ((tid / warpSize) % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    C[tid] = a + b;
}


int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size      = atoi(argv[2]);
    printf("Data size %d\n", size);


    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1)/block.x);
    printf("execution config: %d %d\n", block.x, grid.x);
    
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**) &d_C, nBytes);

    Timer timer;
    timer.start();
    cudaDeviceSynchronize();
    warmingup<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    
    printf("warmup execution time: %f ms\n", elapsedTime);
    
    
    // kernel 1

    Timer timer1;
    timer1.start();
    cudaDeviceSynchronize();
    mathKernel1<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    timer1.stop();
    float elapsedTime1 = timer1.elapsedms();
    
    printf("kernel1 execution time: %f ms\n", elapsedTime1);

    // kernel 2

    Timer timer2;
    timer2.start();
    cudaDeviceSynchronize();
    mathKernel2<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    timer2.stop();
    float elapsedTime2 = timer2.elapsedms();
    
    printf("kernel2 execution time: %f ms\n", elapsedTime2);

    // kernel 3

    Timer timer3;
    timer3.start();
    cudaDeviceSynchronize();
    mathKernel3<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    timer3.stop();
    float elapsedTime3 = timer3.elapsedms();
    
    printf("kernel3 execution time: %f ms\n", elapsedTime3);

    cudaFree(d_C);
    cudaDeviceReset();

    return 0;
}