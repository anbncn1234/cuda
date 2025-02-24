#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>

typedef unsigned long DWORD;

#define CHECK(call) \
 {\
    const cudaError_t error = call; \
    if (error != cudaSuccess)\
    {\
        printf("Error: %s: %d\n", __FILE__, __LINE__);\
        printf("code :%d reason :%s\n", error , cudaGetErrorString(error));\
        exit(1);\
    }\
}

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
    

    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
 
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
 
    // 执行你想要计时的代码
    // ...
    cudaDeviceSynchronize();
    warmingup<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
 
    QueryPerformanceCounter(&end);
 
    double time = (double)(end.QuadPart - start.QuadPart) / (double)frequency.QuadPart * 1.0E6;
    printf("execute time: %f ms\n", time);

/*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("warmup execution time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */
    // kernel 1

    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    
    cudaEventRecord(start1);
    cudaDeviceSynchronize();
    mathKernel1<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    
    printf("Kernel1 execution time: %f ms\n", milliseconds1);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    // kernel 2

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    
    cudaEventRecord(start2);
    cudaDeviceSynchronize();
    mathKernel2<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    
    printf("Kernel2 execution time: %f ms\n", milliseconds2);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    // kernel 3

    cudaEvent_t start3, stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);
    
    cudaEventRecord(start3);
    cudaDeviceSynchronize();
    mathKernel3<<<grid,block>>>(d_C);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);
    
    float milliseconds3 = 0;
    cudaEventElapsedTime(&milliseconds3, start3, stop3);
    
    printf("Kernel3 execution time: %f ms\n", milliseconds3);
    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    cudaFree(d_C);
    cudaDeviceReset();

    return 0;
}