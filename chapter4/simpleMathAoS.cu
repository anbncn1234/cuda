#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>

#define LEN 1 << 20

struct innerStruct{
	float x;
	float y;
};


struct innerArray{
	float x[LEN];
	float y[LEN];
};


__global__ void testInnerStruct(innerStruct *data, innerStruct *result, const int n){
    unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

__global__ void warmup(innerStruct *data, innerStruct *result, const int n){
    unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

void testInnerStructHost(innerStruct *data, innerStruct *result, const int n){
    for (int i = 0; i < n ; i ++){
        innerStruct tmp = data[i];
        tmp.x += 10.f;
        tmp.y += 20.f;
        result[i] = tmp;
    }
}

void initialInnerStruct(innerStruct *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i].x = (float)(rand() & 0xFF) / 100.0f;
        ip[i].y = (float)(rand() & 0xFF) / 100.0f;
    }

    return;
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                    hostRef[i].x, gpuRef[i].x);
            break;
        }

        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i,
                    hostRef[i].y, gpuRef[i].y);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}


int main(int argc, char ** argv){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s \n", dev, deviceprop.name);

    int nElem = LEN;
    size_t nBytes = nElem * sizeof(innerStruct);

    innerStruct     *h_A = (innerStruct *)malloc(nBytes);
    innerStruct *hostRef = (innerStruct *)malloc(nBytes);
    innerStruct *gpuRef  = (innerStruct *)malloc(nBytes);
    
    initialInnerStruct(h_A, nElem);
    testInnerStructHost(h_A, hostRef, nElem);

    innerStruct *d_A, *d_C;
    cudaMalloc((innerStruct**)&d_A, nBytes);
    cudaMalloc((innerStruct**)&d_C, nBytes);
    

    cudaMemcpy(d_A, h_A, nBytes,cudaMemcpyHostToDevice);

    int blocksize = 128;
    if (argc > 1) blocksize = atoi(argv[1]);

    dim3 block(blocksize,1);
    dim3 grid((nElem + block.x - 1)/block.x, 1);

    Timer timer;
    timer.start();
    warmup<<<grid,block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    printf("warmup <<<%4d, %4d>>> elapsed %f ms \n", grid.x, block.x,  elapsedTime);

    timer.start();
    testInnerStruct<<<grid,block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("testInnerStruct <<<%4d, %4d>>> elapsed %f ms \n", grid.x, block.x,  elapsedTime);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerStruct(hostRef, gpuRef, nElem);


    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return 0;
}