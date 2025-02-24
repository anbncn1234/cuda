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

__global__ void testInnerArray( innerArray *data, innerArray *result, const int n){
    unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpx = data -> x[i];
        float tmpy = data -> y[i];

        tmpx += 10.0f;
        tmpy += 20.0f;
        result -> x[i] = tmpx;
        result -> y[i] = tmpy;
    }
}

__global__ void warmup( innerArray *data, innerArray *result, const int n){
    unsigned int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmpx = data -> x[i];
        float tmpy = data -> y[i];

        tmpx += 10.0f;
        tmpy += 20.0f;
        result -> x[i] = tmpx;
        result -> y[i] = tmpy;
    }
}

// functions for inner array outer struct
void initialInnerArray(innerArray *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip->x[i] = (float)( rand() & 0xFF ) / 100.0f;
        ip->y[i] = (float)( rand() & 0xFF ) / 100.0f;
    }

    return;
}

void testInnerArrayHost(innerArray *A, innerArray *C, const int n)
{
    for (int idx = 0; idx < n; idx++)
    {
        C->x[idx] = A->x[idx] + 10.f;
        C->y[idx] = A->y[idx] + 20.f;
    }

    return;
}

void checkInnerArray(innerArray *hostRef, innerArray *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon)
        {
            match = 0;
            printf("different on x %dth element: host %f gpu %f\n", i,
                   hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon)
        {
            match = 0;
            printf("different on y %dth element: host %f gpu %f\n", i,
                   hostRef->y[i], gpuRef->y[i]);
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
    size_t nBytes =   sizeof(innerArray);

    innerArray     *h_A = (innerArray *)malloc(nBytes);
    innerArray *hostRef = (innerArray *)malloc(nBytes);
    innerArray *gpuRef  = (innerArray *)malloc(nBytes);
    
    initialInnerArray(h_A, nElem);
    testInnerArrayHost(h_A, hostRef, nElem);

    innerArray *d_A, *d_C;
    cudaMalloc((innerArray**)&d_A, nBytes);
    cudaMalloc((innerArray**)&d_C, nBytes);
    

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
    testInnerArray<<<grid,block>>>(d_A, d_C, nElem);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("testInnerArray <<<%4d, %4d>>> elapsed %f ms \n", grid.x, block.x,  elapsedTime);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkInnerArray(hostRef, gpuRef, nElem);


    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return 0;
}