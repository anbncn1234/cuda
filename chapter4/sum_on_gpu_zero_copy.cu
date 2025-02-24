#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"



void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i])> epsilon)
        {
            match = 0;
            printf("Array do not match\n");
            printf("host %5.2f gpu % 5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;

        }
    }
    if (match) printf("array matches\n");
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx< N; idx ++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void sumArraysOnGPU_ZeroCopy(float *A, float *B, float *C, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    if (!deviceprop.canMapHostMemory){
        printf("device not support mapping CPU host memory\n");
        return 0;
    }


    int ipower = 10;
    if(argc > 1) ipower = atoi(argv[1]);
    //set up data
    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);
    if (ipower < 18){
        printf("vector size %d power %d nbytes %3.0f KB\n", nElem, ipower, (float) nBytes/1024.0f);
    }
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (float *) malloc (nBytes);
    h_B = (float *) malloc (nBytes);
    hostRef = (float *) malloc (nBytes);
    gpuRef = (float *) malloc (nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef,0, nBytes);
    memset(gpuRef,0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    
    int iLen = 512;

    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1)/block.x);
    /*
    sumArraysOnGPU<<<grid,block>>>(d_A, d_B, d_C);
    printf("execution config <<<%d, %d>>>\n", grid.x, block.x);

    //copy kernel result back to host
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    //checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    */
    //part 2
    unsigned int flags = cudaHostAllocMapped;
    cudaHostAlloc((void**)&h_A, nBytes, flags);
    cudaHostAlloc((void**)&h_B, nBytes, flags);
    
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    cudaHostGetDevicePointer((void**)&d_A, (void *)h_A, 0);
    cudaHostGetDevicePointer((void**)&d_B, (void *)h_B, 0);

    sumArraysOnGPU_ZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_C);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}