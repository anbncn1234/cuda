#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>

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

/*
double cpusec()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double)tp.tv_usec* 1.e-6);
}
*/

int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaSetDevice(dev);


    //set up data
    int nElem = 32;
    size_t nBytes = nElem * sizeof(float);
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

    dim3 block(nElem);
    dim3 grid(nElem/block.x);

    sumArraysOnGPU<<<grid,block>>>(d_A, d_B, d_C);
    printf("execution config <<<%d, %d>>>\n", grid.x, block.x);

    //copy kernel result back to host
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}