#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

__global__ void readOffset(float *A , float *B, float *C, const int N, int offset){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < N) C[i] = A[k] + B[k];
}

__global__ void readOffsetUnroll4(float *A , float *B, float *C, const int N, int offset){
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    unsigned int k = i + offset;
    if (k + 3 * blockDim.x < N) {
        C[i] = A[k] + B[k];
        C[i + blockDim.x] = A[k + blockDim.x ] + B[k + blockDim.x];
        C[i + 2 * blockDim.x] = A[k + 2 * blockDim.x ] + B[k + 2 * blockDim.x];
        C[i + 3 * blockDim.x] = A[k + 3* blockDim.x ] + B[k + 3 * blockDim.x];
    }
}


__global__ void warmup(float *A , float *B, float *C, const int N, int offset){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < N) C[i] = A[k] + B[k];
}


void sumArraysOnHost(float *A, float *B, float *C, const int N, int offset)
{
    for (int idx = offset , k = 0; idx< N; idx ++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv){
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int isize = 1<< 22;
    unsigned int bytes = isize * sizeof(float);

    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    //printf("device %d: %s memory size %d bytes %5.2fMB\n",dev,deviceprop.name, isize, bytes/(1024.0f * 1024.0f) );


    int blocksize = 512;
    int offset = 0;
    if (argc > 1) offset = atoi(argv[1]);
    if (argc > 2) blocksize = atoi(argv[2]);

    dim3 block(blocksize,1);
    dim3 grid((isize + block.x - 1)/ block.x, 1);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *hostRef = (float *)malloc(bytes);
    float *gpuRef = (float *)malloc(bytes);

    initialData(h_A, isize);
    memcpy(h_B, h_A, bytes);
    
    sumArraysOnHost(h_A,h_B, hostRef, isize, offset);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, bytes);
    cudaMalloc((float**)&d_B, bytes);
    cudaMalloc((float**)&d_C, bytes);


    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, bytes, cudaMemcpyHostToDevice);


    Timer timer;
    timer.start();
    warmup<<<grid,block>>>(d_A, d_B, d_C, isize, offset);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    //printf("warmup <<<%4d, %4d>>> offset %4d elapsed %f ms \n", grid.x, block.x, offset, elapsedTime);
    

    //
    timer.start();
    readOffset<<<grid,block>>>(d_A, d_B, d_C, isize, offset);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("readOffset <<<%4d, %4d>>> offset %4d elapsed %f ms \n", grid.x, block.x, offset, elapsedTime);

    timer.start();
    readOffsetUnroll4<<<grid.x/4,block>>>(d_A, d_B, d_C, isize, offset);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("readOffsetUnroll4 <<<%4d, %4d>>> offset %4d elapsed %f ms \n", grid.x, block.x, offset, elapsedTime);


    cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, isize - offset);
    
    
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return 0;

}



