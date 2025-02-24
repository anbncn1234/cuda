#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

__device__ float devData;

__global__ void checkGlobalVariable(){
    printf("device: the value of the global var is %f\n", devData);
    devData += 2.0f;
}

__global__ void readOffset(float *A , float *B, float *C, const int N, int offset){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;
    if (k < N) C[i] = A[i] + B[i];
}

int main(void){
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int isize = 1<< 22;
    unsigned int bytes = isize * sizeof(float);

    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s memory size %d bytes %5.2fMB\n",dev,deviceprop.name, isize, bytes/(1024.0f * 1024.0f) );


    //float *h_a = (float*) malloc(bytes);  //pageable mem
    float *h_aPinned;

    cudaError_t status = cudaMallocHost((void **)& h_aPinned, bytes);

    float * d_a;
    cudaMalloc((float**) &d_a, bytes);

    for (unsigned int i=0;i<isize;i++) h_aPinned[i] = 0.5f;
    Timer timer;
    timer.start();
    cudaMemcpy(d_a, h_aPinned, bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(h_aPinned, d_a, bytes, cudaMemcpyDeviceToHost);
    timer.stop();
    float elapsedTime = timer.elapsedms();
    
    printf("kernel execution time: %f ms\n", elapsedTime);
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);

    cudaDeviceReset();
    return 0;

}



