#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

__device__ float devData;

__global__ void checkGlobalVariable(){
    printf("device: the value of the global var is %f\n", devData);
    devData += 2.0f;
}

int main(void){
    int dev = 0;
    cudaSetDevice(dev);

    unsigned int isize = 1<< 22;
    unsigned int bytes = isize * sizeof(float);

    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s memory size %d bytes %5.2fMB\n",dev,deviceprop.name, isize, bytes/(1024.0f * 1024.0f) );


    float *h_a = (float*) malloc(bytes);

    float * d_a;
    cudaMalloc((float**) &d_a, bytes);

    for (unsigned int i=0;i<isize;i++) h_a[i] = 0.5f;
    Timer timer;
    timer.start();
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    timer.stop();
    float elapsedTime = timer.elapsedms();
    
    printf("kernel execution time: %f ms\n", elapsedTime);
    
    cudaFree(d_a);
    free(h_a);

    cudaDeviceReset();
    return 0;

}



