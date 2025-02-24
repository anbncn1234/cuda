#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

__device__ float devData;

__global__ void checkGlobalVariable(){
    printf("device: the value of the global var is %f\n", devData);
    devData += 2.0f;
}

int main(void){
    float value = 3.14f;
    //这个函数将count (sizeof float)个字节从src (&value)指向的内存复制到symbol (devData)指向的内存中
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host : copy %f to the global variable \n", value);

    checkGlobalVariable<<<1,1>>>();

    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host : the value changed by kernel to %f\n", value);

    cudaDeviceReset();
    return 0;

}



