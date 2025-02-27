#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s \n", dev, deviceprop.name);
    std::cout << "Compute Capability: " << deviceprop.major << "." << deviceprop.minor << std::endl;

    return 0;

}