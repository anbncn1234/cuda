#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>

__global__ void kernel(float *F, double *D)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        *F = 12.1;
        *D = 12.1;
    }
}

int main(int argc, char **argv)
{
    float *deviceF;
    float h_deviceF;
    double *deviceD;
    double h_deviceD;

    float hostF = 12.1;
    double hostD = 12.1;

    CHECK(cudaMalloc((void **)&deviceF, sizeof(float)));
    CHECK(cudaMalloc((void **)&deviceD, sizeof(double)));
    kernel<<<1, 32>>>(deviceF, deviceD);
    CHECK(cudaMemcpy(&h_deviceF, deviceF, sizeof(float),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&h_deviceD, deviceD, sizeof(double),
                     cudaMemcpyDeviceToHost));

    printf("Host single-precision representation of 12.1   = %.20f\n", hostF);
    printf("Host double-precision representation of 12.1   = %.20f\n", hostD);
    printf("Device single-precision representation of 12.1 = %.20f\n", h_deviceF);
    printf("Device double-precision representation of 12.1 = %.20f\n", h_deviceD);
    printf("Device and host single-precision representation equal? %s\n",
           hostF == h_deviceF ? "yes" : "no");
    printf("Device and host double-precision representation equal? %s\n",
           hostD == h_deviceD ? "yes" : "no");

    return 0;
}