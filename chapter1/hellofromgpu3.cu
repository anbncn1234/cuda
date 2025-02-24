#include <stdio.h>


__global__ void hellofromgpu(void)
{
    printf("hello world from gpu\n");
}
int main(void)
{
    printf("hello world from cpu\n");
    hellofromgpu<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}