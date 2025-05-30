#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void checkIndex(void)
{
    printf("threadidx: (%d ,%d ,%d) blockidx:(%d ,%d ,%d) blockdim: (%d ,%d ,%d) gridDim: (%d ,%d ,%d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z,
    blockIdx.x, blockIdx.y, blockIdx.z,
    blockDim.x,blockDim.y,blockDim.z,
    gridDim.x, gridDim.y, gridDim.z
    );
}


int main(int argc , char **argv)
{
    int nElem = 6;
    
    dim3 block(3);
    dim3 grid ((nElem + block.x -1)/block.x);

    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    checkIndex<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}