#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>


#define BDIMX 32
#define BDIMY 16
#define IPAD 1

__global__ void warmup(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = blockIdx.y * blockDim.x + threadIdx.x;
    // smem store
    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();
    // smem load
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setRowReadRow(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = blockIdx.y * blockDim.x + threadIdx.x;
    // smem store
    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();
    // smem load
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = blockIdx.y * blockDim.x + threadIdx.x;
    // smem store
    tile[threadIdx.x][threadIdx.y] = idx;

    __syncthreads();
    // smem load
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = blockIdx.y * blockDim.x + threadIdx.x;
    // smem store
    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();
    // smem load
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int *out){
    //dynamic shared mem
    extern __shared__ int tile[];
    unsigned int row_idx = blockIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.x * blockDim.y + threadIdx.y;

    // smem store
    tile[row_idx] = row_idx;

    __syncthreads();
    // smem load
    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out){
    __shared__ int tile[BDIMY][BDIMX + IPAD];
    unsigned int idx = blockIdx.y * blockDim.x + threadIdx.x;
    // smem store
    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();
    // smem load
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int *out){
    //dynamic shared mem
    extern __shared__ int tile[];
    unsigned int row_idx = blockIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = blockIdx.x * (blockDim.x + IPAD) + threadIdx.y;

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
    // smem store
    tile[row_idx] = g_idx;

    __syncthreads();
    // smem load
    out[row_idx] = tile[col_idx];
}


int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s \n", dev, deviceprop.name);
    std::cout << "Compute Capability: " << deviceprop.major << "." << deviceprop.minor << std::endl;


    dim3 block(BDIMX, BDIMY);
    dim3 grid (1,1); //only 1 block

    int nElem = BDIMX * BDIMX;
    int nBytes = nElem * sizeof(int);

    int *d_A;

    cudaMalloc((int**) &d_A, nBytes);

    Timer timer;
    timer.start();
    warmup<<<grid,block>>>(d_A);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    printf("warmup <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    
    timer.start();
    setRowReadRow<<<grid,block>>>(d_A);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("setRowReadRow <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    timer.start();
    setColReadCol<<<grid,block>>>(d_A);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("setColReadCol <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    timer.start();
    setRowReadCol<<<grid,block>>>(d_A);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("setRowReadCol <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    timer.start();
    setRowReadColDyn<<<grid,block, BDIMX * BDIMY * sizeof(int)>>>(d_A);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("setRowReadColDyn <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    timer.start();
    setRowReadColPad<<<grid,block>>>(d_A);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    printf("setRowReadColPad <<<grid (%4d, %4d), block (%4d, %4d)>>> elapsed %f ms \n", grid.x,grid.y, block.x, block.y, elapsedTime);

    cudaFree(d_A);

    cudaDeviceReset();
    return 0;

}