#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

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

void initialInt( int * ip, int size)
{
    for (int i =0; i < size; i ++)
    {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny)
{
    int *ic = C;
    printf("\n matrix : (%d, %d)\n", nx, ny);
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix =0; ix < nx; ix++){
            printf("%3d",ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;

    printf("threadidx: (%d ,%d) blockidx:(%d ,%d) coordinate: (%d ,%d) global index: (%2d ival %2d)\n", 
    threadIdx.x, threadIdx.y,
    blockIdx.x, blockIdx.y,
    ix, iy,
    idx, A[idx]
    );
}


int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    CHECK(cudaSetDevice(dev));

    // set matrix 
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    int * h_A;
    h_A = (int *) malloc(nBytes);

    //initial int
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // device
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4,2);
    dim3 grid ((nx + block.x - 1)/block.x, (ny + block.y - 1)/ block.y);

    printf("execution config grid (%d, %d), block (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_MatA);
    free(h_A);

    cudaDeviceReset();
    
    return 0;
}