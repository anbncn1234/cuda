#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>
#include "../common/common.h"

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i])> epsilon)
        {
            match = 0;
            printf("Array do not match\n");
            printf("host %5.2f gpu % 5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;

        }
    }
    if (match) printf("array matches\n");
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}



void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix =0; ix < nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; 
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;

    if (ix < nx && iy < ny){
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}


int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);



    int dev = 0;
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    
    int nx = 1 << 14;
    int ny = 1<< 14;
    int nxy = nx * ny;
    int dimx = 32, dimy =32;
    if (argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }


    size_t nBytes = nxy  * sizeof(float);
    //printf("matrix size %d %d\n", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;


    h_A = (float *) malloc (nBytes);
    h_B = (float *) malloc (nBytes);
    hostRef = (float *) malloc (nBytes);
    gpuRef = (float *) malloc (nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef,0, nBytes);
    memset(gpuRef,0, nBytes);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((float**)&d_MatA, nBytes);
    cudaMalloc((float**)&d_MatB, nBytes);
    cudaMalloc((float**)&d_MatC, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/block.x,(ny + block.y - 1)/block.y);
    printf("execution config:  grid (%d %d), block (%d %d)  ", block.x, block.y, grid.x, grid.y);
    
    

    Timer timer;
    timer.start();
    cudaDeviceSynchronize();
    sumMatrixOnGPU2D<<<grid,block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    
    printf("execution time: %f ms\n", elapsedTime);
    
    //copy kernel result back to host
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    sumMatrixOnHost(h_A, h_B, hostRef, nx,ny);

    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);


    return 0;
}