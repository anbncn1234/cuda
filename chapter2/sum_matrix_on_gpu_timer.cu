#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>

typedef unsigned long DWORD;

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

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx< N; idx ++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
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
    CHECK(cudaSetDevice(dev));

    //set up data
    int nx  = 1<<14;
    int ny  = 1<<14;
    int nxy = nx * ny;
    size_t nBytes = nxy  * sizeof(float);
    printf("matrix size %d %d\n", nx, ny);

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


    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    sumMatrixOnGPU2D<<<grid,block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("execution config <<<(%d,%d), (%d,%d)>>>\n", grid.x,grid.y, block.x, block.y);
    printf("Kernel execution time: %f ms\n", milliseconds);



    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
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