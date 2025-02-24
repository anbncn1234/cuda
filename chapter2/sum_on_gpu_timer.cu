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



int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    CHECK(cudaSetDevice(dev));
    

    printf( "Compute capability:  %d.%d\n", deviceprop.major, deviceprop.minor );
    printf( "Clock rate:  %d\n", deviceprop.clockRate );
    printf( "Memory Clock rate:  %d\n", deviceprop.memoryClockRate );
    printf( "Memory busWidth:  %d\n", deviceprop.memoryBusWidth );
    printf( "   --- Memory Information for device  ---\n");
    // printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
    printf( "Total global mem:  %zu\n", deviceprop.totalGlobalMem );
    printf( "Total constant Mem:  %ld\n", deviceprop.totalConstMem );
    printf( "Max mem pitch:  %ld\n", deviceprop.memPitch );
    printf( "Texture Alignment:  %ld\n", deviceprop.textureAlignment );
    
    printf( "   --- MP Information for device  ---\n" );
    printf( "Multiprocessor count:  %d\n",
                deviceprop.multiProcessorCount );
    printf( "Shared mem per mp:  %ld\n", deviceprop.sharedMemPerBlock );
    printf( "Registers per mp:  %d\n", deviceprop.regsPerBlock );
    printf( "Threads in warp:  %d\n", deviceprop.warpSize );
    printf( "Max threads per block:  %d\n",
                deviceprop.maxThreadsPerBlock );
    printf( "Max thread dimensions:  (%d, %d, %d)\n",
                deviceprop.maxThreadsDim[0], deviceprop.maxThreadsDim[1],
                deviceprop.maxThreadsDim[2] );
    printf( "Max grid dimensions:  (%d, %d, %d)\n",
                deviceprop.maxGridSize[0], deviceprop.maxGridSize[1],
                deviceprop.maxGridSize[2] );
    printf( "\n" );

    //set up data
    int nElem = 1<<24;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (float *) malloc (nBytes);
    h_B = (float *) malloc (nBytes);
    hostRef = (float *) malloc (nBytes);
    gpuRef = (float *) malloc (nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef,0, nBytes);
    memset(gpuRef,0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    //transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);


    int Ilen = 1024;
    dim3 block(Ilen);
    dim3 grid((nElem + block.x - 1)/block.x);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    sumArraysOnGPU<<<grid,block>>>(d_A, d_B, d_C);
    printf("execution config <<<%d, %d>>>\n", grid.x, block.x);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);



    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    
    //copy kernel result back to host
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}