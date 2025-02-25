#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void tranposeHost_test(float* out, float* in, const int nx, const int ny){
    for (int i = 0; i < nx ; i ++){
        for (int j = 0; j < ny; j++){
            out[i*ny + j] = in[j * nx + i];
        }
    }

}

void tranposeHost(float* out, float* in, const int nx, const int ny){
    for (int iy = 0; iy < ny ; iy ++){
        for (int ix = 0; ix < nx; ix++){
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }

}

__global__ void warmup( float * out, float* in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny){
        out[iy*nx + ix] = in[iy*nx + ix];
    }
}


// load row store row
__global__ void copyRow( float * out, float* in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny){
        out[iy*nx + ix] = in[iy*nx + ix];
    }
}

// load col store col
__global__ void copyCol( float * out, float* in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny){
        out[ix*ny + iy] = in[ix*ny + iy];
    }
}


// load row, store col
__global__ void tranposeNaiveRow(float* out, float* in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny){
        out[ix*ny + iy] = in[iy*nx + ix];
    }
}

// load col, store row
__global__ void tranposeNaiveCol(float* out, float* in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny){
        out[iy*nx + ix] = in[ix*ny + iy];
    }
}

void initialData(float *ip, int nx, int ny)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < nx; i++) {
        for (int j =0; j < ny; j ++){
            ip[i + j * nx] = (float) (rand() & 0xff) / 10.0f;
        }
    }
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

int main(int argc, char** argv){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s \n", dev, deviceprop.name);
    printf("Peak Memory Bandwidth (GB/s): %f\n",2.0 * deviceprop.memoryClockRate * (deviceprop.memoryBusWidth / 8) / 1.0e6);

    float pbnd = 2.0 * deviceprop.memoryClockRate * (deviceprop.memoryBusWidth / 8) / 1.0e6;

    int nx = 1 << 11;
    int ny = 1 << 11;

    int iKernel = 0;
    int blockx = 16;
    int blocky = 16;

    if(argc > 1) iKernel = atoi(argv[1]);
    if(argc > 2) blockx = atoi(argv[2]);
    if(argc > 3) blocky = atoi(argv[3]);
    if(argc > 4) nx = atoi(argv[4]);
    if(argc > 5) ny = atoi(argv[5]);

    printf("matrix nx %d ny %d with kernel %d\n", nx, ny, iKernel);
    size_t nBytes = nx * ny * sizeof(float);

    dim3 block (blockx, blocky);
    dim3 grid ((nx + block.x-1)/block.x, (ny + block.y -1 )/ block.y);

    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nx, ny);

    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    Timer timer;
    timer.start();
    warmup<<<grid,block>>>(d_A, d_C, nx, ny);
    cudaDeviceSynchronize();
    timer.stop();
    float elapsedTime = timer.elapsedms();
    printf("warmup <<<%4d, %4d>>> elapsed %f ms \n", grid.x, block.x,  elapsedTime);


    // kernel pointer
    void (*kernel)(float*, float*,int, int);
    char *kernelName;

    switch (iKernel){
        case 0:
            kernel = &copyRow;
            kernelName = "COPYROW";
            break;
        case 1:
            kernel = &copyCol;
            kernelName = "COPYCOL";
            break;
        case 2:
            kernel = &tranposeNaiveRow;
            kernelName = "tranposeNaiveRow";
            break;
        case 3:
            kernel = &tranposeNaiveCol;
            kernelName = "tranposeNaiveCol";
            break;
    }

    timer.start();
    kernel<<<grid,block>>>(d_C,d_A , nx, ny);
    cudaDeviceSynchronize();
    timer.stop();
    elapsedTime = timer.elapsedms();
    float ibnd = 2*nx * ny * sizeof(float)/1e6 / elapsedTime; 

    printf("%s elapsed %f ms <<<grid(%d, %d) block (%d, %d)>>> effective bandwidth %f GB/s, ratio %f%% \n", kernelName, elapsedTime, grid.x , grid.y, block.x, block.y, ibnd, ibnd/pbnd * 100);

    if (iKernel > 1){
        cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
        checkResult(hostRef, gpuRef, nx*ny);
    }
    cudaFree(d_A);
    cudaFree(d_C);

    free(h_A);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return 0;
}