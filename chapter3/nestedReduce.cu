#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>
#include "../common/common.h"


int recursiveReduce(int *data, int const size){
    if (size == 1) return data[0];
    int const stride = size /2;
    for (int i = 0; i < stride; i ++){
        data[i] += data[i + stride];
    }
    return recursiveReduce( data, stride);
}
__global__ void warmup( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;  // boundary check

    for(int stride=1;stride<blockDim.x;stride*=2){
        if ((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if  (tid == 0){ 
        g_odata[blockIdx.x] = idata[0];
    }
}

//g input data, g output data
__global__ void reduceNeighbored( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > n) return;  // boundary check

    for (int stride = 1; stride < blockDim.x; stride *= 2){
        if ((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

//g input data, g output data
__global__ void reduceNeighboredLess( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > n) return;  // boundary check

    for (int stride = 1; stride < blockDim.x; stride *= 2){
        int index = 2* stride * tid;
        if (index < blockDim.x){
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

__global__ void reduceInterleaved( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > n) return;  // boundary check

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

__global__ void gpuRecursiveReduceNosync (int *g_idata, int *g_odata,
        unsigned int isize)
{
    // set thread ID
    unsigned int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    // stop condition
    if (isize == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invoke
    int istride = isize >> 1;

    if(istride > 1 && tid < istride)
    {
        idata[tid] += idata[tid + istride];

        if(tid == 0)
        {
            gpuRecursiveReduceNosync<<<1, istride>>>(idata, odata, istride);
        }
    }
}

int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    
    int size = 1 << 24;
    int blocksize = 512;
    
    if (argc > 1){
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);  // 1d
    dim3 grid ((size + block.x - 1) / block.x, 1);
    
    size_t nBytes = size  * sizeof(int);
    int * h_idata = (int*) malloc(nBytes);
    int * h_odata = (int*) malloc( grid.x * sizeof(int));  //you duoshao ge block
    int * temp = (int*) malloc(nBytes);

    //initial the array
    for (int i = 0 ; i < size;i++){
        h_idata[i] = (int)(rand() & 0xff);
    }

    int sum = 0;
    for (int i = 0 ; i < size;i++){
        sum += h_idata[i];
    }
    printf("sum value is : %d\n", sum);

    memcpy(temp, h_idata, nBytes);

    int gpu_sum = 0;

    int *d_idata = NULL;
    int *d_odata = NULL;

    cudaMalloc((void**)&d_idata, nBytes);
    cudaMalloc((void**)&d_odata, grid.x * sizeof(int));

    
    //cpu sum
    Timer timer;
    timer.start();

    int cpu_sum = recursiveReduce(temp, size);
    timer.stop();
    float elapsedTime = timer.elapsedms();

    printf("cpu reduce time: %f,  sum: %d\n", elapsedTime, cpu_sum);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime1 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i ++){
        gpu_sum += h_odata[i];
    }
    printf("warm up reduce time: %f,  sum: %d\n", elapsedTime1, gpu_sum);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime2 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceNeighbored gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime2, gpu_sum, grid.x, block.x);


    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime3 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceNeighboredLess gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime3, gpu_sum, grid.x, block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime4 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceInterleaved gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime4, gpu_sum, grid.x, block.x);


    

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime6 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x; i ++){
        gpu_sum += h_odata[i];
    }
    printf("gpuRecursiveReduceNosync gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime6, gpu_sum, grid.x, block.x);


    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();

    free(h_idata);
    free(h_odata);
    free(temp);

    return 0;
}