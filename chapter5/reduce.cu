#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>

#define DIM 128

int recursiveReduce(int *data, int const size){
    if (size == 1) return data[0];
    int const stride = size /2;
    for (int i = 0; i < stride; i ++){
        data[i] += data[i + stride];
    }
    return recursiveReduce( data, stride);
}

__global__ void warmup( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid  = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (blockDim.x >= 1024 &&  tid < 512) idata[tid] += idata[tid+ 512];
    __syncthreads();
    if (blockDim.x >= 512 &&  tid < 256) idata[tid] += idata[tid+ 256];
    __syncthreads();
    if (blockDim.x >= 256 &&  tid < 128) idata[tid] += idata[tid+ 128];
    __syncthreads();
    if (blockDim.x >= 128 &&  tid < 64) idata[tid] += idata[tid+ 64];
    __syncthreads();

    if (tid < 32){
        volatile int *vmem  = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

__global__ void reduceGmem( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid  = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (blockDim.x >= 1024 &&  tid < 512) idata[tid] += idata[tid+ 512];
    __syncthreads();
    if (blockDim.x >= 512 &&  tid < 256) idata[tid] += idata[tid+ 256];
    __syncthreads();
    if (blockDim.x >= 256 &&  tid < 128) idata[tid] += idata[tid+ 128];
    __syncthreads();
    if (blockDim.x >= 128 &&  tid < 64) idata[tid] += idata[tid+ 64];
    __syncthreads();

    if (tid < 32){
        volatile int *vmem  = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n){
    __shared__ int smem[DIM];

    unsigned int tid  = threadIdx.x;
    // convert global data pointer to local pointer
    int *idata = g_idata + blockIdx.x * blockDim.x;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    //set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    if (blockDim.x >= 1024 &&  tid < 512) smem[tid] += smem[tid+ 512];
    __syncthreads();
    if (blockDim.x >= 512 &&  tid < 256) smem[tid] += smem[tid+ 256];
    __syncthreads();
    if (blockDim.x >= 256 &&  tid < 128) smem[tid] += smem[tid+ 128];
    __syncthreads();
    if (blockDim.x >= 128 &&  tid < 64) smem[tid] += smem[tid+ 64];
    __syncthreads();

    if (tid < 32){
        volatile int *vsmem  = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if  (tid == 0){ g_odata[blockIdx.x] = smem[0];}
}

__global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int n){
    __shared__ int smem[DIM];

    unsigned int tid  = threadIdx.x;
    // convert global data pointer to local pointer
    int *idata = g_idata + blockIdx.x * blockDim.x;

    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    //unrolling 4 blocks
    int tmpSum = 0;
    if (idx + 3 * blockDim.x <= n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    //set to smem by each threads
    smem[tid] = tmpSum;
    __syncthreads();

    if (blockDim.x >= 1024 &&  tid < 512) smem[tid] += smem[tid+ 512];
    __syncthreads();
    if (blockDim.x >= 512 &&  tid < 256) smem[tid] += smem[tid+ 256];
    __syncthreads();
    if (blockDim.x >= 256 &&  tid < 128) smem[tid] += smem[tid+ 128];
    __syncthreads();
    if (blockDim.x >= 128 &&  tid < 64) smem[tid] += smem[tid+ 64];
    __syncthreads();

    if (tid < 32){
        volatile int *vsmem  = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if  (tid == 0){ g_odata[blockIdx.x] = smem[0];}
}


__global__ void reduceSmemUnrollDyn(int *g_idata, int *g_odata, unsigned int n){
    extern __shared__ int smem[];

    unsigned int tid  = threadIdx.x;
    // convert global data pointer to local pointer
    int *idata = g_idata + blockIdx.x * blockDim.x;

    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    //unrolling 4 blocks
    int tmpSum = 0;
    if (idx + 3 * blockDim.x <= n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    //set to smem by each threads
    smem[tid] = tmpSum;
    __syncthreads();

    if (blockDim.x >= 1024 &&  tid < 512) smem[tid] += smem[tid+ 512];
    __syncthreads();
    if (blockDim.x >= 512 &&  tid < 256) smem[tid] += smem[tid+ 256];
    __syncthreads();
    if (blockDim.x >= 256 &&  tid < 128) smem[tid] += smem[tid+ 128];
    __syncthreads();
    if (blockDim.x >= 128 &&  tid < 64) smem[tid] += smem[tid+ 64];
    __syncthreads();

    if (tid < 32){
        volatile int *vsmem  = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if  (tid == 0){ g_odata[blockIdx.x] = smem[0];}
}


int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    
    int size = 1 << 24;
    int blocksize = 512;
    
    if (argc > 1){
        blocksize = atoi(argv[1]);
    }

    dim3 block(DIM, 1);  // 1d
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
    warmup<<<grid.x, block>>>(d_idata, d_odata, size);
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
    reduceGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    elapsedTime1 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x ; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceGmem gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime1, gpu_sum, grid.x , block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    elapsedTime1 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x ; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceSmem gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime1, gpu_sum, grid.x , block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceSmemUnroll<<<grid.x /4 , block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    elapsedTime1 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x /4 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4 ; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceSmemUnroll gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime1, gpu_sum, grid.x / 4, block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceSmemUnrollDyn<<<grid.x /4 , block, DIM * sizeof(int)>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    elapsedTime1 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x /4 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4 ; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceSmemUnrollDyn gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime1, gpu_sum, grid.x / 4, block.x);

    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();

    free(h_idata);
    free(h_odata);
    free(temp);

    return 0;
}