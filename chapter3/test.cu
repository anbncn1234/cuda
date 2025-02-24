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

__global__ void reduceUnrolling2( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x * 2;  //指针的间隔变成2个BLOCKDIM
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    if (idx > n) return;  // boundary check

    //unrolling 2 data blocks  ，先把2个block的数加起来？
    if (idx + blockDim.x < n){
        g_idata[idx] += g_idata[idx + blockDim.x];
        
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){  //
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}


__global__ void reduceUnrolling4( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x * 4;  //指针的间隔变成4个BLOCKDIM
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    if (idx > n) return;  // boundary check

    //unrolling 4 data blocks 
    if (idx +  3 * blockDim.x < n) {
        g_idata[idx] += (g_idata[idx + 1 * blockDim.x]);
        g_idata[idx] += (g_idata[idx + 2 * blockDim.x]);
        g_idata[idx] += (g_idata[idx + 3 * blockDim.x]);
    }
    __syncthreads();

    for (int stride = blockDim.x / 4; stride > 0; stride >>= 1){  //
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

__global__ void reduceUnrolling8( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x * 8;  //指针的间隔变成4个BLOCKDIM
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if (idx > n) return;  // boundary check

    //unrolling 4 data blocks 
    if (idx + 7 * blockDim.x < n){
        g_idata[idx] += (g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] + g_idata[idx + 3 * blockDim.x]+ g_idata[idx + 4 * blockDim.x] + g_idata[idx + 5 * blockDim.x] + g_idata[idx + 6 * blockDim.x] + g_idata[idx + 7 * blockDim.x]);
        //printf("tid %d %d\n", tid, g_idata[idx]);
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){  //
        
        if (tid < stride){
            printf("stride: %d tid %d : %d %d \n",stride, tid, idata[tid], idata[tid + stride]);
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if  (tid == 0){ g_odata[blockIdx.x] = idata[0];}
}

int main(int argc , char **argv)
{
    printf("%s starting\n", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("Using Device %d : %s\n", dev, deviceprop.name);
    
    int size = 1 << 8;
    int blocksize = 32;
    
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
        h_idata[i] = i & 0xff;
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
    reduceUnrolling2<<<grid.x /2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime5 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i ++){
        printf(" %d : %d\n",i,h_odata[i]);
        gpu_sum += h_odata[i];
    }
    printf("reduceUnrolling2 gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime5, gpu_sum, grid.x / 2, block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceUnrolling4<<<grid.x /4, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime6 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x/4 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 4; i ++){
        printf(" %d : %d\n",i,h_odata[i]);
        gpu_sum += h_odata[i];
    }
    printf("reduceUnrolling4 gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime6, gpu_sum, grid.x / 4, block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceUnrolling8<<<grid.x /8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime7 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x /8* sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceUnrolling8 gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime7, gpu_sum, grid.x / 8, block.x);

    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();

    free(h_idata);
    free(h_odata);
    free(temp);

    return 0;
}