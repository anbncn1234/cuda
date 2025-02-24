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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){  //
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

__global__ void reduceUnrolling8Warp( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x * 8;  //指针的间隔变成8个BLOCKDIM
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if (idx > n) return;  // boundary check

    //unrolling 8 data blocks 
    if (idx + 7 * blockDim.x < n){
        g_idata[idx] += (g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] + g_idata[idx + 3 * blockDim.x]+ g_idata[idx + 4 * blockDim.x] + g_idata[idx + 5 * blockDim.x] + g_idata[idx + 6 * blockDim.x] + g_idata[idx + 7 * blockDim.x]);
        
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1){  //  这地方改了 32！
        if (tid < stride){
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

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


__global__ void reduceUnrolling8WarpAll( int *g_idata, int *g_odata, unsigned int n){
    unsigned int tid = threadIdx.x;

    int * idata = g_idata + blockIdx.x * blockDim.x * 8;  //指针的间隔变成8个BLOCKDIM
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if (idx > n) return;  // boundary check

    //unrolling 8 data blocks 
    if (idx + 7 * blockDim.x < n){
        g_idata[idx] += (g_idata[idx + blockDim.x] + g_idata[idx + 2 * blockDim.x] + g_idata[idx + 3 * blockDim.x]+ g_idata[idx + 4 * blockDim.x] + g_idata[idx + 5 * blockDim.x] + g_idata[idx + 6 * blockDim.x] + g_idata[idx + 7 * blockDim.x]);
        
    }
    __syncthreads();

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
    reduceUnrolling2<<<grid.x /2, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime5 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i ++){
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

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceUnrolling8Warp<<<grid.x /8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime8 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x /8* sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceUnrolling8Warp gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime8, gpu_sum, grid.x / 8, block.x);

    //gpu sum
    cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timer.start();
    reduceUnrolling8WarpAll<<<grid.x /8, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize(); 
    timer.stop();
    float elapsedTime9 = timer.elapsedms();
    cudaMemcpy(h_odata, d_odata, grid.x /8* sizeof(int),cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 8; i ++){
        gpu_sum += h_odata[i];
    }
    printf("reduceUnrolling8WarpAll gpu reduce time: %f,  sum: %d, gird ,block (%d %d)\n", elapsedTime9, gpu_sum, grid.x / 8, block.x);


    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaDeviceReset();

    free(h_idata);
    free(h_odata);
    free(temp);

    return 0;
}