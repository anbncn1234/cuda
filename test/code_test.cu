#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include "../common/common.h"

#define BLOCK_X 32
#define BLOCK_Y 32

__global__ void sgemm_v1(float* A, float* B, float* C, const int m,const int n,const int k, float alpha, float beta){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < n && iy < m){
        float tmp = 0.0;
        for (int i =0; i < k; i ++){
            //printf("%d %f\n",i, A[iy * k + i ] * B[ i * n + ix]);
            tmp += A[iy * k + i ] * B[ i * n + ix];
        }
        //printf("%d %d %f\n", ix , iy,A[iy * k + k - 1 ] * B[(k -1) * n + ix]);
        C[ix + iy * n] = alpha * tmp + beta * C[ix + iy * n];
    }
}

__global__ void sgemm_v2(float* A, float* B, float* C, const int m,const int n,const int k, float alpha, float beta){
    const int BM = 8;
    const int BN = 8;
    const int BK = 32;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    

}

__global__ void test(const int a){
    __shared__ float smem[a];
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

void initialData_ez(float *ip, int size)
{
    //time_t t;
    //srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = i & 0xfff;
    }
}


void Checkresult1(float* cpu_m, float* gpu_m, const int m, const int n){
    
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int row = 0; row < m; row ++){
        for (int col = 0; col< n; col ++){
            
            if (abs(cpu_m[row* n + col] - gpu_m[col * m + row])> epsilon){
                match = 0;
                printf("Array do not match\n");
                printf("cpu calc %5.8f gpu calc % 5.8f \n", cpu_m[row* n + col], gpu_m[col * m + row]);
                
                }
            }
        }
    if (match) printf("array matches\n");
    
}


float Checkresult(float* cpu_m, float* gpu_m, const int m, const int n){
    
    float max_error = 0.0;
    for (int row = 0; row < m; row ++){
        for (int col = 0; col< n; col ++){
            float this_error = abs(cpu_m[row* n + col] - gpu_m[col * m + row]);
            if (max_error != max_error || this_error != this_error) // nan
                max_error = -NAN;
            else
                printf("fail point %d %d \n", row, col);
                max_error = max(max_error, this_error);
        }
    }
    return max_error;
}

float Checkresult3(float* cpu_m, float* gpu_m, const int m, const int n){
    
    float max_error = 0.0;
    for (int row = 0; row < m; row ++){
        for (int col = 0; col< n; col ++){
            float this_error = abs(cpu_m[row* n + col] - gpu_m[row* n + col]);
            if (max_error != max_error || this_error != this_error) // nan
                max_error = -NAN;
            else
                //printf("fail point %d %d \n", row, col);
                max_error = max(max_error, this_error);
        }
    }
    return max_error;
}


float Checkresult2(float* cpu_m, float* gpu_m, const int size){
    
    float max_error = 0.0;
    for (int i = 0; i < size; i ++){
            float this_error = abs(cpu_m[i] - gpu_m[i]);
            if (max_error != max_error || this_error != this_error) // nan
                max_error = -NAN;
            else
                max_error = max(max_error, this_error);
    }
    return max_error;
}


void cpu_sgemm(float* A, float* B, float* C, const int m,const int n,const int k, float alpha, float beta){
    for (int y=0 ; y < m; y ++){
        for (int x=0 ; x < n; x++){
            //  x, y
            float tmp = 0;
            for (int i=0; i < k; i++){
                tmp += A[y * k + i] * B[i * n + x];
            }
            C[y * n + x] =alpha * tmp + beta * C[y * n + x];
        }
    }
}

int main(int argc, char** argv) {
    // 初始化 cuBLAS 句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 定义矩阵维度
    int m = 2;  // A 的行数
    int k = 2;  // A 的列数，B 的行数
    int n = 2;  // B 的列数
    int iKernel = 0;
    if(argc > 1) iKernel = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) n = atoi(argv[3]);
    if (argc > 4) k = atoi(argv[4]);

    // 定义矩阵数据（行优先）
    float *A, *B, *C, *h_ret, *h_ret2;
    int nBytes_A = m*k * sizeof(float);
    int nBytes_B = n*k * sizeof(float);
    int nBytes_C = m*n * sizeof(float);

    A = (float *)malloc(nBytes_A);
    B = (float *)malloc(nBytes_B);
    C = (float *)malloc(nBytes_C);
    h_ret = (float *)malloc(nBytes_C);
    h_ret2 = (float *)malloc(nBytes_C);
    
    // kernel pointer
    void (*kernel)(float*, float*,float*,int, int, int, float, float);
    char *kernelName;
    //sgemm_v1(float* A, float* B, float* C, const int m,const int n,const int k, float alpha, float beta)

    switch (iKernel){
        case 0:
            kernel = &sgemm_v1;
            kernelName = "sgemm_v1";
            break;
    }

    // 初始化矩阵 A 和 B
    initialData(A, m*k);
    initialData(B, n*k);
    initialData(C, n*m);

    // 在设备上分配内存
    float *d_A, *d_B, *d_C, *d_D;
    CHECK(cudaMalloc((float**)&d_A, sizeof(float) * m * k));
    CHECK(cudaMalloc((float**)&d_B, sizeof(float) * n * k));
    CHECK(cudaMalloc((float**)&d_C, sizeof(float) * m * n));
    CHECK(cudaMalloc((float**)&d_D, sizeof(float) * m * n));

    CHECK(cudaMemcpy(d_A, A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, nBytes_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, C, nBytes_C, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_D, C, nBytes_C, cudaMemcpyHostToDevice));


    // 定义 cuBLAS 中的标量
    float alpha = 1.0f;
    float beta = 1.0f;
    
    // 开始计时   
    Timer timer;
    timer.start();
    // 执行矩阵乘法 C = alpha * A * B + beta * C
    CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n));
    
    timer.stop();
    float elapsedTime = timer.elapsedms();

    // 计算浮点运算次数
    // 矩阵乘法浮点运算次数: (2k -1) * m * n 
    // 矩阵缩放浮点运算次数: m * n (alpha * AB) + m * n (beta * C)
    // 矩阵加法浮点运算次数: m * n (alpha * AB + beta * C)
    double flops = 2.0 * m * n * k + 2.0 * m * n;

    // 计算 FLOPS
    double tflops = (flops / (elapsedTime / 1000)) / 1e12 ;
    printf("cublasSgemm elapsedTime: %f ms, tflops %lf \n ", elapsedTime, tflops);
    CHECK(cudaMemcpy(h_ret, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    

    dim3 block(BLOCK_X,BLOCK_Y);
    dim3 grid((n + BLOCK_X - 1)/BLOCK_X, (m + BLOCK_Y - 1)/BLOCK_Y );

    Timer timer1;
    timer1.start();
    // 执行矩阵乘法 C = alpha * A * B + beta * C
    kernel<<<grid,block>>>(d_A, d_B, d_D, m, n, k,  alpha,  beta);
    timer1.stop();
    float elapsedTime1 = timer1.elapsedms();
    CHECK(cudaMemcpy(h_ret2, d_D, nBytes_C, cudaMemcpyDeviceToHost));
    double flops1 = 2.0 * m * n * k + 2.0 * m * n;
    
    // 计算 FLOPS
    double tflops1 = (flops1 / (elapsedTime1 / 1000)) / 1e12 ;
    printf("%s  elapsedTime: %f ms, tflops %lf \n " , kernelName,elapsedTime1, tflops1);

    float ret = Checkresult3(h_ret2,  h_ret, m,n);
    printf("max error %f\n", ret);
    
    

    // 释放设备内存
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_D));

    // 释放主机内存
    free(A);
    free(B);
    free(C);
    
    free(h_ret);
    free(h_ret2);

    // 销毁 cuBLAS 句柄
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}