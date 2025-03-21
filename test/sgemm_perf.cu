#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include "../common/common.h"

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
        ip[i] = i & 0xff;
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

int main() {
    // 初始化 cuBLAS 句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 定义矩阵维度
    int m = 4096;  // A 的行数
    int k = 4096;  // A 的列数，B 的行数
    int n = 4096;  // B 的列数
    
    // 定义矩阵数据（行优先）
    float *A, *B, *C, *h_ret;
    int nBytes_A = m*k * sizeof(float);
    int nBytes_B = n*k * sizeof(float);
    int nBytes_C = m*n * sizeof(float);

    A = (float *)malloc(nBytes_A);
    B = (float *)malloc(nBytes_B);
    C = (float *)malloc(nBytes_C);
    h_ret = (float *)malloc(nBytes_C);
    

    // 初始化矩阵 A 和 B

    //for (int i = 0; i < m * k; i++) A[i] = static_cast<float>(rand()) / RAND_MAX;
    //for (int i = 0; i < k * n; i++) B[i] = static_cast<float>(rand()) / RAND_MAX;
    
    initialData(A, m*k);
    initialData(B, n*k);
    initialData(C, n*m);
    
    
    //for (int i = 0; i < m * n; i++) C[i] = 0.0f;
    //for (int i = 0; i < m * n; i++) C[i] = static_cast<float>(rand()) / RAND_MAX;
    
    // 在设备上分配内存

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, sizeof(float) * m * k));
    CHECK(cudaMalloc((float**)&d_B, sizeof(float) * n * k));
    CHECK(cudaMalloc((float**)&d_C, sizeof(float) * m * n));

    CHECK(cudaMemcpy(d_A, A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, nBytes_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, C, nBytes_C, cudaMemcpyHostToDevice));


    // 定义 cuBLAS 中的标量
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 预热（避免冷启动影响性能）
    //CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    //CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m));
    CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n));
    
    // 开始计时
    int repeat = 10;
    for (int i = 0; i < repeat; i++){
        
        Timer timer;
        timer.start();
        // 执行矩阵乘法 C = alpha * A * B + beta * C
        //CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
        //CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m));
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
        printf("run %d elapsedTime: %f ms, tflops %lf \n ", i ,elapsedTime, tflops);
        CHECK(cudaMemcpy(h_ret, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    }
    /*
    for (int i = 0; i < repeat; i++){
        
        Timer timer1;
        timer1.start();
        // 执行矩阵乘法 C = alpha * A * B + beta * C
        cpu_sgemm(A, B, C, m, n, k,  alpha,  beta);
        timer1.stop();
        float elapsedTime1 = timer1.elapsedms();

        double flops1 = 2.0 * m * n * k + 2.0 * m * n;
        
        // 计算 FLOPS
        double tflops1 = (flops1 / (elapsedTime1 / 1000)) / 1e12 ;
        printf("run %d elapsedTime: %f ms, tflops %lf \n ", i ,elapsedTime1, tflops1);
    }
    */

    
    // 释放设备内存
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // 释放主机内存
    free(A);
    free(B);
    free(C);
    
    free(h_ret);

    // 销毁 cuBLAS 句柄
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}