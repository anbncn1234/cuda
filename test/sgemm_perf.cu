#include <cublas_v2.h>
#include <iostream>
#include <chrono>
#include "../common/common.h"



int main() {
    // 初始化 cuBLAS 句柄
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 定义矩阵维度
    int m = 4096;  // A 的行数
    int k = 4096;  // A 的列数，B 的行数
    int n = 4096;  // B 的列数

    // 定义矩阵数据（行优先）
    float *A, *B, *C;
    A = new float[m * k];
    B = new float[k * n];
    C = new float[m * n];

    // 初始化矩阵 A 和 B
    for (int i = 0; i < m * k; i++) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < k * n; i++) B[i] = static_cast<float>(rand()) / RAND_MAX;
    //for (int i = 0; i < m * n; i++) C[i] = 0.0f;
    for (int i = 0; i < m * n; i++) C[i] = static_cast<float>(rand()) / RAND_MAX;
    // 在设备上分配内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, sizeof(float) * m * k));
    CHECK(cudaMalloc((float**)&d_B, sizeof(float) * n * k));
    CHECK(cudaMalloc((float**)&d_C, sizeof(float) * m * n));

    // 将数据从主机复制到设备
    CHECK_CUBLAS(cublasSetMatrix(m, k, sizeof(float), A, m, d_A, m));
    CHECK_CUBLAS(cublasSetMatrix(k, n, sizeof(float), B, k, d_B, k));
    CHECK_CUBLAS(cublasSetMatrix(m, n, sizeof(float), C, m, d_C, m));

    // 定义 cuBLAS 中的标量
    float alpha = 1.0f;
    float beta = 1.0f;

    // 预热（避免冷启动影响性能）
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));

    // 开始计时
    int repeat = 10;
    for (int i = 0; i < repeat; i++){
        Timer timer;
        timer.start();
        // 执行矩阵乘法 C = alpha * A * B + beta * C
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
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
    }

    // 释放设备内存
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // 释放主机内存
    delete[] A;
    delete[] B;
    delete[] C;

    // 销毁 cuBLAS 句柄
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}