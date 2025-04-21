#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#define M 3  // 矩阵 A 的行数
#define N 2  // 矩阵 A 的列数 / 矩阵 B 的行数
#define K 4  // 矩阵 B 的列数

int main() {
    // 定义矩阵 A (M x N), 矩阵 B (N x K), 矩阵 C (M x K)
    float h_A[M * N] = {1, 2, 3, 4, 5, 6};  // 矩阵 A
    float h_B[N * K] = {1, 2, 3, 4, 5, 6, 7, 8};  // 矩阵 B
    float h_C[M * K] = {0};  // 矩阵 C（结果）

    // cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    // 将数据从主机复制到设备
    cublasSetMatrix(M, N, sizeof(float), h_A, M, d_A, M);
    cublasSetMatrix(N, K, sizeof(float), h_B, N, d_B, N);

    // 执行矩阵乘法 C = A * B
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, K, N, &alpha, d_A, M, d_B, N, &beta, d_C, M);

    // 将结果从设备复制回主机
    cublasGetMatrix(M, K, sizeof(float), d_C, M, h_C, M);

    // 打印结果
    printf("Matrix C (Result of A * B):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", h_C[i + j * M]);
        }
        printf("\n");
    }

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 销毁 cuBLAS 句柄
    cublasDestroy(handle);

    return 0;
}