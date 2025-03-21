#include <cublas_v2.h>
#include <iostream>
#include "../common/common.h"

void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " (error code: " << status << ")" << std::endl;
        exit(1);
    }
}

int main() {
    int m = 2, n = 2, k = 2;
    float alpha = 1.0f, beta = 1.0f;
    float* A, *B, *C, *h_ret;
    A = (float *)malloc(sizeof(float) * m * k);
    B = (float *)malloc(sizeof(float) * n * k);
    C = (float *)malloc(sizeof(float) * m * m);
    h_ret = (float *)malloc(sizeof(float) * m * m);
    
    // 输入矩阵（行优先存储）
    for (int i = 0; i < m * k; i ++){
        A[i] = (float)i + 1;
    }
    for (int i = 0; i < n * k; i ++){
        B[i] = (float)i + 5.0;
    }
    for (int i = 0; i < n * m; i ++){
        C[i] = (float)i;
    }
    

    // cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, sizeof(float) * m * k));
    CHECK(cudaMalloc((float**)&d_B, sizeof(float) * n * k));
    CHECK(cudaMalloc((float**)&d_C, sizeof(float) * m * n));

    CHECK(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, C, sizeof(float) * n * m, cudaMemcpyHostToDevice));

    // 执行矩阵乘法
    // 由于A和B是行优先存储的，我们需要对它们进行转置
    // C = A * B （行优先存储）
    // 等价于 C^T = B^T * A^T （列优先存储）
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                       n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
    checkCublasError(status, "cublasSgemm failed");
    CHECK(cudaMemcpy(h_ret, d_C, sizeof(float) * n * m, cudaMemcpyDeviceToHost));

    // 打印结果（行优先存储）
    std::cout << "h_ret = [" << h_ret[0] << ", " << h_ret[1] << "; " << h_ret[2] << ", " << h_ret[3] << "]" << std::endl;

    free(A);
    free(B);
    free(C);
    
    free(h_ret);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    // 释放资源
    cublasDestroy(handle);
    return 0;
}