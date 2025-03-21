#include <cublas_v2.h>
#include <iostream>
#include <cmath>

void checkCublasError(cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " (error code: " << status << ")" << std::endl;
        exit(1);
    }
}

int main() {
    int m = 2;
    int n = 2; 
    int k = 2;
    float alpha = 1.0f, beta = 0.0f;
    int mk = m * k;
    int mn = m * n;
    int kn = k*n;
    // 输入矩阵
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0};

    // cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 执行矩阵乘法
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                        m, n, k, &alpha, A, m, B, k, &beta, C, m);
    checkCublasError(status, "cublasSgemm failed");

    // 打印结果
    std::cout << "C = [" << C[0] << ", " << C[1] << "; " << C[2] << ", " << C[3] << "]" << std::endl;

    // 计算误差
    float expected_C[4] = {19.0f, 22.0f, 43.0f, 50.0f}; // 预期结果
    float error = 0.0f;
    for (int i = 0; i < m * n; i++) {
        error += fabs(C[i] - expected_C[i]);
    }
    std::cout << "Total error: " << error << std::endl;

    // 释放资源
    cublasDestroy(handle);
    return 0;
}