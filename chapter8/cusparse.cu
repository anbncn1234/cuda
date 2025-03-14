#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>

// 检查 CUDA 错误
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

// 检查 cuSPARSE 错误
#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "cuSPARSE Error: " << status << std::endl;                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

int main() {
    // 初始化稠密矩阵 (4x4)
    const int m = 4; // 行数
    const int n = 4; // 列数
    const int nnz = 9; // 非零元素个数
    double h_A[m * n] = {
        1, 0, 0, 2,
        0, 3, 4, 0,
        5, 0, 6, 0,
        0, 7, 0, 8
    };

    // 打印稠密矩阵
    std::cout << "Dense Matrix:" << std::endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // 初始化 cuSPARSE
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // 分配设备内存并拷贝稠密矩阵到设备
    double *d_A;
    CHECK_CUDA(cudaMalloc((void**)&d_A, m * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * n * sizeof(double), cudaMemcpyHostToDevice));

    // 创建稠密矩阵描述符
    cusparseDnMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matA, m, n, n, d_A, CUDA_R_64F, CUSPARSE_ORDER_ROW));

    // 创建稀疏矩阵描述符 (CSR 格式)
    cusparseSpMatDescr_t matB;
    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal;
    CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtr, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrVal, nnz * sizeof(double)));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, m, n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal,
                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // 分配缓冲区
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
    void *d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // 执行稠密矩阵转稀疏矩阵
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

    // 将结果拷贝回主机
    int h_csrRowPtr[m + 1];
    int h_csrColInd[nnz];
    double h_csrVal[nnz];
    CHECK_CUDA(cudaMemcpy(h_csrRowPtr, d_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csrColInd, d_csrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csrVal, d_csrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost));

    // 打印 CSR 格式的结果
    std::cout << "\nCSR RowPtr:" << std::endl;
    for (int i = 0; i <= m; i++) {
        std::cout << h_csrRowPtr[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "CSR ColInd:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        std::cout << h_csrColInd[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "CSR Val:" << std::endl;
    for (int i = 0; i < nnz; i++) {
        std::cout << h_csrVal[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColInd));
    CHECK_CUDA(cudaFree(d_csrVal));
    CHECK_CUDA(cudaFree(d_buffer));

    // 销毁描述符和句柄
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}