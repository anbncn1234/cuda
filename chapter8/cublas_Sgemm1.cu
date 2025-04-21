#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/common.h"
#define M 1024
#define N 1024
#define K 1024

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void printMatrix(int rows, int cols, float *matrix, const char *name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void gemm(const double* g_A, const double* g_B, double* g_C, int method) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0;
    double beta = 0.0;
    switch (method)
    {
    case 0:
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
        &alpha, g_B, N, g_A, K, &beta, g_C, N);
        break;
    case 1:
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
        &alpha, g_A, K, g_B, N, &beta, g_C, M);
        break;
    default:
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
        &alpha, g_B, N, g_A, K, &beta, g_C, N);
        break;
    }
    
    cublasDestroy(handle);
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop, dev));
    printf("device %d: %s \n", dev, deviceprop.name);

    //cublasHandle_t handle;
    //cublasCreate_v2(&handle);  
    int m = M;
    int n = N;
    int k = K;

    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) k = atoi(argv[3]);

    // matrix A size m*K, matrix B size K*N, matrix C size m*N
    int mn = m * n;
    int nk = n * k;
    int mk = m * k;
    float *h_A, *h_B, *h_C, *h_ret;

    int nBytes_A = mk * sizeof(float);
    int nBytes_B = nk * sizeof(float);
    int nBytes_C = mn * sizeof(float);
    h_A = (float *)malloc(nBytes_A);
    h_B = (float *)malloc(nBytes_B);
    h_C = (float *)malloc(nBytes_C);
    h_ret = (float *)malloc(nBytes_C);

    initialData(h_A, mk);
    initialData(h_B, nk);
    initialData(h_C, mn);

    printMatrix(m, k, h_A, "A");
    printMatrix(k, n, h_B, "B");
    printMatrix(m, n, h_C, "C");

    // device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes_A);
    cudaMalloc((float**)&d_B, nBytes_B);
    cudaMalloc((float**)&d_C, nBytes_C);

    cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes_C, cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float beta = 1.0;

    /*
    cublasSgemm(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_T,
                m,
                n,
                k,
                &alpha,
                d_A,
                k,
                d_B,
                n,
                &beta,
                d_C,
                m);
    */

    cudaMemcpy(h_ret, d_C, nBytes_C, cudaMemcpyDeviceToHost);

    printMatrix(m, n, h_ret, "h_ret");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ret);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //cublasDestroy(handle);
    cudaDeviceReset();
    return 0;
}