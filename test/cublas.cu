#include "../common/common.h"
#include <stdio.h>
#include <cublas_v2.h>


namespace {
    int M = 2;
    int K = 2; 
    int N = 3;
    int MK = M * K;
    int KN = K * N;
    int MN = M * N;
}

void printMatrix(int R, int C, double *A, const char *name) {
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            printf("%10.6f", A[r * C + c]);
        }
        printf("\n");
    }
    printf("\n");
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

int main() {

    double *h_A = new double[MK];
    double *h_B = new double[KN];
    double *h_C = new double[MN];
    for (int i=0; i<MK; ++i) h_A[i] = i;
    printMatrix(M, K, h_A, "A");
    for (int i=0; i<KN; i++) h_B[i] = i;
    printMatrix(K, N, h_B, "B");
    for (int i=0; i<MN; ++i) h_C[i] = 0;
    
    double *g_A, *g_B, *g_C;
    CHECK(cudaMalloc((void **)&g_A, sizeof(double) * MK));
    CHECK(cudaMalloc((void **)&g_B, sizeof(double) * KN));
    CHECK(cudaMalloc((void **)&g_C, sizeof(double) * MN));
    
    cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);
    cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);
    cublasSetVector(MN, sizeof(double), h_C, 1, g_C, 1);

    gemm(g_A, g_B, g_C, 0);
    cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);
    printMatrix(M, N, h_C, "C = A * B");

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK(cudaFree(g_A));
    CHECK(cudaFree(g_B));
    CHECK(cudaFree(g_C));
    
    return 0;
}