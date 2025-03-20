#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <iostream>
int main(void){
    int const m = 5;
    int const n = 3;
    int const k = 2;
    float *A ,*B,*C;
    float *d_A,*d_B,*d_C;
    A = (float*)malloc(sizeof(float)*m*k);  //在内存中开辟空间
    B = (float*)malloc(sizeof(float)*n*k);  //在内存中开辟空间
    C = (float*)malloc(sizeof(float)*m*n); //在内存中开辟空间
    printf("A:\n");
    for(int i = 0; i< m*k; i++){
        A[i] = i;
        std::cout <<A[i]<<"\t";
    }
    std::cout <<"\n";

    printf("B:\n");
    for(int i = 0; i< n*k; i++){
        B[i] = i;
        std::cout <<B[i]<<"\t";
    }
    std::cout <<"\n";
    float alpha = 1.0;
    float beta = 0.0;
    cudaMalloc((void**)&d_A,sizeof(float)*m*k);
    cudaMalloc((void**)&d_B,sizeof(float)*n*k);
    cudaMalloc((void**)&d_C,sizeof(float)*m*n);

    printf("C init:\n");
    for (int i = 0; i< m*n;i++){
        std::cout <<C[i]<<"\t";
    }
    std::cout <<"\n";
    cudaMemcpy(d_A,A,sizeof(float)*m*k,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*n*k,cudaMemcpyHostToDevice);
    for (int i = 0; i< m*k;i++){
        std::cout <<A[i]<<"\t";
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);//<测试一>
    //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, d_A, k, d_B, k, &beta, d_C, m);//<测试二>
    cudaMemcpy(C,d_C,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
    
    printf("C finial:\n");
    for (int i = 0; i< m*n;i++){
        std::cout <<C[i]<<"\t";
    }
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}