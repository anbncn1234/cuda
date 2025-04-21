#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include "../common/common.h"
#include <iostream>

int main() {
    // 定义 FFT 变换的相关参数
    int n = 8; // 输入数据的大小
    int size = sizeof(cufftComplex) * n;
 
    // 定义输入和输出数组
    cufftComplex *data;
    cufftComplex *result;
 
    // 分配内存
    cudaMalloc((void**)&data, size);
    cudaMalloc((void**)&result, size);
 
    // 初始化输入数据（假设为一些复数值）
    // 这里只是示例，实际数据可以根据需求进行初始化
    cufftComplex *hostData = (cufftComplex*)malloc(size);
    for (int i = 0; i < n; ++i) {
        hostData[i].x = i;  // 实部
        hostData[i].y = 0;  // 虚部
    }
 
    // 将输入数据从主机内存拷贝到 GPU 内存中
    cudaMemcpy(data, hostData, size, cudaMemcpyHostToDevice);
 
    // 创建 cuFFT 计划
    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, 1);
 
    // 执行 FFT 变换
    cufftExecC2C(plan, data, result, CUFFT_FORWARD);
 
    // 释放 cuFFT 计划
    cufftDestroy(plan);
 
    // 将结果从 GPU 内存拷贝回主机内存
    cudaMemcpy(hostData, result, size, cudaMemcpyDeviceToHost);
 
    // 打印输出结果
    printf("FFT Result:\n");
    for (int i = 0; i < n; ++i) {
        printf("(%f, %f)\n", hostData[i].x, hostData[i].y);
    }
 
    // 释放内存
    cudaFree(data);
    cudaFree(result);
    free(hostData);
 
    return 0;
}