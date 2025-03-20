#include <cufft.h>
#include <iostream>
#include <cmath>
#include <vector>

#define N 64  // 信号长度
#define M_PI 3.14
int main() {
    // 初始化 cuFFT 计划
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, 1);

    // 分配主机内存（输入和输出）
    std::vector<float> h_signal(N);          // 时域信号（实数）
    std::vector<cufftComplex> h_spectrum(N / 2 + 1); // 频域信号（复数）

    // 生成正弦信号
    float frequency = 5.0;  // 正弦波的频率
    for (int i = 0; i < N; ++i) {
        h_signal[i] = sin(2.0 * M_PI * frequency * i / N); // 生成正弦波
    }

    // 分配设备内存
    cufftReal* d_signal;
    cufftComplex* d_spectrum;
    cudaMalloc((void**)&d_signal, sizeof(cufftReal) * N);
    cudaMalloc((void**)&d_spectrum, sizeof(cufftComplex) * (N / 2 + 1));

    // 将数据从主机复制到设备
    cudaMemcpy(d_signal, h_signal.data(), sizeof(cufftReal) * N, cudaMemcpyHostToDevice);

    // 执行 FFT
    cufftExecR2C(plan, d_signal, d_spectrum);

    // 将结果从设备复制回主机
    cudaMemcpy(h_spectrum.data(), d_spectrum, sizeof(cufftComplex) * (N / 2 + 1), cudaMemcpyDeviceToHost);

    // 打印频域结果
    std::cout << "Frequency Domain (FFT Result):" << std::endl;
    for (int i = 0; i < N / 2 + 1; ++i) {
        float magnitude = sqrt(h_spectrum[i].x * h_spectrum[i].x + h_spectrum[i].y * h_spectrum[i].y);
        std::cout << "Bin " << i << ": (" << h_spectrum[i].x << ", " << h_spectrum[i].y << ") Magnitude: " << magnitude << std::endl;
    }

    // 释放资源
    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_spectrum);

    return 0;
}