#include <cufft.h>
#include <iostream>
#include <cmath>
#include <vector>

#define N 64  // 信号长度
#define M_PI 3.14
int main() {
    // 初始化 cuFFT 计划
    cufftHandle plan_forward, plan_inverse;
    cufftPlan1d(&plan_forward, N, CUFFT_R2C, 1);  // 正向 FFT（实数到复数）
    cufftPlan1d(&plan_inverse, N, CUFFT_C2R, 1);  // 反向 FFT（复数到实数）

    // 分配主机内存
    std::vector<float> h_signal(N);                // 时域信号（实数）
    std::vector<cufftComplex> h_spectrum(N / 2 + 1); // 频域信号（复数）
    std::vector<float> h_filtered_signal(N);       // 滤波后的时域信号（实数）

    // 生成频率为 5 的正弦波信号
    float frequency = 5.0;
    for (int i = 0; i < N; ++i) {
        h_signal[i] = sin(2.0 * M_PI * frequency * i / N);
    }

    // 分配设备内存
    cufftReal* d_signal;
    cufftComplex* d_spectrum;
    cufftReal* d_filtered_signal;
    cudaMalloc((void**)&d_signal, sizeof(cufftReal) * N);
    cudaMalloc((void**)&d_spectrum, sizeof(cufftComplex) * (N / 2 + 1));
    cudaMalloc((void**)&d_filtered_signal, sizeof(cufftReal) * N);

    // 将时域信号复制到设备
    cudaMemcpy(d_signal, h_signal.data(), sizeof(cufftReal) * N, cudaMemcpyHostToDevice);

    // 执行正向 FFT（时域 -> 频域）
    cufftExecR2C(plan_forward, d_signal, d_spectrum);

    // 将频域信号复制回主机
    cudaMemcpy(h_spectrum.data(), d_spectrum, sizeof(cufftComplex) * (N / 2 + 1), cudaMemcpyDeviceToHost);

    // 设计滤波器（频率为 4 的理想带通滤波器）
    float filter_frequency = 4.0;
    for (int i = 0; i < N / 2 + 1; ++i) {
        float bin_frequency = static_cast<float>(i);
        if (fabs(bin_frequency - filter_frequency) < 0.5) {
            h_spectrum[i].x *= 1.0;  // 保留频率为 4 的分量
            h_spectrum[i].y *= 1.0;
        } else {
            h_spectrum[i].x = 0.0;  // 滤除其他频率分量
            h_spectrum[i].y = 0.0;
        }
    }

    // 将滤波后的频域信号复制回设备
    cudaMemcpy(d_spectrum, h_spectrum.data(), sizeof(cufftComplex) * (N / 2 + 1), cudaMemcpyHostToDevice);

    // 执行反向 FFT（频域 -> 时域）
    cufftExecC2R(plan_inverse, d_spectrum, d_filtered_signal);

    // 将滤波后的时域信号复制回主机
    cudaMemcpy(h_filtered_signal.data(), d_filtered_signal, sizeof(cufftReal) * N, cudaMemcpyDeviceToHost);

    // 归一化（因为 cuFFT 不自动归一化）
    for (int i = 0; i < N; ++i) {
        h_filtered_signal[i] /= N;
    }

    // 打印滤波后的时域信号
    std::cout << "Filtered Signal (Time Domain):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << h_filtered_signal[i] << std::endl;
    }

    // 释放资源
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    cudaFree(d_signal);
    cudaFree(d_spectrum);
    cudaFree(d_filtered_signal);

    return 0;
}