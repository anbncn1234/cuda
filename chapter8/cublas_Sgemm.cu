#include "../common/common.h"
#include <stdio.h>
#include <cublas_v2.h>


#define M 1024
#define N 1024
#define K 1024


void printMatrix(int R, int C, float *A, const char *name) {
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            printf("%10.6f", A[r * C + c]);
        }
        printf("\n");
    }
    printf("\n");
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

/*
 cublasSgemm(
          handle,
          CUBLAS_OP_T,   //矩阵A的属性参数，转置，按行优先
          CUBLAS_OP_T,   //矩阵B的属性参数，转置，按行优先
          A_ROW,          //矩阵A、C的行数
          B_COL,          //矩阵B、C的列数
          A_COL,          //A的列数，B的行数，此处也可为B_ROW,一样的
          &a,             //alpha的值
          d_A,            //左矩阵，为A
          A_COL,          //A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
          d_B,            //右矩阵，为B
          B_COL,          //B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
          &b,             //beta的值
          d_C,            //结果矩阵C
          A_ROW           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
  );
*/
int main( int argc, char** argv ){

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s \n", dev, deviceprop.name);

    cublasHandle_t handle;
    cublasCreate(&handle);
    int m = M;
    int n = N;
    int k = K;

    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) k = atoi(argv[3]);

    //matrix A size m,K, martix B size K,N martirx C size m,N
    // sgemm    a * A * B + beta * C
    int mn = n * m;  
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
    

    initialData(h_A, nBytes_A);
    initialData(h_B, nBytes_B);
    initialData(h_C, nBytes_C);

    printMatrix(n, m, h_A, "A");
    printMatrix(k, n, h_B, "B");
    printMatrix(k, m, h_C, "C");

    //device memory
    float* d_A, *d_B, *d_C ;
    cudaMalloc((float**)&d_A, nBytes_A);
    cudaMalloc((float**)&d_B, nBytes_B);
    cudaMalloc((float**)&d_C, nBytes_C);

    cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, nBytes_C, cudaMemcpyHostToDevice);

    float alpha = 1.0;
    float beta = 1.0;

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
    m           
    );
    cudaMemcpy(h_ret, d_C, nBytes_C, cudaMemcpyDeviceToHost);

    printMatrix(n, m, h_ret, "h_ret");

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaDeviceReset();
    return 0;
}