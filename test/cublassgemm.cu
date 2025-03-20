#include "../common/common.h"
#include <stdio.h>
#include <cublas_v2.h>

#define M 3
#define N 2
#define K 2

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

void initialData_ez(float *ip, int size)
{
    //time_t t;
    //srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = i;
    }
}


void printMatrixCol(float* A, int row, int col){
    for (int i = 0; i < row; i ++){
        for (int j = 0; j < col; j ++){
            //printf("%f offset %d row %d col %d \t", A[row * j + i], (row * j + i), i, j);
            printf("%f \t", A[row * j + i]);
        }
        printf("\n");
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
    
    
    int m = M;
    int n = N;
    int k = K;
    
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) k = atoi(argv[3]);
    printf("m n k value: %d %d %d\n", m,n,k);
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
    
    
    initialData_ez(h_A, mk);
    initialData_ez(h_B, nk);
    
    printf("A:");
    for(int i = 0; i< m*k; i++){
        
        printf("%f\t", h_A[i]);
    }
    printf("\n");

    printf("B:");
    for(int i = 0; i< n*k; i++){
        
        printf("%f\t", h_B[i]);
    }
    printf("\n");
    printf("C:");
    for(int i = 0; i< n*m; i++){
        h_C[i] = 2;
        printf("%f\t", h_C[i]);
    }
    printf("\n");


    //device memory
    float* d_A, *d_B, *d_C ;
    CHECK(cudaMalloc((float**)&d_A, nBytes_A));
    CHECK(cudaMalloc((float**)&d_B, nBytes_B));
    CHECK(cudaMalloc((float**)&d_C, nBytes_C));

    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, h_C, nBytes_C, cudaMemcpyHostToDevice));

    
    float alpha = 1.0;
    float beta = 0;
    
    // cublas是按列优先的原则存储矩阵信息的，相当于对原始矩阵做了一个转置，要想不干扰正常的矩阵运算，我们就需要再做一次转置，两次转置就得到了原始矩阵，
    
    /*cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,d_A,            
    m,   //LDA       
    d_B,            
    k,   //LDB 
    &beta,             
    d_C,            
    m    //LDC
    );
    */
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    //CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m));
    //CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, d_B, k, d_A, m, &beta, d_C, n));                          
    CHECK(cudaMemcpy(h_ret, d_C, nBytes_C, cudaMemcpyDeviceToHost));
   
    printf("h_ret:");
    for(int i = 0; i< n*m; i++){
        printf("%f\t", h_ret[i]);
    }
    printf("\n");

    printMatrixCol(h_ret, m, n);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ret);
 
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    //cudaDeviceReset();
    
    return 0;
}