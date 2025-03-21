#include "../common/common.h"
#include <stdio.h>
#include <cublas_v2.h>



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
        ip[i] = i & 0xff;
    }
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int) time(&t));

    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0xff) / 10.0f;
    }
}

void Checkresult1(float* cpu_m, float* gpu_m, const int m, const int n){
    
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int row = 0; row < m; row ++){
        for (int col = 0; col< n; col ++){
            
            if (abs(cpu_m[row* n + col] - gpu_m[col * m + row])> epsilon){
                match = 0;
                printf("Array do not match\n");
                printf("cpu calc %5.8f gpu calc % 5.8f \n", cpu_m[row* n + col], gpu_m[col * m + row]);
                break;
                }
            }
        }
    if (match) printf("array matches\n");
    
}

float Checkresult(float* cpu_m, float* gpu_m, const int m, const int n){
    
    float max_error = 0.0;
    for (int row = 0; row < m; row ++){
        for (int col = 0; col< n; col ++){
            float this_error = abs(cpu_m[row* n + col] - gpu_m[col * m + row]);
            if (max_error != max_error || this_error != this_error) // nan
                max_error = -NAN;
            else
                max_error = max(max_error, this_error);
        }
    }
    return max_error;
}

float Checkresult2(float* cpu_m, float* gpu_m, const int size){
    
    float max_error = 0.0;
    for (int i = 0; i < size; i ++){
            float this_error = abs(cpu_m[i] - gpu_m[i]);
            if (max_error != max_error || this_error != this_error) // nan
                max_error = -NAN;
            else
                max_error = max(max_error, this_error);
    }
    return max_error;
}

void cpu_sgemm(float* A, float* B, float* C, const int m,const int n,const int k, float alpha, float beta){
    for (int y=0 ; y < m; y ++){
        for (int x=0 ; x < n; x++){
            //  x, y
            float tmp = 0;
            for (int i=0; i < k; i++){
                tmp += A[y * k + i] * B[i * n + x];
            }
            C[y * n + x] =alpha * tmp + beta * C[y * n + x];
        }
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
    
    
    int m = 3;
    int n = 2;
    int k = 2;
    
    if (argc > 1) m = atoi(argv[1]);
    if (argc > 2) n = atoi(argv[2]);
    if (argc > 3) k = atoi(argv[3]);
    printf("m n k value: %d %d %d\n", m,n,k);
    //matrix A size m,K, martix B size K,N martirx C size m,N
    // sgemm    a * A * B + beta * C
    int mn = n * m;  
    int nk = n * k;
    int mk = m * k;
    float *h_A, *h_B, *h_C, *h_ret, *h_D;
    
    int nBytes_A = mk * sizeof(float);
    int nBytes_B = nk * sizeof(float);
    int nBytes_C = mn * sizeof(float);
    h_A = (float *)malloc(nBytes_A);
    h_B = (float *)malloc(nBytes_B);
    h_C = (float *)malloc(nBytes_C);
    h_ret = (float *)malloc(nBytes_C);
    h_D = (float *)malloc(nBytes_C);
    
    
    initialData_ez(h_A, mk);
    initialData_ez(h_B, nk);
    initialData_ez(h_C, mn);
    
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
        //h_C[i] = 1;
        printf("%f\t", h_C[i]);
    }
    
    //printf("\n");

    

    //device memory
    float* d_A, *d_B, *d_C ;
    CHECK(cudaMalloc((float**)&d_A, nBytes_A));
    CHECK(cudaMalloc((float**)&d_B, nBytes_B));
    CHECK(cudaMalloc((float**)&d_C, nBytes_C));

    CHECK(cudaMemcpy(d_A, h_A, nBytes_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes_B, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, h_C, nBytes_C, cudaMemcpyHostToDevice));

    
    float alpha = 1.0;
    float beta = 1.0;

    cpu_sgemm(h_A, h_B, h_C, m, n,k,alpha,beta);
    
    
    printf("C_out:");
    for(int i = 0; i< n*m; i++){
        
        printf("%f\t", h_C[i]);
    }
    printf("\n");
    

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
    //CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, d_A, k, d_B, n, &beta, d_C, m));
    //CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, d_B, k, d_A, m, &beta, d_C, n));  
    CHECK_CUBLAS( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n));                
    CHECK(cudaMemcpy(h_ret, d_C, nBytes_C, cudaMemcpyDeviceToHost));
    
    printf("h_ret:");
    for(int i = 0; i< n*m; i++){
        printf("%f\t", h_ret[i]);
    }
    printf("\n");
    

    //Checkresult1(h_C, h_ret,  m, n);
    float err;
    err = Checkresult2(h_C, h_ret,  m * n);
    printf("cpu sgemm - cublassgemm max diff %f\n", err);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_ret);
 
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));
    
    //cudaDeviceReset();
    
    return 0;
}