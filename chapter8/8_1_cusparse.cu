#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>

int main(void){
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceprop;
    CHECK(cudaGetDeviceProperties(&deviceprop,dev));
    printf("device %d: %s \n", dev, deviceprop.name);

    cudaStream_t *handle = (cudaStream_t *)malloc(sizeof(cudaStream_t));
    //create cusparse handle
    cusparseCreate(&handle)

    /*
    tensor([[1, 7, 0, 0],
        [0, 2, 8, 0],
        [5, 0, 3, 9],
        [0, 6, 0, 4]])
    >>> sp.csr()
    (row_ptr = tensor([0, 2, 4, 7, 9]), 
    col_ind = tensor([0, 1, 1, 2, 0, 2, 3, 1, 3]), 
    values = tensor([1, 7, 2, 8, 5, 3, 9, 6, 4]))

     */
    int n_vals = 9;
    int n_rows = 4;
    int n_cols = 4;

    float *h_csrVals;
    int *h_csrCols;
    int *h_csrRows;

    h_csrVals[9] =  [1.0, 7.0, 2.0, 8.0, 5.0, 3.0, 9.0, 6.0, 4.0];
    h_csrCols[9] = [0, 1, 1, 2, 0, 2, 3, 1, 3];
    h_csrRows[n_rows + 1] = [0, 2, 4, 7, 9];


    float *d_csrVals;
    int *d_csrCols;
    int *d_csrRows; 
    
    cudaMalloc((void **)&d_csrVals, n_vals * sizeof(float));
    cudaMalloc((void **)&d_csrCols, n_vals * sizeof(int));
    cudaMalloc((void **)&d_csrRows, (n_rows + 1) * sizeof(int));

    cudaMemcpy(d_csrVals, h_csrVals, n_vals* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrCols, h_csrCols, n_vals* sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRows, h_csrRows, (n_rows + 1)* sizeof(int), cudaMemcpyHostToDevice);

    
    free(h_csrVals);
    free(h_csrRows);
    free(h_csrCols);
    cudaFree(d_csrVals);
    cudaFree(d_csrCols);
    cudaFree(d_csrRows);

    cudaDeviceReset();

    return 0;
}