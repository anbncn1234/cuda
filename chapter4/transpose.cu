#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void tranposeHost(float* out, float* in, const int nx, const int ny){
    for (int i = 0; i < nx ; i ++){
        for (int j = 0; j < ny; j++){
            out[i*ny + j] = in[j * nx + i];
        }
    }

}