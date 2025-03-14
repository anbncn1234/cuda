#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <iostream>


int main(void){
    float a = 3.1415927f;
    float b = 3.1415928f;
    if(a == b){
        printf("a equal b\n");
    } else{
        printf("not equal\n");
    }

    double A = 3.1415927;
    double B = 3.1415928;
    if(A == B){
        printf("A equal B\n");
    } else{
        printf("not equal\n");
    }
    return 0;
}