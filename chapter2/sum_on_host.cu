#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx< N; idx ++)
    {
        C[idx] = A[idx] + B[idx];
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

int main(int argc , char **argv)
{
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;

    h_A = (float *) malloc (nBytes);
    h_B = (float *) malloc (nBytes);
    h_C = (float *) malloc (nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}