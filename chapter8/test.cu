#include <cusparse.h>
#include <iostream>
#include <cublas.h>
 
int main() {
    std::cout << "cuSPARSE version: " << CUSPARSE_VERSION / 1000 << "." << (CUSPARSE_VERSION % 1000) / 100 << "." << (CUSPARSE_VERSION % 100) << std::endl;
    std::cout << "cublas version: " << CUBLAS_VERSION / 1000 << "." << (CUBLAS_VERSION % 1000) / 100 << "." << (CUBLAS_VERSION % 100) << std::endl;
    return 0;
}