#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No hay GPUs compatibles con CUDA." << std::endl;
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "GPU detectada: " << prop.name << std::endl;
        std::cout << "Capacidad de computo: " << prop.major << "." << prop.minor << std::endl;
    }
    return 0;
}