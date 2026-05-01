#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cmath>

/**
 * @brief Macro para verificación de errores de CUDA
 * Uso: gpuErrchk( cudaMalloc(&d_ptr, size) );
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/**
 * @brief Verificación de errores después de lanzamiento de kernel (async)
 * OPTIMIZACIÓN PARA ADA: Sin cudaDeviceSynchronize() para mantener pipelining
 * Uso: gpuKernelCheck( kernel<<<blocks, threads>>>(args) );
 */
#define gpuKernelCheck(ans) { gpuKernelAssert((ans), __FILE__, __LINE__); }
inline void gpuKernelAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    // Check para errores de lanzamiento (no requiere sincronización)
    gpuAssert(cudaPeekAtLastError(), file, line, abort);
    // NOTE: cudaDeviceSynchronize() removido - permite pipelining de GPU
    // Para verificación síncrona completa, usar gpuKernelCheckSync() en benchmarks
}

/**
 * @brief Verificación síncrona de kernel (para benchmarking/debugging)
 * NOTA: Impacta performance - usar solo cuando necesites sincronización
 */
#define gpuKernelCheckSync(ans) { gpuKernelAssertSync((ans), __FILE__, __LINE__); }
inline void gpuKernelAssertSync(cudaError_t code, const char *file, int line, bool abort=true) {
    gpuAssert(cudaPeekAtLastError(), file, line, abort);
    gpuAssert(cudaDeviceSynchronize(), file, line, abort);
}

/**
 * @brief Imprime información básica del dispositivo CUDA
 */
inline void printDeviceInfo(int deviceId = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    std::cout << "===========================================\n";
    std::cout << "Dispositivo CUDA [" << deviceId << "]\n";
    std::cout << "  Nombre: " << prop.name << "\n";
    std::cout << "  Capacidad de cómputo: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Memoria global: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Memoria compartida por bloque: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "  Registros por bloque: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n";
    std::cout << "  Max hilos por bloque: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  Max bloques por grid (dim): ["
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << "]\n";
    std::cout << "  Max hilos por bloque (dim): ["
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << "]\n";
    std::cout << "  MultiProcessorCount: " << prop.multiProcessorCount << "\n";
    std::cout << "===========================================\n";
}

/**
 * @brief Imprime información detallada de todos los dispositivos
 */
inline void printAllDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Dispositivos CUDA encontrados: " << deviceCount << "\n";

    for (int i = 0; i < deviceCount; i++) {
        printDeviceInfo(i);
    }
}

/**
 * @brief Inicializa un array en host con un valor
 */
template<typename T>
inline void initHostArray(T *array, size_t n, T value) {
    for (size_t i = 0; i < n; i++) {
        array[i] = value;
    }
}

/**
 * @brief Verifica que dos arrays sean iguales (con tolerancia para float)
 */
inline bool verifyArrays(float *a, float *b, size_t n, float tolerance = 1e-5f) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cerr << "Error en índice " << i << ": " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

#endif // CUDA_UTILS_H
