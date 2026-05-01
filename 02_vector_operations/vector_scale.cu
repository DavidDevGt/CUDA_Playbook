/**
 * @file vector_scale.cu
 * @brief Escalamiento de vector: Y[i] = α * X[i]
 *
 * Similar a la operación SAXPY pero sin el término Y de salida.
 * Muestra cómo pasar argumentos escalares a kernels.
 *
 * Compilación:
 *   nvcc -o vector_scale vector_scale.cu
 *
 * Ejecución:
 *   ./vector_scale [N] [alpha]
 *
 * Ejemplo:
 *   ./vector_scale 1000000 2.5
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel que escala un vector por un factor constante
 *
 * @param X Vector de entrada
 * @param Y Vector de salida
 * @param alpha Factor de escalamiento
 * @param N Número de elementos
 */
__global__ void vector_scale_kernel(const float *X, float *Y, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        Y[idx] = alpha * X[idx];
    }
}

/**
 * @brief Versión con grid-stride para mayor flexibilidad
 */
__global__ void vector_scale_kernel_grid_stride(const float *X, float *Y, float alpha, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride) {
        Y[i] = alpha * X[i];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 02.1: Escalamiento de Vector ===\n\n";

    // ========== PARÁMETROS ==========
    int N = 1 << 22;  // ~4M elementos
    float alpha = 2.5f;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) alpha = atof(argv[2]);

    std::cout << "Parámetros:\n";
    std::cout << "  N (elementos): " << N << "\n";
    std::cout << "  α (alpha):     " << alpha << "\n";
    std::cout << "  Memoria:       " << N * sizeof(float) / (1024.0*1024.0) << " MB\n\n";

    // ========== CONFIGURACIÓN ==========
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 65535) blocksPerGrid = 65535;

    // ========== ASIGNACIÓN ==========
    size_t size = N * sizeof(float);
    float *h_X = new float[N];
    float *h_Y = new float[N];
    float *h_Y_ref = new float[N];
    float *d_X, *d_Y;

    cudaMalloc(&d_X, size);
    cudaMalloc(&d_Y, size);

    // Inicializar X con valores secuenciales: X[i] = i
    for (int i = 0; i < N; i++) {
        h_X[i] = (float)i;
    }

    // Copiar a GPU
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);

    std::cout << "Vector X inicializado (X[i] = i).\n";
    std::cout << "Blocks: " << blocksPerGrid << ", Threads/block: " << threadsPerBlock << "\n\n";

    // ========== EJECUTAR ==========
    Timer timer;
    timer.start();

    vector_scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, alpha, N);
    gpuErrchk( cudaPeekAtLastError() );

    timer.stop();
    float gpuTime = timer.getGpuTime();

    // Copiar resultado
    cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost);

    std::cout << "Kernel ejecutado en " << gpuTime << " ms\n";

    // ========== VERIFICAR ==========
    // Calcular referencia
    for (int i = 0; i < N; i++) {
        h_Y_ref[i] = alpha * h_X[i];
    }

    bool ok = verifyArrays(h_Y, h_Y_ref, N);
    std::cout << (ok ? "✅ " : "❌ ") << "Verificación: " << (ok ? "OK" : "FAIL") << "\n";

    // ========== MUESTRA ==========
    std::cout << "\nPrimeros 5 elementos:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  Y[" << i << "] = " << h_Y[i]
                  << " (esperado: " << h_Y_ref[i] << ")\n";
    }

    // ========== ESTADÍSTICAS ==========
    float throughput = (N * sizeof(float) * 2) / (gpuTime * 1e6);
    std::cout << "\nThroughput: " << throughput << " GB/s\n";

    // ========== COMPARACIÓN CPU ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    for (int i = 0; i < N; i++) {
        h_Y_ref[i] = alpha * h_X[i];
    }
    cpuTimer.stop();
    float cpuTime = cpuTimer.getCpuTime();

    std::cout << "Speedup CPU/GPU: " << (cpuTime / gpuTime) << "x\n";

    // ========== LIMPIEZA ==========
    delete[] h_X;
    delete[] h_Y;
    delete[] h_Y_ref;
    cudaFree(d_X);
    cudaFree(d_Y);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Modifica el kernel para hacer: Y = α*X + β (SAXPY)\n";

    return 0;
}
