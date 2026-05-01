/**
 * @file indexing_1d.cu
 * @brief Indexación 1D - Acceso a datos lineales
 *
 * Muestra los diferentes cálculos de índices para acceder
 * a arrays 1D en CUDA: linear indexing, boundary checks, etc.
 *
 * Compilación:
 *   nvcc -o indexing_1d indexing_1d.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

__global__ void index_example_kernel(float *data, int N) {
    // Variables built-in
    int tid = threadIdx.x;        // Índice local en el bloque (0..blockDim-1)
    int bid = blockIdx.x;         // Índice del bloque (0..gridDim-1)
    int bdim = blockDim.x;        // Tamaño del bloque

    // Cálculo más común: índice global lineal
    int idx = bid * bdim + tid;

    // Si el grid es mayor que el problema, necesitamos boundary check
    if (idx < N) {
        data[idx] = (float)idx;
    }
}

int main() {
    std::cout << "=== Ejemplo 05.3: Indexación 1D ===\n\n";

    int N = 1000;
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    std::cout << "Problema: " << N << " elementos\n";
    std::cout << "Blocks: " << blocks << ", Threads/block: " << threads_per_block << "\n";
    std::cout << "Total hilos lanzados: " << blocks * threads_per_block << "\n\n";

    // Asignar
    float *h_data = new float[N];
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Limpiar
    for (int i = 0; i < N; i++) h_data[i] = -1.0f;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Lanzar kernel
    index_example_kernel<<<blocks, threads_per_block>>>(d_data, N);
    cudaDeviceSynchronize();

    // Copiar de vuelta
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verificar
    std::cout << "Primeros 10 valores:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  data[" << i << "] = " << h_data[i] << "\n";
    }

    // Buscar no inicializados (deberían ser -1 fuera del rango)
    int uninit = 0;
    for (int i = N; i < blocks * threads_per_block; i++) {
        // Nota: no podemos acceder a h_data[i] si i >= N
        // así que este count es teórico
        uninit++;
    }
    std::cout << "\nHilos " << N << " a " << blocks*threads_per_block-1 << ": boundary check los ignoró (" << uninit << " hilos no usados)\n";

    delete[] h_data;
    cudaFree(d_data);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
