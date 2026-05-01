/**
 * @file constant_array.cu
 * @brief Memoria constante (__constant__)
 *
 * La memoria constante es:
 *   - Solo lectura desde kernels
 *   - Cacheada (caché constante, 8KB por SM típicamente)
 *   - Broadcast eficiente: todos los hilos leen la misma dirección
 *
 * Útil para:
 *   - Tablas de consulta (lookup tables)
 *   - Coeficientes fijos
 *   - Parámetros de configuración
 *
 * COMPILACIÓN:
 *   nvcc -O2 -arch=sm_89 -o constant_array constant_array.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

// Memoria constante: visible para todos los kernels, solo lectura
__constant__ float c_constant[1024];

__global__ void kernel_constant(float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = c_constant[idx % 1024];
    }
}

__global__ void kernel_global(const float *c_global, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = c_global[idx % 1024];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Memoria constante: Constant Array ===\n\n";
    
    const int N = 1 << 20;  // 1M elementos
    const int n_constants = 1024;  // Tabla de 1024 constantes
    
    std::cout << "Elementos a procesar: " << N << "\n";
    std::cout << "Tamaño tabla constantes: " << n_constants << "\n\n";
    
    // Inicializar constantes en host
    float *h_constants = new float[n_constants];
    for (int i = 0; i < n_constants; i++) {
        h_constants[i] = sinf(i * 0.01f) * 0.5f + 0.5f;
    }
    
    // Copiar constantes a memoria constante
    gpuErrchk( cudaMemcpyToSymbol(c_constant, h_constants, n_constants * sizeof(float)) );
    
    // Memoria en device para resultados
    float *d_constants_global, *d_output;
    gpuErrchk( cudaMalloc(&d_constants_global, n_constants * sizeof(float)) );
    gpuErrchk( cudaMalloc(&d_output, N * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_constants_global, h_constants, n_constants * sizeof(float), cudaMemcpyHostToDevice) );
    
    // Configuración ejecución
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    Timer timer;
    
    // ========== PRUEBA 1: Memoria constante ==========
    timer.start();
    kernel_constant<<<blocks, threads>>>(d_output, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    timer.stop();
    
    float time_constant = timer.getGpuTime();
    std::cout << "Test 1: Memoria constante (__constant__)\n";
    std::cout << "  Tiempo: " << time_constant << " ms\n";
    
    // ========== PRUEBA 2: Memoria global ==========
    timer.start();
    kernel_global<<<blocks, threads>>>(d_constants_global, d_output, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    timer.stop();
    
    float time_global = timer.getGpuTime();
    std::cout << "\nTest 2: Memoria global (L2/L1 cache)\n";
    std::cout << "  Tiempo: " << time_global << " ms\n";
    std::cout << "\nSpeedup constant memory: " << (time_global / time_constant) << "x\n";
    
    // ========== CARACTERÍSTICAS ==========
    std::cout << "\n📌 Características memoria constante:\n";
    std::cout << "  • Tamaño típico: 64 KB por SM\n";
    std::cout << "  • Caché dedicada: 8 KB por SM\n";
    std::cout << "  • Broadcast eficiente\n";
    std::cout << "  • Solo lectura desde GPU\n\n";
    
    std::cout << "📌 Cuándo usar:\n";
    std::cout << "  • Tablas de consulta (sin, cos, lookup)\n";
    std::cout << "  • Coeficientes de filtro\n";
    std::cout << "  • Parámetros compartidos\n\n";
    
    // ========== LIMPIEZA ==========
    delete[] h_constants;
    cudaFree(d_constants_global);
    cudaFree(d_output);
    
    std::cout << "✅ Ejemplo completado.\n";
    
    return 0;
}
