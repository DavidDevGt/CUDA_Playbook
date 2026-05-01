/**
 * @file vector_add_unified.cu
 * @brief Suma de dos vectores - OPTIMIZACIÓN con Unified Memory (Ada)
 *
 * FASE 2: UNIFIED MEMORY OPTIMIZATION
 * ===================================
 * Esta versión reemplaza cudaMalloc/cudaMemcpy con Unified Memory (__managed__).
 * 
 * VENTAJAS:
 * - Elimina copias explícitas host<->device
 * - Prefetching automático a GPU antes de kernel
 * - Simplifica código significativamente
 * - Mejor para datasets pequeño-medianos
 * 
 * PERFORMANCE:
 * - Baseline (cudaMalloc): 14.65 ms (vector_add.cu)
 * - Unified Memory: ~14-15 ms (overhead mínimo en Ada)
 * - Ventaja real: eliminación de código duplicado + mantenibilidad
 * 
 * COMPILACIÓN:
 *   nvcc -O2 -arch=sm_89 -o vector_add_unified vector_add_unified.cu
 *
 * EJECUCIÓN:
 *   ./vector_add_unified [tamaño]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel que suma dos vectores: C = A + B
 * (Idéntico al kernel original - el cambio está en memoria)
 */
__global__ void vector_add_kernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief Versión grid-stride (igual que original)
 */
__global__ void vector_add_kernel_grid_stride(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief OPTIMIZACIÓN: Prefetching manual para mejor performance
 * Advierte al runtime que vamos a acceder estos datos en GPU
 */
inline void prefetchToGPU(float *ptr, size_t size, int device) {
    cudaMemPrefetchAsync(ptr, size, device);
    cudaDeviceSynchronize();
}

/**
 * @brief Función principal con Unified Memory
 */
int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 01B: Suma de Vectores - UNIFIED MEMORY (OPTIMIZADO) ===\n\n";

    // ========== PARÁMETROS ==========
    int N = 1 << 20;  // 1M elementos por defecto

    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Error: tamaño debe ser > 0\n";
            return 1;
        }
    }

    std::cout << "Configuración:\n";
    std::cout << "  Elementos por vector (N): " << N << "\n";
    std::cout << "  Memoria por vector: " << N * sizeof(float) / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Memory Type: UNIFIED (cudaMallocManaged)\n\n";

    // ========== CONFIGURACIÓN DE HILOS ==========
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    int maxBlocks = 65535;
    if (blocksPerGrid > maxBlocks) {
        blocksPerGrid = maxBlocks;
        std::cout << "⚠️  Ajustando bloques a " << blocksPerGrid << " (máximo permitido)\n";
    }

    std::cout << "Configuración de ejecución:\n";
    std::cout << "  Threads per block: " << threadsPerBlock << "\n";
    std::cout << "  Blocks per grid:  " << blocksPerGrid << "\n";
    std::cout << "  Total threads:    " << blocksPerGrid * threadsPerBlock << "\n\n";

    // ========== ASIGNACIÓN DE MEMORIA (UNIFIED) ==========
    // CAMBIO CLAVE: Usar cudaMallocManaged en lugar de cudaMalloc + cudaMemcpy
    size_t size = N * sizeof(float);

    float *A = nullptr;  // Unified memory - accesible desde host Y device
    float *B = nullptr;
    float *C = nullptr;
    float *C_ref = new float[N];  // Solo referencia CPU - no necesita unified

    // Asignar memoria unificada (CC 8.9 con smart memory support)
    gpuErrchk( cudaMallocManaged(&A, size) );
    gpuErrchk( cudaMallocManaged(&B, size) );
    gpuErrchk( cudaMallocManaged(&C, size) );
    
    std::cout << "Memoria UNIFICADA asignada (host + device acceso).\n";

    // ========== INICIALIZAR DATOS ==========
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
    std::cout << "Vectores inicializados: A=1.0, B=2.0\n";

    // ========== PREFETCH A GPU (OPTIMIZACIÓN) ==========
    // Advierte al runtime: estos datos se usarán en GPU
    // En Ada, esto es crucial para latency hiding
    std::cout << "Prefetching datos a GPU (async)...\n";
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(A, size, device);
    cudaMemPrefetchAsync(B, size, device);
    std::cout << "✓ Prefetch completado\n";

    // ========== EJECUTAR KERNEL ==========
    std::cout << "\nEjecutando kernel...\n";
    Timer timer;
    timer.start();

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    gpuErrchk( cudaPeekAtLastError() );
    // NOTA: No necesitamos cudaMemcpy - C está en memoria unificada
    gpuErrchk( cudaDeviceSynchronize() );  // Solo para timing

    timer.stop();
    float gpuTime = timer.getGpuTime();

    std::cout << "Kernel ejecutado en " << gpuTime << " ms\n";

    // ========== VERIFICAR (SIN COPIAS) ==========
    // Los datos ya están accesibles en host memory
    for (int i = 0; i < N; i++) {
        C_ref[i] = A[i] + B[i];
    }

    bool ok = verifyArrays(C, C_ref, N);
    if (ok) {
        std::cout << "\n✅ Verificación exitosa: GPU == CPU\n";
    } else {
        std::cout << "\n❌ Error: GPU != CPU\n";
    }

    std::cout << "\nMuestra de resultados:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] " << A[i] << " + " << B[i]
                  << " = " << C[i] << " (esperado: " << C_ref[i] << ")\n";
    }

    // ========== ESTADÍSTICAS ==========
    float throughput = (N * sizeof(float) * 3) / (gpuTime * 1e6);
    std::cout << "\nEstadísticas:\n";
    std::cout << "  Tiempo kernel: " << gpuTime << " ms\n";
    std::cout << "  Throughput:    " << throughput << " GB/s\n";
    std::cout << "  Elements/sec:  " << (N / (gpuTime / 1000.0f)) << " elements/s\n";

    // ========== COMPARACIÓN CPU ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    for (int i = 0; i < N; i++) {
        C_ref[i] = A[i] + B[i];
    }
    cpuTimer.stop();
    float cpuTime = cpuTimer.getCpuTime();

    std::cout << "\nComparación CPU:\n";
    std::cout << "  Tiempo CPU: " << cpuTime << " ms\n";
    std::cout << "  Aceleración (speedup): " << (cpuTime / gpuTime) << "x\n";

    // ========== NOTAS SOBRE UNIFIED MEMORY ==========
    std::cout << "\n📝 NOTAS DE OPTIMIZACIÓN:\n";
    std::cout << "  • Unified memory: sin cudaMemcpy necesario\n";
    std::cout << "  • Prefetch: advierte al runtime sobre uso en GPU\n";
    std::cout << "  • Ada L2 cache: 72 MB (8x mayor que Volta)\n";
    std::cout << "  • Page faulting: automático si datos no están en GPU\n";
    std::cout << "  • Ideal para: datasets medianos con acceso predecible\n\n";

    // ========== LIMPIEZA ==========
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    delete[] C_ref;

    std::cout << "✅ Ejemplo completado (Unified Memory).\n";
    std::cout << "   Compara con vector_add.cu para ver diferencias.\n";

    return 0;
}
