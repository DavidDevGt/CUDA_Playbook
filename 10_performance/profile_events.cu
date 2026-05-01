/**
 * @file profile_events.cu
 * @brief Profiling con eventos de CUDA
 *
 * Muestra cómo usar cudaEvent_t para medir tiempos de kernels
 * con alta precisión.
 *
 * Alternativas:
 *   - cudaEvent_t (alta precisión, para código)
 *   - nvprof (herramienta de línea de comandos)
 *   - Nsight Compute/Systems (GUI avanzado)
 *
 * Compilación:
 *   nvcc -o profile_events profile_events.cu
 *
 * Ejecución:
 *   ./profile_events
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * @brief Kernel simple para medir
 */
__global__ void dummy_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = sinf((float)idx) * cosf((float)idx);
    }
}

int main() {
    std::cout << "=== Ejemplo 10.4: Profiling con CUDA Events ===\n\n";

    const int N = 1 << 22;
    float *d_data;
    CHECK( cudaMalloc(&d_data, N * sizeof(float)) );

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    // ========== CREAR EVENTOS ==========
    cudaEvent_t start, stop;
    CHECK( cudaEventCreate(&start) );
    CHECK( cudaEventCreate(&stop) );

    std::cout << "Calentando GPU (warm-up)...\n";
    // Ejecutar una vez para calentar GPU (evita cold-start effects)
    dummy_kernel<<<blocks, threads>>>(d_data, N);
    CHECK( cudaDeviceSynchronize() );

    // ========== MEDICIÓN CON EVENTOS ==========
    std::cout << "\n--- Medición con cudaEvent ---\n";

    int repetitions = 10;
    CHECK( cudaEventRecord(start) );

    for (int i = 0; i < repetitions; i++) {
        dummy_kernel<<<blocks, threads>>>(d_data, N);
    }

    CHECK( cudaEventRecord(stop) );
    CHECK( cudaEventSynchronize(stop) );

    float total_ms = 0.0f;
    CHECK( cudaEventElapsedTime(&total_ms, start, stop) );

    float avg_ms = total_ms / repetitions;
    float gflops = (N * 2.0f) / (avg_ms * 1e6);  // Aproximado

    std::cout << "  Total (" << repetitions << " runs): " << total_ms << " ms\n";
    std::cout << "  Promedio por ejecución: " << avg_ms << " ms\n";
    std::cout << "  Performance aprox.:     " << gflops << " GFLOPS\n";

    // ========== USAR cudaEvent_t con CONCURRENCY ==========
    std::cout << "\n--- Overlap con múltiples streams ---\n";

    cudaStream_t stream1, stream2;
    CHECK( cudaStreamCreate(&stream1) );
    CHECK( cudaStreamCreate(&stream2) );

    cudaEvent_t start1, stop1, start2, stop2;
    CHECK( cudaEventCreate(&start1) );
    CHECK( cudaEventCreate(&stop1) );
    CHECK( cudaEventCreate(&start2) );
    CHECK( cudaEventCreate(&stop2) );

    // Stream 1
    CHECK( cudaEventRecord(start1, stream1) );
    dummy_kernel<<<blocks, threads, 0, stream1>>>(d_data, N);
    CHECK( cudaEventRecord(stop1, stream1) );

    // Stream 2 (concurrente si hay recursos)
    CHECK( cudaEventRecord(start2, stream2) );
    dummy_kernel<<<blocks, threads, 0, stream2>>>(d_data, N);
    CHECK( cudaEventRecord(stop2, stream2) );

    // Esperar ambos
    CHECK( cudaEventSynchronize(stop1) );
    CHECK( cudaEventSynchronize(stop2) );

    float time1, time2;
    CHECK( cudaEventElapsedTime(&time1, start1, stop1) );
    CHECK( cudaEventElapsedTime(&time2, start2, stop2) );

    std::cout << "  Stream 1: " << time1 << " ms\n";
    std::cout << "  Stream 2: " << time2 << " ms\n";
    std::cout << "  (Pueden ser similares o uno más rápido por scheduling)\n";

    // ========== COMPARAR CON nvprof (información) ==========
    std::cout << "\n📌 Consejos de profiling:\n";
    std::cout << "  • cudaEvent_t: para instrumentación en código\n";
    std::cout << "  • nvprof ./a.out         → profiling básico (legacy)\n";
    std::cout << "  • nvprof --print-gpu-trace ./a.out\n";
    std::cout << "  • Nsight Compute:      → kernel-level detallado\n";
    std::cout << "  • Nsight Systems:      → system-wide (CPU + GPU)\n";
    std::cout << "\nComando rápido:\n";
    std::cout << "  nvprof --metrics achieved_occupancy,gld_efficiency,gst_efficiency ./a.out\n";

    // ========== LIMPIEZA ==========
    CHECK( cudaEventDestroy(start) );
    CHECK( cudaEventDestroy(stop) );
    CHECK( cudaEventDestroy(start1) );
    CHECK( cudaEventDestroy(stop1) );
    CHECK( cudaEventDestroy(start2) );
    CHECK( cudaEventDestroy(stop2) );
    CHECK( cudaStreamDestroy(stream1) );
    CHECK( cudaStreamDestroy(stream2) );
    CHECK( cudaFree(d_data) );

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
