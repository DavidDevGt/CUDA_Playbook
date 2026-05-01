/**
 * @file pinned_memory.cu
 * @brief Memoria paginada (Pinned/Page-Locked Host Memory)
 *
 * La memoria normal del host es paginable (paginada), lo que significa
 * que el OS puede swaparla a disco. CUDA puede usar memoria especial
 * page-locked que:
 *   - No se puede swapar
 *   - Permite transferencias más rápidas (hasta 2×)
 *   - Permite acceder desde GPU con DMA (Direct Memory Access)
 *
 * Compilación:
 *   nvcc -o pinned_memory pinned_memory.cu
 *
 * Ejecución:
 *   ./pinned_memory [SIZE_MB]
 *
 * IMPORTANTE: Usa cudaMallocHost() / cudaFreeHost() para memoria pinned.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

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
 * @brief Mide el tiempo de transferencia usando eventos de CUDA
 */
float measureTransferTime(float *h_src, float *d_dest, size_t bytes, bool usePinned) {
    cudaEvent_t start, stop;
    CHECK( cudaEventCreate(&start) );
    CHECK( cudaEventCreate(&stop) );

    CHECK( cudaEventRecord(start) );

    if (usePinned) {
        CHECK( cudaMemcpy(d_dest, h_src, bytes, cudaMemcpyHostToDevice) );
    } else {
        CHECK( cudaMemcpy(d_dest, h_src, bytes, cudaMemcpyHostToDevice) );
    }

    CHECK( cudaEventRecord(stop) );
    CHECK( cudaEventSynchronize(stop) );

    float ms = 0.0f;
    CHECK( cudaEventElapsedTime(&ms, start, stop) );

    CHECK( cudaEventDestroy(start) );
    CHECK( cudaEventDestroy(stop) );

    return ms;
}

/**
 * @brief Mide la latencia de múltiples transfers pequeñas
 */
void testSmallTransfers() {
    const int small_size = 4096;  // 4 KB
    const int iterations = 1000;

    float *h_pageable;
    float *h_pinned;
    float *d_dest;

    CHECK( cudaMalloc(&d_dest, small_size) );
    CHECK( cudaMallocHost(&h_pinned, small_size) );
    h_pageable = new float[small_size];

    // Inicializar
    for (int i = 0; i < small_size / sizeof(float); i++) {
        h_pageable[i] = 1.0f;
        h_pinned[i] = 1.0f;
    }

    // Medir pageable
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        CHECK( cudaMemcpy(d_dest, h_pageable, small_size, cudaMemcpyHostToDevice) );
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_pageable = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Medir pinned
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        CHECK( cudaMemcpy(d_dest, h_pinned, small_size, cudaMemcpyHostToDevice) );
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double time_pinned = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << "=== Transferencias pequeñas (4 KB) ===\n";
    std::cout << "  Pageable: " << time_pageable << " ms (total), "
              << (time_pageable / iterations) << " ms/transfer\n";
    std::cout << "  Pinned:   " << time_pinned << " ms (total), "
              << (time_pinned / iterations) << " ms/transfer\n";
    std::cout << "  Speedup:  " << (time_pageable / time_pinned) << "x\n";

    delete[] h_pageable;
    CHECK( cudaFreeHost(h_pinned) );
    CHECK( cudaFree(d_dest) );
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 04.3: Memoria Paginada (Pinned) ===\n\n";

    size_t size_mb = 100;
    if (argc > 1) size_mb = atoi(argv[1]);

    size_t N = size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);

    std::cout << "Tamaño: " << size_mb << " MB\n\n";

    // ========== ASIGNAR MEMORIA ==========
    // 1. Host normal (paginable)
    float *h_pageable = new float[N];
    for (size_t i = 0; i < N; i++) h_pageable[i] = (float)i;

    // 2. Host paginada (pinned)
    float *h_pinned = nullptr;
    CHECK( cudaMallocHost(&h_pinned, bytes) );
    for (size_t i = 0; i < N; i++) h_pinned[i] = (float)i;

    // 3. Device mem
    float *d_dest;
    CHECK( cudaMalloc(&d_dest, bytes) );

    std::cout << "Memoria asignada (pageable vs pinned).\n\n";

    // ========== PRUEBA DE BANDWIDTH ==========
    std::cout << "--- Transferencias grandes (" << size_mb << " MB) ---\n";

    const int iterations = 10;

    // Pageable
    Timer timer_pageable;
    float total_pageable = 0.0f;
    for (int i = 0; i < iterations; i++) {
        timer_pageable.start();
        CHECK( cudaMemcpy(d_dest, h_pageable, bytes, cudaMemcpyHostToDevice) );
        timer_pageable.stop();
        total_pageable += timer_pageable.getGpuTime();
    }
    float avg_pageable = total_pageable / iterations;
    float bw_pageable = (bytes / (1024.0*1024.0)) / (avg_pageable / 1000.0f);

    // Pinned
    Timer timer_pinned;
    float total_pinned = 0.0f;
    for (int i = 0; i < iterations; i++) {
        timer_pinned.start();
        CHECK( cudaMemcpy(d_dest, h_pinned, bytes, cudaMemcpyHostToDevice) );
        timer_pinned.stop();
        total_pinned += timer_pinned.getGpuTime();
    }
    float avg_pinned = total_pinned / iterations;
    float bw_pinned = (bytes / (1024.0*1024.0)) / (avg_pinned / 1000.0f);

    std::cout << "  Pageable: " << avg_pageable << " ms, " << bw_pageable << " MB/s\n";
    std::cout << "  Pinned:   " << avg_pinned << " ms, " << bw_pinned << " MB/s\n";
    std::cout << "  Speedup:  " << (avg_pageable / avg_pinned) << "x\n";

    // ========== TRANSFERENCIAS PEQUEÑAS ==========
    std::cout << "\n";
    testSmallTransfers();

    // ==========-mapán ==========
    std::cout << "\n📌 Características de memoria pinned:\n";
    std::cout << "  ✓ Transferencias hasta 2× más rápidas\n";
    std::cout << "  ✓ Permite overlap con cómputo (Async + streams)\n";
    std::cout << "  ✗ No se puede swapar (reside siempre en RAM)\n";
    std::cout << "  ✗ Uso elevado de memoria física\n";
    std::cout << "  ✗ Asignaciones más lentas\n";
    std::cout << "\n📌 Cuándo usar:\n";
    std::cout << "  • Transfers grandes (> 4 MB)\n";
    std::cout << "  • transfers frecuentes (IPC)\n";
    std::cout << "  • Cuando necesitas overlap (overlap cómputo/transferencia)\n";

    // ========== LIMPIEZA ==========
    delete[] h_pageable;
    CHECK( cudaFreeHost(h_pinned) );
    CHECK( cudaFree(d_dest) );

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
