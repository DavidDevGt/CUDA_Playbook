/**
 * @file host_to_device.cu
 * @brief Transferencias de memoria: Host → Device
 *
 * Demuestra los diferentes modos de copia de datos desde CPU a GPU:
 * - cudaMemcpy HostToDevice (síncrono)
 * - cudaMemcpyAsync con streams (asíncrono)
 * - Transferencias en paginada (pinned) vs paginable
 *
 * Compilación:
 *   nvcc -o host_to_device host_to_device.cu
 *
 * Ejecución:
 *   ./host_to_device [SIZE_MB]
 */

#include <iostream>
#include <cuda_runtime.h>
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
 * @brief Copia síncrona estándar
 */
void testSyncCopy(float *h_src, float *d_dest, size_t bytes) {
    CHECK( cudaMemcpy(d_dest, h_src, bytes, cudaMemcpyHostToDevice) );
}

/**
 * @brief Copia asíncrona usando stream
 */
void testAsyncCopy(float *h_src, float *d_dest, size_t bytes, cudaStream_t stream) {
    CHECK( cudaMemcpyAsync(d_dest, h_src, bytes, cudaMemcpyHostToDevice, stream) );
}

/**
 * @brief Copia con memoria paginada (pinned)
 */
void testPinnedCopy(float *h_pinned, float *d_dest, size_t bytes) {
    CHECK( cudaMemcpy(d_dest, h_pinned, bytes, cudaMemcpyHostToDevice) );
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 04.1: Transferencia Host → Device ===\n\n";

    // ========== PARÁmetros ==========
    size_t size_mb = 100;  // 100 MB por defecto
    if (argc > 1) size_mb = atoi(argv[1]);

    size_t N = size_mb * 1024 * 1024 / sizeof(float);  // número de floats
    size_t bytes = N * sizeof(float);

    std::cout << "Tamaño de transferencia: " << size_mb << " MB (" << N << " floats)\n\n";

    // ========== ASIGNAR MEMORIA ==========
    // Host: memoria normal (paginable)
    float *h_src = new float[N];
    for (size_t i = 0; i < N; i++) h_src[i] = (float)i;

    // Host: memoria paginada (pinned)
    float *h_pinned = nullptr;
    CHECK( cudaMallocHost(&h_pinned, bytes) );
    for (size_t i = 0; i < N; i++) h_pinned[i] = (float)i;

    // Device
    float *d_dest1, *d_dest2, *d_dest3;
    CHECK( cudaMalloc(&d_dest1, bytes) );
    CHECK( cudaMalloc(&d_dest2, bytes) );
    CHECK( cudaMalloc(&d_dest3, bytes) );

    std::cout << "Memoria asignada:\n";
    std::cout << "  Host paginable: " << N * sizeof(float) / (1024.0*1024.0) << " MB\n";
    std::cout << "  Host pinned:    " << N * sizeof(float) / (1024.0*1024.0) << " MB\n";
    std::cout << "  Device:         " << N * sizeof(float) / (1024.0*1024.0) << " MB × 3\n\n";

    // ========== MODO 1: COPIA SÍNCRONA ==========
    std::cout << "--- Prueba 1: cudaMemcpy síncrono ---\n";

    Timer timer_sync;
    const int iterations = 10;
    float total_time_sync = 0.0f;

    for (int i = 0; i < iterations; i++) {
        timer_sync.start();
        testSyncCopy(h_src, d_dest1, bytes);
        timer_sync.stop();
        total_time_sync += timer_sync.getGpuTime();
    }

    float avg_sync = total_time_sync / iterations;
    float bandwidth_sync = (bytes / (1024.0*1024.0)) / (avg_sync / 1000.0f);

    std::cout << "  Tiempo promedio: " << avg_sync << " ms\n";
    std::cout << "  Bandwidth:       " << bandwidth_sync << " MB/s\n";

    // ========== MODO 2: COPIA ASÍNCRONA (con stream) ==========
    std::cout << "\n--- Prueba 2: cudaMemcpyAsync (con stream) ---\n";

    cudaStream_t stream;
    CHECK( cudaStreamCreate(&stream) );

    Timer timer_async;
    float total_time_async = 0.0f;

    for (int i = 0; i < iterations; i++) {
        timer_async.start();
        testAsyncCopy(h_src, d_dest2, bytes, stream);
        // La copia es asíncrona, pero necesitamos sincronizar para medir
        CHECK( cudaStreamSynchronize(stream) );
        timer_async.stop();
        total_time_async += timer_async.getGpuTime();
    }

    float avg_async = total_time_async / iterations;
    float bandwidth_async = (bytes / (1024.0*1024.0)) / (avg_async / 1000.0f);

    std::cout << "  Tiempo promedio: " << avg_async << " ms\n";
    std::cout << "  Bandwidth:       " << bandwidth_async << " MB/s\n";
    std::cout << "  (Debe ser similar a síncrono si no hay overlap)\n";

    CHECK( cudaStreamDestroy(stream) );

    // ========== MODO 3: COPIA CON MEMORIA PAGINADA ==========
    std::cout << "\n--- Prueba 3: cudaMemcpy con host memory pinned ---\n";

    Timer timer_pinned;
    float total_time_pinned = 0.0f;

    for (int i = 0; i < iterations; i++) {
        timer_pinned.start();
        testPinnedCopy(h_pinned, d_dest3, bytes);
        timer_pinned.stop();
        total_time_pinned += timer_pinned.getGpuTime();
    }

    float avg_pinned = total_time_pinned / iterations;
    float bandwidth_pinned = (bytes / (1024.0*1024.0)) / (avg_pinned / 1000.0f);

    std::cout << "  Tiempo promedio: " << avg_pinned << " ms\n";
    std::cout << "  Bandwidth:       " << bandwidth_pinned << " MB/s\n";

    // ========== COMPARACIÓN ==========
    std::cout << "\n=== Comparación de bandwidth (transferencia H→D) ===\n";
    std::cout << "  Paginable (síncrono): " << bandwidth_sync << " MB/s\n";
    std::cout << "  Pinned (síncrono):    " << bandwidth_pinned << " MB/s\n";
    std::cout << "  Speedup pinned:       " << (bandwidth_pinned / bandwidth_sync) << "x\n";

    float speedup_pinned = avg_sync / avg_pinned;
    if (speedup_pinned > 1.0f) {
        std::cout << "\n✅ Memoria paginada es " << speedup_pinned << "× más rápida\n";
        std::cout << "   (Usa cudaMallocHost para transfers grandes)\n";
    } else {
        std::cout << "\n⚠️  Diferencia mínima (puede deberse a tamaño pequeño o caché)\n";
    }

    // ========== VERIFICAR RESULTADOS ==========
    // Simple check
    bool ok1 = true, ok2 = true, ok3 = true;
    float *h_check1 = new float[N];
    float *h_check2 = new float[N];
    float *h_check3 = new float[N];

    cudaMemcpy(h_check1, d_dest1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_check2, d_dest2, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_check3, d_dest3, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < N; i++) {
        if (fabs(h_check1[i] - h_src[i]) > 1e-5f) ok1 = false;
        if (fabs(h_check2[i] - h_src[i]) > 1e-5f) ok2 = false;
        if (fabs(h_check3[i] - h_src[i]) > 1e-5f) ok3 = false;
    }

    std::cout << "\nVerificación:\n";
    std::cout << "  Síncrono:  " << (ok1 ? "✅ OK" : "❌ FAIL") << "\n";
    std::cout << "  Async:     " << (ok2 ? "✅ OK" : "❌ FAIL") << "\n";
    std::cout << "  Pinned:    " << (ok3 ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== RECOMENDACIONES ==========
    std::cout << "\n📌 Recomendaciones:\n";
    std::cout << "  1. Para transfers grandes (>4MB), usa memoria pinned\n";
    std::cout << "  2. Usa cudaMemcpyAsync + streams para overlap cómputo/transferencia\n";
    std::cout << "  3. Minimiza transfers H↔D, procesa datos en GPU el mayor tiempo posible\n";

    // ========== LIMPIEZA ==========
    delete[] h_src;
    delete[] h_check1;
    delete[] h_check2;
    delete[] h_check3;
    CHECK( cudaFreeHost(h_pinned) );
    CHECK( cudaFree(d_dest1) );
    CHECK( cudaFree(d_dest2) );
    CHECK( cudaFree(d_dest3) );

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
