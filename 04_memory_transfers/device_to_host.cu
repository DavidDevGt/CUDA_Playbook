/**
 * @file device_to_host.cu
 * @brief Transferencias de memoria: Device → Host
 *
 * Muestra patrones de copia desde GPU a CPU:
 * - Copia síncrona estándar
 * - Copia asíncrona con streams
 * - Medición de throughput de lectura
 *
 * Compilación:
 *   nvcc -o device_to_host device_to_host.cu
 *
 * Ejecución:
 *   ./device_to_host [SIZE_MB]
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

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 04.2: Transferencia Device → Host ===\n\n";

    size_t size_mb = 100;
    if (argc > 1) size_mb = atoi(argv[1]);

    size_t N = size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);

    std::cout << "Tamaño: " << size_mb << " MB\n\n";

    // Host y Device
    float *h_src = new float[N];
    float *h_dest = new float[N];
    float *d_src;

    CHECK( cudaMalloc(&d_src, bytes) );

    // Inicializar datos en host y copiar a device
    for (size_t i = 0; i < N; i++) h_src[i] = (float)i;
    CHECK( cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice) );

    std::cout << "Datos en GPU preparados.\n\n";

    // ========== PRUEBA 1: COPIA SÍNCRONA ==========
    std::cout << "--- Prueba 1: cudaMemcpy (DeviceToHost) ---\n";

    Timer timer1;
    const int iters = 10;
    float total1 = 0;

    for (int i = 0; i < iters; i++) {
        timer1.start();
        CHECK( cudaMemcpy(h_dest, d_src, bytes, cudaMemcpyDeviceToHost) );
        timer1.stop();
        total1 += timer1.getGpuTime();
    }

    float avg1 = total1 / iters;
    float bw1 = (bytes / (1024.0*1024.0)) / (avg1 / 1000.0f);

    std::cout << "  Tiempo promedio: " << avg1 << " ms\n";
    std::cout << "  Bandwidth:       " << bw1 << " MB/s\n";

    // ========== PRUEBA 2: COPIA ASÍNCRONA ==========
    std::cout << "\n--- Prueba 2: cudaMemcpyAsync + stream ---\n";

    cudaStream_t stream;
    CHECK( cudaStreamCreate(&stream) );

    Timer timer2;
    float total2 = 0;

    for (int i = 0; i < iters; i++) {
        timer2.start();
        CHECK( cudaMemcpyAsync(h_dest, d_src, bytes, cudaMemcpyDeviceToHost, stream) );
        CHECK( cudaStreamSynchronize(stream) );
        timer2.stop();
        total2 += timer2.getGpuTime();
    }

    CHECK( cudaStreamDestroy(stream) );

    float avg2 = total2 / iters;
    float bw2 = (bytes / (1024.0*1024.0)) / (avg2 / 1000.0f);

    std::cout << "  Tiempo promedio: " << avg2 << " ms\n";
    std::cout << "  Bandwidth:       " << bw2 << " MB/s\n";

    // ========== COMPARACIÓN ==========
    std::cout << "\n=== Resultados ===\n";
    std::cout << "  Síncrono: " << avg1 << " ms (" << bw1 << " MB/s)\n";
    std::cout << "  Async:    " << avg2 << " ms (" << bw2 << " MB/s)\n";
    std::cout << "  Diferencia: " << ((avg2-avg1)/avg1*100.0f) << "%\n";

    // ========== VERIFICACIÓN ==========
    bool ok = true;
    for (size_t i = 0; i < N; i++) {
        if (fabs(h_dest[i] - h_src[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== LIMPIEZA ==========
    delete[] h_src;
    delete[] h_dest;
    CHECK( cudaFree(d_src) );

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
