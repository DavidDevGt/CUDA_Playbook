/**
 * @file streams.cu
 * @brief Múltiples Streams - Concurrencia y Overlap
 *
 * Streams permiten ejecutar kernels y memory transfers de forma
 * concurrente (si hay recursos suficientes).
 *
 * Caso de uso clásico: overlap cómputo con transferencia H↔D.
 *
 * Compilación:
 *   nvcc -o streams streams.cu
 *
 * Ejecución:
 *   ./streams
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel simple
 */
__global__ void compute_kernel(float *data, int N, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * value + 1.0f;
    }
}

int main() {
    std::cout << "=== Ejemplo 11.1: Streams y Overlap ===\n\n";

    const int N = 1 << 22;
    size_t bytes = N * sizeof(float);

    // ========== DATOS ==========
    float *h_data1 = new float[N];
    float *h_data2 = new float[N];
    float *d_data1, *d_data2, *d_data3;

    for (int i = 0; i < N; i++) {
        h_data1[i] = 1.0f;
        h_data2[i] = 2.0f;
    }

    cudaMalloc(&d_data1, bytes);
    cudaMalloc(&d_data2, bytes);
    cudaMalloc(&d_data3, bytes);

    // ========== STREAM 0 (default) ==========
    std::cout << "--- Secuencial (stream por defecto) ---\n";

    Timer timer_seq;
    timer_seq.start();

    // Transferencia + cómputo + transferencia (serie)
    cudaMemcpy(d_data1, h_data1, bytes, cudaMemcpyHostToDevice);
    compute_kernel<<<(N+255)/256, 256>>>(d_data1, N, 2.0f);
    cudaMemcpy(h_data1, d_data1, bytes, cudaMemcpyDeviceToHost);

    timer_seq.stop();
    float time_seq = timer_seq.getGpuTime();
    std::cout << "  Tiempo secuencial: " << time_seq << " ms\n";

    // ========== MÚLTIPLES STREAMS (CONCURRENTES) ==========
    std::cout << "\n--- Concurrente con múltiples streams ---\n";

    const int num_streams = 3;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Dividir datos en chunks
    int chunk_size = N / num_streams;
    Timer timer_async;
    timer_async.start();

    for (int s = 0; s < num_streams; s++) {
        int offset = s * chunk_size;
        int current_N = (s == num_streams - 1) ? (N - offset) : chunk_size;

        // Async copy H→D
        cudaMemcpyAsync(&d_data1[offset], &h_data1[offset], current_N * sizeof(float),
                        cudaMemcpyHostToDevice, streams[s]);

        // Kernel en stream
        compute_kernel<<<(current_N + 255) / 256, 256, 0, streams[s]>>>(
            &d_data1[offset], current_N, 2.0f);

        // Async copy D→H
        cudaMemcpyAsync(&h_data1[offset], &d_data1[offset], current_N * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[s]);
    }

    // Sincronizar todos los streams
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
    }

    timer_async.stop();
    float time_async = timer_async.getGpuTime();
    std::cout << "  Tiempo con streams: " << time_async << " ms\n";
    std::cout << "  Speedup: " << (time_seq / time_async) << "x\n";

    // ========== OVERLAP CÓMPUTO + TRANSFERENCIA ==========
    std::cout << "\n--- Overlap cómputo/transferencia (2 streams) ---\n";

    // Patrón: mientras stream 0 transfiere H→D, stream 1 computa y transfiere D→H
    cudaStream_t s0, s1;
    cudaStreamCreate(&s0);
    cudaStreamCreate(&s1);

    Timer timer_overlap;
    timer_overlap.start();

    // Stage 1: H→D en s0
    cudaMemcpyAsync(d_data2, h_data2, bytes, cudaMemcpyHostToDevice, s0);

    // Stage 2: cómputo en s1 (puede overlap con stage 1 si hay copy engine independiente)
    compute_kernel<<<(N+255)/256, 256, 0, s1>>>(d_data2, N, 2.0f);

    // Stage 3: D→H en s0 (después de que cómputo termine, dependencia)
    cudaStreamWaitEvent(nullptr, nullptr, 0);  // placeholder

    // Copia final en s1 para overlap
    cudaMemcpyAsync(h_data2, d_data2, bytes, cudaMemcpyDeviceToHost, s1);

    cudaStreamSynchronize(s1);
    timer_overlap.stop();

    float time_overlap = timer_overlap.getGpuTime();
    std::cout << "  Tiempo overlap: " << time_overlap << " ms\n";

    // ========== ANÁLISIS ==========
    std::cout << "\n📌 Streams:\n";
    std::cout << "  • Queue de trabajo en orden de envío dentro de un stream\n";
    std::cout << "  • Diferentes streams pueden ejecutarse concurrentemente\n";
    std::cout << "  • Default stream (0) es síncrono con todo\n";
    std::cout << "  • Usar cudaStreamNonBlocking para evitar sincronización con default\n\n";
    std::cout << "📌 Overlap:\n";
    std::cout << "  • copy engine (H↔D) + kernels pueden overlap si hay recursos\n";
    std::cout << "  • Requiere múltiples streams\n";
    std::cout << "  • Ver con Nsight Systems si hay overlap real\n";

    // ========== LIMPIEZA ==========
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);

    delete[] h_data1;
    delete[] h_data2;
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_data3);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
