/**
 * @file events.cu
 * @brief Sincronización fina con eventos (CUDA events)
 *
 * Los eventos permiten:
 *   - Medir tiempo entre puntos específicos
 *   - Sincronizar entre streams
 *   - Control de dependencias explícitas
 *
 * Compilación:
 *   nvcc -o events events.cu
 *
 * Ejecución:
 *   ./events
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

__global__ void kernel_A(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = data[idx] * 2.0f;
}

__global__ void kernel_B(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = data[idx] + 1.0f;
}

int main() {
    std::cout << "=== Ejemplo 11.2: Eventos de CUDA ===\n\n";

    const int N = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Inicializar
    float init_val = 1.0f;
    cudaMemcpy(d_data, &init_val, sizeof(float), cudaMemcpyHostToDevice);

    // ========== CREAR EVENTOS ==========
    cudaEvent_t e_start, e_kernelA, e_kernelB, e_end;
    for (auto &e : {&e_start, &e_kernelA, &e_kernelB, &e_end}) {
        cudaEventCreate(e);
    }

    dim3 threads(256);
    dim3 blocks((N + 255) / 256);

    // ========== EJEMPLO 1: TIMING CON EVENTOS ==========
    std::cout << "--- Medición con eventos ---\n";

    cudaEventRecord(e_start);
    kernel_A<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(e_kernelA);
    kernel_B<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(e_kernelB);
    kernel_A<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(e_end);

    cudaEventSynchronize(e_end);

    float t_total, t_A1, t_B, t_A2;
    cudaEventElapsedTime(&t_total, e_start, e_end);
    cudaEventElapsedTime(&t_A1, e_start, e_kernelA);
    cudaEventElapsedTime(&t_B,  e_kernelA, e_kernelB);
    cudaEventElapsedTime(&t_A2, e_kernelB, e_end);

    std::cout << "  Total: " << t_total << " ms\n";
    std::cout << "  A1:    " << t_A1 << " ms\n";
    std::cout << "  B:     " << t_B << " ms\n";
    std::cout << "  A2:    " << t_A2 << " ms\n";

    // ========== EJEMPLO 2: SINCRONIZACIÓN ENTRE STREAMS ==========
    std::cout << "\n--- Synchronization entre streams ---\n";

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEvent_t e_s1_done;
    cudaEventCreate(&e_s1_done);

    // Stream 1: compute
    kernel_A<<<blocks, threads, 0, s1>>>(d_data, N);
    cudaEventRecord(e_s1_done, s1);

    // Stream 2: esperar a que stream 1 termine
    cudaStreamWaitEvent(s2, e_s1_done, 0);
    kernel_B<<<blocks, threads, 0, s2>>>(d_data, N);

    cudaStreamSynchronize(s2);
    std::cout << "  Stream 2 esperó evento de stream 1 correctamente.\n";

    // ========== EJEMPLO 3: TIMESTAMP ==========
    std::cout << "\n--- Timestamp events (mide en GPU ticks) ---\n";

    cudaEvent_t evt1, evt2;
    cudaEventCreateWithFlags(&evt1, cudaEventDefault);
    cudaEventCreateWithFlags(&evt2, cudaEventDefault);

    cudaEventRecord(evt1);
    kernel_A<<<blocks, threads>>>(d_data, N);
    cudaEventRecord(evt2);

    cudaEventSynchronize(evt2);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, evt1, evt2);
    std::cout << "  Tiempo kernel A: " << ms << " ms\n";

    // ========== LIMPIEZA ==========
    cudaFree(d_data);
    for (auto e : {&e_start, &e_kernelA, &e_kernelB, &e_end, &evt1, &evt2, &e_s1_done}) {
        if (e) cudaEventDestroy(*e);
    }
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    std::cout << "\n📌 Usos de eventos:\n";
    std::cout << "  1. Timing de kernels y transfers\n";
    std::cout << "  2. Sincronización explícita entre streams\n";
    std::cout << "  3. Control de dependencias en DAGs de operaciones\n";

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
