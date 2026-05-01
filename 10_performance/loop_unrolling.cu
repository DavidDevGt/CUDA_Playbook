/**
 * @file loop_unrolling.cu
 * @brief Loop unrolling manual vs automático
 *
 * Loop unrolling: desenrollar bucles para reducir overhead
 * de control y ocultar latency.
 *
 * Compilación:
 *   nvcc -O2 -arch=sm_89 -o loop_unrolling loop_unrolling.cu
 *
 * Ejecución:
 *   ./loop_unrolling [SIZE]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

// Tamaño por defecto del problema (1M elementos)
#define PROBLEM_SIZE (1 << 20)

/**
 * @brief Kernel con loop normal (sin unroll)
 */
__global__ void kernel_no_unroll(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < 4; i++) {
            sum += input[idx + i * N];  // Acceso pattern ficticio
        }
        output[idx] = sum;
    }
}

/**
 * @brief Kernel con loop unroll manual (desenrollado explícitamente)
 */
__global__ void kernel_unroll_manual(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Desenrollado manual
        float sum = 0.0f;
        sum += input[idx + 0 * N];
        sum += input[idx + 1 * N];
        sum += input[idx + 2 * N];
        sum += input[idx + 3 * N];
        output[idx] = sum;
    }
}

/**
 * @brief Kernel con loop unroll (directiva pragma)
 */
__global__ void kernel_unroll_pragma(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            sum += input[idx + i * N];
        }
        output[idx] = sum;
    }
}

/**
 * @brief Kernel con loop unroll completo (full unroll)
 */
__global__ void kernel_unroll_full(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        // Desplegado completo
        sum += input[idx + 0 * N];
        sum += input[idx + 1 * N];
        sum += input[idx + 2 * N];
        sum += input[idx + 3 * N];
        output[idx] = sum;
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 10.3: Loop Unrolling ===\n\n";

    int N = (argc > 1) ? atoi(argv[1]) : PROBLEM_SIZE;
    const int inner_loop = 4;

    std::cout << "N (hilos): " << N << ", Inner loop: " << inner_loop << "\n\n";

    // ========== DATOS ==========
    int total_elements = N * inner_loop;
    float *h_data = new float[total_elements];
    float *h_out[4];
    float *d_data, *d_out[4];

    for (int i = 0; i < total_elements; i++) h_data[i] = (float)(i % 100);

    cudaMalloc(&d_data, total_elements * sizeof(float));
    cudaMemcpy(d_data, h_data, total_elements * sizeof(float), cudaMemcpyHostToDevice);

    for (int k = 0; k < 4; k++) {
        h_out[k] = new float[N];
        cudaMalloc(&d_out[k], N * sizeof(float));
    }

    // ========== CONFIGURACIÓN ==========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    std::cout << "Blocks: " << blocks << ", Threads/block: " << threads << "\n\n";

    // ========== PRUEBA 1: SIN UNROLL ==========
    std::cout << "--- Kernel normal (sin unroll) ---\n";
    Timer t1;
    t1.start();
    kernel_no_unroll<<<blocks, threads>>>(d_data, d_out[0], N);
    cudaDeviceSynchronize();
    t1.stop();
    float time1 = t1.getGpuTime();
    std::cout << "  Tiempo: " << time1 << " ms\n";

    // ========== PRUEBA 2: UNROLL MANUAL ==========
    std::cout << "\n--- Unroll manual ---\n";
    Timer t2;
    t2.start();
    kernel_unroll_manual<<<blocks, threads>>>(d_data, d_out[1], N);
    cudaDeviceSynchronize();
    t2.stop();
    float time2 = t2.getGpuTime();
    std::cout << "  Tiempo: " << time2 << " ms\n";
    std::cout << "  Speedup: " << (time1 / time2) << "x\n";

    // ========== PRUEBA 3: PRAGMA UNROLL ==========
    std::cout << "\n--- #pragma unroll ---\n";
    Timer t3;
    t3.start();
    kernel_unroll_pragma<<<blocks, threads>>>(d_data, d_out[2], N);
    cudaDeviceSynchronize();
    t3.stop();
    float time3 = t3.getGpuTime();
    std::cout << "  Tiempo: " << time3 << " ms\n";
    std::cout << "  Speedup: " << (time1 / time3) << "x\n";

    // ========== PRUEBA 4: FULL UNROLL ==========
    std::cout << "\n--- full unroll ---\n";
    Timer t4;
    t4.start();
    kernel_unroll_full<<<blocks, threads>>>(d_data, d_out[3], N);
    cudaDeviceSynchronize();
    t4.stop();
    float time4 = t4.getGpuTime();
    std::cout << "  Tiempo: " << time4 << " ms\n";
    std::cout << "  Speedup: " << (time1 / time4) << "x\n";

    // ========== RESUMEN ==========
    std::cout << "\n=== Comparación ===\n";
    std::cout << "  Normal:   " << time1 << " ms (baseline)\n";
    std::cout << "  Manual:   " << time2 << " ms (" << (time1/time2) << "x)\n";
    std::cout << "  Pragama:  " << time3 << " ms (" << (time1/time3) << "x)\n";
    std::cout << "  Full:     " << time4 << " ms (" << (time1/time4) << "x)\n";

    // ========== VERIFICAR ==========
    cudaMemcpy(h_out[0], d_out[0], N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out[3], d_out[3], N * sizeof(float), cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_out[0][i] - h_out[3][i]) > 1e-5f) {
            ok = false; break;
        }
    }
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== ANÁLISIS ==========
    std::cout << "\n📌 Loop unrolling:\n";
    std::cout << "  • Reduce branch overhead\n";
    std::cout << "  • Aumenta ILP\n";
    std::cout << "  • Mayor presión en registros\n";
    std::cout << "\n⚠️  Demasiado unroll → register spilling\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    for (int k = 0; k < 4; k++) {
        delete[] h_out[k];
        cudaFree(d_out[k]);
    }
    cudaFree(d_data);

    std::cout << "\n✅ Ejemplo completado.\n";
    return 0;
}
