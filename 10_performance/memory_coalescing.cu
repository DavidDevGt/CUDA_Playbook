/**
 * @file memory_coalescing.cu
 * @brief Accesos de memoria coalesced vs no-coalesced
 *
 * Memory coalescing: accesos consecutivos por hilos consecutivos
 * → se combinan en fewer transactionos de memory → más rápido.
 *
 * Ejemplo:
 *   Coalesced:   thread 0 → addr[0], thread 1 → addr[1], ...
 *   No-coalesced: thread 0 → addr[0], thread 1 → addr[256], ...
 *
 * Compilación:
 *   nvcc -o memory_coalescing memory_coalescing.cu
 *
 * Ejecución:
 *   ./memory_coalescing [SIZE]
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel con acceso coalesced (stride 1)
 */
__global__ void coalesced_access_kernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx];  // Acceso secuencial → coalesced
    }
}

/**
 * @brief Kernel con acceso no-coalesced (stride grande)
 */
__global__ void uncoalesced_access_kernel(const float *input, float *output, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Acceso con stride → posible no-coalesced
        int read_idx = (idx * stride) % N;
        output[idx] = input[read_idx];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 10.2: Memory Coalescing ===\n\n";

    int N = 1 << 22;  // 4M
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Vector size: " << N << " floats (" << N * sizeof(float) / (1024.0*1024.0) << " MB)\n\n";

    size_t bytes = N * sizeof(float);
    float *h_data = new float[N];
    float *h_out1 = new float[N];
    float *h_out2 = new float[N];
    float *d_data, *d_out1, *d_out2;

    // Inicializar secuencial
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out1, bytes);
    cudaMalloc(&d_out2, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // ========== EJEMPLO 1: ACCESO COALESCED ==========
    std::cout << "--- Acceso coalesced (stride = 1) ---\n";

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    Timer timer_coal;
    timer_coal.start();
    coalesced_access_kernel<<<blocks, threads>>>(d_data, d_out1, N);
    cudaDeviceSynchronize();
    timer_coal.stop();

    float time_coal = timer_coal.getGpuTime();
    float bw_coal = (bytes / (1024.0*1024.0)) / (time_coal / 1000.0f);  // MB/s

    std::cout << "  Tiempo: " << time_coal << " ms\n";
    std::cout << "  Bandwidth (teórico): " << bw_coal << " MB/s\n";

    cudaMemcpy(h_out1, d_out1, bytes, cudaMemcpyDeviceToHost);

    // ========== EJEMPLO 2: ACCESO NO COALESCED ==========
    std::cout << "\n--- Acceso no-coalesced (stride = 32) ---\n";

    int stride = 32;
    Timer timer_uncoal;
    timer_uncoal.start();
    uncoalesced_access_kernel<<<blocks, threads>>>(d_data, d_out2, N, stride);
    cudaDeviceSynchronize();
    timer_uncoal.stop();

    float time_uncoal = timer_uncoal.getGpuTime();
    float bw_uncoal = (bytes / (1024.0*1024.0)) / (time_uncoal / 1000.0f);

    std::cout << "  Tiempo: " << time_uncoal << " ms\n";
    std::cout << "  Bandwidth: " << bw_uncoal << " MB/s\n";
    std::cout << "  Slowdown: " << (time_uncoal / time_coal) << "x\n";

    cudaMemcpy(h_out2, d_out2, bytes, cudaMemcpyDeviceToHost);

    // ========== VERIFICAR CORRECTITUD ==========
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_out1[i] - h_data[i]) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "\nVerificación coalesced: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== EXPLICACIÓN ==========
    std::cout << "\n📌 ¿Por qué stride=32 es malo?\n";
    std::cout << "  • Cada warp (32 hilos) accesa 32 direcciones separadas por 32×4=128 bytes\n";
    std::cout << "  • Si cada acceso cae en diferentes 128-byte segments → múltiples memory transactions\n";
    std::cout << "  • En GCC, acceso coalesced: 1 transaction por warp (128B)\n";
    std::cout << "  • No-coalesced: hasta 32 transactions por warp → 32× más lento\n\n";
    std::cout << "📌 Reglas de memoria coalescing:\n";
    std::cout << "  1. Hilos consecutivos acceden direcciones consecutivas\n";
    std::cout << "  2. Acceso stride = 1 es ideal\n";
    std::cout << "  3. Si stride es múltiplo de warpSize (32), revisar alineación\n";
    std::cout << "  4. Usar padding para evitar stride que cause bank conflicts +\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    delete[] h_out1;
    delete[] h_out2;
    cudaFree(d_data);
    cudaFree(d_out1);
    cudaFree(d_out2);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
