/**
 * @file reduction_shared.cu
 * @brief Reducción paralela usando shared memory
 *
 * Suma todos los elementos de un vector (o encontrar min/max).
 * Patrón de reduce: O(N) → O(log N) pasos.
 *
 * Compilación:
 *   nvcc -o reduction_shared reduction_shared.cu
 *
 * Ejecución:
 *   ./reduction_shared [SIZE]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <limits>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Reducción secuencial (CPU)
 */
float reduction_cpu(const float *data, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }
    return sum;
}

/**
 * @brief Kernel de reducción usando shared memory (versión 1)
 *
 * Cada bloque reduce sus datos a un valor, luego se suman en CPU.
 * No es la implementación más optimizada (verOptimized CUDA).
 */
__global__ void reduction_kernel_shared(const float *input, float *block_sums, int N) {
    __shared__ float sdata[256];  // Asume max 256 threads/block

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Cargar en shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reducir en shared memory (patrón de árbol binario)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Escribir suma del bloque
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief Kernel optimizado con sequentially addressing (menos divergence)
 */
__global__ void reduction_optimized(const float *input, float *output, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;

    // Cargar dos elementos por hilo (mejora utilización)
    float mySum = 0.0f;
    if (idx < N) mySum += input[idx];
    if (idx + blockDim.x < N) mySum += input[idx + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // Reducción en el bloque
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 06.3: Reducción con Shared Memory ===\n\n";

    // ========== PARÁMETROS ==========
    int N = 1 << 22;  // ~4M
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Vector size: " << N << "\n";

    // ========== DATOS ==========
    size_t bytes = N * sizeof(float);
    float *h_data = new float[N];
    float *h_block_sums;
    float *d_data, *d_block_sums;

    // Inicializar con valores conocidos: 1.0f cada elemento → suma = N
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    cudaMalloc(&d_data, bytes);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaMalloc(&d_block_sums, blocks * sizeof(float));
    h_block_sums = new float[blocks];

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    std::cout << "Blocks: " << blocks << ", Threads/block: " << threads << "\n\n";

    // ========== KERNEL 1: SHARED SIMPLE ==========
    std::cout << "--- Método 1: Reducción simple ---\n";

    Timer timer1;
    timer1.start();

    reduction_kernel_shared<<<blocks, threads>>>(d_data, d_block_sums, N);
    cudaDeviceSynchronize();

    timer1.stop();
    float time1 = timer1.getGpuTime();

    cudaMemcpy(h_block_sums, d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Sumar parciales en CPU
    float sum_gpu1 = 0.0f;
    for (int i = 0; i < blocks; i++) {
        sum_gpu1 += h_block_sums[i];
    }

    std::cout << "  Tiempo kernel: " << time1 << " ms\n";
    std::cout << "  Suma GPU: " << sum_gpu1 << " (esperado: " << (float)N << ")\n";
    std::cout << "  Error absoluto: " << fabs(sum_gpu1 - N) << "\n";

    // ========== KERNEL 2: OPTIMIZED (2 elementos/hilo) ==========
    std::cout << "\n--- Método 2: Optimizado (2 elementos/hilo) ---\n";

    // Recalcular blocks (cada hilo procesa 2 elementos)
    threads = 256;
    blocks = (N + threads * 2 - 1) / (threads * 2);
    if (blocks > 65535) blocks = 65535;

    Timer timer2;
    timer2.start();

    reduction_optimized<<<blocks, threads, threads * sizeof(float)>>>(d_data, d_block_sums, N);
    cudaDeviceSynchronize();

    timer2.stop();
    float time2 = timer2.getGpuTime();

    cudaMemcpy(h_block_sums, d_block_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum_gpu2 = 0.0f;
    for (int i = 0; i < blocks; i++) {
        sum_gpu2 += h_block_sums[i];
    }

    std::cout << "  Blocks: " << blocks << " (mitad que antes)\n";
    std::cout << "  Tiempo kernel: " << time2 << " ms\n";
    std::cout << "  Suma GPU: " << sum_gpu2 << "\n";

    // ========== CPU REFERENCE ==========
    std::cout << "\n--- CPU reference ---\n";

    CpuTimer cpuTimer;
    cpuTimer.start();
    float sum_cpu = reduction_cpu(h_data, N);
    cpuTimer.stop();

    std::cout << "  Tiempo CPU: " << cpuTimer.getCpuTime() << " ms\n";
    std::cout << "  Suma CPU: " << sum_cpu << "\n";

    // ========== COMPARACIÓN ==========
    std::cout << "\n=== Resultados ===\n";
    std::cout << "  Suma esperada: " << (float)N << "\n";
    std::cout << "  Método 1:     " << sum_gpu1 << " (err: " << fabs(sum_gpu1 - N) << ")\n";
    std::cout << "  Método 2:     " << sum_gpu2 << " (err: " << fabs(sum_gpu2 - N) << ")\n";
    std::cout << "  CPU:          " << sum_cpu << "\n";

    std::cout << "\n  Speedup GPU1: " << (cpuTimer.getCpuTime() / time1) << "x\n";
    std::cout << "  Speedup GPU2: " << (cpuTimer.getCpuTime() / time2) << "x\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    delete[] h_block_sums;
    cudaFree(d_data);
    cudaFree(d_block_sums);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: bank_conflicts.cu\n";

    return 0;
}
