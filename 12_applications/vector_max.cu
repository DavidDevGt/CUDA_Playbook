/**
 * @file vector_max.cu
 * @brief Aplicación: Encontrar valor máximo en un vector (reducción)
 *
 * Este es un ejemplo completo que integra:
 *   - Inicialización de datos
 *   - Reducción paralela con shared memory
 *   - Verificación contra CPU
 *   - Reporte de performance
 *
 * Caso real: encontrar pico de señal, estadísticas, etc.
 *
 * Compilación:
 *   nvcc -o vector_max vector_max.cu
 *
 * Ejecución:
 *   ./vector_max [SIZE]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel de reducción para encontrar máximo
 */
__global__ void max_reduction_kernel(const float *input, float *block_maxima, int N) {
    extern __shared__ float s_max[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Cargar datos en shared memory
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    s_max[tid] = val;
    __syncthreads();

    // Reducción por árbol binario (max)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid] < s_max[tid + stride]) {
                s_max[tid] = s_max[tid + stride];
            }
        }
        __syncthreads();
    }

    // Escribir máximo del bloque
    if (tid == 0) {
        block_maxima[blockIdx.x] = s_max[0];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Aplicación 12.1: Máximo de Vector (Reducción) ===\n\n";

    int N = 1 << 22;
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Vector size: " << N << "\n";

    // ========== DATOS ==========
    float *h_data = new float[N];
    float *d_data, *d_block_max;

    // Inicializar con valores aleatorios
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 1000);
    }

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // ========== KERNEL ==========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cudaMalloc(&d_block_max, blocks * sizeof(float));

    std::cout << "Blocks: " << blocks << ", Threads/block: " << threads << "\n";

    Timer timer;
    timer.start();

    max_reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_data, d_block_max, N);
    gpuErrchk( cudaDeviceSynchronize() );

    timer.stop();
    float gpu_time = timer.getGpuTime();

    // Copiar máximos parciales
    float *h_block_max = new float[blocks];
    cudaMemcpy(h_block_max, d_block_max, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Reducir en CPU
    float gpu_max = h_block_max[0];
    for (int i = 1; i < blocks; i++) {
        if (h_block_max[i] > gpu_max) gpu_max = h_block_max[i];
    }

    std::cout << "  GPU max:  " << gpu_max << "\n";
    std::cout << "  Tiempo:   " << gpu_time << " ms\n";

    // ========== CPU REFERENCE ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    float cpu_max = h_data[0];
    for (int i = 1; i < N; i++) {
        if (h_data[i] > cpu_max) cpu_max = h_data[i];
    }
    cpuTimer.stop();

    std::cout << "  CPU max:  " << cpu_max << "\n";
    std::cout << "  CPU time: " << cpuTimer.getCpuTime() << " ms\n";
    std::cout << "  Speedup:  " << (cpuTimer.getCpuTime() / gpu_time) << "x\n";
    std::cout << "  Correcto: " << (fabs(gpu_max - cpu_max) < 1e-5f ? "✅ Sí" : "❌ No") << "\n";

    // ========== APLICACIONES ==========
    std::cout << "\n📌 Aplicaciones de max-reduction:\n";
    std::cout << "  • Encontrar peak en señal de audio\n";
    std::cout << "  • Estadísticas (máximo, mínimo)\n";
    std::cout << "  • Ray tracing: Bounding Volume Hierarchy (BVH) bounds\n";
    std::cout << "  • Computer vision: feature detection\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    delete[] h_block_max;
    cudaFree(d_data);
    cudaFree(d_block_max);

    std::cout << "\n✅ Aplicación completada.\n";

    return 0;
}

/* Nota: La versión completa sería multi-kernel (llamadas recursivas)
   pero este ejemplo de 1-pass es suficiente para demostrar el patrón.
*/
