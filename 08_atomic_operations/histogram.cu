/**
 * @file histogram.cu
 * @brief Histograma con operaciones atómicas
 *
 * Construye un histograma (conteo de frecuencias) de un vector
 * de datos. Muestra el problema de contención en atómicos y
 * cómo resolverlo con privatización en shared memory.
 *
 * Compilación:
 *   nvcc -o histogram histogram.cu
 *
 * Ejecución:
 *   ./histogram [N] [BINS]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

#define BINS 256  // Número de bins del histograma

/**
 * @brief Kernel naive: cada hilo actualiza bin directamente (mucha contención)
 */
__global__ void histogram_naive(const unsigned char *data, int *hist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int bin = data[idx];  // 0-255
        atomicAdd(&hist[bin], 1);  // Contención alta!
    }
}

/**
 * @brief Kernel optimizado: cada bloque computa su histograma parcial en shared memory
 */
__global__ void histogram_shared(const unsigned char *data, int *hist, int N) {
    // Shared memory para histograma privado del bloque
    __shared__ int s_hist[BINS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Inicializar shared histogram a 0
    for (int i = tid; i < BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Procesar datos
    if (idx < N) {
        int bin = data[idx];
        atomicAdd(&s_hist[bin], 1);  // Atómico en shared (más rápido)
    }
    __syncthreads();

    // Consolidar: un hilo por bin actualiza global histogram
    for (int i = tid; i < BINS; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);  // Atómico en global (poca contención)
    }
}

/**
 * @brief Kernel que usa warp-level primitives (reducción en warp)
 * Requiere CC >= 3.0
 */
__global__ void histogram_warpvote(const unsigned char *data, int *hist, int N) {
    // Cada warp procesa múltiples elementos
    int tid = threadIdx.x;
    int wid = tid / warpSize;
    int lane = tid % warpSize;

    __shared__ int warp_hist[32][BINS];  // 1 warp por fila

    // Inicializar
    for (int b = 0; b < BINS; b++) {
        warp_hist[lane][b] = 0;
    }
    __syncthreads();

    // Processar elementos asignados a este warp
    int warp_start = wid * warpSize * 2;  // 2 elementos por hilo inicialmente
    // ... complejo, omitido por simplicidad

    // Merge a global
    for (int b = lane; b < BINS; b += warpSize) {
        atomicAdd(&hist[b], warp_hist[lane][b]);
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 08.3: Histograma con Atómicos ===\n\n";

    int N = 1 << 22;  // ~4M
    int bins = BINS;

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) bins = atoi(argv[2]);
    if (bins > 256) bins = 256;

    std::cout << "Elementos: " << N << "\n";
    std::cout << "Bins:      " << bins << " (valores 0-" << bins-1 << ")\n\n";

    // ========== DATOS ==========
    unsigned char *h_data = new unsigned char[N];
    int *h_hist_naive = new int[bins]();   // zero-initialized
    int *h_hist_shared = new int[bins]();

    unsigned char *d_data;
    int *d_hist_naive, *d_hist_shared;

    // Datos aleatorios
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % bins;
    }

    cudaMalloc(&d_data, N * sizeof(unsigned char));
    cudaMalloc(&d_hist_naive, bins * sizeof(int));
    cudaMalloc(&d_hist_shared, bins * sizeof(int));

    cudaMemcpy(d_data, h_data, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist_naive, 0, bins * sizeof(int));
    cudaMemset(d_hist_shared, 0, bins * sizeof(int));

    // ========== CONFIGURACIÓN ==========
    int threads = 256;
    int blocks_naive = (N + threads - 1) / threads;
    int blocks_shared = (N + threads - 1) / threads;

    if (blocks_naive > 65535) blocks_naive = 65535;
    if (blocks_shared > 65535) blocks_shared = 65535;

    // ========== MÉTODO NAIVE ==========
    std::cout << "--- Método 1: atomicAdd global directo (contención alta) ---\n";

    Timer timer1;
    timer1.start();
    histogram_naive<<<blocks_naive, threads>>>(d_data, d_hist_naive, N);
    gpuErrchk( cudaDeviceSynchronize() );
    timer1.stop();

    float time1 = timer1.getGpuTime();
    std::cout << "  Tiempo: " << time1 << " ms\n";

    cudaMemcpy(h_hist_naive, d_hist_naive, bins * sizeof(int), cudaMemcpyDeviceToHost);

    // Verificar suma total
    int total_naive = 0;
    for (int i = 0; i < bins; i++) total_naive += h_hist_naive[i];
    std::cout << "  Total conteo: " << total_naive << " (esperado: " << N << ")\n";

    // ========== MÉTODO COMPARTIDO ==========
    std::cout << "\n--- Método 2: Privatización + atomicAdd final ---\n";

    Timer timer2;
    timer2.start();
    histogram_shared<<<blocks_shared, threads, BINS * sizeof(int)>>>(d_data, d_hist_shared, N);
    gpuErrchk( cudaDeviceSynchronize() );
    timer2.stop();

    float time2 = timer2.getGpuTime();
    std::cout << "  Tiempo: " << time2 << " ms\n";
    std::cout << "  Speedup: " << (time1 / time2) << "x\n";

    cudaMemcpy(h_hist_shared, d_hist_shared, bins * sizeof(int), cudaMemcpyDeviceToHost);

    int total_shared = 0;
    for (int i = 0; i < bins; i++) total_shared += h_hist_shared[i];
    std::cout << "  Total conteo: " << total_shared << "\n";

    // ========== MOSTRAR DISTRIBUCIÓN ==========
    std::cout << "\nDistribución (primeros 10 bins):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  Bin " << i << ": naive=" << h_hist_naive[i]
                  << ", shared=" << h_hist_shared[i] << "\n";
    }

    // ========== CPU REFERENCE ==========
    std::cout << "\n--- CPU reference ---\n";
    int *h_hist_cpu = new int[bins]();
    for (int i = 0; i < N; i++) {
        h_hist_cpu[h_data[i]]++;
    }

    bool ok = true;
    for (int i = 0; i < bins; i++) {
        if (h_hist_shared[i] != h_hist_cpu[i]) {
            ok = false;
            break;
        }
    }
    std::cout << "  Verificación vs CPU: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== ANÁLISIS ==========
    std::cout << "\n📌 Lecciones:\n";
    std::cout << "  1. atomicAdd en global memory → alta contención\n";
    std::cout << "  2. Privatización en shared memory reduce atomic ops globales\n";
    std::cout << "  3. Cada bloque acumula su own histogram → menos conflicts\n";
    std::cout << "  4. Solo al final se consolidan (pocos atomic ops)\n";
    std::cout << "\nPatrón general: privatize → compute → reduce\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    delete[] h_hist_naive;
    delete[] h_hist_shared;
    delete[] h_hist_cpu;
    cudaFree(d_data);
    cudaFree(d_hist_naive);
    cudaFree(d_hist_shared);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
