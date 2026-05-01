/**
 * @file scan_prefix_sum.cu
 * @brief Prefix Sum (Scan) - Suma prefija
 *
 * Dado un vector X, calcula Y donde Y[i] = Σ X[j] for j = 0..i.
 * Patrón fundamental para: segmented scan, stream compaction,
 * radix sort, etc.
 *
 * Algoritmo: Blelloch scan (O(log N) pasos).
 *
 * Compilación:
 *   nvcc -o scan_prefix_sum scan_prefix_sum.cu
 *
 * Ejecución:
 *   ./scan_prefix_sum [SIZE]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cstring>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel de scan exclusivo (Blelloch Up-Sweep + Down-Sweep)
 *
 * Versión por bloques. Cada bloque computa un scan parcial exclusivo.
 * Algoritmo correcto: up-sweep reduce, luego down-sweep distribuye.
 */
__global__ void scan_kernel_exclusive(const float *input, float *output, int N) {
    // shared memory: usar 2*blockDim para el algoritmo de Blelloch
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Cargar en shared memory con padding
    temp[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Up-sweep (fase de reducción de árbol)
    // stride avanza: 1, 2, 4, 8, ...
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        // Cada hilo trabaja en un nodo específico del árbol
        // El índice se calcula para saltar stride elementos
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
    }
    __syncthreads();

    // Poner 0 en la raíz (último elemento)
    // Esto prepara para el scan exclusivo
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep (fase de distribución)
    // stride decrece: blockDim/2, blockDim/4, ..., 1
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            float t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] = t + temp[index];
        }
    }
    __syncthreads();

    // Escribir resultado
    if (idx < N) {
        output[idx] = temp[tid];
    }
}

/**
 * @brief Kernel simple de scan inclusivo (para N ≤ blockDim)
 *
 * Algoritmo de Hillis-Steele (O(N log N) work, O(log N) pasos):
 * En cada paso d, cada elemento i suma el elemento i-d si i≥d.
 * Sin race conditions porque lecturas son del paso anterior.
 */
__global__ void scan_inclusive_simple(const float *input, float *output, int N) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    // Cargar datos en shared memory con padding
    s_data[tid] = (tid < N) ? input[tid] : 0.0f;
    __syncthreads();

    // Scan inclusivo de Hillis-Steele
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // Sincronizar antes de leer valores del paso anterior
        if (tid >= stride && tid < N) {
            s_data[tid] += s_data[tid - stride];
        }
    }
    __syncthreads();

    if (tid < N) {
        output[tid] = s_data[tid];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 09.2: Prefix Sum (Scan) ===\n\n";

    const int N = 1024;
    int size = N;

    if (argc > 1) size = atoi(argv[1]);
    if (size > 1024) size = 1024;  // Limitar por shared memory

    std::cout << "Vector size: " << size << " (max 1024 para este ejemplo)\n\n";

    // ========== DATOS ==========
    float *h_in = new float[size];
    float *h_out = new float[size];
    float *h_out_ref = new float[size];
    float *d_in, *d_out;

    // Inicializar con secuencia 0,1,2,...
    for (int i = 0; i < size; i++) {
        h_in[i] = (float)i;
    }
    // Referencia: scan exclusivo
    h_out_ref[0] = 0.0f;
    for (int i = 1; i < size; i++) {
        h_out_ref[i] = h_out_ref[i - 1] + h_in[i - 1];
    }

    cudaMalloc(&d_in, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

    // ========== KERNEL ==========
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    if (blocks > 1) blocks = 1;  // Simplificar: un solo bloque para scan simple

    std::cout << "Blocks: " << blocks << ", Threads: " << threads << "\n";

    // Shared memory size para scan exclusivo (Blelloch necesita 2*threads)
    size_t smem_size = 2 * threads * sizeof(float);

    Timer timer;
    timer.start();

    scan_kernel_exclusive<<<blocks, threads, smem_size>>>(d_in, d_out, size);
    cudaDeviceSynchronize();

    timer.stop();
    float gpu_time = timer.getGpuTime();

    cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << gpu_time << " ms\n";

    // ========== VERIFICAR ==========
    std::cout << "\nResultados (primeros 10):\n";
    std::cout << "  i   input  expected  output\n";
    std::cout << "  --  -----  --------  ------\n";

    bool ok = true;
    for (int i = 0; i < min(10, size); i++) {
        float expected = h_out_ref[i];
        float got = h_out[i];
        std::cout << "  " << i;
        std::cout << "   " << h_in[i];
        std::cout << "    " << expected;
        std::cout << "    " << got;
        if (fabs(got - expected) > 1e-5f) {
            std::cout << " ✗";
            ok = false;
        }
        std::cout << "\n";
    }

    if (ok) {
        std::cout << "\n✅ Scan correcto para primeros 10 elementos.\n";
    } else {
        std::cout << "\n❌ Error en scan.\n";
    }

    // ========== CPU REFERENCE ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float old = h_in[i];
        h_out_ref[i] = sum;
        sum += old;
    }
    cpuTimer.stop();

    std::cout << "\nCPU reference time: " << cpuTimer.getCpuTime() << " ms\n";

    // ========== APLICACIONES ==========
    std::cout << "\n📌 Aplicaciones de Scan:\n";
    std::cout << "  • stream compaction (remove zeros)\n";
    std::cout << "  • Radix sort\n";
    std::cout << "  • Lexicographic ordering\n";
    std::cout << "  • Polynomial evaluation (Horner's method)\n";
    std::cout << "  • Sparse matrix operations\n";

    // ========== LIMPIEZA ==========
    delete[] h_in;
    delete[] h_out;
    delete[] h_out_ref;
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
