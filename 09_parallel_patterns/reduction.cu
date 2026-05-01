/**
 * @file reduction.cu
 * @brief Patrón de Reducción (suma, mín, máx)
 *
 * La reducción es uno de los patrones más comunes: combinar N valores
 * en un solo resultado (suma, producto, min, max, etc.).
 *
 * Algoritmo: árbol binario en O(log N) pasos.
 *
 * Compilación:
 *   nvcc -o reduction reduction.cu
 *
 * Ejecución:
 *   ./reduction [SIZE] [OPERATION]
 *   OPERATION: sum, min, max
 */

#include <iostream>
#include <cuda_runtime.h>
#include <limits>
#include <cfloat>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Reducción en CPU (referencia)
 */
template<typename T, typename Op>
T reduction_cpu(const T *data, int N, Op op, T init) {
    T result = init;
    for (int i = 0; i < N; i++) {
        result = op(result, data[i]);
    }
    return result;
}

/**
 * @brief Kernel de reducción con shared memory (solo suma)
 */
__global__ void reduction_kernel_sum(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;

    float sum = 0.0f;
    if (idx < N) sum = input[idx];
    if (idx + blockDim.x < N) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

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
    std::cout << "=== Ejemplo 09.1: Patrón de Reducción ===\n\n";

    int N = 1 << 24;
    const char *op_name = "sum";

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) op_name = argv[2];

    std::cout << "Vector size: " << N << "\n";
    std::cout << "Operación: " << op_name << "\n\n";

    size_t bytes = N * sizeof(float);
    float *h_data = new float[N];
    float *d_data, *d_buf1, *d_buf2, *d_final;

    // Inicializar datos con valores conocidos
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 100);  // 0..99 repetidos
    }

    // Suma de verificación en CPU directa
    double direct_sum = 0.0;
    for (int i = 0; i < N; i++) direct_sum += h_data[i];
    std::cout << "Suma directa CPU (double): " << direct_sum << "\n";

    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_final, sizeof(float));

    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // ========== CONFIGURACIÓN ==========
    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    if (blocks > 65535) blocks = 65535;

    // Dos buffers para ping-pong en multi-pass
    cudaMalloc(&d_buf1, blocks * sizeof(float));
    cudaMalloc(&d_buf2, blocks * sizeof(float));

    // ========== KERNEL ==========
    std::cout << "Blocks: " << blocks << ", Threads/block: " << threads << "\n";

    Timer timer;

    if (strcmp(op_name, "sum") == 0) {
        // Multi-pass reducción con doble buffer
        int current_N = N;
        float *d_src = d_data;
        float *d_dst = d_buf1;

        timer.start();

        int pass = 0;
        while (current_N > 1) {
            int pass_blocks = (current_N + threads * 2 - 1) / (threads * 2);
            if (pass_blocks < 1) pass_blocks = 1;

            reduction_kernel_sum<<<pass_blocks, threads, threads * sizeof(float)>>>
                (d_src, d_dst, current_N);
            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA error en pass " << pass << ": " 
                         << cudaGetErrorString(err) << "\n";
                break;
            }

            // Swap: el output de esta pasada es el input de la siguiente
            d_src = d_dst;
            d_dst = (d_dst == d_buf1) ? d_buf2 : d_buf1;
            current_N = pass_blocks;
            pass++;
        }

        timer.stop();

        // d_src apunta al buffer con el resultado final (1 elemento)
        float total;
        cudaMemcpy(&total, d_src, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "  Resultado GPU: " << total << "\n";
        std::cout << "  Pasadas: " << pass << "\n";
        std::cout << "  Tiempo kernel(s): " << timer.getGpuTime() << " ms\n";
    } else {
        std::cout << "  (Implementación de min/max similar)\n";
    }

    // ========== CPU REFERENCE ==========
    // Suma con algoritmo simple
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) cpu_sum += h_data[i];

    std::cout << "  CPU sum (double): " << cpu_sum << "\n";
    std::cout << "  CPU sum (float) : " << (float)cpu_sum << "\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    cudaFree(d_data);
    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_final);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   La reducción es base para: sum, min, max, any, all\n";

    return 0;
}
