/**
 * @file vector_dot.cu
 * @brief Producto punto (dot product) de dos vectores
 *
 * Calcula: dot_product = Σ (A[i] * B[i]) para i = 0..N-1
 *
 * Este ejemplo introduce:
 * - Reducción secuencial en paralelo (primeros pasos)
 * - Uso de atomics para acumulación simple
 * - Paralelización de la suma
 *
 * Compilación:
 *   nvcc -o vector_dot vector_dot.cu
 *
 * Ejecución:
 *   ./vector_dot [N]
 *
 * Nota: Para vectores grandes, una implementación sin atómicos
 * sería más eficiente (ver lección 09).
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel naive para producto punto usando atomicAdd
 *
 * Cada hilo calcula un producto parcial y lo agrega atómicamente.
 * No es lo más eficiente para grandes N (mucho contention),
 * pero es simple de entender.
 *
 * @param A Vector A
 * @param B Vector B
 * @param C Resultado (puntero a un único valor en device)
 * @param N Longitud de vectores
 */
__global__ void dot_product_atomic(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float product = A[idx] * B[idx];
        atomicAdd(C, product);
    }
}

/**
 * @brief Kernel optimizado usando reducción en shared memory
 *
 * Cada bloque calcula una suma parcial en shared memory,
 * luego un hilo por bloque escribe el resultado parcial a global memory.
 * Finalmente se suman los parciales en CPU (o segundo kernel).
 */
__global__ void dot_product_reduction(const float *A, const float *B, float *block_sums, int N) {
    // Shared memory para suma dentro del bloque
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Cargar dato en shared memory (cada hilo una posición)
    float value = 0.0f;
    if (idx < N) {
        value = A[idx] * B[idx];
    }
    sdata[tid] = value;
    __syncthreads();

    // Reducción en shared memory (patrón de árbol binario)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // El hilo 0 escribe la suma del bloque a global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 02.2: Producto Punto (Dot Product) ===\n\n";

    // ========== PARÁMETROS ==========
    int N = 1 << 20;  // 1M elementos
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Vector length: " << N << "\n";
    std::cout << "Memoria por vector: " << N * sizeof(float) / (1024.0*1024.0) << " MB\n\n";

    // ========== CONFIGURACIÓN ==========
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > 65535) blocksPerGrid = 65535;

    // ========== ASIGNACIÓN ==========
    size_t size = N * sizeof(float);
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C_ref = new float[N];  // Resultado por bloques (para método reducción)
    float *d_A, *d_B;
    float *d_dot_result;       // Para método atómico
    float *d_block_sums;       // Para método reducción

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_dot_result, sizeof(float));
    cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));

    // ========== INICIALIZAR ==========
    // Usar valores conocidos para verificación fácil
    // A[i] = 1.0, B[i] = 1.0 → dot = N
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    float expected_dot = (float)N;

    // Copiar a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Inicializar resultado en 0
    float zero = 0.0f;
    cudaMemcpy(d_dot_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Vectores inicializados (A[i]=1, B[i]=1).\n";
    std::cout << "Resultado esperado: " << expected_dot << "\n\n";

    // ========== MÉTODO 1: AtomicAdd ==========
    std::cout << "--- Método 1: atomicAdd (simple) ---\n";

    Timer timer1;
    timer1.start();

    dot_product_atomic<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_dot_result, N);
    gpuErrchk( cudaPeekAtLastError() );

    timer1.stop();
    float time1 = timer1.getGpuTime();

    float dot_atomic;
    cudaMemcpy(&dot_atomic, d_dot_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << time1 << " ms\n";
    std::cout << "  Resultado: " << dot_atomic << "\n";
    std::cout << "  Error: " << fabs(dot_atomic - expected_dot) << "\n";

    // ========== MÉTODO 2: Shared Memory Reduction ==========
    std::cout << "\n--- Método 2: Shared Memory Reduction ---\n";

    // Reiniciar resultado
    cudaMemcpy(d_dot_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    Timer timer2;
    timer2.start();

    // Kernel que genera sumas parciales por bloque
    dot_product_reduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_A, d_B, d_block_sums, N);
    gpuErrchk( cudaPeekAtLastError() );

    // Copiar sumas parciales a host
    float *h_block_sums = new float[blocksPerGrid];
    cudaMemcpy(h_block_sums, d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    timer2.stop();
    float time2 = timer2.getGpuTime();

    // Sumar parciales en CPU
    float dot_reduction = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        dot_reduction += h_block_sums[i];
    }

    std::cout << "  Tiempo kernel: " << time2 << " ms\n";
    std::cout << "  Suma en CPU de " << blocksPerGrid << " parciales\n";
    std::cout << "  Resultado: " << dot_reduction << "\n";
    std::cout << "  Error: " << fabs(dot_reduction - expected_dot) << "\n";

    // ========== COMPARACIÓN ==========
    std::cout << "\n=== Comparación ===\n";
    std::cout << "  Método atomicAdd:    " << time1 << " ms\n";
    std::cout << "  Método reduction:    " << time2 << " ms\n";

    if (time1 < time2) {
        std::cout << "  → atomicAdd es más rápido en este caso\n";
        std::cout << "    (puede deberse a que N es pequeño o hay pocos hilos por bloque)\n";
    } else {
        std::cout << "  → Reduction es más rápido (reducción de contención)\n";
    }

    // ========== CPU REFERENCE ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    float cpu_dot = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu_dot += h_A[i] * h_B[i];
    }
    cpuTimer.stop();

    std::cout << "\nCPU reference:\n";
    std::cout << "  Tiempo: " << cpuTimer.getCpuTime() << " ms\n";
    std::cout << "  Resultado: " << cpu_dot << "\n";

    float speedup1 = cpuTimer.getCpuTime() / time1;
    float speedup2 = cpuTimer.getCpuTime() / time2;
    std::cout << "  Speedup (atomic): " << speedup1 << "x\n";
    std::cout << "  Speedup (reduction): " << speedup2 << "x\n";

    // ========== LIMPIEZA ==========
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_ref;
    delete[] h_block_sums;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_dot_result);
    cudaFree(d_block_sums);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: Matrices (lección 03)\n";

    return 0;
}
