/**
 * @file matrix_tiling.cu
 * @brief Tiling de matrices usando shared memory
 *
 * Mejora del ejemplo matrix_multiply.cu usando shared memory
 * para cachear bloques de A y B, reduciendo accesos a global memory.
 *
 * Cada hilo calcula un elemento del resultado usando un tile
 * de A (M×K) y B (K×N) en shared memory.
 *
 * Compilación:
 *   nvcc -o matrix_tiling matrix_tiling.cu
 *
 * Ejecución:
 *   ./matrix_tiling [M] [N] [K]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

// Tamaño del tile (debe ser múltiplo de warpSize para evitar bank conflicts)
#define TILE_SIZE 16

/**
 * @brief Kernel de multiplicación con tiling
 *
 * Cada bloque calcula TILE_SIZE×TILE_SIZE elementos de C.
 */
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    // Shared memory para tiles de A y B
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    // Coordenadas del hilo en el resultado
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop sobre los tiles de K
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cargar tile de A [row][t*TILE_SIZE + tx] en sA
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            sA[ty][tx] = A[row * K + a_col];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Cargar tile de B [t*TILE_SIZE + ty][col] en sB
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            sB[ty][tx] = B[b_row * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiplicar los tiles cargados
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Escribir resultado
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Kernel optimizado (sin condiciones en el loop)
 * Requiere que K sea múltiplo de TILE_SIZE
 */
__global__ void matmul_tiled_perfect(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < K / TILE_SIZE; t++) {
        // Carga directa sin boundary check (K/TILE_SIZE es entero)
        sA[ty][tx] = A[row * K + (t * TILE_SIZE + tx)];
        sB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 06.2: Tiling con Shared Memory ===\n\n";

    // ========== PARÁMETROS ==========
    int M = 512, N = 512, K = 512;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    std::cout << "Dimensiones: A(" << M << "×" << K << ") × B(" << K << "×" << N << ") = C(" << M << "×" << N << ")\n";

    // Ajustar a múltiplo de TILE_SIZE
    int M_orig = M, N_orig = N, K_orig = K;
    M = ((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    N = ((N + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    K = ((K + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

    if (M != M_orig || N != N_orig || K != K_orig) {
        std::cout << "Ajustando a múltiplos de " << TILE_SIZE << ": "
                  << M_orig << "→" << M << ", "
                  << N_orig << "→" << N << ", "
                  << K_orig << "→" << K << "\n\n";
    }

    // ========== MEMORIA ==========
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // ========== INICIALIZAR ==========
    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // ========== CONFIGURACIÓN ==========
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, M / TILE_SIZE);

    std::cout << "Blocks por grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")\n";
    std::cout << "Threads por block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")\n";

    // ========== KERNEL TILED ==========
    std::cout << "\n--- Ejecutando kernel con tiling ---\n";

    Timer timer;
    timer.start();

    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A, d_B, d_C, M, N, K);
    gpuErrchk( cudaPeekAtLastError() );

    timer.stop();
    float gpu_time = timer.getGpuTime();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    float gflops = (float)M * N * K * 2.0f / (gpu_time * 1e6);
    std::cout << "  Tiempo: " << gpu_time << " ms\n";
    std::cout << "  Performance: " << gflops << " GFLOPS\n";

    // ========== CPU REFERENCE ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i * K + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
    cpuTimer.stop();
    float cpu_time = cpuTimer.getCpuTime();
    float cpu_gflops = (float)M * N * K * 2.0f / (cpu_time * 1e6);

    // ========== VERIFICAR (solo subconjunto por speed) ==========
    bool ok = true;
    for (int i = 0; i < M_orig; i++) {
        for (int j = 0; j < N_orig; j++) {
            if (fabs(h_C[i * N + j] - h_C_ref[i * N + j]) > 1e-2f) {
                ok = false;
                break;
            }
        }
    }

    std::cout << "\n=== Resultados ===\n";
    std::cout << "  Verificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";
    std::cout << "  CPU:          " << cpu_time << " ms, " << cpu_gflops << " GFLOPS\n";
    std::cout << "  GPU tiled:    " << gpu_time << " ms, " << gflops << " GFLOPS\n";
    std::cout << "  Speedup:      " << (cpu_time / gpu_time) << "x\n";

    // ========== ANÁLISIS DE SHARED MEMORY ==========
    std::cout << "\n📊 Uso de shared memory por bloque:\n";
    int smem_per_block = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    std::cout << "  sA[" << TILE_SIZE << "][" << TILE_SIZE << "] + sB[" << TILE_SIZE << "][" << TILE_SIZE << "] = "
              << smem_per_block / 1024.0f << " KB\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  Shared memory disponible por SM: " << prop.sharedMemPerBlock / 1024.0f << " KB\n";
    int max_blocks_sm = prop.sharedMemPerBlock / smem_per_block;
    if (max_blocks_sm > 0) {
        std::cout << "  Máx bloques simultáneos por SM (shared mem bound): " << max_blocks_sm << "\n";
    }

    // ========== LIMPIEZA ==========
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Key insight: Shared memory reduce accesos a global memory\n";
    std::cout << "   desde O(K) por hilo → O(1) por dato (reutilización).\n";

    return 0;
}
