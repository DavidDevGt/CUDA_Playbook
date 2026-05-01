/**
 * @file matrix_multiply.cu
 * @brief Multiplicación de matrices (C = A × B)
 *
 * Muestra el algoritmo naive (O(n³)) y una versión optimizada
 * usando tiling y shared memory.
 *
 * Ecuación: C[i][j] = Σ (A[i][k] × B[k][j]) para k = 0..K-1
 *
 * Compilación:
 *   nvcc -o matrix_multiply matrix_multiply.cu
 *
 * Ejecución:
 *   ./matrix_multiply [M] [N] [K]
 *
 * Ejemplo:
 *   ./matrix_multiply 1024 1024 1024
 *
 * Nota: Para optimización avanzada con tiling, ver lección 06.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel naive de multiplicación de matrices
 *
 * Cada hilo calcula un elemento C[i][j].
 * Acceso no coalesced a B → performance pobre.
 *
 * @param A Matriz M×K
 * @param B Matriz K×N
 * @param C Matriz resultado M×N
 * @param M Filas de A y C
 * @param N Columnas de B y C
 * @param K Columnas de A / filas de B
 */
__global__ void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    // Índices 2D del hilo
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..N-1

    if (row < M && col < N) {
        float sum = 0.0f;
        // calcular C[row][col]
        for (int k = 0; k < K; k++) {
            // A[row][k] = A[row * K + k]
            // B[k][col] = B[k * N + col]  <-- acceso no coalesced!
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief Kernel optimizado con tiling en shared memory
 *
 * Cada bloque calcula un tile de la matriz resultante.
 * Los tiles de A y B se cargan en shared memory para reutilización.
 *
 * TILE_DIM debe ser múltiplo de warpSize (32) y ≤ sharedMemPerBlock
 */
#define TILE_DIM 16

__global__ void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    // Shared memory para tiles de A y B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Índices globales del hilo
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop sobre los tiles
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        // Cargar tile de A y B en shared memory
        int tiledA_col = t * TILE_DIM + threadIdx.x;
        int tiledA_row = blockIdx.y * TILE_DIM + threadIdx.y;
        if (tiledA_row < M && tiledA_col < K) {
            As[threadIdx.y][threadIdx.x] = A[tiledA_row * K + tiledA_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int tiledB_row = t * TILE_DIM + threadIdx.y;
        int tiledB_col = blockIdx.x * TILE_DIM + threadIdx.x;
        if (tiledB_row < K && tiledB_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledB_row * N + tiledB_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiplicar los tiles cargados
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Escribir resultado
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * @brief Kernel con tiling y sin condición en el loop (para K múltiplo de TILE_DIM)
 */
__global__ void matmul_tiled_perfect(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    #pragma unroll
    for (int t = 0; t < K / TILE_DIM; t++) {
        As[threadIdx.y][threadIdx.x] = A[(blockIdx.y * TILE_DIM + threadIdx.y) * K + (t * TILE_DIM + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + (blockIdx.x * TILE_DIM + threadIdx.x)];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 03.2: Multiplicación de Matrices ===\n\n";

    // ========== PARÁMETROS ==========
    int M = 512;  // filas de A, filas de C
    int N = 512;  // columnas de B, columnas de C
    int K = 512;  // columnas de A, filas de B

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    std::cout << "Dimensiones:\n";
    std::cout << "  A: " << M << " × " << K << "\n";
    std::cout << "  B: " << K << " × " << N << "\n";
    std::cout << "  C: " << M << " × " << N << "\n";
    std::cout << "  Total elementos C: " << M * N << "\n";
    std::cout << "  Operaciones FLOPs: " << (float)M * N * K * 2 << " (2× por multiplicación+suma)\n\n";

    // ========== ASIGNACIÓN ==========
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
    // Inicializar con datos aleatorios simples pero predecibles
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 10 + 1);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 10 + 1);
    }

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    std::cout << "Matrices A y B inicializadas con valores 1-10.\n";

    // ========== CONFIGURACIÓN DE BLOQUES ==========
    // Método naive: 1D
    int threadsPerBlock = 256;
    int blocksPerGrid = (M * N + threadsPerBlock - 1) / threadsPerBlock;

    // Método tiled: 2D
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim(
        (N + TILE_DIM - 1) / TILE_DIM,
        (M + TILE_DIM - 1) / TILE_DIM
    );

    std::cout << "Configuración tiled:\n";
    std::cout << "  Block: " << TILE_DIM << "×" << TILE_DIM << " = " << TILE_DIM*TILE_DIM << " threads\n";
    std::cout << "  Grid:  " << gridDim.x << "×" << gridDim.y << " = " << gridDim.x*gridDim.y << " blocks\n\n";

    // ========== MÉTODO NAIVE ==========
    std::cout << "--- Método naive (sin shared memory) ---\n";

    Timer timer_naive;
    timer_naive.start();

    matmul_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk( cudaPeekAtLastError() );

    timer_naive.stop();
    float time_naive = timer_naive.getGpuTime();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << time_naive << " ms\n";
    float gflops_naive = (M * N * K * 2.0f) / (time_naive * 1e6);
    std::cout << "  Performance: " << gflops_naive << " GFLOPS\n";

    // ========== MÉTODO TILED ==========
    std::cout << "\n--- Método tiled (con shared memory) ---\n";

    Timer timer_tiled;
    timer_tiled.start();

    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    gpuErrchk( cudaPeekAtLastError() );

    timer_tiled.stop();
    float time_tiled = timer_tiled.getGpuTime();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << time_tiled << " ms\n";
    float gflops_tiled = (M * N * K * 2.0f) / (time_tiled * 1e6);
    std::cout << "  Performance: " << gflops_tiled << " GFLOPS\n";

    // ========== COMPARACIÓN ==========
    std::cout << "\n=== Resultados ===\n";
    std::cout << "  Speedup (tiled vs naive): " << (time_naive / time_tiled) << "x\n";
    std::cout << "  Mejora: " << ((time_naive - time_tiled) / time_naive * 100.0f) << "%\n";

    // ========== VERIFICACIÓN (CPU) ==========
    std::cout << "\nVerificando resultado (CPU reference)...\n";

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
    float cpuTime = cpuTimer.getCpuTime();

    float cpu_gflops = (M * N * K * 2.0f) / (cpuTime * 1e6);
    std::cout << "  CPU time: " << cpuTime << " ms\n";
    std::cout << "  CPU GFLOPS: " << cpu_gflops << "\n";

    // Verificación de precisión
    bool ok = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3f) {
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "✅ Resultados correctos (dentro de tolerancia)\n";
    } else {
        std::cout << "❌ Error en verificación\n";
    }

    // ========== RESUMEN ==========
    std::cout << "\n=== Resumen de performance ===\n";
    std::cout << "  CPU:                 " << cpuTime << " ms  (" << cpu_gflops << " GFLOPS)\n";
    std::cout << "  GPU naive:           " << time_naive << " ms  (" << gflops_naive << " GFLOPS)\n";
    std::cout << "  GPU tiled:           " << time_tiled << " ms  (" << gflops_tiled << " GFLOPS)\n";
    std::cout << "  Speedup naive:       " << (cpuTime / time_naive) << "x\n";
    std::cout << "  Speedup tiled:       " << (cpuTime / time_tiled) << "x\n";
    std::cout << "  Mejora tiled/naive:  " << (time_naive / time_tiled) << "x\n";

    // ========== LIMPIEZA ==========
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Para optimización avanzada, ver lección 06 (Shared Memory).\n";

    return 0;
}
