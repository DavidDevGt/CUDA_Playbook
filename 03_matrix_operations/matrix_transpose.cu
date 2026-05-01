/**
 * @file matrix_transpose.cu
 * @brief Transposición de matriz: B = A^T
 *
 * Muestra cómo indexar correctamente una matriz en 2D
 * y el impacto de los patrones de acceso a memoria.
 *
 * Compilación:
 *   nvcc -o matrix_transpose matrix_transpose.cu
 *
 * Ejecución:
 *   ./matrix_transpose [M] [N]
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel naive para transposición
 *
 * Cada hilo lee A[row][col] y escribe a B[col][row].
 * Puede causar conflictos de memoria no coalesced.
 */
__global__ void transpose_naive(const float *A, float *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Lectura: A[row][col] = A[row * cols + col]
        // Escritura: B[col][row] = B[col * rows + row]
        B[col * rows + row] = A[row * cols + col];
    }
}

/**
 * @brief Kernel optimizado usando shared memory para evitar bank conflicts
 *
 * Cada bloque procesa un tile de la matriz.
 */
#define TILE_DIM 16

__global__ void transpose_shared(const float *A, float *B, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // columna en A
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // fila en A

    // Leer tile de A en shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = A[y * cols + x];
    }

    __syncthreads();

    // Escribir tile transpuesto a B
    // Nota: invertir índices para transposición
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;  // columna en B
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;  // fila en B

    if (x_out < rows && y_out < cols) {
        B[y_out * rows + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 03.3: Transposición de Matriz ===\n\n";

    // ========== PARÁMETROS ==========
    int rows = 1024;
    int cols = 1024;

    if (argc > 1) rows = atoi(argv[1]);
    if (argc > 2) cols = atoi(argv[2]);

    int total = rows * cols;

    std::cout << "Matriz: " << rows << " × " << cols << "\n";
    std::cout << "Total elementos: " << total << "\n\n";

    // ========== ASIGNACIÓN ==========
    size_t size = total * sizeof(float);
    float *h_A = new float[total];
    float *h_B = new float[total];
    float *h_B_ref = new float[total];
    float *d_A, *d_B;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // ========== INICIALIZAR ==========
    // Inicializar A con valores pattern: A[i,j] = i * cols + j
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_A[i * cols + j] = (float)(i * cols + j);
        }
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    std::cout << "Matriz A inicializada con patrón A[i,j]=i*cols+j.\n";

    // ========== CONFIGURACIÓN ==========
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // ========== MÉTODO NAIVE ==========
    std::cout << "\n--- Transposición naive ---\n";

    Timer timer_naive;
    timer_naive.start();

    transpose_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, rows, cols);
    gpuErrchk( cudaPeekAtLastError() );

    timer_naive.stop();
    float time_naive = timer_naive.getGpuTime();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << time_naive << " ms\n";

    // ========== MÉTODO CON SHARED MEMORY ==========
    std::cout << "\n--- Transposición con shared memory ---\n";

    dim3 blocksPerGrid_shm(
        (cols + TILE_DIM - 1) / TILE_DIM,
        (rows + TILE_DIM - 1) / TILE_DIM
    );

    Timer timer_shm;
    timer_shm.start();

    transpose_shared<<<blocksPerGrid_shm, TILE_DIM, TILE_DIM * TILE_DIM * sizeof(float)>>>(
        d_A, d_B, rows, cols);
    gpuErrchk( cudaPeekAtLastError() );

    timer_shm.stop();
    float time_shm = timer_shm.getGpuTime();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << time_shm << " ms\n";

    // ========== VERIFICAR ==========
    // Calcular referencia: B_ref[i][j] = A[j][i]
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_B_ref[j * rows + i] = h_A[i * cols + j];
        }
    }

    bool ok = verifyArrays(h_B, h_B_ref, total);
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== MOSTRAR EJEMPLO ==========
    std::cout << "\nEjemplo de transposición (2×2):\n";
    std::cout << "  A[0][0] = " << h_A[0] << "  A[0][1] = " << h_A[1] << "\n";
    std::cout << "  A[1][0] = " << h_A[cols + 0] << "  A[1][1] = " << h_A[cols + 1] << "\n";
    std::cout << "\n";
    std::cout << "  B[0][0] = " << h_B[0] << "  B[0][1] = " << h_B[rows + 0] << "\n";
    std::cout << "  B[1][0] = " << h_B[1] << "  B[1][1] = " << h_B[rows + 1] << "\n";

    // ========== COMPARACIÓN ==========
    std::cout << "\n=== Comparación ===\n";
    std::cout << "  Naive:           " << time_naive << " ms\n";
    std::cout << "  Shared memory:   " << time_shm << " ms\n";
    if (time_naive > 0) {
        std::cout << "  Speedup:         " << (time_naive / time_shm) << "x\n";
    }

    // ========== CPU REFERENCE ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            h_B_ref[j * rows + i] = h_A[i * cols + j];
        }
    }
    cpuTimer.stop();

    std::cout << "  CPU time:        " << cpuTimer.getCpuTime() << " ms\n";
    std::cout << "  Speedup GPU:     " << (cpuTimer.getCpuTime() / time_shm) << "x\n";

    // ========== LIMPIEZA ==========
    delete[] h_A;
    delete[] h_B;
    delete[] h_B_ref;
    cudaFree(d_A);
    cudaFree(d_B);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: memory_transfers (gestión de memoria)\n";

    return 0;
}
