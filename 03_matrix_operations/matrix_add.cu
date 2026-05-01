/**
 * @file matrix_add.cu
 * @brief Suma de matrices: C = A + B
 *
 * Ejemplo introductorio a operaciones 2D en CUDA.
 * Cada hilo procesa un elemento (i, j) de la matriz.
 *
 * Compilación:
 *   nvcc -o matrix_add matrix_add.cu
 *
 * Ejecución:
 *   ./matrix_add [M] [N]
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel para suma de matrices
 *
 * @param A Matriz A (filas * columnas)
 * @param B Matriz B
 * @param C Matriz resultado
 * @param rows Número de filas
 * @param cols Número de columnas
 */
__global__ void matrix_add_kernel(const float *A, const float *B, float *C, int rows, int cols) {
    // Calcular coordenadas 2D del hilo
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // fila
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // columna

    // Verificar límites
    if (row < rows && col < cols) {
        // Índice linearizado en matriz row-major
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief Kernel alternativo usando indexación 1D (más simple)
 *
 * Aquí tratamos la matriz como un vector y cada hilo procesa un elemento.
 */
__global__ void matrix_add_kernel_1d(const float *A, const float *B, float *C, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 03.1: Suma de Matrices ===\n\n";

    // ========== PARÁMETROS ==========
    int rows = 1024;
    int cols = 1024;

    if (argc > 1) rows = atoi(argv[1]);
    if (argc > 2) cols = atoi(argv[2]);

    int total = rows * cols;

    std::cout << "Matriz: " << rows << " × " << cols << " = " << total << " elementos\n";
    std::cout << "Memoria por matriz: " << total * sizeof(float) / (1024.0*1024.0) << " MB\n\n";

    // ========== CONFIGURACIÓN DE HILOS ==========
    // Para matrices 2D, usamos bloques 2D
    dim3 threadsPerBlock(16, 16);  // 256 hilos por bloque
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    std::cout << "Configuración 2D:\n";
    std::cout << "  Threads per block:  (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")\n";
    std::cout << "  Blocks per grid:   (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")\n";
    std::cout << "  Total threads:     " << blocksPerGrid.x * blocksPerGrid.y * threadsPerBlock.x * threadsPerBlock.y << "\n\n";

    // ========== ASIGNACIÓN ==========
    size_t size = total * sizeof(float);
    float *h_A = new float[total];
    float *h_B = new float[total];
    float *h_C = new float[total];
    float *h_C_ref = new float[total];
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // ========== INICIALIZAR ==========
    // A[i] = sqrt(i), B[i] = 1/sqrt(i) → C[i] debería ser 2*sqrt(i) ?
    // Mejor: A[i]=i, B[i]=i*2 → C[i]=3*i
    for (int i = 0; i < total; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    std::cout << "Matrices A y B inicializadas.\n";

    // ========== EJECUTAR (MÉTODO 2D) ==========
    std::cout << "\n--- Lanzando kernel 2D ---\n";

    Timer timer;
    timer.start();

    matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
    gpuErrchk( cudaPeekAtLastError() );

    timer.stop();
    float time2d = timer.getGpuTime();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Tiempo (2D indexing): " << time2d << " ms\n";

    // ========== EJECUTAR (MÉTODO 1D) ==========
    // Reiniciar resultado
    cudaMemcpy(d_C, h_C_ref, size, cudaMemcpyHostToDevice);  // dummy

    threadsPerBlock.x = 256;
    blocksPerGrid.x = (total + threadsPerBlock.x - 1) / threadsPerBlock.x;
    if (blocksPerGrid.x > 65535) blocksPerGrid.x = 65535;

    timer.start();

    matrix_add_kernel_1d<<<blocksPerGrid.x, 256>>>(d_A, d_B, d_C, total);
    gpuErrchk( cudaPeekAtLastError() );

    timer.stop();
    float time1d = timer.getGpuTime();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Tiempo (1D indexing): " << time1d << " ms\n";

    // ========== VERIFICAR ==========
    for (int i = 0; i < total; i++) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    bool ok = verifyArrays(h_C, h_C_ref, total);
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== MUESTRA ==========
    std::cout << "\nMuestra de resultados (primeros 5 elementos):\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  C[" << i << "] = " << h_C[i]
                  << " (esperado: " << h_C_ref[i] << ")\n";
    }

    // ========== ESTADÍSTICAS ==========
    std::cout << "\nEstadísticas:\n";
    std::cout << "  Throughput: " << (total * sizeof(float) * 3) / (time2d * 1e6) << " GB/s\n";

    // ========== CPU REFERENCE ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    for (int i = 0; i < total; i++) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }
    cpuTimer.stop();

    std::cout << "  CPU time: " << cpuTimer.getCpuTime() << " ms\n";
    std::cout << "  Speedup:  " << (cpuTimer.getCpuTime() / time2d) << "x\n";

    // ========== LIMPIEZA ==========
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: matrix_multiply.cu (multiplicación de matrices)\n";

    return 0;
}
