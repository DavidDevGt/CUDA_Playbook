/**
 * @file indexing_2d.cu
 * @brief Indexación 2D - Matrices y datos bidimensionales
 *
 * Muestra cómo transformar coordenadas 2D (fila, col) en
 * índice lineal, y viceversa.
 *
 * Fórmulas:
 *   idx = row * width + col           (row-major)
 *   row = idx / width
 *   col = idx % width
 *
 * Compilación:
 *   nvcc -o indexing_2d indexing_2d.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

__global__ void matrix_index_kernel(float *matrix, int rows, int cols) {
    // Coordenadas 2D del hilo
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < rows && col < cols) {
        // Convertir (row, col) a índice lineal (row-major)
        int idx = row * cols + col;

        // Escribir patrón para verificar
        matrix[idx] = (float)(row * cols + col);  // Patrón: 0, 1, 2, ...
    }
}

int main() {
    std::cout << "=== Ejemplo 05.4: Indexación 2D ===\n\n";

    int rows = 8, cols = 8;
    int total = rows * cols;

    std::cout << "Matriz " << rows << "×" << cols << " (" << total << " elementos)\n\n";

    // ========== CONFIGURACIÓN 2D ==========
    dim3 threads(16, 16);
    dim3 blocks(
        (cols + threads.x - 1) / threads.x,
        (rows + threads.y - 1) / threads.y
    );

    std::cout << "Blocks: (" << blocks.x << ", " << blocks.y << ")\n";
    std::cout << "Threads/block: (" << threads.x << ", " << threads.y << ")\n\n";

    // ========== ASIGNAR ==========
    float *h_matrix = new float[total];
    float *d_matrix;
    cudaMalloc(&d_matrix, total * sizeof(float));

    // Inicializar con -1
    for (int i = 0; i < total; i++) h_matrix[i] = -1.0f;
    cudaMemcpy(d_matrix, h_matrix, total * sizeof(float), cudaMemcpyHostToDevice);

    // ========== LANZAR KERNEL ==========
    matrix_index_kernel<<<blocks, threads>>>(d_matrix, rows, cols);
    cudaDeviceSynchronize();

    // Copiar de vuelta
    cudaMemcpy(h_matrix, d_matrix, total * sizeof(float), cudaMemcpyDeviceToHost);

    // ========== MOSTRAR MATRIZ ==========
    std::cout << "Resultado (patrón row-major):\n";
    std::cout << "     ";
    for (int c = 0; c < cols; c++) {
        std::cout << "Col " << c;
        for (int pad = 6 - std::to_string(c).length(); pad > 1; pad--) std::cout << " ";
    }
    std::cout << "\n";

    for (int r = 0; r < rows; r++) {
        std::cout << "Row " << r << " ";
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            std::cout << h_matrix[idx];
            for (int pad = 8 - std::to_string((int)h_matrix[idx]).length(); pad > 0; pad--) std::cout << " ";
        }
        std::cout << "\n";
    }

    // ========== VERIFICAR ==========
    bool ok = true;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            if (fabs(h_matrix[idx] - (float)(r * cols + c)) > 1e-5f) {
                ok = false;
                break;
            }
        }
    }
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== EJEMPLO: TRANSPONER ÍNDICES ==========
    std::cout << "\n=== Cálculo de índices ===\n";
    int r = 5, c = 3;
    int idx_rc = r * cols + c;
    std::cout << "  Coordenada (" << r << "," << c << ") → idx = " << idx_rc << "\n";

    // Recuperar coordenadas
    int row_back = idx_rc / cols;
    int col_back = idx_rc % cols;
    std::cout << "  idx " << idx_rc << " → (row=" << row_back << ", col=" << col_back << ")\n";

    // ========== LIMPIEZA ==========
    delete[] h_matrix;
    cudaFree(d_matrix);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}

#include <string>
std::string to_string(int x) {
    if (x == 0) return "0";
    std::string s;
    while (x > 0) {
        s = char('0' + x % 10) + s;
        x /= 10;
    }
    return s;
}
