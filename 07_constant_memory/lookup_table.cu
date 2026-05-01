/**
 * @file lookup_table.cu
 * @brief Tabla de consulta en memoria constante
 *
 * Ejemplo práctico: evaluación de polynomial
 * usando una tabla de valores precalculados en memoria constante.
 *
 * COMPI_VALLACIÓN:
 *   nvcc -O2 -arch=sm_89 -o lookup_table lookup_table.cu
 *
 * EJECUCIÓN:
 *   ./lookup_table
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

// Tamaño de la tabla de consulta
#define TABLE_SIZE 1024
#define PI_VAL 3.14159265358979323846f

// Declarar tabla en memoria constante
__constant__ float c_sin_table[TABLE_SIZE];
__constant__ float c_cos_table[TABLE_SIZE];

__global__ void sin_kernel_directo(const float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = sinf(x[idx]);
    }
}

/**
 * @brief Kernel que interpola valores usando tabla de consulta
 *
 * Evalúa sin(x) aproximando con tabla + interpolación lineal
 */
__global__ void sin_lookup_kernel(const float *x, float *y, int N, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Mapear x a índice en tabla
        float normalized = x[idx] * scale;  // 0..TABLE_SIZE-1
        int i0 = (int)floorf(normalized);
        int i1 = i0 + 1;
        if (i1 >= TABLE_SIZE) i1 = TABLE_SIZE - 1;

        float frac = normalized - i0;

        // Interpolación lineal entre tabla[i0] y tabla[i1]
        y[idx] = c_sin_table[i0] * (1.0f - frac) + c_sin_table[i1] * frac;
    }
}

/**
 * @brief Kernel que usa tabla directa (sin interpolación)
 */
__global__ void direct_lookup_kernel(const float *indices, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int table_idx = (int)indices[idx] % TABLE_SIZE;
        output[idx] = c_cos_table[table_idx];
    }
}

int main() {
    std::cout << "=== Ejemplo 07.2: Tabla de Consulta (Lookup Table) ===\n\n";

    const int N = 1 << 20;
    const float PI_VAL_VAL = 3.14159265358979f;

    // ========== CONSTRUIR TABLAS ==========
    std::cout << "Construyendo tablas de seno y coseno...\n";

    float h_sin[TABLE_SIZE];
    float h_cos[TABLE_SIZE];
    for (int i = 0; i < TABLE_SIZE; i++) {
        float angle = (float)i * PI_VAL_VAL / (float)TABLE_SIZE;
        h_sin[i] = sinf(angle);
        h_cos[i] = cosf(angle);
    }

    // Copiar tablas a constant memory
    gpuErrchk( cudaMemcpyToSymbol(c_sin_table, h_sin, TABLE_SIZE * sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(c_cos_table, h_cos, TABLE_SIZE * sizeof(float)) );
    std::cout << "  Tablas cargadas en memoria constante.\n\n";

    // ========== CREAR DATOS ==========
    float *h_x = new float[N];
    float *h_y_sin = new float[N];
    float *h_y_cos = new float[N];
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i * PI_VAL / (float)N;  // 0..PI_VAL
    }

    float *d_x, *d_y_sin, *d_y_cos;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y_sin, N * sizeof(float));
    cudaMalloc(&d_y_cos, N * sizeof(float));
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // ========== EJEMPLO 1: SIN CON TABLA ==========
    std::cout << "--- Benchmark: Cálculo directo (sin tabla) ---\n";

    Timer timer1;
    timer1.start();

    sin_kernel_directo<<<blocks, threads>>>(d_x, d_y_sin, N);
    cudaDeviceSynchronize();
    timer1.stop();

    float time_direct = timer1.getGpuTime();
    std::cout << "  Tiempo directo (sin tabla): " << time_direct << " ms\n";

    // ========== EJEMPLO 2: CON TABLA DE CONSULTA ==========
    std::cout << "\n--- Benchmark: Con lookup table (const memory) ---\n";

    timer1.start();
    sin_lookup_kernel<<<blocks, threads>>>(d_x, d_y_sin, N, (float)TABLE_SIZE / (2.0f * PI_VAL_VAL));
    cudaDeviceSynchronize();
    timer1.stop();

    float time_table = timer1.getGpuTime();
    std::cout << "  Tiempo con tabla:    " << time_table << " ms\n";
    std::cout << "  Speedup:             " << (time_direct / time_table) << "x\n";

    // ========== VERIFICACIÓN ==========
    cudaMemcpy(h_y_sin, d_y_sin, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calcular referencia en CPU
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        float expected = sinf(h_x[i]);
        float error = fabs(h_y_sin[i] - expected);
        if (error > max_error) max_error = error;
    }

    std::cout << "  Error máximo (tabla vs directo): " << max_error << "\n";

    // ========== EJEMPLO 3: COS CON TABLA DIRECTA ==========
    std::cout << "\n--- Ejemplo 3: Acceso directo a tabla coseno ---\n";

    // índices enteros para acceso directo
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(i % TABLE_SIZE);
    }
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    direct_lookup_kernel<<<blocks, threads>>>(d_x, d_y_cos, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y_cos, d_y_cos, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int idx = (int)h_x[i] % TABLE_SIZE;
        if (fabs(h_y_cos[i] - h_cos[idx]) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "  Verificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== ANÁLISIS ==========
    std::cout << "\n📌 Ventajas de constant memory:\n";
    std::cout << "  • Caché constante (8KB por SM) → acceso rápido\n";
    std::cout << "  • Broadcast eficiente: todos los hilos leen misma dirección simultáneamente\n";
    std::cout << "  • Ideal para tablas small (<8KB) que se consultan frecuentemente\n\n";
    std::cout << "📌 Limitaciones:\n";
    std::cout << "  • Tamaño limitado (~64KB total, 8KB por banco)\n";
    std::cout << "  • Solo lectura desde kernels\n";
    std::cout << "  • Debe conocerse en compilación (tamaño estático) o usar cudaMemcpyToSymbol\n";
    std::cout << "  • Performance óptimo cuando todos los warps leen las mismas posiciones\n";

    // ========== LIMPI_VALEZA ==========
    delete[] h_x;
    delete[] h_y_sin;
    delete[] h_y_cos;
    cudaFree(d_x);
    cudaFree(d_y_sin);
    cudaFree(d_y_cos);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: 08_atomic_operations\n";

    return 0;
}
