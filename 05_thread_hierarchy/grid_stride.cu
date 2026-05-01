/**
 * @file grid_stride.cu
 * @brief Grid-Stride Loops - Patrón flexible de paralelización
 *
 * Grid-stride loops permiten que cada hilo procese múltiples elementos,
 * haciendo el kernel flexible a cualquier tamaño de problema sin
 * preocuparse por el número exacto de hilos lanzados.
 *
 * Ventajas:
 *   - Funciona para cualquier N (no múltiplo de TPB*blocks)
 *   - Mejor utilización de SMs si N << (blocks × threads)
 *   - Fácil de escalar
 *
 * Compilación:
 *   nvcc -o grid_stride grid_stride.cu
 *
 * Ejecución:
 *   ./grid_stride [SIZE]
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Grid-stride loop: cada hilo procesa múltiples elementos
 *
 * stride = total_threads = gridDim.x * blockDim.x
 *
 * @param data Puntero a datos
 * @param N Total de elementos
 */
__global__ void grid_stride_kernel(float *data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Cada hilo procesa elementos con stride
    for (int i = tid; i < N; i += stride) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

/**
 * @brief Kernel sin grid-stride (método tradicional)
 * Solo funciona si N ≤ (blocks × threads)
 */
__global__ void traditional_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 05.2: Grid-Stride Loops ===\n\n";

    int N = 1 << 24;  // 16M elementos
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Elementos: " << N << "\n\n";

    // ========== CONFIGURACIÓN ==========
    // Podemos lanzar MENOS hilos que elementos (grid-stride lo compensa)
    int threads_per_block = 256;

    // Opción A: bloques mínimos (menos hilos totales)
    int blocks_grid_stride = 64;  // Solo 64×256 = 16384 hilos

    // Opción B: bloques máximos (muchos hilos)
    int blocks_traditional = (N + threads_per_block - 1) / threads_per_block;
    if (blocks_traditional > 65535) blocks_traditional = 65535;

    std::cout << "Configuración:\n";
    std::cout << "  Grid-stride: " << blocks_grid_stride << " bloques × " << threads_per_block
              << " = " << blocks_grid_stride * threads_per_block << " hilos\n";
    std::cout << "  Traditional: " << blocks_traditional << " bloques × " << threads_per_block
              << " = " << blocks_traditional * threads_per_block << " hilos\n";
    std::cout << "  Ratio hilos/elementos (grid-stride): "
              << float(blocks_grid_stride * threads_per_block) / N << "\n\n";

    // ========== ASIGNAR ==========
    size_t bytes = N * sizeof(float);
    float *h_data = new float[N];
    float *d_data1, *d_data2;

    cudaMalloc(&d_data1, bytes);
    cudaMalloc(&d_data2, bytes);

    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    cudaMemcpy(d_data1, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data, bytes, cudaMemcpyHostToDevice);

    // ========== PRUEBA 1: TRADITIONAL ==========
    std::cout << "--- Método tradicional (sin grid-stride) ---\n";

    Timer timer1;
    timer1.start();
    traditional_kernel<<<blocks_traditional, threads_per_block>>>(d_data1, N);
    cudaDeviceSynchronize();
    timer1.stop();

    float time1 = timer1.getGpuTime();
    std::cout << "  Tiempo: " << time1 << " ms\n";

    // ========== PRUEBA 2: GRID-STRIDE ==========
    std::cout << "\n--- Grid-Stride Loop ---\n";

    Timer timer2;
    timer2.start();
    grid_stride_kernel<<<blocks_grid_stride, threads_per_block>>>(d_data2, N);
    cudaDeviceSynchronize();
    timer2.stop();

    float time2 = timer2.getGpuTime();
    std::cout << "  Tiempo: " << time2 << " ms\n";

    // ========== RESULTADOS ==========
    std::cout << "\n=== Comparación ===\n";
    std::cout << "  Traditional: " << time1 << " ms, " << (N/(time1/1000.0f))/1e9 << " G elements/s\n";
    std::cout << "  Grid-stride: " << time2 << " ms, " << (N/(time2/1000.0f))/1e9 << " G elements/s\n";

    if (time2 < time1) {
        std::cout << "  → Grid-stride más rápido en este caso\n";
    } else if (time1 < time2) {
        std::cout << "  → Traditional más rápido (probablemente N es grande)\n";
    } else {
        std::cout << "  → Similar performance\n";
    }

    // ========== VERIFICAR ==========
    cudaMemcpy(h_data, d_data2, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_data[i] - (1.0f * 2.0f + 1.0f)) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "  Verificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== EJEMPLO ADICIONAL: STRIDE LÓGICO ==========
    std::cout << "\n=== Ventajas grid-stride ===\n";
    std::cout << "  1. Flexible: funciona para cualquier N\n";
    std::cout << "  2. Eficiente: cada hilo trabaja continuamente\n";
    std::cout << "  3. Escalable: adapta a cualquier configuración de GPU\n";
    std::cout << "  4. Menor overhead (menos blocks activation)\n";

    // ========== LIMPIEZA ==========
    delete[] h_data;
    cudaFree(d_data1);
    cudaFree(d_data2);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
