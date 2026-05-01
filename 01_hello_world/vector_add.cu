/**
 * @file vector_add.cu
 * @brief Suma de dos vectores en paralelo (Y = A + B)
 *
 * Este es el primer ejemplo útil de computación paralela en GPU.
 * Cada hilo calcula un elemento del resultado:
 *   Y[i] = A[i] + B[i]
 *
 * COMPILACIÓN:
 *   nvcc -o vector_add vector_add.cu
 *
 * EJECUCIÓN:
 *   ./vector_add [tamaño]
 *
 * Ejemplo:
 *   ./vector_add 1000000
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel que suma dos vectores: C = A + B
 *
 * @param A Primer vector (entrada)
 * @param B Segundo vector (entrada)
 * @param C Vector resultado (salida)
 * @param N Número de elementos
 */
__global__ void vector_add_kernel(const float *A, const float *B, float *C, int N) {
    // Calcular índice global único para este hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Condición de frontera: verificar que idx esté dentro del array
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief Versión alternativa usando grid-stride loop
 *
 * Esta versión es más flexible: funciona incluso si N > (blocks * threads)
 * y se adapta automáticamente a cualquier tamaño.
 */
__global__ void vector_add_kernel_grid_stride(const float *A, const float *B, float *C, int N) {
    // Índice inicial para este hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Stride = total de hilos en el grid
    int stride = gridDim.x * blockDim.x;

    // Cada hilo procesa múltiples elementos si es necesario
    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief Función principal
 */
int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 01: Suma de Vectores (Vector Addition) ===\n\n";

    // ========== PARÁMETROS ==========
    int N = 1 << 20;  // 1M elementos por defecto

    // Opcional: tomar tamaño desde línea de comandos
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0) {
            std::cerr << "Error: tamaño debe ser > 0\n";
            return 1;
        }
    }

    std::cout << "Configuración:\n";
    std::cout << "  Elementos por vector (N): " << N << "\n";
    std::cout << "  Memoria por vector: " << N * sizeof(float) / (1024.0 * 1024.0) << " MB\n\n";

    // ========== CONFIGURACIÓN DE HILOS ==========
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Limitar blocksPerGrid si es demasiado grande
    int maxBlocks = 65535;
    if (blocksPerGrid > maxBlocks) {
        blocksPerGrid = maxBlocks;
        std::cout << "⚠️  Ajustando bloques a " << blocksPerGrid << " (máximo permitido)\n";
    }

    std::cout << "Configuración de ejecución:\n";
    std::cout << "  Threads per block: " << threadsPerBlock << "\n";
    std::cout << "  Blocks per grid:  " << blocksPerGrid << "\n";
    std::cout << "  Total threads:    " << blocksPerGrid * threadsPerBlock << "\n";
    std::cout << "  (Grid-stride: " << blocksPerGrid * threadsPerBlock << ")\n\n";

    // ========== ASIGNACIÓN DE MEMORIA ==========
    size_t size = N * sizeof(float);

    // Host
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];
    float *h_C_ref = new float[N];  // Para verificar

    std::cout << "Memoria host (CPU) asignada.\n";

    // Device
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    gpuErrchk( cudaMalloc(&d_A, size) );
    gpuErrchk( cudaMalloc(&d_B, size) );
    gpuErrchk( cudaMalloc(&d_C, size) );
    std::cout << "Memoria device (GPU) asignada.\n";

    // ========== INICIALIZAR DATOS ==========
    // Inicializar vectores con valores conocidos
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;      // Vector A = [1, 1, 1, ...]
        h_B[i] = 2.0f;      // Vector B = [2, 2, 2, ...]
        // Resultado esperado: C = [3, 3, 3, ...]
    }
    std::cout << "Vectores inicializados: A=1.0, B=2.0\n";

    // ========== COPIAR A GPU ==========
    gpuErrchk( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );
    std::cout << "Datos copiados: Host → Device\n";

    // ========== EJECUTAR KERNEL ==========
    Timer timer;
    timer.start();

    // Lanzar kernel
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    gpuErrchk( cudaPeekAtLastError() );

    timer.stop();
    float gpuTime = timer.getGpuTime();

    std::cout << "\nKernel ejecutado en " << gpuTime << " ms\n";

    // ========== COPIAR RESULTADOS ==========
    gpuErrchk( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );
    std::cout << "Resultados copiados: Device → Host\n";

    // ========== VERIFICAR ==========
    // Calcular referencia en CPU
    for (int i = 0; i < N; i++) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }

    // Comparar
    bool ok = verifyArrays(h_C, h_C_ref, N);
    if (ok) {
        std::cout << "\n✅ Verificación exitosa: GPU == CPU\n";
    } else {
        std::cout << "\n❌ Error: GPU != CPU\n";
    }

    // Mostrar algunos valores
    std::cout << "\nMuestra de resultados:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  [" << i << "] " << h_A[i] << " + " << h_B[i]
                  << " = " << h_C[i] << " (esperado: " << h_C_ref[i] << ")\n";
    }

    // ========== ESTADÍSTICAS ==========
    float throughput = (N * sizeof(float) * 3) / (gpuTime * 1e6);  // GB/s
    std::cout << "\nEstadísticas:\n";
    std::cout << "  Tiempo kernel: " << gpuTime << " ms\n";
    std::cout << "  Throughput:    " << throughput << " GB/s\n";
    std::cout << "  Elements/sec:  " << (N / (gpuTime / 1000.0f)) << " elements/s\n";

    // ========== COMPARACIÓN CPU ==========
    CpuTimer cpuTimer;
    cpuTimer.start();
    for (int i = 0; i < N; i++) {
        h_C_ref[i] = h_A[i] + h_B[i];
    }
    cpuTimer.stop();
    float cpuTime = cpuTimer.getCpuTime();

    std::cout << "\nComparación CPU:\n";
    std::cout << "  Tiempo CPU: " << cpuTime << " ms\n";
    std::cout << "  Aceleración (speedup): " << (cpuTime / gpuTime) << "x\n";

    // ========== LIMPIEZA ==========
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente paso: modifica el kernel para hacer Y = α·A + β·B\n";

    return 0;
}
