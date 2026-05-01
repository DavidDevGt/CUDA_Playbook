/**
 * @file hello.cu
 * @brief Tu primer kernel CUDA - "Hola Mundo" en la GPU
 *
 * Este ejemplo demuestra los conceptos más básicos:
 * 1. Declaración de un kernel con __global__
 * 2. Lanzamiento de kernel con <<<blocks, threads>>>
 * 3. Acceso a variables de hilo (threadIdx, blockIdx)
 * 4. Transferencia de datos CPU ↔ GPU
 *
 * COMPILACIÓN:
 *   nvcc -o hello hello.cu
 *
 * EJECUCIÓN:
 *   ./hello
 *
 * SALIDA ESPERADA:
 *   Lanzando kernel con 256 hilos...
 *   Kernel ejecutado. Resultados:
 *   h_data[0] = 42.0
 *   h_data[1] = 43.0
 *   ...
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

/**
 * @brief Kernel que modifica datos en paralelo
 *
 * Cada hilo calcula su índice único y modifica un elemento del array.
 *
 * @param data Puntero a datos en memoria de dispositivo (GPU)
 * @param N Número de elementos en el array
 */
__global__ void hello_kernel(float *data, int N) {
    // Calcular índice global del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Verificar que no nos salgamos del array
    if (idx < N) {
        // Modificar el dato: cada hilo pone su idx + 42
        data[idx] = (float)(idx + 42);
    }
}

/**
 * @brief Función principal
 */
int main() {
    std::cout << "=== Ejemplo 01: Hola Mundo CUDA ===\n\n";

    // ========== CONFIGURACIÓN ==========
    const int N = 256;                    // Número total de elementos
    const int threadsPerBlock = 256;      // Hilos por bloque
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Configuración:\n";
    std::cout << "  Elementos totales: " << N << "\n";
    std::cout << "  Hilos por bloque: " << threadsPerBlock << "\n";
    std::cout << "  Bloques en grid:  " << blocksPerGrid << "\n";
    std::cout << "  Total hilos:      " << blocksPerGrid * threadsPerBlock << "\n\n";

    // ========== MEMORIA HOST (CPU) ==========
    // Asignar memoria en el host
    float *h_data = new float[N];

    // Inicializar datos en el host
    for (int i = 0; i < N; i++) {
        h_data[i] = 0.0f;
    }
    std::cout << "Memoria host (CPU) asignada e inicializada.\n";

    // ========== MEMORIA DEVICE (GPU) ==========
    float *d_data = nullptr;
    gpuErrchk( cudaMalloc(&d_data, N * sizeof(float)) );
    std::cout << "Memoria device (GPU) asignada.\n";

    // ========== COPIA HOST → DEVICE ==========
    gpuErrchk( cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice) );
    std::cout << "Datos copiados: Host → Device\n";

    // ========== LANZAMIENTO DEL KERNEL ==========
    std::cout << "\nLanzando kernel...\n";

    // Configurar eventos para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // EJECUCIÓN DEL KERNEL
    // Sintaxis: kernel<<<num_bloques, hilos_por_bloque>>>(argumentos...);
    hello_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Verificar errores de lanzamiento
    gpuErrchk( cudaPeekAtLastError() );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, start, stop);

    std::cout << "✅ Kernel completado en " << kernelTime << " ms\n";

    // ========== COPIA DEVICE → HOST ==========
    gpuErrchk( cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost) );
    std::cout << "Resultados copiados: Device → Host\n\n";

    // ========== VERIFICACIÓN ==========
    std::cout << "Resultados (primeros 10 elementos):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "  h_data[" << i << "] = " << h_data[i];
        if (h_data[i] == i + 42.0f) {
            std::cout << " ✓\n";
        } else {
            std::cout << " ✗ (esperado " << i + 42.0f << ")\n";
        }
    }

    // ========== LIMPIEZA ==========
    cudaFree(d_data);
    delete[] h_data;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "\n✅ Ejemplo completado exitosamente.\n";
    std::cout << "   Continúa con vector_add.cu para ver operaciones útiles.\n";

    return 0;
}
