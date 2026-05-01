/**
 * @file atomic_add.cu
 * @brief Operaciones atómicas - Suma atómica
 *
 * atomicAdd() garantiza que múltiples hilos puedan actualizar
 * la misma variable global sin condiciones de carrera.
 *
 * Caso de uso: acumulación de valores desde múltiples hilos.
 *
 * Compilación:
 *   nvcc -o atomic_add atomic_add.cu
 *
 * Ejecución:
 *   ./atomic_add [N]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel que suma valores atómicamente
 *
 * Cada hilo genera un valor aleatorio y lo suma atómicamente
 * a un contador global.
 */
__global__ void atomic_add_kernel(unsigned int *global_counter, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        // Cada hilo suma 1 (o un valor) atómicamente
        unsigned int old = atomicAdd(global_counter, 1);
        // old contiene el valor anterior
    }
}

/**
 * @brief Kernel que suma valores variables atómicamente
 */
__global__ void atomic_add_values_kernel(float *global_sum, const float *values, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(global_sum, values[idx]);
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 08.1: Suma Atómica (atomicAdd) ===\n\n";

    int N = 1 << 20;  // 1M hilos
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Hilos: " << N << "\n";
    std::cout << "Cada hilo suma 1 a un contador global.\n";
    std::cout << "Resultado esperado: " << N << "\n\n";

    // ========== CONFIGURACIÓN ==========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // ========== EJEMPLO 1: CONTADOR SIMPLE ==========
    std::cout << "--- Ejemplo 1: Contador atómico ---\n";

    unsigned int *d_counter;
    unsigned int h_counter = 0;

    cudaMalloc(&d_counter, sizeof(unsigned int));
    cudaMemcpy(d_counter, &h_counter, sizeof(unsigned int), cudaMemcpyHostToDevice);

    Timer timer;
    timer.start();

    atomic_add_kernel<<<blocks, threads>>>(d_counter, N);
    gpuErrchk( cudaPeekAtLastError() );

    timer.stop();
    cudaMemcpy(&h_counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "  Tiempo: " << timer.getGpuTime() << " ms\n";
    std::cout << "  Resultado: " << h_counter << "\n";
    std::cout << "  Esperado:  " << N << "\n";
    std::cout << "  Correcto:  " << (h_counter == (unsigned int)N ? "✅ Sí" : "❌ No") << "\n";

    cudaFree(d_counter);

    // ========== EJEMPLO 2: SUMA DE VALORES ==========
    std::cout << "\n--- Ejemplo 2: Suma de valores variables ---\n";

    float *h_vals = new float[N];
    float *d_vals, *d_sum;
    float h_sum;

    // Inicializar con valores aleatorios
    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_vals[i] = (float)(rand() % 100);
        expected_sum += h_vals[i];
    }

    cudaMalloc(&d_vals, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemcpy(d_vals, h_vals, N * sizeof(float), cudaMemcpyHostToDevice);

    float zero = 0.0f;
    cudaMemcpy(d_sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    timer.start();
    atomic_add_values_kernel<<<blocks, threads>>>(d_sum, d_vals, N);
    gpuErrchk( cudaPeekAtLastError() );
    timer.stop();

    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "  Suma esperada: " << expected_sum << "\n";
    std::cout << "  Suma GPU:      " << h_sum << "\n";
    std::cout << "  Error relativo: " << fabs(h_sum - expected_sum) / expected_sum * 100.0f << "%\n";

    delete[] h_vals;
    cudaFree(d_vals);
    cudaFree(d_sum);

    // ========== CONTENCIÓN ==========
    std::cout << "\n📌 Consideraciones sobre atomic operations:\n";
    std::cout << "  • Contención: múltiples hilos accesan misma dirección → serialización\n";
    std::cout << "  • Performance: atomicAdd de floats requiere CC 2.0+\n";
    std::cout << "  • Alternativas: reducción en shared memory (lección 09)\n";
    std::cout << "  • Uso: cuando no hay forma de evitar escrituras concurrentes\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\n  Tu GPU soporta atomicAdd en float? ";
    if (prop.major >= 2) {
        std::cout << "✅ Sí (CC >= 2.0)\n";
    } else {
        std::cout << "❌ No (CC < 2.0)\n";
    }

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
