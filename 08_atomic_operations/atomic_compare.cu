/**
 * @file atomic_compare.cu
 * @brief Operaciones atómicas - Compare and Swap (CAS)
 *
 * atomicCAS() (Compare And Swap) permite implementar operaciones
 * atómicas más complejas como fetch-and-add,锁, etc.
 *
 * Caso de uso: actualización condicional thread-safe.
 *
 * Compilación:
 *   nvcc -o atomic_compare atomic_compare.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel que usa atomicCAS para un contador lock-free
 *
 * Implementa un contador que solo se incrementa si el valor
 * anterior satisface una condición.
 */
__global__ void atomic_cas_kernel(int *counter, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int old = *counter;
        // Intentar actualizar hasta que el CAS sea exitoso
        while (atomicCAS(counter, old, old + 1) != old) {
            old = *counter;  // Releer y reintentar
        }
    }
}

/**
 * @brief Usando atomicCAS para implementar fetch-and-add
 */
__device__ int atomic_fetch_and_add(int *ptr, int value) {
    int old = *ptr;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(ptr, assumed, assumed + value);
    } while (assumed != old);
    return assumed;
}

/**
 * @brief Kernel con fetch-and-add personalizado
 */
__global__ void custom_atomic_kernel(int *counter, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomic_fetch_and_add(counter, 1);
    }
}

/**
 * @brief Implementar un spinlock simple con atomicCAS
 */
__global__ void spinlock_kernel(int *lock, int *counter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Adquirir lock
    while (atomicCAS(lock, 0, 1) != 0) {
        // Spin until lock acquired
    }

    // Sección crítica
    atomicAdd(counter, 1);

    // Liberar lock
    atomicExch(lock, 0);
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 08.2: Compare and Swap (CAS) ===\n\n";

    int N = 10000;
    if (argc > 1) N = atoi(argv[1]);

    std::cout << "Hilos: " << N << "\n\n";

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // ========== EJEMPLO 1: atomicCAS básico ==========
    std::cout << "--- Ejemplo 1: Contador con atomicCAS ---\n";

    int *d_counter1;
    int h_counter1 = 0;
    cudaMalloc(&d_counter1, sizeof(int));
    cudaMemcpy(d_counter1, &h_counter1, sizeof(int), cudaMemcpyHostToDevice);

    Timer timer1;
    timer1.start();
    atomic_cas_kernel<<<blocks, threads>>>(d_counter1, N);
    gpuErrchk( cudaDeviceSynchronize() );
    timer1.stop();

    cudaMemcpy(&h_counter1, d_counter1, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  atomicCAS contador: " << h_counter1 << " (esperado: " << N << ")\n";
    std::cout << "  Tiempo: " << timer1.getGpuTime() << " ms\n";
    cudaFree(d_counter1);

    // ========== EJEMPLO 2: fetch-and-add personalizado ==========
    std::cout << "\n--- Ejemplo 2: Fetch-and-add implementado con atomicCAS ---\n";

    int *d_counter2;
    cudaMalloc(&d_counter2, sizeof(int));
    h_counter1 = 0;
    cudaMemcpy(d_counter2, &h_counter1, sizeof(int), cudaMemcpyHostToDevice);

    timer1.start();
    custom_atomic_kernel<<<blocks, threads>>>(d_counter2, N);
    gpuErrchk( cudaDeviceSynchronize() );
    timer1.stop();

    cudaMemcpy(&h_counter1, d_counter2, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  fetch-and-add: " << h_counter1 << " (esperado: " << N << ")\n";
    std::cout << "  Tiempo: " << timer1.getGpuTime() << " ms\n";
    cudaFree(d_counter2);

    // ========== EJEMPLO 3: Spinlock ==========
    std::cout << "\n--- Ejemplo 3: Spinlock (protección de sección crítica) ---\n";

    int *d_lock, *d_counter3;
    cudaMalloc(&d_lock, sizeof(int));
    cudaMalloc(&d_counter3, sizeof(int));

    h_counter1 = 0;
    cudaMemcpy(d_lock, &h_counter1, sizeof(int), cudaMemcpyHostToDevice);  // lock = 0 (unlocked)
    cudaMemcpy(d_counter3, &h_counter1, sizeof(int), cudaMemcpyHostToDevice);

    timer1.start();
    spinlock_kernel<<<blocks, threads>>>(d_lock, d_counter3);
    gpuErrchk( cudaDeviceSynchronize() );
    timer1.stop();

    cudaMemcpy(&h_counter1, d_counter3, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  Contador protegido: " << h_counter1 << " (esperado: " << N << ")\n";
    std::cout << "  Tiempo: " << timer1.getGpuTime() << " ms\n";
    std::cout << "  (Spinlock es más lento que atomicAdd simple)\n";

    cudaFree(d_lock);
    cudaFree(d_counter3);

    // ========== RESUMEN ==========
    std::cout << "\n📌 atomicCAS(ptr, expected, desired):\n";
    std::cout << "  Lee *ptr, si == expected → escribe desired y retorna old\n";
    std::cout << "  Si no → retorna el valor actual, hay que reintentar\n\n";
    std::cout << "📌 Implementa:\n";
    std::cout << "  • Lock-free data structures\n";
    std::cout << "  • Spinlocks\n";
    std::cout << "  • Fetch-and-add, compare-and-swap\n";
    std::cout << "  • Referencias counting, etc.\n";
    std::cout << "\n⚠️  Cuidado: spinlocks pueden causar deadlock o livelock\n";

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
