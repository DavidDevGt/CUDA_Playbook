/**
 * @file dynamic_parallelism.cu
 * @brief Paralelismo Dinámico - Kernels que lanzan kernels
 *
 * Dynamic Parallel (DP) permite que un kernel en ejecución
 * lance otros kernels (child kernels) sin volver al host.
 *
 * Beneficios:
 *   - Adaptatividad: lanzar solo lo necesario
 *   - Recursividad paralela
 *   - División y conquista en GPU
 *
 * Requisitos:
 *   - GPU con Compute Capability >= 3.5 (Kepler)
 *   - Compilar con -rdc=true y -lcudadevrt
 *
 * Compilación:
 *   nvcc -rdc=true -lcudadevrt -o dynamic_parallelism dynamic_parallelism.cu
 *
 * Ejecución:
 *   ./dynamic_parallelism [DEPTH]
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel hijo simple
 */
__global__ void child_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

/**
 * @brief Kernel padre que lanza kernel hijo dinámicamente
 */
__global__ void parent_kernel(float *data, int N, int depth) {
    if (depth <= 0) return;

    // Kernel padre trabaja en su porción
    int elements_per_block = (N + gridDim.x - 1) / gridDim.x;
    int start = blockIdx.x * elements_per_block;
    int end = (start + elements_per_block < N) ? start + elements_per_block : N;

    // Procesar datos
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        data[i] = data[i] * 0.5f;
    }
    __syncthreads();

    // Lanzar kernel hijo con menos datos (solo un hilo lo lanza)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Configurar child grid
        dim3 child_blocks(2);
        dim3 child_threads(128);

        // Lanzar kernel hijo (dynamic parallelism)
        child_kernel<<<child_blocks, child_threads>>>(data, N);
        // Nota: cudaDeviceSynchronize() no está disponible siempre en DP
        // En este ejemplo ilustrativo omitimos la sincronización device-side
    }
    __syncthreads();
}

/**
 * @brief Kernel ejemplo simple de recursión (Fibonacci ilustrativo)
 */
__global__ void fib_kernel(int *result, int n) {
    if (n <= 1) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *result = n;
        }
        return;
    }

    // Este ejemplo es ilustrativo. DP real requiere más cuidado
    // con jerarquía de bloques y sincronización.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Dynamic parallelism: compute fib(%d)\n", n);
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 11.3: Paralelismo Dinámico ===\n\n";

    // Verificar soporte
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    bool hasDP = (prop.major >= 3);
    std::cout << "  Soporta dynamic parallelism: " << (hasDP ? "✅ Sí (CC >= 3.0)" : "❌ No") << "\n";

    if (!hasDP) {
        std::cout << "\n❌ Esta GPU NO soporta dynamic parallelism.\n";
        std::cout << "   Requiere CC >= 3.5 (Kepler)\n";
        return 0;
    }

    int depth = (argc > 1) ? atoi(argv[1]) : 2;

    // ========== DATOS ==========
    const int N = 1 << 20;
    float *h_data = new float[N];
    float *d_data;

    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // ========== LANZAR KERNEL PADRE ==========
    std::cout << "\nLanzando kernel padre (depth=" << depth << ")\n";

    dim3 parent_blocks(1);
    dim3 parent_threads(128);

    Timer timer;
    timer.start();

    parent_kernel<<<parent_blocks, parent_threads>>>(d_data, N, depth);
    gpuErrchk( cudaPeekAtLastError() );

    // Sincronización host para timing
    cudaDeviceSynchronize();

    timer.stop();
    std::cout << "  Tiempo: " << timer.getGpuTime() << " ms\n";

    // ========== VERIFICAR ==========
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < 10; i++) {
        // Esperamos que data[i] = i * 0.5 * 2 + 1 = i + 1
        float expected = (float)i + 1.0f;
        if (fabs(h_data[i] - expected) > 0.01f) {
            ok = false;
            break;
        }
    }
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    std::cout << "\n📌 Dynamic Parallelism:\n";
    std::cout << "  • Kernels pueden lanzar kernels\n";
    std::cout << "  • Adaptatividad en tiempo de ejecución\n";
    std::cout << "  • Útil para división y conquista\n";
    std::cout << "  • Requiere -rdc=true y -lcudadevrt\n";
    std::cout << "  • Sincronización device-side compleja\n\n";

    cudaFree(d_data);
    delete[] h_data;

    std::cout << "✅ Ejemplo completado.\n";
    return 0;
}
