/**
 * @file bank_conflicts.cu
 * @brief Conflictos de bancos en shared memory
 *
 * La memoria shared está dividida en 32 bancos (uno por hilo del warp).
 * Si múltiples hilos acceden al mismo banco simultáneamente → serialización.
 *
 * Ejemplos:
 *   • Sin conflictos: acceso secuencial (stride = 1)
 *   • Conflictos 2-way: stride = 32 (todos accesan mismo banco)
 *   • Conflictos broadcast: todos accesan misma dirección (OK en CC ≥ 2.0)
 *
 * Compilación:
 *   nvcc -o bank_conflicts bank_conflicts.cu
 *
 * Ejecución:
 *   ./bank_conflicts
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

#define ARRAY_SIZE 1024
#define BLOCK_SIZE 256

/**
 * @brief Kernel que muestra acceso secuencial (sin conflictos)
 *
 * Stride = 1 → cada hilo accesa banco diferente
 */
__global__ void no_conflict_kernel(float *data) {
    __shared__ float s_data[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Acceso secuencial: banco = (dirección / 4) % 32
    // Como cada hilo escribe s_data[tid], cada uno va a banco distinto
    s_data[tid] = (float)tid;

    __syncthreads();

    // Leer en patrón secuencial
    float val = s_data[tid];
    data[tid] = val;
}

/**
 * @brief Kernel que genera conflictos de 2 vías
 *
 * Stride = 32 → todos los hilos pares accesan banco X, impares banco Y
 */
__global__ void conflict_2way_kernel(float *data) {
    __shared__ float s_data[BLOCK_SIZE * 2];  // 512 elementos

    int tid = threadIdx.x;
    int idx = tid * 2;  // Stride 2 en el bloque

    // idx: 0, 2, 4, 6, ... → todos accesan bancos pares
    s_data[idx] = (float)idx;

    __syncthreads();

    data[tid] = s_data[idx];
}

/**
 * @brief Kernel que genera broadcast (todos leen misma dirección)
 *
 * Todos los hilos del warp leen s_data[0] → puede ser broadcast (sin conflicto en CC ≥ 2.0)
 */
__global__ void broadcast_kernel(float *data) {
    __shared__ float s_data[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Todos los hilos del primer warp escriben en s_data[0]
    if (tid == 0) {
        s_data[0] = 42.0f;
    }
    __syncthreads();

    // Todos los hilos leen s_data[0] (broadcast)
    data[tid] = s_data[0];
}

/**
 * @brief Kernel que muestra patrón estríado real
 *
 * Acceso strided: stride = 32 → conflicto severo
 */
__global__ void stride_conflict_kernel(float *data, int stride) {
    __shared__ float s_data[BLOCK_SIZE * 32];  // grande

    int tid = threadIdx.x;
    int idx = tid * stride;  // stride configurable

    s_data[idx] = (float)idx;
    __syncthreads();

    data[tid] = s_data[idx];
}

int main() {
    std::cout << "=== Ejemplo 06.4: Conflictos de Bancos ===\n\n";

    float *h_out;
    float *d_out;
    Timer timer;
    const int trials = 1000;

    // ========== EJEMPLO 1: SIN CONFLICTOS ==========
    std::cout << "--- Ejemplo 1: Acceso secuencial (stride=1) ---\n";
    h_out = new float[BLOCK_SIZE];
    cudaMalloc(&d_out, BLOCK_SIZE * sizeof(float));

    no_conflict_kernel<<<1, BLOCK_SIZE>>>(d_out);
    cudaDeviceSynchronize();

    timer.start();
    for (int t = 0; t < trials; t++) {
        no_conflict_kernel<<<1, BLOCK_SIZE>>>(d_out);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float time_no_conflict = timer.getGpuTime() / trials;
    std::cout << "  Tiempo por launch: " << time_no_conflict << " ms\n";
    std::cout << "  Acceso: s_data[tid] (stride 1) → sin conflictos\n";

    delete[] h_out;
    cudaFree(d_out);

    // ========== EJEMPLO 2: CONFLICTO 2-WAY ==========
    std::cout << "\n--- Ejemplo 2: Acceso con stride 2 ---\n";
    h_out = new float[BLOCK_SIZE];
    cudaMalloc(&d_out, BLOCK_SIZE * sizeof(float));

    timer.start();
    for (int t = 0; t < trials; t++) {
        conflict_2way_kernel<<<1, BLOCK_SIZE>>>(d_out);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float time_conflict = timer.getGpuTime() / trials;
    std::cout << "  Tiempo por launch: " << time_conflict << " ms\n";
    std::cout << "  Acceso: s_data[tid*2] (stride 2) → conflictos de bancos\n";

    delete[] h_out;
    cudaFree(d_out);

    // ========== EJEMPLO 3: BROADCAST ==========
    std::cout << "\n--- Ejemplo 3: Broadcast (todos leen misma dirección) ---\n";
    h_out = new float[BLOCK_SIZE];
    cudaMalloc(&d_out, BLOCK_SIZE * sizeof(float));

    timer.start();
    for (int t = 0; t < trials; t++) {
        broadcast_kernel<<<1, BLOCK_SIZE>>>(d_out);
    }
    cudaDeviceSynchronize();
    timer.stop();

    float time_broadcast = timer.getGpuTime() / trials;
    std::cout << "  Tiempo por launch: " << time_broadcast << " ms\n";
    std::cout << "  Acceso: todos leen s_data[0] → puede ser broadcast (rápido en CC≥2.0)\n";

    delete[] h_out;
    cudaFree(d_out);

    // ========== EJEMPLO 4: STRIDE GRANDE ==========
    std::cout << "\n--- Ejemplo 4: Diferentes strides ---\n";
    for (int stride : {1, 2, 4, 8, 16, 32}) {
        h_out = new float[BLOCK_SIZE];
        cudaMalloc(&d_out, BLOCK_SIZE * sizeof(float));

        timer.start();
        for (int t = 0; t < trials; t++) {
            stride_conflict_kernel<<<1, BLOCK_SIZE>>>(d_out, stride);
        }
        cudaDeviceSynchronize();
        timer.stop();

        float time_stride = timer.getGpuTime() / trials;

        // Calcular banco: (address / 4) % 32
        // stride de 32 significa que todos accesan el mismo banco
        std::cout << "  Stride " << stride;
        for (int pad = 5 - std::to_string(stride).length(); pad > 0; pad--) std::cout << " ";
        std::cout << ": " << time_stride << " ms";

        if (stride == 1) {
            std::cout << " (sin conflictos, baseline)";
        } else if (stride % 32 == 0) {
            std::cout << " (máximo conflicto: todos al mismo banco)";
        } else if (stride % 16 == 0) {
            std::cout << " (conflicto 2-way)";
        }
        std::cout << "\n";

        delete[] h_out;
        cudaFree(d_out);
    }

    // ========== RESUMEN ==========
    std::cout << "\n📌 Reglas de conflictos:\n";
    std::cout << "  • Bank = (address / 4) % 32 (para floats, 4 bytes cada uno)\n";
    std::cout << "  • Si 2+ hilos accesan el mismo banco en la misma instrucción → serialización\n";
    std::cout << "  • Solución: padding (añadir padding a structs/arrays)\n";
    std::cout << "  • Broadcast (todos leen misma dirección) → sin conflicto en CC ≥ 2.0\n\n";
    std::cout << "📌 Cómo evitar:\n";
    std::cout << "  struct { float x; float y; float z; float w; }  // ya es 16 bytes = 4 bancos\n";
    std::cout << "  struct { float data[33]; }  // pad de 1 para evitar stride=32\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\nTu GPU: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")\n";
    if (prop.major >= 2) {
        std::cout << "→ Broadcast de misma dirección es rápido (sin conflicto)\n";
    }

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
