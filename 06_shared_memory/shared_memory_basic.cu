/**
 * @file shared_memory_basic.cu
 * @brief Uso básico de memoria compartida (shared memory)
 *
 * Shared memory es memoria on-chip muy rápida (≈ 100× global memory).
 * Es compartida por todos los hilos de un bloque.
 *
 * Características:
 *   - Declarada con __shared__ en el kernel
 *   - Vida: solo durante ejecución del bloque
 *   - Tamaño limitado: 48KB por SM (varía por GPU)
 *   - Se debe sincronizar con __syncthreads()
 *
 * Compilación:
 *   nvcc -o shared_memory_basic shared_memory_basic.cu
 *
 * Ejecución:
 *   ./shared_memory_basic
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

/**
 * @brief Ejemplo 1: Copiar datos de global a shared, operar, copiar de vuelta
 */
__global__ void shared_copy_kernel(const float *input, float *output, int N) {
    // Declarar shared memory (tamaño estático)
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. Cargar desde global memory a shared memory
    if (idx < N) {
        s_data[tid] = input[idx];
    } else {
        s_data[tid] = 0.0f;
    }

    // 2. Sincronizar: asegurar que todos los hilos cargaron sus datos
    __syncthreads();

    // 3. Operar en shared memory (ej: multiplicar por 2)
    s_data[tid] = s_data[tid] * 2.0f;

    // 4. Sincronizar de nuevo si hay dependencias
    __syncthreads();

    // 5. Escribir de vuelta a global memory
    if (idx < N) {
        output[idx] = s_data[tid];
    }
}

/**
 * @brief Ejemplo 2: Shared memory dinámico
 * Tamaño especificado en el lanzamiento del kernel
 */
__global__ void shared_dynamic_kernel(const float *input, float *output, int N) {
    // Declarar shared memory dinámico (extern)
    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. Cargar desde global memory a shared memory con bounds checking
    if (idx < N) {
        s_data[tid] = input[idx];
    } else {
        s_data[tid] = 0.0f;  // Padding seguro para out-of-bounds
    }

    // 2. Sincronizar: asegurar que TODOS los hilos del bloque cargaron
    __syncthreads();

    // 3. Operar en shared memory y escribir resultado
    if (idx < N) {
        output[idx] = s_data[tid] + 1.0f;
    }

    // Nota: se podría añadir otro __syncthreads() si más hilos reutilizaran s_data
}

/**
 * @brief Ejemplo 3: Dos arrays en shared memory
 */
__global__ void dual_shared_kernel(const float *A, const float *B, float *C, int N) {
    __shared__ float s_A[256];  // Tamaño debe ser al menos blockDim.x
    __shared__ float s_B[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 1. Cargar datos desde global memory
    if (idx < N) {
        s_A[tid] = A[idx];
        s_B[tid] = B[idx];
    } else {
        s_A[tid] = 0.0f;
        s_B[tid] = 0.0f;
    }
    // 2. Sincronizar: asegurar que TODOS los hilos del bloque cargaron
    __syncthreads();

    // 3. Operar en shared memory (todos los datos ya están disponibles)
    if (idx < N) {
        C[idx] = s_A[tid] + s_B[tid];
    }
}

int main() {
    std::cout << "=== Ejemplo 06.1: Memoria Compartida Básica ===\n\n";

    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // ========== DATOS ==========
    float *h_in = new float[N];
    float *h_out = new float[N];
    float *d_in, *d_out;

    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // ========== EJEMPLO 1: Shared estático ==========
    std::cout << "--- Ejemplo 1: Shared memory estático ---\n";

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    shared_copy_kernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verificar: output = input * 2
    bool ok1 = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_in[i] * 2.0f;
        if (fabs(h_out[i] - expected) > 1e-5f) {
            ok1 = false;
            break;
        }
    }
    std::cout << "  Verificación primeras 10: " << (ok1 ? "✅ OK" : "❌ FAIL") << "\n";
    std::cout << "  (output[i] = input[i] * 2)\n";

    // ========== EJEMPLO 2: Shared dinámico ==========
    std::cout << "\n--- Ejemplo 2: Shared memory dinámico ---\n";

    shared_dynamic_kernel<<<blocks, threads, threads.x * sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verificar: output = input + 1
    bool ok2 = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_in[i] + 1.0f;
        if (fabs(h_out[i] - expected) > 1e-5f) {
            ok2 = false;
        }
    }
    std::cout << "  Verificación: " << (ok2 ? "✅ OK" : "❌ FAIL") << "\n";
    std::cout << "  (output[i] = input[i] + 1)\n";

    // ========== EJEMPLO 3: Dos arrays ==========
    std::cout << "\n--- Ejemplo 3: Dos arrays en shared memory ---\n";

    float *d_B;
    cudaMalloc(&d_B, bytes);
    float *h_B = new float[N];
    for (int i = 0; i < N; i++) h_B[i] = 10.0f;
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    float *d_C;
    cudaMalloc(&d_C, bytes);

    dual_shared_kernel<<<blocks, threads>>>(d_in, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_C, bytes, cudaMemcpyDeviceToHost);

    bool ok3 = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_in[i] + h_B[i];
        if (fabs(h_out[i] - expected) > 1e-5f) {
            ok3 = false;
        }
    }
    std::cout << "  Verificación: " << (ok3 ? "✅ OK" : "❌ FAIL") << "\n";
    std::cout << "  (C[i] = A[i] + B[i])\n";

    // ========== RESUMEN ==========
    std::cout << "\n📌 Puntos clave:\n";
    std::cout << "  • __shared__ declara variables en shared memory\n";
    std::cout << "  __syncthreads() sincroniza hilos del bloque\n";
    std::cout << "  • Tamaño estático: conocido en compilación\n";
    std::cout << "  • Tamaño dinámico: especificado en lanzamiento (tercer arg de <<<>>>)\n";
    std::cout << "  • Acceso mucho más rápido que global memory\n";
    std::cout << "  • Usar para: tiling, reducción, buffers intermedios\n";

    // ========== LIMPIEZA ==========
    delete[] h_in;
    delete[] h_out;
    delete[] h_B;
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
