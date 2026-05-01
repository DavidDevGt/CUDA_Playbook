/**
 * @file zero_copy.cu
 * @brief Zero-copy memory - accessing host memory from GPU
 *
 * Zero-copy allows GPU to directly access page-locked host memory
 * without explicit cudaMemcpy. Useful for:
 *   - Infrequent GPU access to large datasets
 *   - Initial prototyping before optimizing data movement
 *   - Memory-bound kernels where copy overhead dominates
 *
 * NOTE: On modern GPUs (Ampere+), unified memory with prefetching
 * often performs better than traditional zero-copy.
 *
 * COMPILATION:
 *   nvcc -O2 -arch=sm_89 -o zero_copy zero_copy.cu
 *
 * EXECUTION:
 *   ./zero_copy [size_mb]
 *
 * Example:
 *   ./zero_copy 16
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

/**
 * @brief Kernel that doubles array elements
 */
__global__ void double_kernel(float *data, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main(int argc, char **argv) {
    size_t size_mb = 16;
    if (argc > 1) size_mb = atoi(argv[1]);
    
    size_t N = size_mb * 1024 * 1024 / sizeof(float);
    size_t bytes = N * sizeof(float);
    
    std::cout << "=== Zero-Copy Memory Demo ===\n\n";
    std::cout << "Data size: " << size_mb << " MB (" << N << " floats)\n\n";
    
    // ========== METHOD 1: cudaHostAlloc (Zero-Copy Pinned) ==========
    std::cout << "--- Method 1: cudaHostAlloc (Zero-Copy Pinned) ---\n";
    
    float *h_zero_copy = nullptr;
    gpuErrchk( cudaHostAlloc(&h_zero_copy, bytes, cudaHostAllocMapped) );
    
    // Initialize
    for (size_t i = 0; i < N; i++) h_zero_copy[i] = (float)i;
    
    // Get device pointer for the mapped memory
    float *d_zero_copy = nullptr;
    gpuErrchk( cudaHostGetDevicePointer(&d_zero_copy, h_zero_copy, 0) );
    
    dim3 threads(256);
    dim3 blocks((unsigned int)((N + threads.x - 1) / threads.x));
    
    Timer timer;
    timer.start();
    double_kernel<<<blocks, threads>>>(d_zero_copy, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    timer.stop();
    
    float time = timer.getGpuTime();
    std::cout << "  Kernel executed in " << time << " ms\n";
    
    // Verify (already in host memory)
    bool ok = true;
    for (size_t i = 0; i < N; i++) {
        if (fabsf(h_zero_copy[i] - (float)(i * 2)) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "  Verification: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";
    
    gpuErrchk( cudaFreeHost(h_zero_copy) );
    
    // ========== METHOD 2: cudaMallocManaged (Unified Memory) ==========
    std::cout << "\n--- Method 2: cudaMallocManaged (Unified Memory) ---\n";
    
    float *um_ptr = nullptr;
    gpuErrchk( cudaMallocManaged(&um_ptr, bytes) );
    
    for (size_t i = 0; i < N; i++) um_ptr[i] = (float)i;
    
    // Prefetch to GPU (optional but recommended)
    gpuErrchk( cudaMemPrefetchAsync(um_ptr, bytes, 0) );
    
    timer.start();
    double_kernel<<<blocks, threads>>>(um_ptr, (int)N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    timer.stop();
    
    time = timer.getGpuTime();
    std::cout << "  Kernel executed in " << time << " ms\n";
    
    // Verify
    ok = true;
    for (size_t i = 0; i < N; i++) {
        if (fabsf(um_ptr[i] - (float)(i * 2)) > 1e-5f) {
            ok = false;
            break;
        }
    }
    std::cout << "  Verification: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";
    
    gpuErrchk( cudaFree(um_ptr) );
    
    // ========== COMPARACIÓN CON COPIA TRADICIONAL ==========
    std::cout << "\n=== Comparación con cudaMemcpy ===\n";
    std::cout << "  Zero-Copy:\n";
    std::cout << "    • Pros: Sin copias explícitas, código más simple\n";
    std::cout << "    • Cons: Acceso más lento (PCIe), menor throughput\n";
    std::cout << "    • Usar: Para depuración, prototipado rápido, datos grandes\n";
    std::cout << "           que no caben en GPU\n";
    std::cout << "\n  cudaMalloc + cudaMemcpy:\n";
    std::cout << "    • Pros: Máximo performance, transfers asíncronas\n";
    std::cout << "    • Cons: Código más verboso, overhead de copies\n";
    std::cout << "    • Usar: Production code, performance crítica\n";
    
    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: thread_hierarchy (jerarquía de hilos)\n";
    
    return 0;
}
