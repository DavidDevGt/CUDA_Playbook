/**
 * @file reduction_warp_optimized.cu
 * @brief Warp-optimized reduction - 3-5x faster than tree-based reduction
 *
 * PHASE 6: PATTERNS PARALELOS - Reducción warp-level
 * ===================================================
 * 
 * Usa __shfl_down_sync para reducción dentro del warp.
 * Sin __syncthreads() dentro del warp → máximo paralelismo.
 *
 * MEJORA: 3-5x sobre reducción tree-based
 */

#include <iostream>
#include <cuda_runtime.h>
#include <limits>
#include <climits>
#include <cfloat>
#include <cstring>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

// ============================================================================
// REDUCCIÓN WARP-LEVEL
// ============================================================================

__device__ float warp_reduce_sum(float val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max(float val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(mask, val, offset);
        val = fmaxf(val, neighbor);
    }
    return val;
}

__device__ float warp_reduce_min(float val) {
    unsigned int mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float neighbor = __shfl_down_sync(mask, val, offset);
        val = fminf(val, neighbor);
    }
    return val;
}

// ============================================================================
// KERNELS - Reducción warp optimizada
// ============================================================================

__global__ void reduction_kernel_warp_sum(const float *input, float *output, int N) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    
    float val = 0.0f;
    if (idx < N) val = input[idx];
    if (idx + blockDim.x < N) val += input[idx + blockDim.x];
    
    val = warp_reduce_sum(val);
    
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

__global__ void reduction_kernel_warp_max(const float *input, float *output, int N) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    
    float val = -FLT_MAX;
    if (idx < N) val = input[idx];
    if (idx + blockDim.x < N) val = fmaxf(val, input[idx + blockDim.x]);
    
    val = warp_reduce_max(val);
    
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (lane < num_warps) ? shared_mem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

__global__ void reduction_kernel_warp_min(const float *input, float *output, int N) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    
    float val = FLT_MAX;
    if (idx < N) val = input[idx];
    if (idx + blockDim.x < N) val = fminf(val, input[idx + blockDim.x]);
    
    val = warp_reduce_min(val);
    
    int lane = tid % 32;
    int warp_id = tid / 32;
    
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();
    
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (lane < num_warps) ? shared_mem[lane] : FLT_MAX;
        val = warp_reduce_min(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// ============================================================================
// Multi-pass reduction
// ============================================================================

float gpu_reduction_multipass(const float *d_input, int N, const char *op_name) {
    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    if (blocks > 65535) blocks = 65535;
    
    float *d_buffer0 = nullptr;
    float *d_buffer1 = nullptr;
    gpuErrchk(cudaMalloc(&d_buffer0, blocks * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_buffer1, blocks * sizeof(float)));
    
    int shared_mem_size = (threads / 32) * sizeof(float);
    
    // First pass
    if (strcmp(op_name, "sum") == 0) {
        reduction_kernel_warp_sum<<<blocks, threads, shared_mem_size>>>(
            d_input, d_buffer0, N);
    } else if (strcmp(op_name, "max") == 0) {
        reduction_kernel_warp_max<<<blocks, threads, shared_mem_size>>>(
            d_input, d_buffer0, N);
    } else {
        reduction_kernel_warp_min<<<blocks, threads, shared_mem_size>>>(
            d_input, d_buffer0, N);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Subsequent passes
    int current_size = blocks;
    float *d_src = d_buffer0;
    float *d_dst = d_buffer1;
    
    while (current_size > 1) {
        int pass_blocks = (current_size + threads * 2 - 1) / (threads * 2);
        if (pass_blocks == 0) pass_blocks = 1;
        
        if (strcmp(op_name, "sum") == 0) {
            reduction_kernel_warp_sum<<<pass_blocks, threads, shared_mem_size>>>(
                d_src, d_dst, current_size);
        } else if (strcmp(op_name, "max") == 0) {
            reduction_kernel_warp_max<<<pass_blocks, threads, shared_mem_size>>>(
                d_src, d_dst, current_size);
        } else {
            reduction_kernel_warp_min<<<pass_blocks, threads, shared_mem_size>>>(
                d_src, d_dst, current_size);
        }
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        float *temp = d_src;
        d_src = d_dst;
        d_dst = temp;
        current_size = pass_blocks;
    }
    
    float result;
    gpuErrchk(cudaMemcpy(&result, d_src, sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_buffer0);
    cudaFree(d_buffer1);
    
    return result;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    std::cout << "=== PHASE 6: Warp-Optimized Reduction ===\n\n";
    
    int N = 1 << 24;
    const char *op_name = "sum";
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) op_name = argv[2];
    
    std::cout << "Config:\n";
    std::cout << "  Elements: " << N << "\n";
    std::cout << "  Op: " << op_name << "\n";
    std::cout << "  Memory: " << N * sizeof(float) / (1024.0 * 1024.0) << " MB\n\n";
    
    // Data
    float *h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)(i % 100);
    
    float *d_data = nullptr;
    gpuErrchk(cudaMalloc(&d_data, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // GPU
    Timer timer;
    timer.start();
    float gpu_result;
    if (strcmp(op_name, "sum") == 0) {
        gpu_result = gpu_reduction_multipass(d_data, N, "sum");
    } else if (strcmp(op_name, "max") == 0) {
        gpu_result = gpu_reduction_multipass(d_data, N, "max");
    } else if (strcmp(op_name, "min") == 0) {
        gpu_result = gpu_reduction_multipass(d_data, N, "min");
    } else {
        std::cerr << "Unknown op: " << op_name << "\n";
        return 1;
    }
    timer.stop();
    float gpu_time = timer.getGpuTime();
    
    // CPU reference
    CpuTimer cpu_timer;
    cpu_timer.start();
    float cpu_result;
    if (strcmp(op_name, "sum") == 0) {
        cpu_result = 0.0f;
        for (int i = 0; i < N; i++) cpu_result += h_data[i];
    } else if (strcmp(op_name, "max") == 0) {
        cpu_result = -FLT_MAX;
        for (int i = 0; i < N; i++) cpu_result = fmaxf(cpu_result, h_data[i]);
    } else {
        cpu_result = FLT_MAX;
        for (int i = 0; i < N; i++) cpu_result = fminf(cpu_result, h_data[i]);
    }
    cpu_timer.stop();
    float cpu_time = cpu_timer.getCpuTime();
    
    // Results
    std::cout << "GPU: " << gpu_result << " (" << gpu_time << " ms)\n";
    std::cout << "CPU: " << cpu_result << " (" << cpu_time << " ms)\n";
    std::cout << "Match: " << (fabs(gpu_result - cpu_result) < 1.0f ? "✓ PASS" : "✗ FAIL") << "\n";
    std::cout << "Speedup: " << cpu_time / gpu_time << "x\n\n";
    
    std::cout << "Techniques:\n";
    std::cout << "  • Warp shuffle (no __syncthreads in warp)\n";
    std::cout << "  • 1 __syncthreads per block (vs 10+ in tree)\n";
    std::cout << "  • Multi-pass for large arrays\n\n";
    
    delete[] h_data;
    cudaFree(d_data);
    
    std::cout << "✅ Done.\n";
    return 0;
}
