/**
 * @file occupancy.cu
 * @brief Cálculo de Occupancy (utilización de S Ms)
 *
 * Occupancy = (threads activos por SM) / (máximo threads por SM).
 *
 * Factores que afectan occupancy:
 *   - Registers por hilo
 *   - Shared memory por bloque
 *   - Threads por bloque
 *
 * Compilación:
 *   nvcc -o occupancy occupancy.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include "../common/cuda_utils.h"

/**
 * @brief Calcula occupancy teórico para una configuración
 */
void calculateOccupancy(const cudaDeviceProp &prop, int threads_per_block, int shared_mem_per_block) {
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
    int max_shared_mem_per_block = prop.sharedMemPerBlock;
    int max_regs_per_block = prop.regsPerBlock;

    // Cálculo de occupancy limitado por threads
    int blocks_per_sm_by_threads = max_threads_per_sm / threads_per_block;

    // Cálculo limitado por shared memory
    int blocks_per_sm_by_shmem = max_shared_mem_per_block / shared_mem_per_block;
    if (shared_mem_per_block == 0) blocks_per_sm_by_shmem = max_blocks_per_sm;

    // Cálculo limitado por registros (asumiendo regs_per_thread = ?)
    // No calculamos por registros por simplicidad

    int actual_blocks_per_sm = min(blocks_per_sm_by_threads, blocks_per_sm_by_shmem);
    actual_blocks_per_sm = min(actual_blocks_per_sm, max_blocks_per_sm);

    int active_threads = actual_blocks_per_sm * threads_per_block;
    float occupancy = (float)active_threads / max_threads_per_sm * 100.0f;

    std::cout << "TPB=" << threads_per_block;
    std::cout << ", Smem/block=" << shared_mem_per_block / 1024.0f << "KB";
    std::cout << " → Blocks/SM=" << actual_blocks_per_sm;
    std::cout << ", Active threads/SM=" << active_threads;
    std::cout << ", Occupancy=" << occupancy << "%\n";
}

int main() {
    std::cout << "=== Ejemplo 10.1: Cálculo de Occupancy ===\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "  Max threads/SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "  Max blocks/SM:  " << prop.maxBlocksPerMultiProcessor << "\n";
    std::cout << "  Shared mem/block: " << prop.sharedMemPerBlock / 1024.0f << " KB\n";
    std::cout << "  Warp size: " << prop.warpSize << "\n\n";

    std::cout << "=== Simulación de diferentes configuraciones ===\n\n";

    // Probar diferentes TPB y shared memory
    std::cout << "--- Variando Threads Per Block (sin shared memory) ---\n";
    for (int tpb : {32, 64, 128, 256, 512, 1024}) {
        if (tpb > prop.maxThreadsPerBlock) continue;
        int max_blocks = prop.maxThreadsPerMultiProcessor / tpb;
        max_blocks = min(max_blocks, prop.maxBlocksPerMultiProcessor);
        int active_threads = max_blocks * tpb;
        float occ = (float)active_threads / prop.maxThreadsPerMultiProcessor * 100.0f;
        std::cout << "  TPB=" << tpb;
        std::cout << " → Max blocks/SM=" << max_blocks;
        std::cout << ", Occupancy=" << occ << "%\n";
    }

    std::cout << "\n--- Efecto de shared memory (TPB=256) ---\n";
    int tpb = 256;
    for (int smem_kb = 0; smem_kb <= 48; smem_kb += 8) {
        int smem_bytes = smem_kb * 1024;
        int blocks_by_smem = (smem_bytes > 0) ? (prop.sharedMemPerBlock / smem_bytes) : prop.maxBlocksPerMultiProcessor;
        if (smem_bytes == 0) blocks_by_smem = prop.maxBlocksPerMultiProcessor;

        int blocks_by_threads = prop.maxThreadsPerMultiProcessor / tpb;
        int actual_blocks = min(blocks_by_smem, min(blocks_by_threads, prop.maxBlocksPerMultiProcessor));
        int active_threads = actual_blocks * tpb;
        float occ = (float)active_threads / prop.maxThreadsPerMultiProcessor * 100.0f;

        std::cout << "  Shared/block=" << smem_kb << "KB";
        std::cout << " → Blocks/SM=" << actual_blocks;
        std::cout << ", Occupancy=" << occ << "%\n";
    }

    // ========== USAR CUDA OCCUPANCY API ==========
    std::cout << "\n--- Usando CUDA Occupancy API ---\n";

    int actual_blocks;
    // Calcula blocks/SM para un bloque de 256 threads
    int tpb_estimate = 256;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&actual_blocks, nullptr, tpb_estimate, 0);
    std::cout << "  Blocks/SM (para TPB=" << tpb_estimate << "): " << actual_blocks << "\n";
    std::cout << "  Active threads/SM: " << actual_blocks * tpb_estimate << "\n";

    // ========== RECOMENDACIONES ==========
    std::cout << "\n📌 Guía rápida de occupancy:\n";
    std::cout << "  • 100% occupancy: ideal pero no siempre óptimo (puede causar pressure en recursos)\n";
    std::cout << "  • 50-75% occupancy: a veces mejor para kernels latency-bound\n";
    std::cout << "  • Más shared memory/registros → menor occupancy\n";
    std::cout << "  • Usa Nsight Compute para análisis detallado\n";
    std::cout << "\nComando útil:\n";
    std::cout << "  nvcc -Xptxas -v arch=sm_XX kernel.cu  # Ver register/shared usage\n";

    std::cout << "\n✅ Ejemplo completado.\n";

    return 0;
}
