/**
 * @file block_dimensions.cu
 * @brief Cálculo de dimensiones de bloque optimizadas
 *
 * Muestra cómo elegir el número de hilos por bloque (TPB)
 * y bloques por grid de forma óptima.
 *
 * Reglas generales:
 *   - TPB debe ser múltiplo de warpSize (32)
 *   - TPB típicos: 128, 256, 512 (máximo 1024)
 *   - Blocks per grid: cubrir todo el problema, limitado a 65535 por dimensión
 *
 * Compilación:
 *   nvcc -o block_dimensions block_dimensions.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

__global__ void dummy_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

int main(int argc, char **argv) {
    std::cout << "=== Ejemplo 05.1: Dimensiones de Bloque ===\n\n";

    int N = 1 << 20;  // 1M elementos

    std::cout << "Problema: " << N << " elementos\n\n";

    std::cout << "=== Estrategias de configuración ===\n\n";

    // Estrategia 1: TPB fijo, bloques ajustados
    int tpb_fixed = 256;
    int blocks_fixed = (N + tpb_fixed - 1) / tpb_fixed;

    std::cout << "1. TPB fijo = " << tpb_fixed << "\n";
    std::cout << "   Blocks necesarios: " << blocks_fixed << "\n";
    std::cout << "   Total threads: " << blocks_fixed * tpb_fixed << "\n";
    std::cout << "   Sobran threads: " << (blocks_fixed * tpb_fixed) - N << " (waste)\n\n";

    // Estrategia 2: Mínimo número de bloques
    int min_blocks = (N + 1023) / 1024;  // máximo TPB = 1024
    std::cout << "2. TPB máximo (1024)\n";
    std::cout << "   Blocks mínimos: " << min_blocks << "\n";
    std::cout << "   Total threads: " << min_blocks * 1024 << "\n";
    std::cout << "   Waste: " << (min_blocks * 1024) - N << "\n\n";

    // Estrategia 3: Optimizar occupancy (veremos en 10_performance)
    // Por ahora: usar 256 o 512 TPB

    // Calcular occupancy teórico (requiere conocer SM count, pero podemos estimar)
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;

    std::cout << "=== Capacidades de tu GPU ===\n";
    std::cout << "  SMs: " << sm_count << "\n";
    std::cout << "  Max threads por SM: " << max_threads_per_sm << "\n";
    std::cout << "  Max blocks por SM:  " << prop.maxBlocksPerMultiProcessor << "\n";

    // Calcular occupancy máximo teórico para diferentes TPB
    std::cout << "\n=== Occupancy teórico (max active warps/SM) ===\n";
    std::cout << "TPB  |  Warps/block | Max blocks/SM | Threads/SM | Occupancy %\n";
    std::cout << "-----|--------------|---------------|------------|------------\n";

    for (int tpb : {32, 64, 128, 256, 512, 1024}) {
        int warps_per_block = (tpb + prop.warpSize - 1) / prop.warpSize;
        int max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;
        // Limitar por threads y blocks
        int blocks_by_threads = max_threads_per_sm / tpb;
        int actual_blocks = (max_blocks_per_sm < blocks_by_threads) ? max_blocks_per_sm : blocks_by_threads;
        int total_threads = actual_blocks * tpb;
        float occupancy = (float)total_threads / max_threads_per_sm * 100.0f;

        std::cout << " " << tpb;
        std::cout << "   |     " << warps_per_block;
        std::cout << "      |      " << actual_blocks;
        std::cout << "       |   " << total_threads;
        std::cout << "    |  " << occupancy << "%\n";
    }

    std::cout << "\n📌 Recomendaciones:\n";
    std::cout << "  • TPB = 256 es buen punto de partida (balance occupancy/overhead)\n";
    std::cout << "  • Evita TPB que no sean múltiplos de 32 (warpSize)\n";
    std::cout << "  • Para kernels con mucha register/shared memory, TPB puede necesitar ajuste\n";
    std::cout << "  • Usa la calculadora de occupancy de Nsight Compute para optimizar\n";

    // ========== EJEMPLO PRÁCTICO ==========
    std::cout << "\n=== Ejemplo práctico ===\n";

    float *h_data = new float[N];
    float *d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Probar diferentes TPB
    int test_tpbs[] = {64, 128, 256, 512};
    Timer timer;

    std::cout << "TPB  | Tiempo (ms) | Throughput (Melements/s)\n";
    std::cout << "-----|-------------|--------------------------\n";

    for (int tpb : test_tpbs) {
        int blocks = (N + tpb - 1) / tpb;
        if (blocks > 65535) blocks = 65535;

        timer.start();
        dummy_kernel<<<blocks, tpb>>>(d_data, N);
        cudaDeviceSynchronize();
        timer.stop();

        float time = timer.getGpuTime();
        float throughput = N / (time / 1000.0f) / 1e6;  // Melements/s

         std::cout << " " << tpb;
         for (int pad = 4 - std::to_string(tpb).length(); pad > 0; pad--) std::cout << " ";
         std::cout << " | " << time;
         for (int pad = 9 - std::to_string((int)time).length(); pad > 0; pad--) std::cout << " ";
        std::cout << " | " << throughput << "\n";
    }

    // ========== LIMPIEZA ==========
    delete[] h_data;
    cudaFree(d_data);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: grid_stride.cu (patrones flexibles)\n";

    return 0;
}

// Helper para convertir int a string
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
