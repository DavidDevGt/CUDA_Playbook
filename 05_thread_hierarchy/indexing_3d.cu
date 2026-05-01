/**
 * @file indexing_3d.cu
 * @brief Indexación 3D - Datos tridimensionales
 *
 * Muestra cómo acceder a arrays 3D (volúmenes) en CUDA.
 *
 * Fórmulas (row-major):
 *   idx = x + y * width + z * width * height
 *
 * Compilación:
 *   nvcc -o indexing_3d indexing_3d.cu
 */

#include <iostream>
#include <cuda_runtime.h>
#include "../common/cuda_utils.h"

/**
 * @brief Kernel que itera sobre un volumen 3D
 *
 * @param data Puntero al volumen
 * @param width  Dimensión X
 * @param height Dimensión Y
 * @param depth  Dimensión Z
 */
__global__ void volume_index_kernel(float *data, int width, int height, int depth) {
    // Coordenadas 3D
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Boundary check
    if (x < width && y < height && z < depth) {
        // Indexación row-major: x + y*width + z*width*height
        int idx = x + y * width + z * width * height;
        data[idx] = (float)(x + y * 100 + z * 10000);  // Patrón identificable
    }
}

int main() {
    std::cout << "=== Ejemplo 05.5: Indexación 3D ===\n\n";

    int width = 4, height = 4, depth = 4;
    int total = width * height * depth;

    std::cout << "Volumen: " << width << " × " << height << " × " << depth << " = " << total << " voxeles\n\n";

    // ========== CONFIGURACIÓN 3D ==========
    // Límites: cada dimensión de bloque ≤ 1024
    dim3 threads(4, 4, 4);    // 64 hilos por bloque
    dim3 blocks(
        (width  + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        (depth  + threads.z - 1) / threads.z
    );

    std::cout << "Blocks: (" << blocks.x << ", " << blocks.y << ", " << blocks.z << ")\n";
    std::cout << "Threads/block: (" << threads.x << ", " << threads.y << ", " << threads.z << ")\n";
    std::cout << "Total hilos: " << blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z << "\n\n";

    // ========== ASIGNAR ==========
    float *h_volume = new float[total];
    float *d_volume;
    cudaMalloc(&d_volume, total * sizeof(float));

    for (int i = 0; i < total; i++) h_volume[i] = -1.0f;
    cudaMemcpy(d_volume, h_volume, total * sizeof(float), cudaMemcpyHostToDevice);

    // ========== LANZAR ==========
    volume_index_kernel<<<blocks, threads>>>(d_volume, width, height, depth);
    cudaDeviceSynchronize();

    // Copiar de vuelta
    cudaMemcpy(h_volume, d_volume, total * sizeof(float), cudaMemcpyDeviceToHost);

    // ========== MOSTRAR VOLUMEN ==========
    std::cout << "Volumen (cada voxel = x + y*100 + z*10000):\n\n";

    for (int z = 0; z < depth; z++) {
        std::cout << "=== Slice Z=" << z << " ===\n";
        std::cout << "     ";
        for (int x = 0; x < width; x++) {
            std::cout << "x" << x;
            for (int pad = 4 - std::to_string(x).length(); pad > 0; pad--) std::cout << " ";
        }
        std::cout << "\n";

        for (int y = 0; y < height; y++) {
            std::cout << "y" << y << "  ";
            for (int x = 0; x < width; x++) {
                int idx = x + y * width + z * width * height;
                std::cout << h_volume[idx];
                for (int pad = 6 - std::to_string((int)h_volume[idx]).length(); pad > 0; pad--) std::cout << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    // ========== VERIFICAR ==========
    bool ok = true;
    for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = x + y * width + z * width * height;
                float expected = (float)(x + y * 100 + z * 10000);
                if (fabs(h_volume[idx] - expected) > 1e-5f) {
                    ok = false;
                    std::cout << "Mismatch en (" << x << "," << y << "," << z << "): "
                              << h_volume[idx] << " vs " << expected << "\n";
                }
            }
        }
    }
    std::cout << "Verificación: " << (ok ? "✅ OK" : "❌ FAIL") << "\n";

    // ========== EXPLICACIÓN ==========
    std::cout << "\n📌 Fórmulas de indexación:\n";
    std::cout << "  idx_1d(x,y)   = y * width + x\n";
    std::cout << "  idx_1d(x,y,z) = x + y*width + z*width*height\n\n";
    std::cout << "  Recuperar coordenadas desde idx:\n";
    std::cout << "    x = idx % width\n";
    std::cout << "    y = (idx / width) % height\n";
    std::cout << "    z = idx / (width * height)\n";

    // ========== LIMPIEZA ==========
    delete[] h_volume;
    cudaFree(d_volume);

    std::cout << "\n✅ Ejemplo completado.\n";
    std::cout << "   Siguiente: 06_shared_memory\n";

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
