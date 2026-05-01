/**
 * @file check_cuda.cu
 * @brief Verifica la instalación de CUDA y detecta dispositivos
 *
 * Este es el primer programa que debes ejecutar para verificar que:
 * 1. Tienes un driver NVIDIA instalado
 * 2. CUDA Toolkit está correctamente configurado
 * 3. Tu GPU es compatible con CUDA
 *
 * Compilación:
 *   nvcc -o check_cuda check_cuda.cu
 *
 * Uso:
 *   ./check_cuda
 */

#include <iostream>
#include <cuda_runtime.h>

// Kernel de prueba simple
__global__ void test_kernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = idx;
    }
}

int main() {
    std::cout << "=== Verificación de CUDA ===\n\n";

    // 1. Contar dispositivos CUDA disponibles
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "ERROR: No se pudo inicializar CUDA.\n";
        std::cerr << "       " << cudaGetErrorString(err) << "\n";
        std::cerr << "\nPosibles causas:\n";
        std::cerr << "  - Driver NVIDIA no instalado o desactualizado\n";
        std::cerr << "  - CUDA Toolkit no instalado\n";
        std::cerr << "  - Conflictos con versiones\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "❌ No se detectaron GPUs compatibles con CUDA.\n";
        std::cout << "\nRevisa que:\n";
        std::cout << "  1. Tu GPU sea NVIDIA (AMD/Intel no son compatibles)\n";
        std::cout << "  2. El driver NVIDIA esté instalado: nvidia-smi\n";
        std::cout << "  3. La GPU tenga soporte CUDA (Compute Capability >= 3.5)\n";
        return 1;
    }

    std::cout << "✅ Dispositivos CUDA detectados: " << deviceCount << "\n\n";

    // 2. Información del dispositivo 0 (por defecto)
    std::cout << "--- Dispositivo 0 ---\n";
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "  Nombre: " << prop.name << "\n";
    std::cout << "  Capacidad de cómputo: " << prop.major << "." << prop.minor;
    std::cout << "  (CC " << prop.major << prop.minor << ")\n";

    // Clasificación por generación
    if (prop.major == 1) {
        std::cout << "     → Generación Tesla (muy antigua)\n";
    } else if (prop.major == 2) {
        std::cout << "     → Generación Fermi (antigua)\n";
    } else if (prop.major == 3) {
        std::cout << "     → Generación Kepler (obsoleta)\n";
    } else if (prop.major == 5) {
        std::cout << "     → Generación Maxwell\n";
    } else if (prop.major == 6) {
        std::cout << "     → Generación Pascal\n";
    } else if (prop.major == 7) {
        if (prop.minor == 0) {
            std::cout << "     → Generación Volta\n";
        } else {
            std::cout << "     → Generación Turing\n";
        }
    } else if (prop.major == 8) {
        if (prop.minor == 0) {
            std::cout << "     → Generación Ampere (GA100)\n";
        } else if (prop.minor == 6) {
            std::cout << "     → Generación Ampere (GA106/104/102)\n";
        } else {
            std::cout << "     → Generación Ampere\n";
        }
    } else if (prop.major == 9) {
        std::cout << "     → Generación Ada Lovelace / Hopper\n";
    } else {
        std::cout << "     → Arquitectura nueva/desconocida\n";
    }

    std::cout << "  Memoria global: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB\n";
    std::cout << "  Memoria compartida por bloque: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "  Registros por bloque: " << prop.regsPerBlock << "\n";
    std::cout << "  Warp size: " << prop.warpSize << " hilos\n";
    std::cout << "  Max hilos por bloque: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  MultiProcessors (SMs): " << prop.multiProcessorCount << "\n";

    // 3. Verificar capacidad mínima recomendada
    std::cout << "\n";
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        std::cout << "⚠️  Advertencia: Tu GPU tiene Compute Capability " << prop.major << "." << prop.minor << "\n";
        std::cout << "   Se recomienda CC 3.5 o superior para ejemplos avanzados.\n";
        std::cout << "   Algunos ejemplos pueden no compilar o funcionar incorrectamente.\n";
    }

    // 4. Probar un kernel simple (opcional)
    std::cout << "\n=== Test de kernel simple ===\n";

    // Kernel de prueba: cada hilo escribe su ID
    int N = 256;
    int *d_test;
    int *h_test = new int[N];

    cudaMalloc(&d_test, N * sizeof(int));
    std::cout << "  Lanzando kernel con " << N << " hilos...\n";

    // Lanzar kernel simple
    test_kernel<<<1, N>>>(d_test, N);
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "❌ Error al lanzar kernel: " << cudaGetErrorString(kernelErr) << "\n";
    } else {
        std::cout << "✅ Kernel lanzado exitosamente\n";
    }

    // Copiar resultado
    cudaMemcpy(h_test, d_test, N * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  Primeros 5 valores: ";
    for (int i = 0; i < 5 && i < N; i++) {
        std::cout << h_test[i] << " ";
    }
    std::cout << "\n";

    // Limpiar
    cudaFree(d_test);
    delete[] h_test;

    std::cout << "\n=== Resultado ===\n";
    std::cout << "✅ CUDA está funcionando correctamente.\n";
    std::cout << "   Puedes proceder a la lección 01_hello_world.\n";

    return 0;
}
