/**
 * @file device_info.cu
 * @brief Muestra información detallada de TODOS los dispositivos CUDA
 *
 * Este programa extiende check_cuda.cu mostrando:
 * - Información de cada GPU en el sistema
 * - Capacidades específicas
 * - Límites de ejecución
 * - Propiedades de memoria y caché
 *
 * Útil para entender las capacidades de tu hardware
 * y optimizar kernels según las especificaciones.
 *
 * Compilación:
 *   nvcc -o device_info device_info.cu
 *
 * Uso:
 *   ./device_info
 */

#include <iostream>
#include <cuda_runtime.h>

void printDeviceProperties(const cudaDeviceProp& prop, int id) {
    std::cout << "╔═══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Dispositivo CUDA [" << id << "]                              ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";

    std::cout << "║  Nombre:                      " << prop.name;
    for (size_t i = strlen(prop.name) + 1; i < 40; i++) std::cout << " ";
    std::cout << "║\n";

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  ARQUITECTURA                                           ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Compute Capability:          " << prop.major << "." << prop.minor;
    for (size_t i = 0; i < 33; i++) std::cout << " ";
    std::cout << "║\n";

    char arch[50];
    if (prop.major == 1) sprintf(arch, "Tesla");
    else if (prop.major == 2) sprintf(arch, "Fermi");
    else if (prop.major == 3) sprintf(arch, "Kepler");
    else if (prop.major == 5) sprintf(arch, "Maxwell");
    else if (prop.major == 6) sprintf(arch, "Pascal");
    else if (prop.major == 7) {
        if (prop.minor == 0) sprintf(arch, "Volta");
        else sprintf(arch, "Turing");
    }
    else if (prop.major == 8) sprintf(arch, "Ampere");
    else if (prop.major == 9) sprintf(arch, "Ada/Hopper");
    else sprintf(arch, "Nueva arquitectura");
    std::cout << "║  Arquitectura:                " << arch;
    for (size_t i = strlen(arch); i < 33; i++) std::cout << " ";
    std::cout << "║\n";

    std::cout << "║  Compute Mode:                " << prop.computeMode;
    for (size_t i = 0; i < 33; i++) std::cout << " ";
    std::cout << "║\n";

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  RECURSOS                                               ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  MultiProcessors (SMs):       " << prop.multiProcessorCount;
    std::cout << "                               ║\n";
    std::cout << "║  CUDA Cores (SM count × cores/): Ver tabla abajo ║\n";
    std::cout << "║  Max Threads per Block:       " << prop.maxThreadsPerBlock;
    std::cout << "                               ║\n";
    std::cout << "║  Max Threads Dim:             [" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]";
    for (size_t i = 0; i < 20; i++) std::cout << " ";
    std::cout << "║\n";
    std::cout << "║  Max Grid Size:               [" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]";
    for (size_t i = 0; i < 20; i++) std::cout << " ";
    std::cout << "║\n";
    std::cout << "║  Max Threads per SM:          " << prop.maxThreadsPerMultiProcessor;
    std::cout << "                               ║\n";
    std::cout << "║  Max Blocks per SM:           " << prop.maxBlocksPerMultiProcessor;
    std::cout << "                               ║\n";

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  MEMORIA                                                ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Global Memory:               " << prop.totalGlobalMem / (1024.0*1024.0) << " MB";
    for (size_t i = 0; i < 27; i++) std::cout << " ";
    std::cout << "║\n";
    std::cout << "║  Shared Memory per Block:     " << prop.sharedMemPerBlock / 1024.0 << " KB";
    std::cout << "                               ║\n";
    std::cout << "║  Shared Memory per SM:        ";
    // sharedMemPerMultiprocessor solo disponible en CC >= 3.5
    if (prop.sharedMemPerMultiprocessor > 0) {
        std::cout << prop.sharedMemPerMultiprocessor / 1024.0 << " KB";
    } else {
        std::cout << "N/A";
    }
    std::cout << "                               ║\n";
    std::cout << "║  Constant Memory:             " << prop.totalConstMem / 1024.0 << " KB";
    std::cout << "                               ║\n";
    std::cout << "║  Texture Memory (global):     " << prop.totalGlobalMem / (1024.0*1024.0) << " MB";
    std::cout << "                               ║\n";
    std::cout << "║  Texture Alignment:           " << prop.textureAlignment << " bytes";
    std::cout << "                               ║\n";
    std::cout << "║  Pitch Limit:                 " << prop.memPitch / (1024.0*1024.0) << " MB";
    std::cout << "                               ║\n";

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  WARPS Y SCHEDULER                                     ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Warp Size:                   " << prop.warpSize;
    std::cout << " hilos                         ║\n";
    std::cout << "║  Max Warps per SM:            " << prop.maxThreadsPerMultiProcessor / prop.warpSize;
    std::cout << "                               ║\n";
    std::cout << "║  Max Warps per Block:         " << prop.maxThreadsPerBlock / prop.warpSize;
    std::cout << "                               ║\n";

    int threadsPerSM = prop.maxThreadsPerMultiProcessor;
    int warpsPerSM = threadsPerSM / prop.warpSize;
    int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxWarpsPerBlock = maxThreadsPerBlock / prop.warpSize;

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  LIMITES DE LANZAMIENTO                                ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Blocks por SM:               " << maxBlocksPerSM;
    std::cout << "                               ║\n";
    std::cout << "║  Threads por SM:              " << threadsPerSM;
    std::cout << "                               ║\n";
    std::cout << "║  Warps por SM:                " << warpsPerSM;
    std::cout << "                               ║\n";
    std::cout << "║  Threads por bloque (máx):    " << maxThreadsPerBlock;
    std::cout << "                               ║\n";
    std::cout << "║  Warps por bloque (máx):      " << maxWarpsPerBlock;
    std::cout << "                               ║\n";

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  REGISTROS Y MEMORIA COMPARTIDA                        ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  Registros por bloque:        " << prop.regsPerBlock;
    std::cout << "                               ║\n";
    std::cout << "║  Registros por SM:            " << prop.regsPerMultiprocessor;
    std::cout << "                               ║\n";
    std::cout << "║  Regs por hilo (máx):         " << prop.regsPerBlock / maxThreadsPerBlock;
    std::cout << "                               ║\n";
    std::cout << "║  SharedMem por bloque:        " << prop.sharedMemPerBlock / 1024.0 << " KB";
    std::cout << "                               ║\n";

    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  CACHÉS Y MEMORIA                                      ║\n";
    std::cout << "╠═══════════════════════════════════════════════════════╣\n";
    std::cout << "║  L1 Cache per SM:             Variable (ver Nsight)\n";
    std::cout << "                               ║\n";
    std::cout << "║  L2 Cache Size:               ";
    if (prop.l2CacheSize > 0) {
        std::cout << prop.l2CacheSize / 1024.0 << " KB";
    } else {
        std::cout << "N/A";
    }
    std::cout << "                               ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════╝\n";
}

void printCoresPerSM(int major, int minor) {
    // Tabla de núcleos CUDA por SM según arquitectura
    // Fuente: NVIDIA docs y tablas públicas
    int cores = 0;
    const char* arch = "";

    if (major == 1) { arch = "Tesla"; cores =  8; }
    else if (major == 2) { arch = "Fermi"; cores = 32; }
    else if (major == 3) {
        if (minor == 0) { arch = "Kepler GK104"; cores = 192; }
        else if (minor == 2) { arch = "Kepler GK110"; cores = 192; }
        else if (minor == 5) { arch = "Kepler GK210"; cores = 192; }
        else { arch = "Kepler"; cores = 192; }
    }
    else if (major == 5) {
        if (minor == 0) { arch = "Maxwell GM107"; cores = 128; }
        else if (minor == 2) { arch = "Maxwell GM204"; cores = 128; }
        else { arch = "Maxwell"; cores = 128; }
    }
    else if (major == 6) {
        if (minor == 0) { arch = "Pascal GP100"; cores = 64; }
        else if (minor == 1) { arch = "Pascal GP104"; cores = 128; }
        else if (minor == 2) { arch = "Pascal GP102"; cores = 128; }
        else { arch = "Pascal"; cores = 128; }
    }
    else if (major == 7) {
        if (minor == 0) { arch = "Volta GV100"; cores = 64; }
        else if (minor == 5) { arch = "Turing TU104"; cores = 64; }
        else if (minor == 6) { arch = "Turing TU106"; cores = 64; }
        else { arch = "Turing/Ampere"; cores = 64; }
    }
    else if (major == 8) {
        if (minor == 0) { arch = "Ampere A100"; cores = 64; }
        else { arch = "Ampere"; cores = 64; }
    }
    else if (major == 9) {
        arch = "Ada/Hopper"; cores = 128;
    }
    else {
        arch = "Unknown";
        cores = 0;
    }

    std::cout << "  (Aproximadamente " << cores << " CUDA Cores por SM para " << arch << ")\n";
    std::cout << "  Total CUDA Cores (aprox): " << cores << " × " << "MultiProcessorCount = "
              << 0 << " (debes multiplicar manualmente)\n";
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     INFORMACIÓN DETALLADA DE DISPOSITIVOS CUDA               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "❌ No se encontraron dispositivos CUDA.\n";
        return 1;
    }

    std::cout << "Dispositivos encontrados: " << deviceCount << "\n\n";

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printDeviceProperties(prop, i);

        // Mostrar cores por SM
        printCoresPerSM(prop.major, prop.minor);

        std::cout << "\n";
    }

    // Mostrar dispositivo activo actual
    int currentDevice;
    cudaGetDevice(&currentDevice);
    std::cout << "Dispositivo activo actual: " << currentDevice << "\n";
    std::cout << "\n✅ Información completa mostrada.\n";
    std::cout << "   Usa esta información para:\n";
    std::cout << "   - Seleccionar arquitectura de compilación (-arch=sm_XX)\n";
    std::cout << "   - Calcular occupancy teórico\n";
    std::cout << "   - Diseñar kernels optimizados\n";

    return 0;
}
