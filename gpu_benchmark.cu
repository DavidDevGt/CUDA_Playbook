#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void dummy_kernel(float *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = data[i] * 2.0f;
    }
}

int main() {
    // 1. Configuración del problema (1 millón de elementos)
    int N = 1 << 20; 
    size_t size = N * sizeof(float);
    std::cout << "Procesando " << N << " elementos (" << size / (1024*1024) << " MB)..." << std::endl;

    // 2. Memoria en el Host (CPU)
    float *h_data = (float*)malloc(size);
    for(int i=0; i<N; i++) h_data[i] = 1.0f;

    // 3. Memoria en el Device (GPU)
    float *d_data;
    gpuErrchk( cudaMalloc(&d_data, size) );

    // 4. Copiar datos: Host -> Device
    gpuErrchk( cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice) );

    // 5. Configuración de la ejecución
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // --- MEDICIÓN DE TIEMPO ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // Marca de inicio

    // Lanzamiento del Kernel
    dummy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    
    // Verificamos si hubo errores en el lanzamiento
    gpuErrchk( cudaPeekAtLastError() );

    cudaEventRecord(stop); // Marca de fin
    cudaEventSynchronize(stop); // Esperamos a que la GPU termine

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // ---------------------------

    // 6. Traer resultados: Device -> Host
    gpuErrchk( cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost) );

    // 7. Verificación final
    std::cout << "Resultado en indice 10: " << h_data[10] << std::endl;
    std::cout << "Resultado en indice 999: " << h_data[999] << std::endl;
    std::cout << "Tiempo de ejecucion del kernel: " << milliseconds << " ms" << std::endl;

    // 8. Limpieza
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}