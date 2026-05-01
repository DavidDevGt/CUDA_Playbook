/**
 * @file image_convolution.cu
 * @brief Aplicación: Convolución 2D (filtro de imagen)
 *
 * Aplica un filtro de convolución (ej: blur, edge detect) a una imagen.
 * Usa shared memory con tiling para optimizar accesos.
 *
 * Compilación:
 *   nvcc -o image_convolution image_convolution.cu
 *
 * Ejecución:
 *   ./image_convolution [WIDTH] [HEIGHT]
 *
 * Nota: Genera imagen sintética (patrón) y aplica filtro Sobel.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/timer.h"

// Kernel de convolución (Sobel X) usando shared memory
#define FILTER_RADIUS 1
#define TILE_SIZE 16

__constant__ float c_sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

// Función device para cargar con clamping (misma lógica que CPU)
__device__ float clamped_load(const float *img, int x, int y, int w, int h) {
    int cx = min(max(x, 0), w-1);
    int cy = min(max(y, 0), h-1);
    return img[cy * w + cx];
}

__global__ void convolution_sobel_x(const float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Coordenadas dentro del halo de este hilo:
    // Cada hilo es responsable de cargar una celda del tile (incluyendo halo)
    // Mapeamos threadIdx.x,y a posición en tile con offset -1..TILE_SIZE
    int sx = threadIdx.x;  // 0..TILE_SIZE-1 para centro
    int sy = threadIdx.y;

    // Cargar celda central (sx, sy) -> (x,y)
    // Pero necesitamos cargar también los halos.
    // Estrategia: cada hilo carga su propia celda del tile (que incluye halo).
    // Definimos índices globales para esta celda del tile:
    int gx = x - 1 + threadIdx.x;  // desde x-1 hasta x+TILE_SIZE-1
    int gy = y - 1 + threadIdx.y;  // desde y-1 hasta y+TILE_SIZE-1

    // Cargamos usando clamped_load para asegurar consistencia con CPU
    tile[threadIdx.y][threadIdx.x] = clamped_load(input, gx, gy, width, height);
    __syncthreads();

    // Aplicar convolución 3x3 solo para píxeles válidos (dentro de la imagen)
    if (x < width && y < height) {
        float sum = 0.0f;
        // Desplazamientos dentro del tile: el píxel central está en (threadIdx.x+1, threadIdx.y+1)
        // Pero el tile se cargó con un desplazamiento de -1, así que el píxel (x,y) está en (threadIdx.x+1, threadIdx.y+1)?
        // Revisar: gx = x-1 + threadIdx.x. Si threadIdx.x = 1, entonces gx = x, que es el píxel central.
        // Así que el píxel central está en tile[threadIdx.y][1] si threadIdx.x=1.
        // En general, el píxel central está en tile[threadIdx.y][threadIdx.x] con threadIdx.x=1? No.
        // Necesitamos reindexar: dado que threadIdx.x va de 0 a TILE_SIZE-1, y x = blockIdx.x*TILE_SIZE + threadIdx.x,
        // entonces el píxel central corresponde a threadIdx.x (sin -1). Pero al cargar con gx = x-1+threadIdx.x,
        // cuando threadIdx.x=0, gx=x-1 (halo izquierdo); threadIdx.x=1, gx=x (central); etc.
        // Por lo tanto, el píxel central está en tile[threadIdx.y][1] si asumimos threadIdx.x=1 para central.
        // Esto es confuso.
        // Simplificación: recalcular índices del kernel usando el tile ya cargado.
        // Sabemos que el tile contiene un halo de 1 alrededor del bloque TILE_SIZE x TILE_SIZE.
        // El bloque abarca píxeles [blockIdx.x*TILE_SIZE .. blockIdx.x*TILE_SIZE+TILE_SIZE-1] en x e y.
        // El tile en shared memory tiene tamaño (TILE_SIZE+2) y contiene esos píxeles más halo.
        // La posición del píxel (x,y) dentro del tile es (threadIdx.x+1, threadIdx.y+1)?
        // Porque threadIdx.x corre de 0 a TILE_SIZE-1 para el bloque. Si añadimos 1, obtenemos 1..TILE_SIZE.
        // Y el halo está en 0 y TILE_SIZE+1.
        // Así que la convolución para este píxel debe usar:
        //   tile[threadIdx.y+1 + ky][threadIdx.x+1 + kx] para ky,kx en {-1,0,1}.
        // Pero nuestro tile se cargó con gx = x-1+threadIdx.x, es decir, con threadIdx.x=0 corresponde a x-1.
        // Entonces tile[threadIdx.y][threadIdx.x] corresponde a la posición (x-1+threadIdx.x, y-1+threadIdx.y).
        // Por lo tanto, el píxel (x,y) se encuentra cuando threadIdx.x=1 y threadIdx.y=1? No, porque si threadIdx.x=1, gx=x.
        // Así que efectivamente, para que gx=x, necesitamos threadIdx.x=1. Pero threadIdx.x varía de 0 a TILE_SIZE-1, que incluye 1.
        // Esto significa que cada hilo tiene un threadIdx.x único, y solo un hilo por bloque tiene threadIdx.x=1 (si TILE_SIZE>1).
        // Eso no es correcto: queremos que todos los hilos del bloque contribuyan a cargar el tile, y luego cada hilo compute su píxel.
        // En nuestro esquema, cada hilo carga una celda del tile (incluyendo halo) y luego, después de __syncthreads(),
        // cada hilo puede leer del tile para computar su píxel.
        // Así que para el hilo con threadIdx.x, su píxel central está en (x,y). En el tile, ese píxel está en qué posición?
        // x = blockIdx.x*TILE_SIZE + threadIdx.x. Entonces la coordenada x dentro del bloque es threadIdx.x.
        // El tile tiene un halo de 1 a la izquierda, por lo que la columna dentro del bloque (threadIdx.x) se mapea a threadIdx.x+1 en el tile.
        // Así que tile[threadIdx.y+1][threadIdx.x+1] corresponde al píxel (x,y).
        // Sin embargo, en nuestra carga, asignamos tile[threadIdx.y][threadIdx.x] = clamped_load(...). Eso está mal.
        // Necesitamos corregir la asignación: tile[threadIdx.y+1][threadIdx.x+1] = clamped_load(input, x, y, ...).
        // Y para los halos, usamos otros hilos con desplazamientos.
        // Replanteamos: cada hilo carga su propio píxel central en tile[ty+1][tx+1], y también ayuda a cargar halos
        // usando condiciones en tx,ty. Esto se hace habitualmente con ifs.
        // Dado el tiempo, podemos adoptar una versión más simple y conocida que funcione.
        // Vamos a reescribir el kernel con un enfoque estándar de tiling con halo.
    }
    // ... Este enfoque está resultando confuso. Mejor reemplazamos por una versión estándar corregida.
}

// Versión estándar corregida de convolución con shared memory (Sobel X)
__global__ void convolution_sobel_x_fixed(const float *input, float *output, int width, int height) {
    __shared__ float tile[TILE_SIZE + 2 * FILTER_RADIUS][TILE_SIZE + 2 * FILTER_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_SIZE + ty;  // fila de salida
    int col_o = blockIdx.x * TILE_SIZE + tx;  // columna de salida
    int row_i = row_o - FILTER_RADIUS;        // fila de entrada (con halo)
    int col_i = col_o - FILTER_RADIUS;        // columna de entrada (con halo)

    // Cargar datos en shared memory con clamping
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        tile[ty][tx] = input[row_i * width + col_i];
    } else {
        // Clamp a borde
        int clamped_row = min(max(row_i, 0), height-1);
        int clamped_col = min(max(col_i, 0), width-1);
        tile[ty][tx] = input[clamped_row * width + clamped_col];
    }
    __syncthreads();

    // Aplicar convolución solo para píxeles de salida válidos
    if (ty < TILE_SIZE && tx < TILE_SIZE && row_o < height && col_o < width) {
        float sum = 0.0f;
        for (int ky = -FILTER_RADIUS; ky <= FILTER_RADIUS; ky++) {
            for (int kx = -FILTER_RADIUS; kx <= FILTER_RADIUS; kx++) {
                float val = tile[ty + FILTER_RADIUS + ky][tx + FILTER_RADIUS + kx];
                float coeff = c_sobel_x[ky + FILTER_RADIUS][kx + FILTER_RADIUS];
                sum += val * coeff;
            }
        }
        output[row_o * width + col_o] = fabsf(sum);
    }
}

int main() {
    std::cout << "=== Aplicación 12.2: Convolución de Imagen (Sobel X) ===\n\n";

    int width = 512, height = 512;
    std::cout << "Imagen: " << width << "×" << height << "\n\n";

    int total_pixels = width * height;
    float *h_input = new float[total_pixels];
    float *h_output = new float[total_pixels];
    float *d_input, *d_output;

    // Generar imagen sintética: gradiente radial
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = x - width/2.0f;
            float dy = y - height/2.0f;
            float dist = sqrtf(dx*dx + dy*dy);
            h_input[y*width + x] = sinf(dist * 0.1f) * 100.0f + 128.0f;
        }
    }

    cudaMalloc(&d_input, total_pixels * sizeof(float));
    cudaMalloc(&d_output, total_pixels * sizeof(float));
    cudaMemcpy(d_input, h_input, total_pixels * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE + 2 * FILTER_RADIUS, TILE_SIZE + 2 * FILTER_RADIUS);
    dim3 blocks((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Blocks: (" << blocks.x << ", " << blocks.y << ")\n";
    std::cout << "Threads/block: (" << threads.x << ", " << threads.y << ")\n";

    Timer timer;
    timer.start();
    convolution_sobel_x_fixed<<<blocks, threads>>>(d_input, d_output, width, height);
    gpuErrchk( cudaDeviceSynchronize() );
    timer.stop();
    float gpu_time = timer.getGpuTime();
    std::cout << "  Tiempo GPU: " << gpu_time << " ms\n";
    std::cout << "  Throughput: " << (total_pixels / (gpu_time / 1000.0f)) / 1e6 << " Mpixels/s\n";

    // ========== CPU REFERENCE ==========
    std::cout << "\n--- CPU reference ---\n";
    CpuTimer cpuTimer;
    cpuTimer.start();

    float *h_out_cpu = new float[total_pixels];
    float sobel_cpu[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = min(max(x + kx, 0), width-1);
                    int iy = min(max(y + ky, 0), height-1);
                    float pixel = h_input[iy*width + ix];
                    sum += pixel * sobel_cpu[ky+1][kx+1];
                }
            }
            h_out_cpu[y*width + x] = fabsf(sum);
        }
    }
    cpuTimer.stop();
    std::cout << "  CPU time: " << cpuTimer.getCpuTime() << " ms\n";
    std::cout << "  Speedup:  " << (cpuTimer.getCpuTime() / gpu_time) << "x\n";

    // ========== VERIFICAR ==========
    cudaMemcpy(h_output, d_output, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (fabs(h_output[idx] - h_out_cpu[idx]) > 1.0f) {
                ok = false;
                mismatches++;
                if (mismatches <= 5) {
                    std::cout << "Mismatch at (" << x << "," << y << "): GPU=" << h_output[idx] << " CPU=" << h_out_cpu[idx] << "\n";
                }
            }
        }
    }
    std::cout << "\nVerificación: " << (ok ? "✅ OK" : "❌ FAIL") << " (mismatches: " << mismatches << ")\n";

    float min_val = h_output[0], max_val = h_output[0];
    for (int i = 1; i < total_pixels; i++) {
        if (h_output[i] < min_val) min_val = h_output[i];
        if (h_output[i] > max_val) max_val = h_output[i];
    }
    std::cout << "Rango de valores: [" << min_val << ", " << max_val << "]\n";

    delete[] h_input; delete[] h_output; delete[] h_out_cpu;
    cudaFree(d_input); cudaFree(d_output);

    std::cout << "\n✅ Aplicación completada.\n";
    return 0;
}
