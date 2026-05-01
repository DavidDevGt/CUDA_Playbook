#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief Clase para medición de tiempo en GPU y CPU
 */
class Timer {
private:
    cudaEvent_t start_gpu, stop_gpu;
    float gpu_time;

public:
    Timer() : gpu_time(0.0f) {
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
    }

    ~Timer() {
        cudaEventDestroy(start_gpu);
        cudaEventDestroy(stop_gpu);
    }

    /**
     * @brief Inicia medición de tiempo GPU
     */
    void start() {
        gpu_time = 0.0f;
        cudaEventRecord(start_gpu);
    }

    /**
     * @brief Detiene medición de tiempo GPU
     */
    void stop() {
        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);
        cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    }

    /**
     * @brief Obtiene tiempo en milisegundos
     */
    float getGpuTime() const {
        return gpu_time;
    }

    /**
     * @brief Imprime tiempo en formato amigable
     */
    void printElapsedTime(const std::string& label = "Tiempo") const {
        std::cout << label << ": " << gpu_time << " ms";
        if (gpu_time < 1.0f) {
            std::cout << " (" << gpu_time * 1000.0f << " μs)";
        } else if (gpu_time > 1000.0f) {
            std::cout << " (" << gpu_time / 1000.0f << " s)";
        }
        std::cout << "\n";
    }
};

/**
 * @brief Timer simple para CPU usando clock()
 */
class CpuTimer {
private:
    clock_t start_cpu, end_cpu;
    float cpu_time;

public:
    CpuTimer() : cpu_time(0.0f) {}

    void start() {
        start_cpu = clock();
    }

    void stop() {
        end_cpu = clock();
        cpu_time = ((float)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0f;
    }

    float getCpuTime() const {
        return cpu_time;
    }

    void printElapsedTime(const std::string& label = "Tiempo CPU") const {
        std::cout << label << ": " << cpu_time << " ms";
        if (cpu_time < 1.0f) {
            std::cout << " (" << cpu_time * 1000.0f << " μs)";
        } else if (cpu_time > 1000.0f) {
            std::cout << " (" << cpu_time / 1000.0f << " s)";
        }
        std::cout << "\n";
    }
};

#endif // TIMER_H
