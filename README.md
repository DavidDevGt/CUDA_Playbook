# Aprender CUDA desde Cero

Repositorio completo y estructurado para aprender programación CUDA, desde los conceptos más básicos hasta técnicas avanzadas de optimización.

## 📋 Tabla de Contenidos

1. [Prerrequisitos](#prerrequisitos)
2. [Estructura del Repositorio](#estructura-del-repositorio)
3. [Configuración](#configuración)
4. [Lecciones](#lecciones)
5. [Recursos Adicionales](#recursos-adicionales)

## Prerrequisitos

### Hardware
- GPU NVIDIA con soporte CUDA (Compute Capability 3.5 o superior recomendado)
- Verificar compatibilidad: `nvidia-smi`

### Software
- **Sistema Operativo**: Linux, Windows o macOS
- **Driver NVIDIA**: Actualizado a la última versión
- **CUDA Toolkit**: Versión 11.0 o superior ([descargar](https://developer.nvidia.com/cuda-downloads))
- **Compilador**: `gcc`/`g++` (Linux/macOS) o Visual Studio (Windows)
- **Make** (opcional, pero recomendado)

### Verificar Instalación

```bash
# Verificar driver NVIDIA
nvidia-smi

# Verificar instalación de CUDA
nvcc --version

# Probar compilación (ejecutar desde la raíz del repositorio)
make 00_check
./build/00_check
```

## Estructura del Repositorio

```
.
├── README.md                    # Este archivo
├── Makefile                     # Sistema de compilación unificado
├── .gitignore                   # Archivos ignorados por git
├── common/                      # Utilidades compartidas
│   ├── cuda_utils.cu            # Funciones auxiliares CUDA
│   ├── cuda_utils.h
│   ├── timer.cu                 # Clase para medición de tiempo
│   └── timer.h
├── 00_prerequisites/            # Verificar entorno CUDA
│   ├── check_cuda.cu            # Detectar GPU y capacidades
│   ├── device_info.cu           # Información detallada del dispositivo
│   └── Makefile
├── 01_hello_world/              # Tu primer kernel
│   ├── hello.cu                 # "Hola Mundo" en GPU
│   ├── vector_add.cu            # Suma de vectores simple
│   └── Makefile
├── 02_vector_operations/        # Operaciones con vectores
│   ├── vector_add.cu            # Suma de vectores (SAXPY)
│   ├── vector_scale.cu          # Escalamiento de vectores
│   ├── vector_dot.cu            # Producto punto
│   └── Makefile
├── 03_matrix_operations/        # Operaciones con matrices
│   ├── matrix_add.cu            # Suma de matrices
│   ├── matrix_multiply.cu       # Multiplicación de matrices
│   ├── matrix_transpose.cu      # Transposición
│   └── Makefile
├── 04_memory_transfers/         # Gestión de memoria
│   ├── host_to_device.cu        # Copia CPU → GPU
│   ├── device_to_host.cu        # Copia GPU → CPU
│   ├── pinned_memory.cu         # Memoria paginada (pinned)
│   ├── zero_copy.cu             # Zero-copy memory
│   └── Makefile
├── 05_thread_hierarchy/         # Jerarquía de hilos
│   ├── block_dimensions.cu      # Calculando dimensiones de bloque
│   ├── grid_stride.cu           # Grid-stride loops
│   ├── indexing_1d.cu           # Indexación 1D
│   ├── indexing_2d.cu           # Indexación 2D (matrices)
│   ├── indexing_3d.cu           # Indexación 3D
│   └── Makefile
├── 06_shared_memory/            # Memoria compartida
│   ├── shared_memory_basic.cu   # Uso básico
│   ├── matrix_tiling.cu         # Tiling para matrices
│   ├── reduction_shared.cu      # Reducción con shared memory
│   ├── bank_conflicts.cu        # Conflictos de bancos
│   └── Makefile
├── 07_constant_memory/          # Memoria constante
│   ├── constant_array.cu        # Array en memoria constante
│   ├── lookup_table.cu          # Tabla de consulta
│   └── Makefile
├── 08_atomic_operations/        # Operaciones atómicas
│   ├── atomic_add.cu            # Adición atómica
│   ├── atomic_compare.cu        # Comparación e intercambio
│   ├── histogram.cu             // Histograma con atómicos
│   └── Makefile
├── 09_parallel_patterns/        # Patrones paralelos
│   ├── reduction.cu             # Reducción (suma, min, max)
│   ├── scan_prefix_sum.cu       # Prefix sum / scan
│   ├── segmented_scan.cu        # Scan segmentado
│   └── Makefile
├── 10_performance/              # Optimización y profiling
│   ├── occupancy.cu             # Cálculo de occupancy
│   ├── memory_coalescing.cu     # Accesos coalesced
│   ├── loop_unrolling.cu        # Loop unrolling
│   ├── profile_events.cu        // Profiling con eventos
│   └── Makefile
├── 11_advanced/                 # Temas avanzados
│   ├── streams.cu               # Múltiples streams
│   ├── events.cu                # Sincronización con eventos
│   ├── dynamic_parallelism.cu   # Paralelismo dinámico (requiere CC 3.5+)
│   └── Makefile
├── 12_applications/             # Aplicaciones completas
│   ├── nbody_simulation.cu      // Simulación N-Cuerpos
│   ├── image_processing.cu      // Filtros de imagen
│   └── Makefile
└── exercises/                   # Ejercicios propuestos
    ├── basic/
    ├── intermediate/
    └── advanced/
```

## Configuración

### Compilación

El repositorio incluye un `Makefile` unificado que compila todos los ejemplos:

```bash
# Compilar todos los ejemplos
make all

# Compilar una lección específica
make 00_check
make 01_hello
make 02_vector

# Limpiar archivos compilados
make clean

# Ver todas las opciones
make help
```

### Compilación Manual

Si prefieres compilar manualmente:

```bash
# Ejemplo básico
nvcc -o hello hello.cu

  # Con optimización
  nvcc -O3 -arch=sm_89 -o matrix_multiply matrix_multiply.cu

  # Con depuración
  nvcc -G -g -o debug_example debug.cu
```

**Flags importantes:**
- `-arch=sm_XX`: Especifica la arquitectura de tu GPU (sm_70 = Volta, sm_80 = Ampere, sm_89 = Ada, etc.)
- `-O3`: Optimización agresiva
- `-G`: Debug info (desactiva optimizaciones)
- `-lineinfo`: Info de línea para profiler

## Lecciones

### 00. Prerrequisitos
Verifica que tu entorno CUDA esté correctamente configurado.

- **check_cuda.cu**: Detecta GPU y muestra capacidades
- **device_info.cu**: Muestra información detallada (memoria, cores, etc.)

**Objetivo**: Asegurarse de que CUDA está instalado y funcionando.

### 01. Hello World
Tu primer kernel CUDA.

- **hello.cu**: Kernel que modifica datos en paralelo
- **vector_add.cu**: Suma de dos vectores

**Conceptos clave**:
- `__global__`:Funciones que se ejecutan en la GPU
- `<<<blocks, threads>>>`:Sintaxis de lanzamiento
- `cudaMalloc`, `cudaMemcpy`:Gestión de memoria

### 02. Vector Operations
Operaciones fundamentales con vectores.

- **vector_add.cu**: `C = A + B`
- **vector_scale.cu**: `C = α · A`
- **vector_dot.cu**: Producto punto `A · B`

**Conceptos clave**:
- Indexación de hilos
- Condiciones de frontera
- Rendimiento vs CPU

### 03. Matrix Operations
Extension a datos 2D.

- **matrix_add.cu**: Suma de matrices
- **matrix_multiply.cu**: Multiplicación naive
- **matrix_transpose.cu**: Transposición

**Conceptos clave**:
- Indexación 2D
- Patrones de acceso a memoria
- Tiling introductorio

### 04. Memory Transfers
Gestión eficiente de memoria.

- **host_to_device.cu**: Copias H→D
- **device_to_host.cu**: Copias D→H
- **pinned_memory.cu**: Memoria paginada
- **zero_copy.cu**: Zero-copy mapping

**Conceptos clave**:
- `cudaMemcpy` modos
- Latencia de transferencia
- Memoria unificada (opcional)

### 05. Thread Hierarchy
Dominar la jerarquía de ejecución.

- **block_dimensions.cu**: Cálculo óptimo de bloques
- **grid_stride.cu**: Grid-stride loops (para datasets grandes)
- **indexing_1d/2d/3d.cu**: Diferentes dimensiones

**Conceptos clave**:
- `blockIdx`, `threadIdx`, `blockDim`, `gridDim`
- Grid-stride para flexibilidad
- División del trabajo

### 06. Shared Memory
Memoria rápida on-chip.

- **shared_memory_basic.cu**: Declaración y uso
- **matrix_tiling.cu**: Tiling para cachear bloques
- **reduction_shared.cu**: Reducción eficiente
- **bank_conflicts.cu**: Identificación y solución

**Conceptos clave**:
- `__shared__`
- Sincronización `__syncthreads()`
- Bank conflicts

### 07. Constant Memory
Lecturas de solo lectura caché.

- **constant_array.cu**: Declaración y acceso
- **lookup_table.cu**: Tabla de consulta

**Conceptos clave**:
- `__constant__`
- Caché constante
- Cuándo usarlo

### 08. Atomic Operations
Operaciones thread-safe.

- **atomic_add.cu**: Suma atómica
- **atomic_compare.cu**: CAS (Compare and Swap)
- **histogram.cu**: Construcción de histograma

**Conceptos clave**:
- `atomicAdd`, `atomicCAS`, etc.
- Contención
- Alternativas sin atómicos

### 09. Parallel Patterns
Patrones comunes de paralelización.

- **reduction.cu**: Suma, min, max
- **scan_prefix_sum.cu**: Scan exclusivo/inclusivo
- **segmented_scan.cu**: Scan por segmentos

**Conceptos clave**:
- Algoritmos paralelos
- División y conquista
- pami local

### 10. Performance
Métricas y optimización.

- **occupancy.cu**: Cálculo de occupancy
- **memory_coalescing.cu**: Accesos secuenciales
- **loop_unrolling.cu**: Desenrollado manual
- **profile_events.cu**: Uso de eventos para timing

**Conceptos clave**:
- Warps, occupancy
- Coalesced memory access
- Profiling con `nvprof`/`nsight`

### 11. Advanced
Características avanzadas.

- **streams.cu**: Concurrencia con múltiples streams
- **events.cu**: Sincronización fine-grained
- **dynamic_parallelism.cu**: Kernels que lanzan kernels

**Conceptos clave**:
- Overlap cómputo/transferencia
- Dependencias
- GPU envío deGPU

### 12. Applications
Aplicaciones completas integradoras.

- **nbody_simulation.cu**: Simulación de partículas
- **image_processing.cu**: Filtros en tiempo real

## Recursos Adicionales

### Documentación Oficial
- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)

### Tutoriales
- [NVIDIA CUDA Toolkit Tutorials](https://github.com/NVIDIA/cuda-samples)
- [CUDA by Example](https://developer.nvidia.com/cuda-example) (libro)

### Herramientas
- `nvprof`: Profiler básico (legacy)
- **Nsight Systems**: System-wide profiling
- **Nsight Compute**: Kernel-level profiling
- `cuda-gdb`: Debugger para CUDA

### Comunidad
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/cuda/c/68)
- [Stack Overflow: cuda](https://stackoverflow.com/questions/tagged/cuda)

## Consejos de Aprendizaje

1. **Orden**: Sigue las lecciones en orden. Cada una construye sobre la anterior.
2. **Experimenta**: Modifica los ejemplos, cambia parámetros, rompe cosas.
3. **Mide**: Siempre usa `cudaEvent_t` o `nvprof` para medir rendimiento.
4. **Lee el código**: Analiza cada línea, pregunta por qué.
5. **Implementa**: Después de cada tema, intenta resolver un problema Similar.

## Licencia

Este repositorio está bajo licencia MIT. Puedes usar el código libremente para aprender y enseñar.

---

**¡Comienza con [00_prerequisites/check_cuda.cu](./00_prerequisites/check_cuda.cu) y verifica tu entorno!**
