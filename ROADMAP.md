# 🚀 CUDA Learning Roadmap - RTX 5070 Ti (Ada Lovelace)

## Guía Interactiva de Aprendizaje CUDA

**Script principal:** `./roadmap.sh`

### ¿Qué es esto?

Una guía paso-a-paso que te enseña CUDA desde cero hasta nivel intermedio-avanzado, ejecutando código real en tu RTX 5070 Ti. Cada fase incluye:

- 📚 **Conceptos teóricos** - qué y por qué
- 💻 **Código funcional** - ejemplos compilados y listos
- 📊 **Ejecuciones reales** - ver resultados en tu GPU
- 🎯 **Key insights** - conclusiones prácticas

### Fases del Roadmap

| Fase | Tema | Ejemplos | Duración estimada |
|------|------|----------|-------------------|
| **0** | Prerrequisitos | `check_cuda`, `device_info` | 5 min |
| **1** | Hola Mundo | `hello`, `vector_add` (tradicional + unified) | 10 min |
| **2** | Vector Ops | `vector_scale`, `vector_dot`, `vector_operations_fused` | 10 min |
| **3** | Transferencias | `host_to_device`, `zero_copy`, `async_copy_example` | 15 min |
| **4** | Jerarquía | `indexing_1d/2d/3d`, `grid_stride` | 15 min |
| **5** | Shared Memory | `shared_memory_basic`, `matrix_tiling`, `reduction_shared` | 20 min |
| **6** | Patrones | `reduction`, `scan_prefix_sum`, `reduction_warp_optimized` | 20 min |
| **7** | Constante | `constant_array`, `lookup_table` | 15 min |
| **8** | Atómicas | `atomic_add`, `atomic_compare` | 10 min |
| **9** | Optimización | `occupancy`, `loop_unrolling`, `profile_events` | 20 min |
| **10** | Avanzado | `streams`, `events`, `dynamic_parallelism` | 25 min |
| **11** | Aplicaciones | `vector_max`, `image_convolution` | 15 min |

**Total:** ~3.5 horas de aprendizaje práctico

### Requisitos

```bash
# Ya configurado en este repositorio:
- GPU: NVIDIA RTX 5070 Ti (CC 12.0)
- CUDA Toolkit: 12.4
- Arquitectura target: sm_89 (Ada Lovelace)
```

### Cómo usar

#### Opción 1: Modo interactivo (Recomendado)

```bash
cd /home/rou/Projects/C
./roadmap.sh
```

El script te guiará fase por fase, explicando conceptos y ejecutando ejemplos. Presiona Enter para avanzar.

#### Opción 2: Manual por fases

```bash
# Compila todo
make all

# Ejecuta una fase específica
./build/hello                    # Fase 1
./build/vector_add_unified       # Fase 1-2
./build/reduction_warp_optimized # Fase 6
./build/streams                  # Fase 10
```

#### Opción 3: Ejecutar binarios específicos

```bash
# Suma de vectores (1M elementos)
./build/vector_add 1000000

# Reducción (16M elementos)
./build/reduction_warp_optimized 16777216 sum

# Multiplicación de matrices
./build/matrix_multiply_optimized_ada 1024 1024 1024

# Streams (con tamaño)
./build/streams 16777216
```

### Conceptos Clave Aprendidos

#### 1. Arquitectura Ada Lovelace (sm_89)
- ✅ 96 KB shared memory por SM (2× Volta)
- ✅ 72 MB L2 cache (8× Volta)
- ✅ Tensor Cores 4ª generación
- ✅ cp.async para copias asíncronas hardware

#### 2. Jerarquía de Memoria (rápido → lento)
```
Registros (por hilo)
    ↓
Shared Memory (por bloque, 96 KB/SM)
    ↓
L1 Cache / Constant Memory
    ↓
L2 Cache (72 MB, compartida)
    ↓
Global Memory (15.8 GB GDDR7)
    ↓
Host Memory (PCIe)
```

#### 3. Optimizaciones Clave

| Técnica | Speedup | Cuándo usar |
|---------|---------|-------------|
| Kernel fusion | 2-3× | Multiples operaciones secuenciales |
| Warp shuffle | 3-5× | Reducciones, escaneos |
| Tiling (32×32) | 31× | Multiplicación matrices |
| Pinned memory | 2× | Transferencias frecuentes |
| Async streams | 1.5-2× | Overlap H2D/K/D2H |
| Unified Memory | Similar | Código más simple |

#### 4. Reglas de Oro

- 📏 **Tamaño bloque**: múltiplo de 32 (tamaño warp)
- 🔄 **Grid-stride loops**: para datasets de cualquier tamaño
- 💾 **Minimiza transferencias**: CPU ↔ GPU es lento (PCIe)
- 🎯 **Coalescing**: accesos secuenciales de memoria global
- ⚡ **Occupancy**: 50-75% suele ser óptimo (no maximizar siempre)
- 🔀 **Warp divergence**: evitar if/else divergentes en warp

### Ejemplos Destacados

#### 🚀 Mejor rendimiento
```bash
# Kernel fusionado: 1200+ GB/s
./build/vector_operations_fused 100000
```

#### 🧠 Speedup impresionante
```bash
# Matrix multiply: 31.3× más rápido (32×32 tiles)
./build/matrix_multiply_optimized_ada 1024 1024 1024
```

#### ⚡ Reducción warp-level
```bash
# 3-5× más rápido que tree reduction
./build/reduction_warp_optimized 16777216 sum
```

#### 📊 Overlap computación/transferencia
```bash
# Multi-stream para esconder latencia PCIe
./build/streams 16777216
```

### Troubleshooting

**Problema**: `nvcc: error: unrecognized option '-arch=sm_120'`
- **Solución**: CUDA 12.4 no soporta sm_120 aún. Usa `-arch=sm_89` (máximo disponible)

**Problema**: `too many resources requested for launch`
- **Solución**: Reducir hilos/bloque o memoria compartida por bloque

**Problema**: Resultados incorrectos en reducción
- **Solución**: Verificar condición de frontera en kernel

**Problema**: Bajo rendimiento
- **Solución**: 
  1. Verificar coalescing (accesos secuenciales)
  2. Minimizar transferencias CPU-GPU
  3. Usar pinned memory para async
  4. Perfilamiento con Nsight

### Siguientes Pasos

Después de completar este roadmap:

1. **Tensor Cores** (WMMA API)
   - 4×-16× speedup en matmul
   - Requiere código específico

2. **CUDA Graphs**
   - Capturar y repetir pipelines
   - Reducir overhead de lanzamiento

3. **Multi-GPU**
   - NVLink para P2P rápido
   - Escalabilidad horizontal

4. **Frameworks**
   - Thrust (algoritmos paralelos STL-like)
   - CUB (building blocks)
   - Integración PyTorch/TensorFlow

5. **Lenguajes**
   - CUDA C++ (actual)
   - Numba Python (prototipado rápido)
   - CuPy (arrays NumPy-like)

### Recursos

**Documentación:**
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

**Tools:**
- Nsight Compute (profiling kernel)
- Nsight Systems (análisis sistema)
- nvprof (legacy pero útil)

**Comunidad:**
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/cuda/c/68)
- [Stack Overflow: cuda](https://stackoverflow.com/questions/tagged/cuda)
- [CUDA Samples GitHub](https://github.com/NVIDIA/cuda-samples)

### Archivos del Repositorio

```
.
├── roadmap.sh              # ← Script interactivo (EMPIEZA AQUÍ)
├── README.md               # Documentación general
├── Makefile                # Build system principal
├── common/                 # Utilidades compartidas
│   ├── cuda_utils.h        # Macros verificación errores
│   ├── cuda_warp_ops.h     # Operaciones warp-level
│   └── timer.h             # Timing GPU/CPU
├── 00_prerequisites/       # Verificación CUDA
├── 01_hello_world/         # Primeros pasos
├── 02_vector_operations/   # Operaciones vectoriales
├── 03_matrix_operations/   # Matrices
├── 04_memory_transfers/    # Gestión memoria
├── 05_thread_hierarchy/    # Jerarquía hilos
├── 06_shared_memory/       # Memoria compartida
├── 07_constant_memory/     # Memoria constante
├── 08_atomic_operations/   # Operaciones atómicas
├── 09_parallel_patterns/   # Patrones (scan, reduce)
├── 10_performance/         # Optimización
├── 11_advanced/            # Streams, DP
└── 12_applications/        # Aplicaciones reales
```

### Licencia

MIT - Libre para uso educativo y comercial

---

**¡Empieza tu viaje CUDA hoy!** 🚀

```bash
cd /home/rou/Projects/C
./roadmap.sh
```
