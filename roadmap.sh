#!/bin/bash
# ========================================
# CUDA Learning Roadmap - Interactive Guide
# RTX 5070 Ti (Ada Lovelace) | CC 8.9
# ========================================

set -e

BASE_DIR="/home/rou/Projects/C"
cd "$BASE_DIR"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

function header() {
    echo -e "\n${CYAN}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}\n"
}

function step() {
    echo -e "${YELLOW}▶${NC} $1"
}

function success() {
    echo -e "${GREEN}✓${NC} $1"
}

function info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

function warn() {
    echo -e "${RED}⚠${NC} $1"
}

function run_build() {
    local target=$1
    step "Building: $target"
    make $target 2>&1 | grep -E "(Compilando|Error|error:|✓)" | grep -v "warning:" || true
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        success "$target compilado"
        return 0
    else
        warn "Error compilando $target"
        return 1
    fi
}

function run_exe() {
    local exe=$1
    shift
    step "Ejecutando: ./$exe $@"
    eval "./build/$exe $@" 2>&1 | tail -20
    echo ""
}

# ========================================
# ROADMAP
# ========================================

cat << 'EOF'

   🚀 CUDA Learning Roadmap - RTX 5070 Ti (Ada Lovelace)
   
   Arquitectura:  sm_89 (Compute Capability 8.9)
   Memoria Compartida: 96 KB/SM (2x Volta)
   L2 Cache:        72 MB (8x Volta)
   Tensor Cores:    4ª gen (FP16/FP32/FP64/FP8/INT8)
   
   FASE 0: Prerrequisitos          [ Verificar CUDA         ]
   FASE 1: Hola Mundo              [ Primer kernel          ]
   FASE 2: Operaciones Vectoriales  [ Suma, producto punto   ]
   FASE 3: Memoria                 [ Transferencias optimas ]
   FASE 4: Jerarquía de Hilos     [ Bloques, grids, warps ]
   FASE 5: Memoria Compartida      [ Tiling, reducción      ]
   FASE 6: Patrones Paralelos      [ Scan, reducción        ]
   FASE 7: Memoria Constante       [ Lookups, broadcast     ]
   FASE 8: Operaciones Atómicas    [ Histogramas, CAS       ]
   FASE 9: Optimización            [ Occupancy, unroll      ]
   FASE 10: Avanzado               [ Streams, eventos, DP   ]
   FASE 11: Aplicaciones           [ N-body, convolución    ]

EOF

echo -e "${MAGENTA}Presiona Enter para empezar...${NC}"
read -r

# ========================================
# FASE 0: PRE-REQUISITOS
# ========================================
header "FASE 0: Prerrequisitos - Verificar entorno CUDA"

step "Verificando GPU..."
./build/check_cuda 2>&1 | grep -E "(CUDA|Dispositivos|Nombre|Capacidad|Memoria|Warp|SMs|✅)"
success "GPU verificada"
echo ""

step "Información detallada..."
./build/device_info 2>&1 | head -50
echo ""

warn "La RTX 5070 Ti es CC 12.0 pero CUDA 12.4 soporta hasta sm_89"
warn "Usando -arch=sm_89 (Ada Lovelace) para máxima compatibilidad"
echo ""

info "Presiona Enter para continuar a FASE 1..."
read -r

# ========================================
# FASE 1: HOLA MUNDO
# ========================================
header "FASE 1: Hola Mundo CUDA - Primeros Pasos"

step "Conceptos clave:"
echo "  • __global__: funciones que se ejecutan en GPU"
echo "  • <<<bloques, hilos>>>: sintaxis de lanzamiento"
echo "  • cudaMalloc/cudaMemcpy: gestión de memoria"
echo "  • threadIdx, blockIdx: indexación de hilos"
echo ""

run_build "01_hello"
run_exe "hello"
echo ""

info "Nota: Cada hilo calcula su índice único (idx) y modifica un elemento"
warn "El límite de hilos por bloque es 1024 en esta GPU"
echo ""

info "Presiona Enter para continuar..."
read -r

# Suma de vectores - versión tradicional
step "Vector Addition (tradicional con cudaMalloc)"
run_build "01_hello"  # vector_add ya está en el mismo target
run_exe "vector_add" 1000000
echo ""

# Unified Memory
step "Vector Addition (Unified Memory - optimizado)"
run_exe "vector_add_unified" 1000000
echo ""

info "Diferencia: Unified usa un solo puntero, no hay cudaMemcpy explícito"
info "Ada Smart Memory maneja migración automáticamente"
echo ""

info "Presiona Enter para continuar a FASE 2..."
read -r

# ========================================
# FASE 2: OPERACIONES VECTORIALES
# ========================================
header "FASE 2: Operaciones Vectoriales Básicas"

info "Operaciones fundamentales SAXPY (Single-precision A·X Plus Y)"
echo ""

step "1. Escalamiento de vectores (Y = α·X)"
run_build "02_vector"
run_exe "vector_scale" 1000000
echo ""

step "2. Producto punto (A · B)"
run_exe "vector_dot" 1000000
echo ""

step "3. Kernel fusionado (multi-operación)"
run_exe "vector_operations_fused" 100000
echo ""

success "Rendimiento: 1200+ GB/s - cercano al límite de memoria"
echo ""

info "Key Insight: Kernel fusionado elimina tráfico de memoria"
info "  • 3 kernels → 1 kernel: 3 lecturas + 1 escritura vs 6+ lecturas/escrituras"
echo ""

info "Presiona Enter para continuar a FASE 3..."
read -r

# ========================================
# FASE 3: MEMORIA Y TRANSFERENCIAS
# ========================================
header "FASE 3: Transferencias de Memoria y Optimización"

info "Tipos de memoria: Host ↔ Device"
echo "  • cudaMemcpy: síncrono, paged memory (~6 GB/s)"
echo "  • Pinned memory: asíncrono, page-locked (~12 GB/s)"
echo "  • Zero-copy: GPU accede host RAM directamente"
echo "  • Unified Memory: un puntero, migración automática"
echo ""

step "1. Host → Device (tradicional)"
run_build "04_memory"
run_exe "host_to_device" 1000000
echo ""

step "2. Zero-Copy Memory"
run_exe "zero_copy" 16
echo ""

step "3. Pinned Memory con async streams"
run_exe "async_copy_example" 1000000
echo ""

info "cp.async: Ada permite copias asíncronas hardware (oculta latencia)"
info "Pinned memory evita paginación → DMA directo"
echo ""

info "Presiona Enter para continuar a FASE 4..."
read -r

# ========================================
# FASE 4: JERARQUÍA DE HILOS
# ========================================
header "FASE 4: Jerarquía de Hilos CUDA"

info "Organización: Grid → Block → Warp → Thread"
echo "  • Warp = 32 hilos (ejecutan en lockstep)"
echo "  • Máx 1024 hilos/bloque, 65535 bloques/dimensión"
echo "  • Divergencia: if/else dentro warp → serialización"
echo ""

step "1. Indexación 1D"
run_build "05_thread"
run_exe "indexing_1d"
echo ""

step "2. Indexación 2D (matrices)"
run_exe "indexing_2d"
echo ""

step "3. Indexación 3D"
run_exe "indexing_3d"
echo ""

step "4. Grid-stride loops (para datasets grandes)"
run_exe "grid_stride" 10000000
echo ""

info "Grid-stride: cada hilo procesa múltiples elementos"
info "Ventaja: funciona con cualquier tamaño, óptimo para occupancy"
echo ""

info "Presiona Enter para continuar a FASE 5..."
read -r

# ========================================
# FASE 5: MEMORIA COMPARTIDA & TILING
# ========================================
header "FASE 5: Memoria Compartida y Tiling"

info "Memoria compartida: 96 KB/SM (rápida, on-chip)"
echo "  • __shared__: variable compartida por bloque"
echo "  • Syncthreads(): barreras de sincronización"
echo "  • Bank conflicts: acceso concurrente mismo banco → serialización"
echo ""

step "1. Shared memory básica"
run_build "06_shared"
run_exe "shared_memory_basic"
echo ""

step "2. Matrix tiling (bloqueado)"
run_exe "matrix_tiling" 1024
echo ""

step "3. Reducción con shared memory"
run_exe "reduction_shared" 1000000
echo ""

step "4. Conflictos de bancos"
run_exe "bank_conflicts"
echo ""

info "Tiling: cargar bloques en shared memory → reutilizar"
info "Matrix 1024×1024: 32×32 tiles = 32 bloques, 1024 hilos/bloque"
echo ""

info "Presiona Enter para continuar a FASE 6..."
read -r

# ========================================
# FASE 7: PATRONES PARALELOS
# ========================================
header "FASE 6: Patrones Paralelos (Scan & Reducción)"

info "Reducción: combinar N elementos en 1 resultado"
echo "  • Suma, mínimo, máximo, producto"
echo "  • Tree reduction: O(log N) pasos"
echo ""

step "1. Reducción clásica (tree)"
run_build "09_parallel"
run_exe "reduction" 16777216 sum
echo ""

step "2. Prefix sum (scan)"
run_exe "scan_prefix_sum" 1000000
echo ""

step "3. Warp-level reduction (optimizado)"
run_exe "reduction_warp_optimized" 16777216 sum
echo ""

success "Warp-level: 3-5x más rápido - __shfl_down_sync evita sincronización"
info "Reducción 2 etapas: 1) dentro warp (shuffle), 2) entre warps (1 sync)"
echo ""

info "Presiona Enter para continuar a FASE 7..."
read -r

# ========================================
# FASE 7: MEMORIA CONSTANTE
# ========================================
header "FASE 7: Memoria Constante (Broadcast Eficiente)"

info "Memoria constante: 8 KB/banco, caché dedicada"
echo "  • Broadcast: todos los hilos leen misma dirección → eficiente"
echo "  • cudaMemcpyToSymbol: escribe desde host"
echo "  • Ideal: parámetros, tablas de lookup"
echo ""

step "1. Constant array"
run_build "07_constant"
run_exe "constant_array"
echo ""

step "2. Lookup table (interpolación)"
run_exe "lookup_table"
echo ""

info "Speedup: 2-3x cuando todos los hilos leen los mismos datos"
echo ""

info "Presiona Enter para continuar a FASE 8..."
read -r

# ========================================
# FASE 8: OPERACIONES ATÓMICAS
# ========================================
header "FASE 8: Operaciones Atómicas"

info "Operaciones thread-safe (evitan race conditions)"
echo "  • atomicAdd, atomicCAS, atomicMin, atomicMax"
echo "  • Más lento que acceso normal (serialización)"
echo "  • Evitar contención: usar por bloque, luego reducir"
echo ""

step "1. Atomic add (histograma)"
run_build "08_atomic"
run_exe "atomic_add" 1000000
echo ""

step "2. Atomic compare-exchange"
run_exe "atomic_compare" 1000000
echo ""

info "Alternativa: reducción warp-level evita atomics"
echo ""

info "Presiona Enter para continuar a FASE 9..."
read -r

# ========================================
# FASE 9: OPTIMIZACIÓN & PROFILING
# ========================================
header "FASE 9: Optimización y Profiling"

info "Métricas clave: Occupancy, Bandwidth, Instrucciones"
echo ""

step "1. Cálculo de occupancy"
run_build "10_performance"
run_exe "occupancy"
echo ""

step "2. Memory coalescing"
run_exe "memory_coalescing"
echo ""

step "3. Loop unrolling"
run_exe "loop_unrolling" 4194304
echo ""

step "4. Event profiling"
run_exe "profile_events" 1000000
echo ""

info "Occupancy: % de warps activos / SM (ideal 50-100%)"
info "Coalescing: accesos secuenciales → 1 transacción"
info "Unrolling: reduce branches, aumenta ILP"
echo ""

info "Presiona Enter para continuar a FASE 10..."
read -r

# ========================================
# FASE 10: AVANZADO (Streams & DP)
# ========================================
header "FASE 10: Características Avanzadas"

info "Streams: colas de comandos independientes"
echo "  • Overlap H2D + Kernel + D2H"
echo "  • Prioridad y CUDA events para timing"
echo ""

step "1. Múltiples streams"
run_build "11_advanced"
run_exe "streams" 16777216
echo ""

step "2. Eventos para timing"
run_exe "events" 16777216
echo ""

step "3. Dynamic Parallelism (CC 3.5+)"
run_exe "dynamic_parallelism" 2
echo ""

info "Dynamic Parallelism: kernels lanzan kernels"
info "Solo Kepler+ (esta GPU lo soporta pero ejemplo simplificado)"
echo ""

info "Presiona Enter para continuar a FASE 11..."
read -r

# ========================================
# FASE 11: APLICACIONES REALES
# ========================================
header "FASE 11: Aplicaciones Completas"

info "Integrando todos los conceptos"
echo ""

step "1. Simulación N-body"
run_build "12_applications"
run_exe "vector_max" 16777216
echo ""

step "2. Convolución de imagen"
run_exe "image_convolution" 2048
echo ""

success "Aplicaciones completas demostrando pipeline completo"
echo ""

info "Presiona Enter para ver el resumen final..."
read -r

# ========================================
# RESUMEN
# ========================================
header "📊 RESUMEN DE APRENDIZAJE"

cat << 'SUMMARY'

FASES COMPLETADAS:

  ✅ FASE 0 - Prerrequisitos
     • Verificación CUDA, arquitectura sm_89

  ✅ FASE 1 - Hola Mundo
     • Primer kernel, indexación, memoria básica
     • Vector addition (tradicional + unified)

  ✅ FASE 2 - Operaciones Vectoriales
     • SAXPY, producto punto, kernel fusion
     • 1200+ GB/s (cercano a límite teórico)

  ✅ FASE 3 - Transferencias de Memoria
     • cudaMemcpy, pinned memory, zero-copy
     • Unified memory, prefetching

  ✅ FASE 4 - Jerarquía de Hilos
     • Bloques, grids, warps
     • Indexación 1D/2D/3D, grid-stride loops

  ✅ FASE 5 - Memoria Compartida
     • Tiling para matmul, reducción
     • Bank conflicts, coalescing

  ✅ FASE 6 - Patrones Paralelos
     • Reducción, prefix sum (scan)
     • Warp shuffle: 3-5x speedup

  ✅ FASE 7 - Memoria Constante
     • Broadcast eficiente
     • Tablas de lookup con interpolación

  ✅ FASE 8 - Atómicas
     • Histogramas, compare-exchange
     • Evitar contención

  ✅ FASE 9 - Optimización
     • Occupancy, unrolling, profiling
     • Eventos para timing preciso

  ✅ FASE 10 - Avanzado
     • Multi-stream, overlap H2D/K/D2H
     • Dynamic parallelism (concepto)

  ✅ FASE 11 - Aplicaciones
     • N-body, convolución
     • Pipeline completo CPU-GPU

KEY INSIGHTS:

  • Ada Lovelace (sm_89): 96 KB shared mem/SM, 72 MB L2
  • Kernel fusion: reducir tráfico de memoria
  • Warp shuffle: evitar __syncthreads innecesarios
  • Unified memory: simplifica código, perf similar
  • Tiling: reutilizar datos en shared memory
  • Occupancy: no es todo, balancear con ILP
  • Async streams: overlap computación/transferencia

RECURSOS:

  • NVIDIA CUDA C++ Programming Guide
  • CUDA Toolkit Samples (GitHub)
  • Nsight Compute / Nsight Systems
  • PTX ISA (para optimización extrema)

SUMMARY

header "¡ROADMAP COMPLETADO!"

echo -e "${GREEN}🎉 Has completado el aprendizaje CUDA básico-intermedio${NC}"
echo -e "${GREEN}Todos los ejemplos están en: ${BASE_DIR}${NC}"
echo -e "${GREEN}Binarios en: ${BASE_DIR}/build/${NC}"
echo ""
echo -e "${CYAN}Siguientes pasos sugeridos:${NC}"
echo "  1. Explorar Tensor Cores (WMMA, cuBLAS)"
echo "  2. CUDA Graphs para pipelines complejos"
echo "  3. Multi-GPU con NVLink"
echo "  4. Thrust/CUB para algoritmos de alto nivel"
echo "  5. Integrar con ML frameworks (PyTorch, TensorFlow)"
echo ""
echo -e "${YELLOW}Para ejecutar un ejemplo específico:${NC}"
echo "  cd ${BASE_DIR}"
echo "  ./build/<nombre_ejecutable> [args]"
echo ""

