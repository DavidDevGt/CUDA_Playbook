# ============================================================
# CUDA Learning Repository - Main Makefile
# ============================================================
# Este Makefile permite construir todos los ejemplos organizados
# por lecciones. Cada subdirectorio tiene su propio Makefile.
#
# Uso:
#   make all        # Compilar todos los ejemplos
#   make 00_check   # Compilar lección 00
#   make clean      # Limpiar todo
#   make help       # Mostrar ayuda
# ============================================================

# NVCC compiler
NVCC = nvcc

# Flags por defecto (pueden sobreescribirse desde línea de comandos)
# Use -arch=sm_89 for auto-detection (works with RTX 5070 Ti CC 12.0)
NVCCFLAGS = -O2 -arch=sm_89

# Directorios
BUILD_DIR = build

# Crear build dir si no existe
$(shell mkdir -p $(BUILD_DIR))

# Lecciones en orden
LESSONS = \
    00_prerequisites \
    01_hello_world \
    02_vector_operations \
    03_matrix_operations \
    04_memory_transfers \
    05_thread_hierarchy \
    06_shared_memory \
    07_constant_memory \
    08_atomic_operations \
    09_parallel_patterns \
    10_performance \
    11_advanced \
    12_applications

# Targets por lección (se delegan a los Makefiles locales)
.PHONY: all clean help $(LESSONS)

# Construir todos los targets de todas las lecciones
all: $(LESSONS)
	@echo ""
	@echo "✅ Todos los ejemplos compilados en $(BUILD_DIR)/"
	@echo "   Ejecuta: ./build/00_check para verificar CUDA"

# Regla general: cada lección se construye con su propio Makefile
$(LESSONS):
	@echo "=== Compilando lección $@ ==="
	$(MAKE) -C $@ NVCC="$(NVCC)" NVCCFLAGS="$(NVCCFLAGS)" BUILD_DIR="../$(BUILD_DIR)"
	@echo ""

# Limpiar todo
clean:
	@echo "Limpiando todos los archivos compilados..."
	rm -rf $(BUILD_DIR)/*
	@for dir in $(LESSONS); do \
		$(MAKE) -C $$dir clean BUILD_DIR="../$(BUILD_DIR)" || true; \
	done
	@echo "Limpieza completada."

# Mostrar ayuda
help:
	@echo "=========================================="
	@echo "  CUDA Learning Repository - Makefile"
	@echo "=========================================="
	@echo ""
	@echo "Targets principales:"
	@echo "  make all              - Compilar TODOS los ejemplos"
	@echo "  make clean            - Eliminar archivos compilados"
	@echo "  make help             - Mostrar esta ayuda"
	@echo ""
	@echo "Targets por lección (ejemplo):"
	@echo "  make 00_check         - Verificar instalación CUDA"
	@echo "  make 01_hello         - Hola Mundo"
	@echo "  make 02_vector        - Operaciones con vectores"
	@echo "  make 03_matrix        - Operaciones con matrices"
	@echo "  make 04_memory        - Transferencias de memoria"
	@echo "  make 05_thread        - Jerarquía de hilos"
	@echo "  make 06_shared        - Memoria compartida"
	@echo "  make 07_constant      - Memoria constante"
	@echo "  make 08_atomic        - Operaciones atómicas"
	@echo "  make 09_parallel      - Patrones paralelos"
	@echo "  make 10_performance   - Optimización y profiling"
	@echo "  make 11_advanced      - Topics avanzados"
	@echo "  make 12_applications  - Aplicaciones completas"
	@echo ""
	@echo "Variables de entorno:"
	@echo "  NVCCFLAGS='-O3 -arch=sm_80'  # Cambiar optimización/arquitectura"
	@echo ""
	@echo "Ejemplos:"
	@echo "  make all NVCCFLAGS='-O3 -arch=sm_80'"
	@echo "  make 06_shared"
	@echo ""
	@echo "Todos los binarios se guardan en: ./build/"

# Atajos (alias) conveniente
00_check: 00_prerequisites
01_hello: 01_hello_world
02_vector: 02_vector_operations
03_matrix: 03_matrix_operations
04_memory: 04_memory_transfers
05_thread: 05_thread_hierarchy
06_shared: 06_shared_memory
07_constant: 07_constant_memory
08_atomic: 08_atomic_operations
09_parallel: 09_parallel_patterns
10_performance: 10_performance
11_advanced: 11_advanced
12_applications: 12_applications

.PHONY: 00_check 01_hello 02_vector 03_matrix 04_memory 05_thread \
         06_shared 07_constant 08_atomic 09_parallel \
         10_performance 11_advanced 12_applications
