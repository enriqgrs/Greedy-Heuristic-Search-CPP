#!/bin/bash
#  USO:
#    ./ejecutar.sh [opciones]
#
#  OPCIONES (todas opcionales, tienen valores por defecto):
#    -c <n_clavos>          Número de clavos        (default: 300)
#    -H <n_hilos_calculo>   Máx. hilos a calcular   (default: 5000)
#    -s <n_hilos_seleccion> Top-S hilos/iteración   (default: 20)
#    -i <intensidad>        Intensidad por hilo      (default: 17)
#    -d <directorio>        Directorio de imágenes   (default: pruebas/)
#    -v <variantes>         Variantes a ejecutar     (default: todas)
#                           Ej: -v "v_base,v_all"
#    -h                     Mostrar esta ayuda
#
#  EJEMPLO:
#    ./ejecutar.sh -c 200 -H 3000 -s 10 -i 20
#    ./ejecutar.sh -v "v_base,v_all" -d mis_imagenes/
# fin

export LC_ALL=C

# Valores por defecto
N_CLAVOS=300
N_HILOS_CALCULO=5000
N_HILOS_SELECCION=20
INTENSIDAD=17
IMG_DIR="pruebas/"
VARIANTES_ELEGIDAS=""   # vacío = todas

BENCH_SRC="main_bench.cpp"
MSE_TOOL="./calc_mse"
CXX="g++"
CXXFLAGS="-O2 -std=c++17 -pthread"
OUT_DIR="bench_resultados"   # solo imágenes .pgm
BIN_DIR="/tmp/bench_bins"    # ejecutables compilados

# Colores
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# Parseo de argumentos
while getopts "c:H:s:i:d:v:h" opt; do
    case $opt in
        c) N_CLAVOS="$OPTARG" ;;
        H) N_HILOS_CALCULO="$OPTARG" ;;
        s) N_HILOS_SELECCION="$OPTARG" ;;
        i) INTENSIDAD="$OPTARG" ;;
        d) IMG_DIR="$OPTARG" ;;
        v) VARIANTES_ELEGIDAS="$OPTARG" ;;
        h)
            sed -n '/^#  USO:/,/^# fin/p' "$0" | sed 's/^#  \?//'
            exit 0
            ;;
        *) echo "Opción inválida. Usa -h para ayuda." >&2; exit 1 ;;
    esac
done

# Variantes
declare -A VARIANTES
VARIANTES["v_base"]=""
VARIANTES["v_weighted"]="-DUSE_WEIGHTED_NAIL"
VARIANTES["v_heap"]="-DUSE_TOP_S_HEAP"
VARIANTES["v_gradient"]="-DUSE_GRADIENT_MAP"
VARIANTES["v_threads"]="-DUSE_MULTITHREADING -DUSE_TOP_S_HEAP"
VARIANTES["v_all"]="-DUSE_WEIGHTED_NAIL -DUSE_TOP_S_HEAP -DUSE_GRADIENT_MAP -DUSE_MULTITHREADING"

ORDEN_COMPLETO=("v_base" "v_weighted" "v_heap" "v_gradient" "v_threads" "v_all")

declare -A NOMBRES_LEGIBLES
NOMBRES_LEGIBLES["v_base"]="Sin heurísticas (base)"
NOMBRES_LEGIBLES["v_weighted"]="+ Clavo Ponderado"
NOMBRES_LEGIBLES["v_heap"]="+ Min-Heap Top-S"
NOMBRES_LEGIBLES["v_gradient"]="+ Gradiente"
NOMBRES_LEGIBLES["v_threads"]="+ Threads + Heap"
NOMBRES_LEGIBLES["v_all"]="Todas las heurísticas"

# Seleccionar orden de variantes
if [ -n "$VARIANTES_ELEGIDAS" ]; then
    IFS=',' read -ra ORDEN <<< "$VARIANTES_ELEGIDAS"
else
    ORDEN=("${ORDEN_COMPLETO[@]}")
fi

# Verificar variantes válidas
for nombre in "${ORDEN[@]}"; do
    if [ -z "${VARIANTES[$nombre]+x}" ]; then
        echo -e "${RED}Error: variante desconocida '$nombre'.${NC}"
        echo "Variantes disponibles: ${ORDEN_COMPLETO[*]}"
        exit 1
    fi
done

# Imágenes
mapfile -t IMAGENES < <(find "$IMG_DIR" -name '*.pgm' | sort)

if [ ${#IMAGENES[@]} -eq 0 ]; then
    echo -e "${RED}Error: no se encontraron imágenes PGM en '$IMG_DIR'.${NC}"
    exit 1
fi

# Preparar directorio de binarios
mkdir -p "$OUT_DIR"
mkdir -p "$BIN_DIR"

# Compilar calc_mse
echo -e "${BOLD}[1/3] Compilando calc_mse...${NC}"
$CXX $CXXFLAGS -o calc_mse calc_mse.cpp image_loader.cpp
echo -e "  ${GREEN}OK${NC}"

# Compilar variantes
echo -e "\n${BOLD}[2/3] Compilando variantes del benchmark...${NC}"
for nombre in "${ORDEN[@]}"; do
    flags="${VARIANTES[$nombre]}"
    echo -ne "  Compilando ${CYAN}${nombre}${NC} (flags: ${flags:-ninguno})... "
    $CXX $CXXFLAGS $flags -o "$BIN_DIR/$nombre" "$BENCH_SRC" image_loader.cpp
    echo -e "${GREEN}OK${NC}"
done

# Ejecutar benchmark
echo -e "\n${BOLD}[3/3] Ejecutando benchmark...${NC}"
echo -e "  Parámetros: clavos=${N_CLAVOS}, hilos_max=${N_HILOS_CALCULO}, top_s=${N_HILOS_SELECCION}, intensidad=${INTENSIDAD}\n"

for imagen in "${IMAGENES[@]}"; do
    if [ ! -f "$imagen" ]; then
        echo -e "  ${YELLOW}⚠ Imagen no encontrada: $imagen — omitida${NC}"
        continue
    fi

    nombre_img=$(basename "$imagen" .pgm)
    echo -e "  ${BOLD}▶ Imagen: ${nombre_img}${NC}"

    for nombre in "${ORDEN[@]}"; do
        exe="$BIN_DIR/$nombre"
        out_pgm="$OUT_DIR/${nombre}_${nombre_img}.pgm"

        printf "    %-38s" "${NOMBRES_LEGIBLES[$nombre]}..."

        BENCH_OUTPUT=$(timeout 300 "$exe" \
            "$N_CLAVOS" "$N_HILOS_CALCULO" "$N_HILOS_SELECCION" \
            "$imagen" "$out_pgm" "$INTENSIDAD" 2>&1 || echo "TIMEOUT")

        BENCH_LINE=$(echo "$BENCH_OUTPUT" | grep "^BENCH:" || true)

        if [ -z "$BENCH_LINE" ]; then
            echo -e "${RED}TIMEOUT/ERROR${NC}"
            continue
        fi

        tiempo_ms=$(echo "$BENCH_LINE" | sed 's/.*tiempo_ms=\([0-9]*\).*/\1/')
        hilos_d=$(echo "$BENCH_LINE"   | sed 's/.*hilos_dibujados=\([0-9]*\).*/\1/')

        mse_line=$(LC_ALL=C "$MSE_TOOL" "$imagen" "$out_pgm" 2>/dev/null || echo "MSE=0")
        mse_real=$(echo "$mse_line" | sed 's/MSE=\([0-9.]*\).*/\1/')



        printf "${GREEN}%6d ms${NC} | hilos: %4d | MSE: %s\n" \
               "$tiempo_ms" "$hilos_d" "$mse_real"
    done
    echo ""
done

echo -e "${GREEN}${BOLD}Benchmark completado.${NC}"
