#!/bin/bash
# ======================================================================
# profile.sh - Профилирование производительности BPE токенизатора
# ======================================================================
#
# @file profile.sh
# @brief Запуск различных профилировщиков для анализа производительности
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @usage ./profile.sh [options]
#   --tool TOOL     Инструмент профилирования (builtin|perf|callgrind|gprof|flamegraph|all)
#   --target TARGET Цель для профилирования (tokenizer|encode|decode|train)
#   --size SIZE     Размер тестовых данных (small|medium|large|huge)
#   --output DIR    Директория для сохранения отчетов (по умолч. reports)
#   --open          Открыть отчет после генерации
#   --compare       Сравнить до/после оптимизации
#   --help          Показать справку
#
# @example
#   ./profile.sh --tool builtin --target encode --size large
#   ./profile.sh --tool perf --target decode --open
#   ./profile.sh --tool all --compare
#
# ======================================================================

set -e  # Прерывать при ошибке

# ======================================================================
# Цвета для вывода
# ======================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ======================================================================
# Функции для вывода
# ======================================================================
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${MAGENTA}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${MAGENTA}========================================${NC}\n"
}

# ======================================================================
# Парсинг аргументов
# ======================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Значения по умолчанию
TOOL="builtin"
TARGET="encode"
SIZE="medium"
OUTPUT_DIR="$PROJECT_ROOT/reports"
OPEN_REPORT=0
COMPARE=0

# Тестовые данные для разных размеров
declare -A TEST_SIZES
TEST_SIZES["small"]="10000"    # 10KB
TEST_SIZES["medium"]="100000"   # 100KB
TEST_SIZES["large"]="1000000"   # 1MB
TEST_SIZES["huge"]="10000000"   # 10MB

# Обработка аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --tool)
            TOOL="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --open)
            OPEN_REPORT=1
            shift
            ;;
        --compare)
            COMPARE=1
            shift
            ;;
        --help)
            echo "Использование: $0 [options]"
            echo ""
            echo "Опции:"
            echo "  --tool TOOL     Инструмент профилирования:"
            echo "                  builtin   - встроенный SimpleProfiler (по умолчанию)"
            echo "                  perf      - Linux perf"
            echo "                  callgrind - Valgrind Callgrind"
            echo "                  gprof     - GNU gprof"
            echo "                  flamegraph - Flame Graph"
            echo "                  all       - все инструменты"
            echo ""
            echo "  --target TARGET Цель для профилирования:"
            echo "                  tokenizer - полный токенизатор"
            echo "                  encode    - только encode"
            echo "                  decode    - только decode"
            echo "                  train     - обучение"
            echo ""
            echo "  --size SIZE     Размер тестовых данных:"
            echo "                  small  (10KB)"
            echo "                  medium (100KB, по умолчанию)"
            echo "                  large  (1MB)"
            echo "                  huge   (10MB)"
            echo ""
            echo "  --output DIR    Директория для сохранения отчетов"
            echo "  --open          Открыть отчет после генерации"
            echo "  --compare       Сравнить до/после оптимизации"
            echo "  --help          Показать справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --tool builtin --target encode --size large"
            echo "  $0 --tool perf --open"
            echo "  $0 --tool all --compare"
            exit 0
            ;;
        *)
            print_error "Неизвестная опция: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# ======================================================================
# Проверка инструментов
# ======================================================================
print_header "🔧 ПРОФИЛИРОВАНИЕ BPE TOKENIZER"

print_info "Инструмент: $TOOL"
print_info "Цель: $TARGET"
print_info "Размер данных: $SIZE (${TEST_SIZES[$SIZE]} байт)"
print_info "Директория отчетов: $OUTPUT_DIR"

# Создаем директорию для отчетов
mkdir -p "$OUTPUT_DIR"

# Проверка наличия необходимых инструментов
check_tool() {
    if ! command -v $1 &> /dev/null; then
        print_warning "$1 не найден"
        return 1
    else
        print_success "$1 найден"
        return 0
    fi
}

# ======================================================================
# Сборка с флагами профилирования
# ======================================================================
build_with_profiling() {
    print_header "СБОРКА С ПРОФИЛИРОВАНИЕМ"
    
    cd "$PROJECT_ROOT/bpe_cpp/build"
    
    print_info "Конфигурация CMake..."
    
    if [ "$TOOL" == "builtin" ] || [ "$TOOL" == "all" ]; then
        # Для встроенного профайлера
        cmake .. \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DBUILD_EXAMPLES=ON
    else
        # Для внешних профайлеров
        cmake .. \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DBUILD_WITH_PROFILING=ON \
            -DBUILD_EXAMPLES=ON
    fi
    
    print_info "Сборка..."
    make clean
    make -j$(nproc) fast_tokenizer_demo
    
    if [ $? -eq 0 ]; then
        print_success "Сборка завершена"
    else
        print_error "Ошибка сборки"
        exit 1
    fi
}

# ======================================================================
# Генерация тестовых данных
# ======================================================================
generate_test_data() {
    local size=${TEST_SIZES[$SIZE]}
    local test_file="$OUTPUT_DIR/test_data_${SIZE}.txt"
    
    print_info "Генерация тестовых данных (${size} байт)..."
    
    # Создаем C++ код нужного размера
    python3 -c "
import sys
import os

# Базовый C++ код
code = '''#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

class Test {
public:
    void process() {
        for (int i = 0; i < 1000; ++i) {
            data.push_back(i);
        }
    }
private:
    std::vector<int> data;
};

int main() {
    Test t;
    t.process();
    return 0;
}
'''

target_size = $size
result = code
while len(result.encode('utf-8')) < target_size:
    result += '\\n// ' + 'x' * 100

with open('$test_file', 'w') as f:
    f.write(result[:target_size])
"
    
    echo "$test_file"
}

# ======================================================================
# Встроенный профайлер (SimpleProfiler)
# ======================================================================
profile_with_builtin() {
    print_header "ВСТРОЕННЫЙ ПРОФАЙЛЕР (SimpleProfiler)"
    
    local test_file="$1"
    local output="$OUTPUT_DIR/builtin_${TARGET}_${SIZE}"
    
    print_info "Запуск с встроенным профайлером..."
    
    # Копируем тестовый файл в нужное место
    cp "$test_file" ./test_input.txt
    
    # Запускаем демо с флагом профилирования
    ./examples/fast_tokenizer_demo --profile --input ./test_input.txt
    
    # Копируем отчет
    if [ -f "profiler_report.txt" ]; then
        cp profiler_report.txt "${output}.txt"
        print_success "Отчет сохранен: ${output}.txt"
        
        # Показываем топ-10 самых медленных операций
        echo -e "\n🔍 Топ-10 самых медленных операций:"
        grep -A 15 "ОТЧЕТ ПРОФИЛИРОВАНИЯ" "${output}.txt" | head -20
    else
        print_error "Отчет не создан"
    fi
    
    # Очистка
    rm -f ./test_input.txt
}

# ======================================================================
# Профилирование с perf
# ======================================================================
profile_with_perf() {
    print_header "ПРОФИЛИРОВАНИЕ С PERF"
    
    check_tool "perf" || return 1
    
    local test_file="$1"
    local output="$OUTPUT_DIR/perf_${TARGET}_${SIZE}"
    
    print_info "Запуск perf record..."
    perf record -g \
        -o "${output}.data" \
        ./examples/fast_tokenizer_demo "$(cat "$test_file")"
    
    print_info "Генерация отчета..."
    perf report -i "${output}.data" \
        --stdio \
        --sort=dso,symbol \
        --no-children > "${output}_report.txt"
    
    perf report -i "${output}.data" \
        --graph \
        --no-children > "${output}_graph.txt"
    
    print_success "Perf отчеты сохранены"
    
    # Показываем топ-10 функций
    echo -e "\n🔍 Топ-10 функций (по времени):"
    head -20 "${output}_report.txt" | grep -E "^[ ]+[0-9.]+%"
}

# ======================================================================
# Профилирование с Callgrind
# ======================================================================
profile_with_callgrind() {
    print_header "ПРОФИЛИРОВАНИЕ С CALLGRIND"
    
    check_tool "valgrind" || return 1
    
    local test_file="$1"
    local output="$OUTPUT_DIR/callgrind_${TARGET}_${SIZE}"
    
    print_info "Запуск callgrind (может быть медленно)..."
    valgrind --tool=callgrind \
        --dump-instr=yes \
        --collect-jumps=yes \
        --callgrind-out-file="${output}.out" \
        ./examples/fast_tokenizer_demo "$(cat "$test_file")"
    
    print_success "Callgrind отчет: ${output}.out"
    print_info "Для просмотра: kcachegrind ${output}.out"
}

# ======================================================================
# Профилирование с gprof
# ======================================================================
profile_with_gprof() {
    print_header "ПРОФИЛИРОВАНИЕ С GPROF"
    
    check_tool "gprof" || return 1
    
    local test_file="$1"
    local output="$OUTPUT_DIR/gprof_${TARGET}_${SIZE}"
    
    print_info "Запуск с gmon.out генерацией..."
    ./examples/fast_tokenizer_demo "$(cat "$test_file")"
    
    if [ -f "gmon.out" ]; then
        gprof ./examples/fast_tokenizer_demo gmon.out > "${output}.txt"
        rm gmon.out
        print_success "gprof отчет: ${output}.txt"
        
        # Показываем топ-10 функций
        echo -e "\n🔍 Топ-10 функций (по времени):"
        grep -A 10 "index % time" "${output}.txt" | head -15
    else
        print_error "gmon.out не создан"
    fi
}

# ======================================================================
# Генерация flame graph
# ======================================================================
generate_flamegraph() {
    print_header "ГЕНЕРАЦИЯ FLAME GRAPH"
    
    check_tool "perf" || return 1
    
    local test_file="$1"
    local output="$OUTPUT_DIR/flame_${TARGET}_${SIZE}"
    
    # Проверка наличия FlameGraph
    FLAMEGRAPH_DIR="$PROJECT_ROOT/third_party/FlameGraph"
    if [ ! -d "$FLAMEGRAPH_DIR" ]; then
        print_info "Загрузка FlameGraph..."
        mkdir -p "$PROJECT_ROOT/third_party"
        git clone https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
    fi
    
    print_info "Сбор данных perf..."
    perf record -F 99 -g \
        -o "${output}.data" \
        ./examples/fast_tokenizer_demo "$(cat "$test_file")"
    
    print_info "Генерация flamegraph..."
    perf script -i "${output}.data" > "${output}.perf"
    "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" "${output}.perf" > "${output}.folded"
    "$FLAMEGRAPH_DIR/flamegraph.pl" "${output}.folded" > "${output}.svg"
    
    print_success "FlameGraph: ${output}.svg"
    
    if [ $OPEN_REPORT -eq 1 ] && command -v firefox &> /dev/null; then
        firefox "${output}.svg"
    fi
}

# ======================================================================
# Сравнение до/после оптимизации
# ======================================================================
compare_before_after() {
    print_header "СРАВНЕНИЕ ДО/ПОСЛЕ ОПТИМИЗАЦИИ"
    
    local output="$OUTPUT_DIR/comparison_${TARGET}_${SIZE}"
    
    # Ищем предыдущие отчеты
    local before_report=$(ls -t "$OUTPUT_DIR"/builtin_* 2>/dev/null | head -2 | tail -1)
    local after_report=$(ls -t "$OUTPUT_DIR"/builtin_* 2>/dev/null | head -1)
    
    if [ -z "$before_report" ] || [ -z "$after_report" ]; then
        print_warning "Недостаточно данных для сравнения"
        print_info "Запустите профилирование дважды: до и после оптимизации"
        return
    fi
    
    print_info "Сравнение:"
    echo "  До:    $(basename "$before_report")"
    echo "  После: $(basename "$after_report")"
    
    # Извлекаем общее время из отчетов
    local before_time=$(grep "Общее время" "$before_report" | awk '{print $3}')
    local after_time=$(grep "Общее время" "$after_report" | awk '{print $3}')
    
    if [ -n "$before_time" ] && [ -n "$after_time" ]; then
        local speedup=$(echo "scale=2; $before_time / $after_time" | bc)
        echo -e "\nУскорение: ${speedup}x"
        
        if (( $(echo "$speedup > 5" | bc -l) )); then
            echo "Цель достигнута ( >5x )"
        else
            echo "Нужно больше оптимизаций"
        fi
    fi
    
    # Сравнение топ-5 функций
    echo -e "\nТоп-5 функций ДО:"
    grep -A 7 "ОТЧЕТ ПРОФИЛИРОВАНИЯ" "$before_report" | tail -n +3 | head -5
    
    echo -e "\nТоп-5 функций ПОСЛЕ:"
    grep -A 7 "ОТЧЕТ ПРОФИЛИРОВАНИЯ" "$after_report" | tail -n +3 | head -5
}

# ======================================================================
# Основная логика
# ======================================================================

# Сборка
build_with_profiling

# Генерация тестовых данных
TEST_FILE=$(generate_test_data)

# Запуск профилирования в зависимости от выбранного инструмента
case $TOOL in
    builtin)
        profile_with_builtin "$TEST_FILE"
        ;;
    perf)
        profile_with_perf "$TEST_FILE"
        ;;
    callgrind)
        profile_with_callgrind "$TEST_FILE"
        ;;
    gprof)
        profile_with_gprof "$TEST_FILE"
        ;;
    flamegraph)
        generate_flamegraph "$TEST_FILE"
        ;;
    all)
        profile_with_builtin "$TEST_FILE"
        profile_with_perf "$TEST_FILE"
        profile_with_callgrind "$TEST_FILE"
        profile_with_gprof "$TEST_FILE"
        generate_flamegraph "$TEST_FILE"
        ;;
    *)
        print_error "Неизвестный инструмент: $TOOL"
        exit 1
        ;;
esac

# Сравнение если нужно
if [ $COMPARE -eq 1 ]; then
    compare_before_after
fi

# ======================================================================
# Итог
# ======================================================================
print_header "ПРОФИЛИРОВАНИЕ ЗАВЕРШЕНО"

print_info "Отчеты сохранены в: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" | grep -E "(builtin|perf|callgrind|gprof|flame)" | tail -5

echo ""
print_success "Готово!"