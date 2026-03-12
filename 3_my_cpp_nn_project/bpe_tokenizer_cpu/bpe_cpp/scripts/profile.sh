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
# @version 3.3.0
#
# @details Этот скрипт предоставляет единый интерфейс для запуска различных
#          инструментов профилирования, позволяя анализировать производительность
#          BPE токенизатора на разных уровнях:
#
#          **Поддерживаемые инструменты:**
#           - builtin       - встроенный SimpleProfiler (статистика операций)
#           - perf          - Linux perf (анализ на уровне процессора)
#           - callgrind     - Valgrind Callgrind (симуляция кэша)
#           - gprof         - GNU gprof (профилирование функций)
#           - flamegraph    - генерация Flame Graphs (визуализация)
#           - all           - все инструменты последовательно
#
#          **Цели профилирования:**
#          - tokenizer    - полный цикл токенизации
#          - encode       - только операция кодирования
#          - decode       - только операция декодирования
#          - train        - обучение токенизатора
#
#          **Размеры тестовых данных:**
#          - small     - 10 КБ  (быстрое тестирование)
#          - medium    - 100 КБ (баланс)
#          - large     - 1 МБ   (реалистичная нагрузка)
#          - huge      - 10 МБ  (стресс-тест)
#
# @usage ./profile.sh [options]
#   --tool TOOL        Инструмент профилирования (builtin|perf|callgrind|gprof|flamegraph|all)
#   --target TARGET    Цель для профилирования (tokenizer|encode|decode|train)
#   --size SIZE        Размер тестовых данных (small|medium|large|huge)
#   --output DIR       Директория для сохранения отчетов (по умолч. reports)
#   --open             Открыть отчет после генерации
#   --compare          Сравнить до/после оптимизации
#   --no-build         Пропустить сборку (использовать существующие бинарники)
#   --clean            Очистить перед сборкой
#   --verbose          Подробный вывод
#   --help             Показать справку
#
# @example
#   ./profile.sh --tool builtin --target encode --size large
#   ./profile.sh --tool perf --target decode --open
#   ./profile.sh --tool all --compare
#   ./profile.sh --tool flamegraph --size huge --open
#
# ======================================================================

set -euo pipefail  # Прерывать при ошибке, неопределенных переменных и сбоях в пайпах

# ======================================================================
# Цвета для красивого вывода
# ======================================================================
RED='\033[0;31m'        # Красный      - ошибки
GREEN='\033[0;32m'      # Зеленый      - успех
YELLOW='\033[1;33m'     # Желтый       - предупреждения
BLUE='\033[0;34m'       # Синий        - информация
MAGENTA='\033[0;35m'    # Пурпурный    - заголовки
CYAN='\033[0;36m'       # Голубой      - команды
NC='\033[0m'            # No Color     - сброс цвета

# ======================================================================
# Функции для форматированного вывода
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
    echo -e "\n${MAGENTA}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${MAGENTA}════════════════════════════════════════════════════════════${NC}\n"
}

print_command() {
    echo -e "${CYAN}  > $1${NC}"
}

# ======================================================================
# Парсинг аргументов
# ======================================================================

# Определение путей (скрипт может быть запущен из любой директории)
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
NO_BUILD=0
CLEAN_BUILD=0
VERBOSE=0

# Ассоциативный массив с размерами тестовых данных (в байтах)
declare -A TEST_SIZES
TEST_SIZES["small"]="10000"      # 10 КБ
TEST_SIZES["medium"]="100000"    # 100 КБ
TEST_SIZES["large"]="1000000"    # 1 МБ
TEST_SIZES["huge"]="10000000"    # 10 МБ

# Обработка аргументов командной строки
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
        --no-build)
            NO_BUILD=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            echo "Использование: $0 [options]"
            echo ""
            echo "Опции:"
            echo "  --tool TOOL     Инструмент профилирования:"
            echo "                  builtin       - встроенный SimpleProfiler (по умолчанию)"
            echo "                  perf          - Linux perf"
            echo "                  callgrind     - Valgrind Callgrind"
            echo "                  gprof         - GNU gprof"
            echo "                  flamegraph    - Flame Graph"
            echo "                  all           - все инструменты"
            echo ""
            echo "  --target TARGET Цель для профилирования:"
            echo "                  tokenizer    - полный токенизатор"
            echo "                  encode       - только encode"
            echo "                  decode       - только decode"
            echo "                  train        - обучение"
            echo ""
            echo "  --size SIZE     Размер тестовых данных:"
            echo "                  small     - 10 КБ"
            echo "                  medium    - 100 КБ (по умолчанию)"
            echo "                  large     - 1 МБ"
            echo "                  huge      - 10 МБ"
            echo ""
            echo "  --output DIR    Директория для сохранения отчетов"
            echo "  --open          Открыть отчет после генерации"
            echo "  --compare       Сравнить до/после оптимизации"
            echo "  --no-build      Пропустить сборку (использовать существующие бинарники)"
            echo "  --clean         Очистить перед сборкой"
            echo "  --verbose       Подробный вывод"
            echo "  --help          Показать справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --tool builtin --target encode --size large"
            echo "  $0 --tool perf --open"
            echo "  $0 --tool all --compare"
            echo "  $0 --tool flamegraph --size huge --open"
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
print_header "ПРОФИЛИРОВАНИЕ BPE TOKENIZER"

print_info "Инструмент: $TOOL"
print_info "Цель: $TARGET"
print_info "Размер данных: $SIZE (${TEST_SIZES[$SIZE]} байт)"
print_info "Директория отчетов: $OUTPUT_DIR"

# Создаем директорию для отчетов (если не существует)
mkdir -p "$OUTPUT_DIR"

# Функция проверки наличия инструмента
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        print_warning "$1 не найден"
        return 1
    else
        if [ $VERBOSE -eq 1 ]; then
            print_success "$1 найден: $(which "$1")"
        fi
        return 0
    fi
}

# ======================================================================
# Сборка с флагами профилирования
# ======================================================================
build_with_profiling() {
    if [ $NO_BUILD -eq 1 ]; then
        print_info "Сборка пропущена (--no-build)"
        return 0
    fi
    
    print_header "СБОРКА С ПРОФИЛИРОВАНИЕМ"
    
    # Проверяем существование директории build
    if [ ! -d "$PROJECT_ROOT/build" ]; then
        mkdir -p "$PROJECT_ROOT/build"
    fi
    
    cd "$PROJECT_ROOT/build"
    
    if [ $CLEAN_BUILD -eq 1 ]; then
        print_info "Очистка предыдущей сборки..."
        rm -rf *
    fi
    
    print_info "Конфигурация CMake..."
    
    local cmake_options=(
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
        "-DBUILD_EXAMPLES=ON"
        "-DBUILD_BENCHMARKS=ON"
    )
    
    if [ "$TOOL" != "builtin" ] && [ "$TOOL" != "all" ]; then
        # Для внешних профайлеров
        cmake_options+=("-DBUILD_WITH_PROFILING=ON")
    fi
    
    if [ $VERBOSE -eq 1 ]; then
        cmake_options+=("--log-level=VERBOSE")
    fi
    
    print_command "cmake .. ${cmake_options[*]}"
    cmake .. "${cmake_options[@]}"
    
    print_info "Сборка..."
    
    # Определяем цель сборки в зависимости от TARGET
    local build_target="fast_tokenizer_demo"
    if [ "$TARGET" == "train" ]; then
        build_target="train_example"
    fi
    
    print_command "make -j$(nproc) $build_target"
    make -j$(nproc) "$build_target"
    
    if [ $? -eq 0 ]; then
        print_success "Сборка завершена!"
    else
        print_error "Ошибка сборки!"
        exit 1
    fi
}

# ======================================================================
# Генерация тестовых данных
# ======================================================================
generate_test_data() {
    local size=${TEST_SIZES[$SIZE]}
    local test_file="$OUTPUT_DIR/test_data_${SIZE}_$(date +%Y%m%d_%H%M%S).txt"
    
    print_info "Генерация тестовых данных (${size} байт)..."
    
    # Создаем C++ код нужного размера через Python
    python3 -c "
import sys

# Базовый C++ код
code = '''#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

class Test {
private:
    std::vector<int> data;
    std::string name;
    
public:
    Test(const std::string& n = \"test\") : name(n) {}
    
    void process() {
        for (int i = 0; i < 1000; ++i) {
            data.push_back(i * i);
        }
    }
    
    double calculate() {
        double sum = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            sum += data[i];
        }
        return data.empty() ? 0 : sum / data.size();
    }
};

int main() {
    Test t;
    t.process();
    std::cout << \"Result: \" << t.calculate() << std::endl;
    return 0;
}
'''

target_size = $size
result = code
multiplier = 1

while len(result.encode('utf-8')) < target_size:
    multiplier += 1
    modified = code.replace('Test', f'Test{multiplier}')
    result += f'\\n// Block {multiplier}\\n' + modified

result = result[:target_size]

with open('$test_file', 'w') as f:
    f.write(result)

print('$test_file')
" | tail -n1
    
    # Проверяем, что файл создан
    if [ -f "$test_file" ]; then
        print_success "Тестовые данные созданы: $test_file"
        echo "  размер: $(wc -c < "$test_file") байт"
        echo "  строк: $(wc -l < "$test_file")"
    else
        print_error "Ошибка создания тестовых данных!"
        exit 1
    fi
    
    # Возвращаем путь к файлу
    echo "$test_file"
}

# ======================================================================
# Встроенный профайлер (SimpleProfiler)
# ======================================================================
profile_with_builtin() {
    print_header "ВСТРОЕННЫЙ ПРОФАЙЛЕР (SimpleProfiler)"
    
    local test_file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output="$OUTPUT_DIR/builtin_${TARGET}_${SIZE}_${timestamp}"
    
    print_info "Запуск с встроенным профайлером..."
    
    # Определяем исполняемый файл
    local executable="./examples/fast_tokenizer_demo"
    if [ "$TARGET" == "train" ]; then
        executable="./examples/train_example"
    fi
    
    # Копируем тестовый файл в рабочую директорию
    cp "$test_file" ./test_input.txt
    
    # Запускаем демо с флагом профилирования
    print_command "$executable --profile --input ./test_input.txt"
    $executable --profile --input ./test_input.txt 2>&1 | tee "${output}.log"
    
    # Копируем отчет
    if [ -f "profiler_report.txt" ]; then
        cp profiler_report.txt "${output}.txt"
        print_success "Отчет сохранен: ${output}.txt"
        
        # Показываем топ-10 самых медленных операций
        echo -e "\nТоп-10 самых медленных операций:"
        echo "----------------------------------------"
        grep -A 15 "ОТЧЕТ ПРОФИЛИРОВАНИЯ" "${output}.txt" | head -20 | sed 's/^/  /'
        
        # Открываем если нужно
        if [ $OPEN_REPORT -eq 1 ]; then
            less "${output}.txt"
        fi
    else
        print_warning "Отчет не создан!"
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
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output="$OUTPUT_DIR/perf_${TARGET}_${SIZE}_${timestamp}"
    
    # Определяем исполняемый файл
    local executable="./examples/fast_tokenizer_demo"
    if [ "$TARGET" == "train" ]; then
        executable="./examples/train_example"
    fi
    
    print_info "Запуск perf record..."
    print_command "perf record -g -o ${output}.data $executable < $test_file"
    perf record -g \
        -o "${output}.data" \
        "$executable" < "$test_file"
    
    print_info "Генерация отчета..."
    print_command "perf report -i ${output}.data --stdio --sort=dso,symbol --no-children > ${output}_report.txt"
    perf report -i "${output}.data" \
        --stdio \
        --sort=dso,symbol \
        --no-children > "${output}_report.txt"
    
    print_command "perf report -i ${output}.data --graph --no-children > ${output}_graph.txt"
    perf report -i "${output}.data" \
        --graph \
        --no-children > "${output}_graph.txt"
    
    print_success "Perf отчеты сохранены:"
    echo "  Файл: ${output}_report.txt"
    echo "  Файл: ${output}_graph.txt"
    
    # Показываем топ-10 функций
    echo -e "\nТоп-10 функций (по времени):"
    echo "----------------------------------------"
    grep -E "^[ ]+[0-9.]+%" "${output}_report.txt" | head -10 | sed 's/^/  /'
}

# ======================================================================
# Профилирование с Callgrind
# ======================================================================
profile_with_callgrind() {
    print_header "ПРОФИЛИРОВАНИЕ С CALLGRIND"
    
    check_tool "valgrind" || return 1
    
    local test_file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output="$OUTPUT_DIR/callgrind_${TARGET}_${SIZE}_${timestamp}"
    
    # Определяем исполняемый файл
    local executable="./examples/fast_tokenizer_demo"
    if [ "$TARGET" == "train" ]; then
        executable="./examples/train_example"
    fi
    
    print_info "Запуск callgrind (может быть медленно)..."
    print_command "valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes --callgrind-out-file=${output}.out $executable < $test_file"
    valgrind --tool=callgrind \
        --dump-instr=yes \
        --collect-jumps=yes \
        --callgrind-out-file="${output}.out" \
        "$executable" < "$test_file" 2>&1 | tee "${output}.log"
    
    print_success "Callgrind отчет: ${output}.out"
    echo "  Файл: ${output}.out"
    echo ""
    echo "  Для просмотра используйте:"
    echo "    kcachegrind ${output}.out"
    echo "    qcachegrind ${output}.out"
}

# ======================================================================
# Профилирование с gprof
# ======================================================================
profile_with_gprof() {
    print_header "ПРОФИЛИРОВАНИЕ С GPROF"
    
    check_tool "gprof" || return 1
    
    local test_file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output="$OUTPUT_DIR/gprof_${TARGET}_${SIZE}_${timestamp}"
    
    # Определяем исполняемый файл
    local executable="./examples/fast_tokenizer_demo"
    if [ "$TARGET" == "train" ]; then
        executable="./examples/train_example"
    fi
    
    print_info "Запуск с gmon.out генерацией..."
    print_command "$executable < $test_file"
    "$executable" < "$test_file"
    
    if [ -f "gmon.out" ]; then
        print_command "gprof $executable gmon.out > ${output}.txt"
        gprof "$executable" gmon.out > "${output}.txt"
        rm gmon.out
        print_success "gprof отчет: ${output}.txt"
        
        # Показываем топ-10 функций
        echo -e "\nТоп-10 функций (по времени):"
        echo "----------------------------------------"
        grep -A 15 "index % time" "${output}.txt" | grep -E "^\[[0-9]+\]" | head -10 | sed 's/^/  /'
    else
        print_error "gmon.out не создан!"
    fi
}

# ======================================================================
# Генерация flame graph
# ======================================================================
generate_flamegraph() {
    print_header "ГЕНЕРАЦИЯ FLAME GRAPH"
    
    check_tool "perf" || return 1
    
    local test_file="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output="$OUTPUT_DIR/flame_${TARGET}_${SIZE}_${timestamp}"
    
    # Определяем исполняемый файл
    local executable="./examples/fast_tokenizer_demo"
    if [ "$TARGET" == "train" ]; then
        executable="./examples/train_example"
    fi
    
    # Проверка наличия FlameGraph
    FLAMEGRAPH_DIR="$PROJECT_ROOT/third_party/FlameGraph"
    if [ ! -d "$FLAMEGRAPH_DIR" ]; then
        print_info "Загрузка FlameGraph..."
        mkdir -p "$PROJECT_ROOT/third_party"
        if [ -d "$PROJECT_ROOT/third_party" ]; then
            git clone https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
        else
            print_error "Не удалось создать директорию third_party"
            return 1
        fi
    fi
    
    if [ ! -f "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" ] || [ ! -f "$FLAMEGRAPH_DIR/flamegraph.pl" ]; then
        print_error "FlameGraph скрипты не найдены в $FLAMEGRAPH_DIR"
        return 1
    fi
    
    print_info "Сбор данных perf..."
    print_command "perf record -F 99 -g -o ${output}.data $executable < $test_file"
    perf record -F 99 -g \
        -o "${output}.data" \
        "$executable" < "$test_file"
    
    print_info "Генерация flamegraph..."
    print_command "perf script -i ${output}.data > ${output}.perf"
    perf script -i "${output}.data" > "${output}.perf"
    
    print_command "$FLAMEGRAPH_DIR/stackcollapse-perf.pl ${output}.perf > ${output}.folded"
    "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" "${output}.perf" > "${output}.folded"
    
    print_command "$FLAMEGRAPH_DIR/flamegraph.pl ${output}.folded > ${output}.svg"
    "$FLAMEGRAPH_DIR/flamegraph.pl" "${output}.folded" > "${output}.svg"
    
    print_success "FlameGraph создан: ${output}.svg"
    echo "  Файл: ${output}.svg"
    
    if [ $OPEN_REPORT -eq 1 ]; then
        if command -v firefox &> /dev/null; then
            firefox "${output}.svg" &
        elif command -v google-chrome &> /dev/null; then
            google-chrome "${output}.svg" &
        elif command -v open &> /dev/null; then
            open "${output}.svg"
        else
            print_warning "Не удалось открыть браузер!"
        fi
    fi
    
    # Очистка промежуточных файлов
    rm -f "${output}.perf" "${output}.folded"
}

# ======================================================================
# Сравнение до/после оптимизации
# ======================================================================
compare_before_after() {
    print_header "СРАВНЕНИЕ ДО/ПОСЛЕ ОПТИМИЗАЦИИ"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output="$OUTPUT_DIR/comparison_${TARGET}_${SIZE}_${timestamp}.txt"
    
    # Ищем предыдущие отчеты builtin
    local reports=($(ls -t "$OUTPUT_DIR"/builtin_*_*.txt 2>/dev/null))
    
    if [ ${#reports[@]} -lt 2 ]; then
        print_warning "Недостаточно данных для сравнения! (нужно минимум 2 отчета)"
        print_info "Запустите профилирование дважды: до и после оптимизации"
        print_info "Например:"
        echo "  ./profile.sh --tool builtin --target $TARGET --size $SIZE  # до"
        echo "  # ... внесите оптимизации ..."
        echo "  ./profile.sh --tool builtin --target $TARGET --size $SIZE  # после"
        return
    fi
    
    local before_report="${reports[1]}"
    local after_report="${reports[0]}"
    
    print_info "Сравнение:"
    echo "  до:    $(basename "$before_report")"
    echo "  после: $(basename "$after_report")"
    
    # Извлекаем общее время из отчетов
    local before_time=$(grep "Общее время" "$before_report" | awk '{print $3}')
    local after_time=$(grep "Общее время" "$after_report" | awk '{print $3}')
    
    if [ -n "$before_time" ] && [ -n "$after_time" ]; then
        local speedup=$(echo "scale=2; $before_time / $after_time" | bc 2>/dev/null || echo "0")
        
        # Сохраняем сравнение в файл
        {
            echo "СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ"
            echo "================================"
            echo "Дата: $(date)"
            echo "Цель: $TARGET"
            echo "Размер: $SIZE"
            echo ""
            echo "До:    $(basename "$before_report")"
            echo "После: $(basename "$after_report")"
            echo ""
            echo "Время до:    $before_time мс"
            echo "Время после: $after_time мс"
            echo "Ускорение:   ${speedup}x"
            echo ""
            echo "Топ-5 функций ДО:"
            grep -A 7 "ОТЧЕТ ПРОФИЛИРОВАНИЯ" "$before_report" | tail -n +3 | head -5 | sed 's/^/  /'
            echo ""
            echo "Топ-5 функций ПОСЛЕ:"
            grep -A 7 "ОТЧЕТ ПРОФИЛИРОВАНИЯ" "$after_report" | tail -n +3 | head -5 | sed 's/^/  /'
        } > "$output"
        
        print_success "Сравнение сохранено: $output"
        
        if (( $(echo "$speedup > 5.0" | bc -l 2>/dev/null) )); then
            echo "  ✅ Цель достигнута (>5x)"
        elif (( $(echo "$speedup > 3.0" | bc -l 2>/dev/null) )); then
            echo "  ⚠ Хорошо, но можно лучше (3-5x)"
        elif (( $(echo "$speedup > 1.0" | bc -l 2>/dev/null) )); then
            echo "  ⚠ Слабое ускорение (1-3x)"
        else
            echo "  ❌ Ускорение отсутствует или отрицательное!"
        fi
    else
        print_error "Не удалось извлечь время из отчетов!"
    fi
}

# ======================================================================
# Основная логика
# ======================================================================

# Переходим в директорию build
cd "$PROJECT_ROOT/build"

# Сборка (если нужно)
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
print_header "ПРОФИЛИРОВАНИЕ ЗАВЕРШЕНО!"

print_info "Отчеты сохранены в: $OUTPUT_DIR"
echo ""
echo "Последние отчеты:"
ls -la "$OUTPUT_DIR" | grep -E "(builtin|perf|callgrind|gprof|flame)" | tail -5 | sed 's/^/  /'

echo ""
print_success "Готово!"