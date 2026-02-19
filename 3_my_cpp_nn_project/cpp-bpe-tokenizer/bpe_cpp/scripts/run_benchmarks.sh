#!/bin/bash
# ======================================================================
# run_benchmarks.sh - Запуск бенчмарков BPE токенизатора
# ======================================================================
#
# @file run_benchmarks.sh
# @brief Запуск всех бенчмарков и анализ производительности
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @usage ./run_benchmarks.sh [options]
#   --type TYPE      Тип бенчмарка (all|fast|original|comparison|parallel)
#   --iterations N   Количество итераций для каждого бенчмарка
#   --format FORMAT  Формат вывода (console|json|csv|plot)
#   --output FILE    Сохранить результаты в файл
#   --compare        Сравнить с предыдущими результатами
#   --html           Сгенерировать HTML отчет
#   --help           Показать справку
#
# @example
#   ./run_benchmarks.sh --type all --format plot
#   ./run_benchmarks.sh --type fast --iterations 10 --output results.json
#   ./run_benchmarks.sh --type comparison --html --compare
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
BENCH_TYPE="all"
ITERATIONS=""
FORMAT="console"
OUTPUT_FILE=""
COMPARE=0
GEN_HTML=0
VERBOSE=0

# Обработка аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            BENCH_TYPE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --compare)
            COMPARE=1
            shift
            ;;
        --html)
            GEN_HTML=1
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
            echo "  --type TYPE      Тип бенчмарка:"
            echo "                  all        - все бенчмарки (по умолчанию)"
            echo "                  fast       - только быстрый токенизатор"
            echo "                  original   - только оригинальный"
            echo "                  comparison - сравнение версий"
            echo "                  parallel   - многопоточность"
            echo ""
            echo "  --iterations N   Количество итераций"
            echo "  --format FORMAT  Формат вывода (console|json|csv|plot)"
            echo "  --output FILE    Сохранить результаты в файл"
            echo "  --compare        Сравнить с предыдущими результатами"
            echo "  --html           Сгенерировать HTML отчет"
            echo "  --verbose        Подробный вывод"
            echo "  --help           Показать справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --type all --format plot"
            echo "  $0 --type fast --iterations 10 --output results.json"
            echo "  $0 --type comparison --html --compare"
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
# Проверка окружения
# ======================================================================
print_header "📊 ЗАПУСК БЕНЧМАРКОВ BPE TOKENIZER"

print_info "Тип бенчмарков: $BENCH_TYPE"
print_info "Формат вывода: $FORMAT"

BUILD_DIR="$PROJECT_ROOT/cpp/build"
REPORTS_DIR="$PROJECT_ROOT/reports/benchmarks"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$REPORTS_DIR"

# Проверка сборки
if [ ! -d "$BUILD_DIR" ]; then
    print_warning "Директория сборки не найдена. Запуск сборки..."
    "$SCRIPT_DIR/build.sh" Release
fi

# Проверка наличия бенчмарков
cd "$BUILD_DIR"

BENCHMARKS=()
case $BENCH_TYPE in
    all)
        BENCHMARKS=(
            "benchmarks/bpe_benchmarks"
            "benchmarks/bench_fast_tokenizer"
            "benchmarks/bench_comparison"
        )
        ;;
    fast)
        BENCHMARKS=("benchmarks/bench_fast_tokenizer")
        ;;
    original)
        BENCHMARKS=("benchmarks/bpe_benchmarks")
        ;;
    comparison)
        BENCHMARKS=("benchmarks/bench_comparison")
        ;;
    parallel)
        if [ -f "benchmarks/bench_parallel" ]; then
            BENCHMARKS=("benchmarks/bench_parallel")
        else
            print_error "Parallel benchmarks not found"
            exit 1
        fi
        ;;
    *)
        print_error "Неизвестный тип бенчмарка: $BENCH_TYPE"
        exit 1
        ;;
esac

# Проверка наличия моделей
if [ ! -f "$PROJECT_ROOT/models/cpp_vocab.json" ]; then
    print_warning "Модели не найдены. Конвертация из Python..."
    python3 "$PROJECT_ROOT/tools/convert_vocab.py"
fi

# ======================================================================
# Функции для запуска бенчмарков
# ======================================================================

run_benchmark() {
    local bench=$1
    local bench_name=$(basename "$bench")
    local output_base="$REPORTS_DIR/${bench_name}_${TIMESTAMP}"
    
    print_header "🏃 ЗАПУСК $bench_name"
    
    # Формируем команду
    CMD="$bench"
    
    if [ -n "$ITERATIONS" ]; then
        CMD="$CMD --benchmark_repetitions=$ITERATIONS"
    fi
    
    # Запуск в зависимости от формата
    case $FORMAT in
        json)
            CMD="$CMD --benchmark_out=${output_base}.json --benchmark_out_format=json"
            ;;
        csv)
            CMD="$CMD --benchmark_out=${output_base}.csv --benchmark_out_format=csv"
            ;;
        plot|html)
            CMD="$CMD --benchmark_out=${output_base}.json --benchmark_out_format=json"
            ;;
    esac
    
    if [ $VERBOSE -eq 1 ]; then
        echo "  Команда: $CMD"
    fi
    
    # Запускаем
    eval $CMD
    
    # Если нужен plot, генерируем график
    if [ "$FORMAT" = "plot" ] || [ "$FORMAT" = "html" ]; then
        generate_plot "${output_base}.json" "$bench_name"
    fi
    
    print_success "Бенчмарк $bench_name завершен"
    echo "  Результаты: ${output_base}.*"
}

generate_plot() {
    local json_file=$1
    local bench_name=$2
    local plot_file="${json_file%.json}.png"
    
    print_info "Генерация графика..."
    
    # Python скрипт для визуализации
    python3 -c "
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

with open('$json_file') as f:
    data = json.load(f)

benchmarks = data.get('benchmarks', [])
names = []
times = []
errors = []

for b in benchmarks:
    if 'real_time' in b:
        names.append(b.get('name', 'unknown')[:30])
        times.append(b['real_time'] / 1000000)  # convert to ms
        if 'time_unit' in b and b['time_unit'] == 'ns':
            times[-1] /= 1000

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(names)), times)
plt.xlabel('Benchmark')
plt.ylabel('Time (ms)')
plt.title('$bench_name Performance')
plt.xticks(range(len(names)), names, rotation=45, ha='right')

# Add value labels
for i, (bar, t) in enumerate(zip(bars, times)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{t:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('$plot_file', dpi=150)
print(f'  График сохранен: $plot_file')
"
}

generate_html_report() {
    local report_file="$REPORTS_DIR/report_${TIMESTAMP}.html"
    
    print_info "Генерация HTML отчета..."
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BPE Tokenizer Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .chart { margin: 20px 0; }
        .speedup { font-weight: bold; color: #4CAF50; }
    </style>
</head>
<body>
    <h1>📊 BPE Tokenizer Benchmark Report</h1>
    <p>Generated: $(date)</p>
    <p>Type: $BENCH_TYPE</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <ul>
EOF
    
    # Добавляем информацию о каждом бенчмарке
    for json in "$REPORTS_DIR"/*_${TIMESTAMP}.json; do
        if [ -f "$json" ]; then
            name=$(basename "$json" .json)
            echo "<li><a href=\"#${name}\">${name}</a></li>" >> "$report_file"
        fi
    done
    
    echo "        </ul>" >> "$report_file"
    echo "    </div>" >> "$report_file"
    
    # Добавляем графики
    for png in "$REPORTS_DIR"/*_${TIMESTAMP}.png; do
        if [ -f "$png" ]; then
            name=$(basename "$png" .png)
            echo "<div class='chart'>" >> "$report_file"
            echo "<h2 id='${name}'>${name}</h2>" >> "$report_file"
            echo "<img src='${png}' style='max-width:100%'>" >> "$report_file"
            echo "</div>" >> "$report_file"
        fi
    done
    
    # Добавляем таблицу с результатами
    echo "<h2>Detailed Results</h2>" >> "$report_file"
    echo "<table>" >> "$report_file"
    echo "<tr><th>Benchmark</th><th>Time (ms)</th><th>CPU Time (ms)</th><th>Iterations</th></tr>" >> "$report_file"
    
    for json in "$REPORTS_DIR"/*_${TIMESTAMP}.json; do
        if [ -f "$json" ]; then
            python3 -c "
import json
with open('$json') as f:
    data = json.load(f)
for b in data.get('benchmarks', []):
    name = b.get('name', 'unknown')
    time = b.get('real_time', 0) / 1000000
    cpu = b.get('cpu_time', 0) / 1000000
    iter = b.get('iterations', 0)
    print(f\"<tr><td>{name}</td><td>{time:.3f}</td><td>{cpu:.3f}</td><td>{iter}</td></tr>\")
" >> "$report_file"
        fi
    done
    
    echo "</table>" >> "$report_file"
    echo "</body></html>" >> "$report_file"
    
    print_success "HTML отчет сгенерирован: $report_file"
    
    if command -v firefox &> /dev/null; then
        firefox "$report_file" &
    fi
}

# ======================================================================
# Основной запуск
# ======================================================================

# Запускаем каждый бенчмарк
for bench in "${BENCHMARKS[@]}"; do
    if [ -f "$bench" ]; then
        run_benchmark "$bench"
    else
        print_warning "Бенчмарк не найден: $bench"
    fi
done

# Сравнение с предыдущими результатами
if [ $COMPARE -eq 1 ]; then
    print_header "📈 СРАВНЕНИЕ С ПРЕДЫДУЩИМИ РЕЗУЛЬТАТАМИ"
    
    # Находим предыдущие результаты
    PREV_JSON=$(ls -t "$REPORTS_DIR"/*.json 2>/dev/null | head -2 | tail -1)
    CURR_JSON=$(ls -t "$REPORTS_DIR"/*.json 2>/dev/null | head -1)
    
    if [ -f "$PREV_JSON" ] && [ -f "$CURR_JSON" ]; then
        python3 -c "
import json
import sys

def get_avg_time(json_file):
    with open(json_file) as f:
        data = json.load(f)
    times = [b.get('real_time', 0) for b in data.get('benchmarks', [])]
    return sum(times) / len(times) if times else 0

prev_avg = get_avg_time('$PREV_JSON')
curr_avg = get_avg_time('$CURR_JSON')

if prev_avg > 0:
    change = ((curr_avg - prev_avg) / prev_avg) * 100
    print(f'  Предыдущий средний: {prev_avg/1000000:.3f} ms')
    print(f'  Текущий средний:    {curr_avg/1000000:.3f} ms')
    print(f'  Изменение:          {change:+.1f}%')
    
    if change < -5:
        print('  ✅ Производительность улучшилась!')
    elif change > 5:
        print('  ⚠️ Производительность ухудшилась!')
    else:
        print('  📊 Производительность стабильна')
"
    else
        print_warning "Недостаточно данных для сравнения"
    fi
fi

# Генерация HTML отчета
if [ $GEN_HTML -eq 1 ]; then
    generate_html_report
fi

# ======================================================================
# Итог
# ======================================================================
print_header "✅ БЕНЧМАРКИ ЗАВЕРШЕНЫ"

print_success "Результаты сохранены в: $REPORTS_DIR"

# Показываем последние результаты
echo ""
echo "📁 Последние файлы:"
ls -lt "$REPORTS_DIR" | head -10 | awk '{printf "  %s %s\n", $5, $9}'

echo ""
print_success "Готово!"