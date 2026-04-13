#!/bin/bash
# ============================================================================
# run_benchmarks.sh - Запуск бенчмарков BPE токенизатора
# ============================================================================
#
# @file run_benchmarks.sh
# @brief Запуск всех бенчмарков и анализ производительности
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Этот скрипт автоматизирует запуск всех бенчмарков проекта,
#          собирает результаты и генерирует отчеты в различных форматах.
#
#          **Поддерживаемые типы бенчмарков:**
#          ┌────────────┬───────────────────────────────────────┐
#          │ all        │ Все доступные бенчмарки               │
#          │ fast       │ Только оптимизированная версия (Fast) │
#          │ original   │ Только базовая версия (BPE)           │
#          │ comparison │ Прямое сравнение версий               │
#          │ parallel   │ Тестирование многопоточности          │
#          │ cache      │ Тестирование эффективности кэша       │
#          │ memory     │ Тестирование использования памяти     │
#          └────────────┴───────────────────────────────────────┘
#
#          **Форматы вывода:**
#          ┌────────────┬───────────────────────────────────────┐
#          │ console    │ Вывод в консоль (по умолчанию)        │
#          │ json       │ Сохранение в JSON                     │
#          │ csv        │ сохранение в CSV                      │
#          │ plot       │ генерация графиков (matplotlib)       │
#          │ html       │ генерация HTML отчета                 │
#          │ markdown   │ генерация Markdown отчета             │
#          └────────────┴───────────────────────────────────────┘
#
# @usage ./run_benchmarks.sh [options]
#   --type TYPE     - Тип бенчмарка (all|fast|original|comparison|parallel|cache|memory)
#   --iterations N  - Количество итераций для каждого бенчмарка
#   --format FORMAT - Формат вывода (console|json|csv|plot|html|markdown)
#   --output FILE   - Сохранить результаты в файл
#   --compare       - Сравнить с предыдущими результатами
#   --html          - Сгенерировать HTML отчет (то же что --format html)
#   --open          - Открыть HTML отчет в браузере
#   --verbose       - Подробный вывод
#   --quick         - Быстрый режим (меньше итераций)
#   --help          - Показать справку
#
# @example
#   ./run_benchmarks.sh --type all --format plot
#   ./run_benchmarks.sh --type fast --iterations 10 --output results.json
#   ./run_benchmarks.sh --type comparison --html --open
#
# ============================================================================

set -euo pipefail    # Прерывать при ошибке, неопределенных переменных и сбоях в пайпах

# ============================================================================
# Цвета для красивого вывода
# ============================================================================
RED='\033[0;31m'        # Красный   - Ошибки
GREEN='\033[0;32m'      # Зеленый   - Успех
YELLOW='\033[1;33m'     # Желтый    - Предупреждения
BLUE='\033[0;34m'       # Синий     - Информация
MAGENTA='\033[0;35m'    # Пурпурный - Заголовки
CYAN='\033[0;36m'       # Голубой   - Команды
NC='\033[0m'            # No Color  - Сброс цвета

# ============================================================================
# Функции для форматированного вывода
# ============================================================================

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
    echo -e "\n${MAGENTA}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${MAGENTA}============================================================${NC}\n"
}

print_command() {
    echo -e "${CYAN}  > $1${NC}"
}

# ============================================================================
# Парсинг аргументов
# ============================================================================

# Определение путей (скрипт может быть запущен из любой директории)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Значения по умолчанию
BENCH_TYPE="all"
ITERATIONS=""
FORMAT="json"
OUTPUT_FILE=""
COMPARE=0
GEN_HTML=0
OPEN_REPORT=0
VERBOSE=0
QUICK_MODE=0

# Обработка аргументов командной строки
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
            FORMAT="html"
            shift
            ;;
        --open)
            OPEN_REPORT=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --quick)
            QUICK_MODE=1
            ITERATIONS="5"
            shift
            ;;
        --help)
            echo "Использование: $0 [options]"
            echo ""
            echo "Опции:"
            echo "  --type TYPE      Тип бенчмарка:"
            echo "                       all        - Все бенчмарки (по умолчанию)"
            echo "                       fast       - Только быстрый токенизатор"
            echo "                       original   - Только оригинальный"
            echo "                       comparison - Сравнение версий"
            echo "                       parallel   - Многопоточность"
            echo "                       cache      - Эффективность кэша"
            echo "                       memory     - Использование памяти"
            echo ""
            echo "  --iterations N  - Количество итераций для каждого бенчмарка"
            echo "  --format FORMAT - Формат вывода (console|json|csv|plot|html|markdown)"
            echo "  --output FILE   - Сохранить результаты в файл"
            echo "  --compare       - Сравнить с предыдущими результатами"
            echo "  --html          - Сгенерировать HTML отчет"
            echo "  --open          - Открыть HTML отчет в браузере"
            echo "  --verbose       - Подробный вывод команд"
            echo "  --quick         - Быстрый режим (5 итераций)"
            echo "  --help          - Показать справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --type all --format plot"
            echo "  $0 --type fast --iterations 10 --output results.json"
            echo "  $0 --type comparison --html --open"
            exit 0
            ;;
        *)
            print_error "Неизвестная опция: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# ============================================================================
# Проверка окружения
# ============================================================================
print_header "ЗАПУСК БЕНЧМАРКОВ BPE TOKENIZER"

print_info "Тип бенчмарков: $BENCH_TYPE"
print_info "Формат вывода:  $FORMAT"

# Пути к директориям
BPE_CPP_DIR="$PROJECT_ROOT/bpe_cpp"
BUILD_DIR="$BPE_CPP_DIR/build"
REPORTS_DIR="$BPE_CPP_DIR/reports/benchmarks"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_BASE="$REPORTS_DIR/${BENCH_TYPE}_${TIMESTAMP}"

# Создаем директорию для отчетов
mkdir -p "$REPORTS_DIR"

# Проверка прав на запись
if [ ! -w "$REPORTS_DIR" ]; then
    print_error "Нет прав на запись в директорию: $REPORTS_DIR!"
    exit 1
fi

echo -e "${CYAN}[DIAG] Директория отчетов:${NC} $REPORTS_DIR"
echo -e "${CYAN}[DIAG] Права:${NC} $(ls -ld "$REPORTS_DIR")"

# Переходим в директорию сборки
cd "$BPE_CPP_DIR/build"

# Проверка сборки
if [ ! -d "$BUILD_DIR" ]; then
    print_warning "Директория сборки не найдена. Запуск сборки..."
    if [ -f "$SCRIPT_DIR/build.sh" ]; then
        "$SCRIPT_DIR/build.sh" Release
    else
        print_error "Скрипт сборки не найден!"
        exit 1
    fi
fi

# Переходим в директорию сборки
cd "$BUILD_DIR"

# Определяем список бенчмарков для запуска
BENCHMARKS=()
case $BENCH_TYPE in
    all)
        BENCHMARKS=(
            "benchmarks/bench_tokenizer"
            "benchmarks/bench_fast_tokenizer"
            "benchmarks/bench_comparison"
        )
        # Проверяем наличие дополнительных бенчмарков
        for extra in "bench_parallel" "bench_cache" "bench_memory"; do
            if [ -f "benchmarks/$extra" ]; then
                BENCHMARKS+=("benchmarks/$extra")
            fi
        done
        ;;
    fast)
        BENCHMARKS=("benchmarks/bench_fast_tokenizer")
        ;;
    original)
        BENCHMARKS=("benchmarks/bench_tokenizer")
        ;;
    comparison)
        BENCHMARKS=("benchmarks/bench_comparison")
        ;;
    parallel)
        if [ -f "benchmarks/bench_parallel" ]; then
            BENCHMARKS=("benchmarks/bench_parallel")
        else
            print_error "Parallel benchmarks не найдены"
            exit 1
        fi
        ;;
    cache)
        if [ -f "benchmarks/bench_cache" ]; then
            BENCHMARKS=("benchmarks/bench_cache")
        else
            print_error "Cache benchmarks не найдены"
            exit 1
        fi
        ;;
    memory)
        if [ -f "benchmarks/bench_memory" ]; then
            BENCHMARKS=("benchmarks/bench_memory")
        else
            print_error "Memory benchmarks не найдены"
            exit 1
        fi
        ;;
    *)
        print_error "Неизвестный тип бенчмарка: $BENCH_TYPE"
        exit 1
        ;;
esac

# Проверка наличия моделей
MODELS_DIR="$PROJECT_ROOT/bpe_cpp/models"
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
    print_warning "Модели не найдены. Конвертация из Python..."
    CONVERT_SCRIPT="$PROJECT_ROOT/bpe_cpp/tools/convert_vocab.py"
    if [ -f "$CONVERT_SCRIPT" ]; then
        python3 "$CONVERT_SCRIPT" --model-size 8000
        python3 "$CONVERT_SCRIPT" --model-size 10000
        python3 "$CONVERT_SCRIPT" --model-size 12000
    else
        print_error "Скрипт конвертации не найден: $CONVERT_SCRIPT"
        exit 1
    fi
fi

# ============================================================================
# Функции для запуска бенчмарков
# ============================================================================

run_benchmark() {
    local bench=$1
    local bench_name=$(basename "$bench")
    local output_base="$REPORTS_DIR/${bench_name}_${TIMESTAMP}"
    
    print_header "ЗАПУСК $bench_name"
    
    # Проверяем существование бенчмарка
    if [ ! -f "$bench" ]; then
        print_error "Бенчмарк не найден: $bench!"
        return 1
    fi
    
    # Формируем команду
    CMD="$bench"
    
    if [ -n "$ITERATIONS" ]; then
        CMD="$CMD --benchmark_repetitions=$ITERATIONS"
    fi
    
    # Запуск в зависимости от формата
    case $FORMAT in
        json|csv|plot|html|markdown)
            CMD="$CMD --benchmark_out=${output_base}.json --benchmark_out_format=json"
            ;;
        console)
            CMD="$CMD --benchmark_format=console"
            ;;
    esac
    
    # ДИАГНОСТИКА: выводим команду и пути
    echo -e "\n${CYAN}[DIAG] Команда:${NC} $CMD"
    echo -e "${CYAN}[DIAG] Рабочая директория:${NC} $(pwd)"
    echo -e "${CYAN}[DIAG] Выходной файл:${NC} ${output_base}.json"
    
    if [ $VERBOSE -eq 1 ]; then
        echo "Команда: $CMD"
        eval $CMD
    else
        # Сохраняем stdout/stderr во временный файл для диагностики
        local temp_out=$(mktemp)
        eval $CMD > "$temp_out" 2>&1
        local exit_code=$?
        
        echo -e "${CYAN}[DIAG] Выход бенчмарка (первые 20 строк):${NC}"
        head -20 "$temp_out" | sed 's/^/  /'
        
        if [ $exit_code -ne 0 ]; then
            print_error "Ошибка при запуске бенчмарка $bench_name (код: $exit_code)!"
            echo "Полный вывод сохранен в: $temp_out"
            return 1
        else
            rm -f "$temp_out"
        fi
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Ошибка при запуске бенчмарка $bench_name!"
        return 1
    fi
    
    # Проверяем, создался ли файл
    if [ -f "${output_base}.json" ]; then
        local file_size=$(stat -c%s "${output_base}.json" 2>/dev/null || stat -f%z "${output_base}.json" 2>/dev/null)
        print_success "Бенчмарк $bench_name завершен!"
        echo "Результаты: ${output_base}.json (${file_size} байт)"
        
        # Показываем первые несколько строк JSON
        echo "Первые строки JSON:"
        head -5 "${output_base}.json" | sed 's/^/    /'
    else
        print_error "Файл результатов не создан: ${output_base}.json"
        return 1
    fi
    
    # Если нужен plot, генерируем график
    if [ "$FORMAT" = "plot" ] || [ "$FORMAT" = "html" ] || [ "$FORMAT" = "markdown" ]; then
        generate_plot "${output_base}.json" "$bench_name"
    fi
    
    # Если нужен CSV, конвертируем
    if [ "$FORMAT" = "csv" ]; then
        convert_to_csv "${output_base}.json"
    fi
}

generate_plot() {
    local json_file=$1
    local bench_name=$2
    local plot_file="${json_file%.json}.png"
    
    print_info "Генерация графика..."
    
    # Проверяем наличие matplotlib
    python3 -c "import matplotlib" 2>/dev/null || {
        print_warning "matplotlib не установлен. Пропускаем генерацию графика."
        return
    }
    
    # Python скрипт для визуализации
    python3 -c "
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def generate_plot(json_file, plot_file, bench_name):
    try:
        with open(json_file) as f:
            data = json.load(f)
        
        benchmarks = data.get('benchmarks', [])
        if not benchmarks:
            print(f'Нет данных в {json_file}!')
            return
        
        names = []
        times = []
        errors = []
        
        for b in benchmarks:
            if 'real_time' in b:
                # Обрезаем длинные имена
                name = b.get('name', 'unknown')
                if len(name) > 50:
                    name = name[:47] + '...'
                names.append(name)
                
                # Конвертируем в миллисекунды
                time_ns = b['real_time']
                time_ms = time_ns / 1_000_000.0
                times.append(time_ms)
                
                if 'cpu_time' in b:
                    cpu_ms = b['cpu_time'] / 1_000_000.0
                    errors.append(abs(time_ms - cpu_ms))
                else:
                    errors.append(0)
        
        if not times:
            print(f'Нет временных данных в {json_file}')
            return
        
        # Сортируем по времени
        sorted_indices = np.argsort(times)[::-1]
        names = [names[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
        errors = [errors[i] for i in sorted_indices]
        
        plt.figure(figsize=(12, max(6, len(names) * 0.3)))
        
        # Горизонтальные полосы для лучшей читаемости
        y_pos = np.arange(len(names))
        bars = plt.barh(y_pos, times, xerr=errors, capsize=3, 
                       color='steelblue', edgecolor='navy', alpha=0.7)
        
        plt.xlabel('Время (мс)', fontsize=12)
        plt.ylabel('Бенчмарк', fontsize=12)
        plt.title(f'{bench_name} - Производительность\n{os.path.basename(json_file)}', 
                  fontsize=14, fontweight='bold')
        plt.yticks(y_pos, names, fontsize=9)
        
        # Добавляем значения на полосы
        for i, (bar, t) in enumerate(zip(bars, times)):
            plt.text(bar.get_width() + max(times)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{t:.2f} мс', va='center', fontsize=8)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f'График сохранен: {plot_file}')
        
    except Exception as e:
        print(f'Ошибка генерации графика: {e}!')

generate_plot('$json_file', '$plot_file', '$bench_name')
"
}

convert_to_csv() {
    local json_file=$1
    local csv_file="${json_file%.json}.csv"
    
    print_info "Конвертация в CSV..."
    
    python3 -c "
import json
import csv

with open('$json_file') as f:
    data = json.load(f)

with open('$csv_file', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'real_time_ms', 'cpu_time_ms', 'iterations', 'bytes_per_second'])
    
    for b in data.get('benchmarks', []):
        name = b.get('name', 'unknown')
        real_time_ns = b.get('real_time', 0)
        cpu_time_ns = b.get('cpu_time', 0)
        iterations = b.get('iterations', 0)
        bytes_per_second = b.get('bytes_per_second', 0)
        
        writer.writerow([
            name,
            real_time_ns / 1_000_000.0,
            cpu_time_ns / 1_000_000.0,
            iterations,
            bytes_per_second
        ])
    
print(f'  CSV сохранен: $csv_file')
"
}

generate_markdown() {
    local md_file="$REPORT_BASE.md"
    
    print_info "Генерация Markdown отчета..."
    
    {
        echo "# BPE Tokenizer Benchmark Report"
        echo ""
        echo "**Generated:** $(date)"
        echo "**Type:**      $BENCH_TYPE"
        echo "**Format:**    $FORMAT"
        echo ""
        echo "## Summary"
        echo ""
        
        # Собираем все JSON файлы
        for json in "$REPORTS_DIR"/*_${TIMESTAMP}.json; do
            if [ -f "$json" ]; then
                name=$(basename "$json" .json)
                echo "### $name"
                echo ""
                echo "| Benchmark | Real Time (мс) | CPU Time (мс) | Iterations |"
                echo "|-----------|----------------|---------------|------------|"
                
                python3 -c "
import json
with open('$json') as f:
    data = json.load(f)
for b in data.get('benchmarks', []):
    name = b.get('name', 'unknown')
    real_ns = b.get('real_time', 0)
    cpu_ns = b.get('cpu_time', 0)
    iter_count = b.get('iterations', 0)
    print(f'| {name} | {real_ns/1_000_000:.3f} | {cpu_ns/1_000_000:.3f} | {iter_count} |')
" >> "$md_file"
                
                echo "" >> "$md_file"
                
                # Добавляем ссылку на график если есть
                if [ -f "${json%.json}.png" ]; then
                    echo "![${name} plot]($(basename "${json%.json}.png"))" >> "$md_file"
                    echo "" >> "$md_file"
                fi
            fi
        done
        
        echo "## System Information" >> "$md_file"
        echo "" >> "$md_file"
        echo "- **OS:**     $(uname -a)" >> "$md_file"
        echo "- **CPU:**    $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)" >> "$md_file"
        echo "- **Cores:**  $(nproc)" >> "$md_file"
        echo "- **Memory:** $(free -h | grep Mem | awk '{print $2}')" >> "$md_file"
        
    } > "$md_file"
    
    print_success "Markdown отчет сгенерирован: $md_file"
}

generate_html_report() {
    local report_file="$REPORT_BASE.html"
    
    print_info "Генерация HTML отчета..."
    
    # Собираем информацию о системе
    local cpu_info=$(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
    local cores=$(nproc)
    local memory=$(free -h | grep Mem | awk '{print $2}')
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BPE Tokenizer Benchmark Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            box-sizing: border-box;
        }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 { 
            color: white; 
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        h2 {
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h3 {
            color: #34495e;
            margin: 20px 0 10px;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h3 {
            margin: 0 0 10px;
            color: #4CAF50;
        }
        .card .value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .card .label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .benchmark-section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        th, td { 
            border: none; 
            padding: 12px; 
            text-align: left; 
        }
        th { 
            background-color: #4CAF50; 
            color: white;
            font-weight: 600;
        }
        tr:nth-child(even) { 
            background-color: #f8f9fa; 
        }
        tr:hover {
            background-color: #e9ecef;
        }
        .chart { 
            margin: 30px 0; 
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .chart img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge.success {
            background: #4CAF50;
            color: white;
        }
        .badge.warning {
            background: #f39c12;
            color: white;
        }
        .badge.error {
            background: #e74c3c;
            color: white;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 20px;
            text-align: right;
        }
        .footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
        }
        @media (max-width: 768px) {
            .summary-cards {
                grid-template-columns: 1fr;
            }
            body {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ BPE Tokenizer Benchmark Report</h1>
        
        <div class="summary-cards">
            <div class="card">
                <h3>Тип</h3>
                <div class="value">$BENCH_TYPE</div>
                <div class="label">benchmark type</div>
            </div>
            <div class="card">
                <h3>Дата</h3>
                <div class="value">$(date +"%d.%m.%Y")</div>
                <div class="label">$(date +"%H:%M:%S")</div>
            </div>
            <div class="card">
                <h3>CPU</h3>
                <div class="value">$cores ядер</div>
                <div class="label">$cpu_info</div>
            </div>
            <div class="card">
                <h3>Память</h3>
                <div class="value">$memory</div>
                <div class="label">total RAM</div>
            </div>
        </div>
EOF
    
    # Добавляем информацию о каждом бенчмарке
    for json in "$REPORTS_DIR"/*_${TIMESTAMP}.json; do
        if [ -f "$json" ]; then
            name=$(basename "$json" .json)
            
            cat >> "$report_file" << EOF
        
        <div class="benchmark-section">
            <h2>${name}</h2>
EOF
            
            # Добавляем таблицу
            python3 -c "
import json
with open('$json') as f:
    data = json.load(f)

benchmarks = data.get('benchmarks', [])
if benchmarks:
    print('            <table>')
    print('                <tr><th>Benchmark</th><th>Real Time (ms)</th><th>CPU Time (ms)</th><th>Iterations</th></tr>')
    for b in benchmarks:
        name = b.get('name', 'unknown')
        real_ns = b.get('real_time', 0)
        cpu_ns = b.get('cpu_time', 0)
        iter_count = b.get('iterations', 0)
        print(f'                <tr><td>{name}</td><td>{real_ns/1_000_000:.3f}</td><td>{cpu_ns/1_000_000:.3f}</td><td>{iter_count}</td></tr>')
    print('            </table>')
" >> "$report_file"
            
            # Добавляем график если есть
            if [ -f "${json%.json}.png" ]; then
                cp "${json%.json}.png" "$REPORTS_DIR/${name}.png"
                echo "            <div class='chart'>" >> "$report_file"
                echo "                <h3>Визуализация</h3>" >> "$report_file"
                echo "                <img src='${name}.png' alt='${name} benchmark results'>" >> "$report_file"
                echo "            </div>" >> "$report_file"
            fi
            
            echo "        </div>" >> "$report_file"
        fi
    done
    
    # Закрываем HTML
    cat >> "$report_file" << EOF
        
        <div class="footer">
            <p>Generated by BPE Tokenizer Benchmark Suite • $(date)</p>
        </div>
    </div>
</body>
</html>
EOF
    
    print_success "HTML отчет сгенерирован: $report_file"
    
    # Открываем в браузере если нужно
    if [ $OPEN_REPORT -eq 1 ]; then
        if command -v firefox &> /dev/null; then
            firefox "$report_file" &
        elif command -v google-chrome &> /dev/null; then
            google-chrome "$report_file" &
        elif command -v xdg-open &> /dev/null; then
            xdg-open "$report_file" &
        else
            print_info "Отчет сохранен: $report_file"
        fi
    fi
}

# ============================================================================
# Основной запуск
# ============================================================================

# Запускаем каждый бенчмарк
for bench in "${BENCHMARKS[@]}"; do
    if [ -f "$bench" ]; then
        run_benchmark "$bench"
    else
        print_warning "Бенчмарк не найден: $bench!"
    fi
done

# Сравнение с предыдущими результатами
if [ $COMPARE -eq 1 ]; then
    print_header "СРАВНЕНИЕ С ПРЕДЫДУЩИМИ РЕЗУЛЬТАТАМИ"
    
    # Находим предыдущие результаты (последние два файла)
    PREV_JSON=$(ls -t "$REPORTS_DIR"/*.json 2>/dev/null | head -2 | tail -1)
    CURR_JSON=$(ls -t "$REPORTS_DIR"/*.json 2>/dev/null | head -1)
    
    if [ -f "$PREV_JSON" ] && [ -f "$CURR_JSON" ] && [ "$PREV_JSON" != "$CURR_JSON" ]; then
        python3 -c "
import json
import sys
import statistics
import os

def get_avg_time(json_file):
    with open(json_file) as f:
        data = json.load(f)
    times = [b.get('real_time', 0) for b in data.get('benchmarks', []) if b.get('real_time', 0) > 0]
    return statistics.mean(times) if times else 0

prev_avg = get_avg_time('$PREV_JSON')
curr_avg = get_avg_time('$CURR_JSON')

if prev_avg > 0:
    change = ((curr_avg - prev_avg) / prev_avg) * 100
    print(f'\nСравнение:')
    print(f'- Предыдущий: {os.path.basename("$PREV_JSON")}')
    print(f'- Текущий:    {os.path.basename("$CURR_JSON")}')
    print(f'\n- Среднее время:')
    print(f'    До:        {prev_avg/1_000_000:.3f} мс')
    print(f'    После:     {curr_avg/1_000_000:.3f} мс')
    print(f'    Изменение: {change:+.1f}%')
    
    if change < -5:
        print('\nПроизводительность УЛУЧШИЛАСЬ!')
    elif change > 5:
        print('\nПроизводительность УХУДШИЛАСЬ!')
    else:
        print('\nПроизводительность СТАБИЛЬНА')
"
    else
        print_warning "Недостаточно данных для сравнения (нужно минимум 2 файла)!"
    fi
fi

# Генерация отчетов
if [ "$FORMAT" = "html" ] || [ $GEN_HTML -eq 1 ]; then
    generate_html_report
fi

if [ "$FORMAT" = "markdown" ]; then
    generate_markdown
fi

# Если указан выходной файл, копируем результаты
if [ -n "$OUTPUT_FILE" ]; then
    if [ -f "$REPORTS_DIR/${BENCH_TYPE}_${TIMESTAMP}.json" ]; then
        cp "$REPORTS_DIR/${BENCH_TYPE}_${TIMESTAMP}.json" "$OUTPUT_FILE"
        print_success "Результаты сохранены в: $OUTPUT_FILE"
    fi
fi

# ============================================================================
# Итог
# ============================================================================
print_header "БЕНЧМАРКИ ЗАВЕРШЕНЫ!"

print_success "Результаты сохранены в: $REPORTS_DIR"

echo ""
echo "Последние файлы:"
ls -lt "$REPORTS_DIR" | grep -E "\.(json|csv|png|html|md)$" | head -10 | awk '{printf "  %s %s %s\n", $6, $7, $9}'

echo ""
print_success "Готово!"