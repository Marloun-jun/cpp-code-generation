#!/bin/bash
# ======================================================================
# run_tests.sh - Запуск тестов BPE токенизатора
# ======================================================================
#
# @file run_tests.sh
# @brief Запуск всех модульных тестов и анализ покрытия кода
#
# @author Евгений П.
# @date 2026
# @version 3.4.0
#
# @details Этот скрипт автоматизирует запуск всех тестов проекта,
#          обеспечивая различные режимы тестирования:
#
#          **Режимы тестирования:**
#           - Обычный запуск    - все тесты с отчетом о провалах
#           - Фильтрация        - запуск только определенных тестов
#           - Параллельный      - ускорение на многоядерных системах
#           - Повторный         - выявление flaky тестов
#
#          **Дополнительные возможности:**
#          Покрытие кода      - анализ с помощью gcov/lcov
#          Проверка памяти    - Valgrind memcheck
#          Отчеты             - JUnit XML, HTML и Markdown форматы
#
# @usage ./run_tests.sh [options]
#   --filter FILTER    Фильтр для запуска конкретных тестов (например: *Tokenizer*)
#   --coverage         Запустить с анализом покрытия кода (требует gcov/lcov)
#   --parallel N       Запустить тесты в N параллельных потоков
#   --repeat N         Повторить каждый тест N раз
#   --verbose          Подробный вывод
#   --xml              Сгенерировать JUnit XML отчет
#   --html             Сгенерировать HTML отчет о тестах
#   --markdown         Сгенерировать Markdown отчет о тестах
#   --memcheck         Запустить с Valgrind для проверки утечек памяти
#   --open             Открыть HTML отчет в браузере
#   --build-type TYPE  Тип сборки (Debug|Release|RelWithDebInfo)
#   --no-build         Не запускать сборку (использовать существующие бинарники)
#   --clean            Очистить перед сборкой
#   --help             Показать справку
#
# @example
#   ./run_tests.sh --filter *Tokenizer* --verbose
#   ./run_tests.sh --coverage --html
#   ./run_tests.sh --memcheck --parallel 4
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
TEST_FILTER=""
RUN_COVERAGE=0
PARALLEL_JOBS=1
REPEAT_COUNT=1
VERBOSE=0
GEN_XML=0
GEN_HTML=0
GEN_MARKDOWN=0
RUN_MEMCHECK=0
OPEN_REPORT=0
BUILD_TYPE="Debug"
NO_BUILD=0
CLEAN_BUILD=0

# Директории
BUILD_DIR="$PROJECT_ROOT/build"
REPORTS_DIR="$PROJECT_ROOT/reports/tests"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_RESULT=0

# Создаем директорию для отчетов
mkdir -p "$REPORTS_DIR"

# Обработка аргументов командной строки
while [[ $# -gt 0 ]]; do
    case $1 in
        --filter)
            TEST_FILTER="$2"
            shift 2
            ;;
        --coverage)
            RUN_COVERAGE=1
            shift
            ;;
        --parallel)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --repeat)
            REPEAT_COUNT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --xml)
            GEN_XML=1
            shift
            ;;
        --html)
            GEN_HTML=1
            shift
            ;;
        --markdown)
            GEN_MARKDOWN=1
            shift
            ;;
        --memcheck)
            RUN_MEMCHECK=1
            shift
            ;;
        --open)
            OPEN_REPORT=1
            shift
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-build)
            NO_BUILD=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --help)
            echo "Использование: $0 [options]"
            echo ""
            echo "Опции:"
            echo "  --filter FILTER    Фильтр тестов (например: *Tokenizer*)"
            echo "  --coverage         Анализ покрытия кода (gcov/lcov)"
            echo "  --parallel N       Параллельный запуск N потоков"
            echo "  --repeat N         Повторить тесты N раз"
            echo "  --verbose          Подробный вывод"
            echo "  --xml              JUnit XML отчет"
            echo "  --html             HTML отчет о тестах"
            echo "  --markdown         Markdown отчет о тестах"
            echo "  --memcheck         Проверка утечек памяти (Valgrind)"
            echo "  --open             Открыть HTML отчет в браузере"
            echo "  --build-type TYPE  Тип сборки (Debug|Release|RelWithDebInfo)"
            echo "  --no-build         Не запускать сборку"
            echo "  --clean            Очистить перед сборкой"
            echo "  --help             Показать справку"
            echo ""
            echo "Примеры:"
            echo "  $0 --filter *Tokenizer* --verbose"
            echo "  $0 --coverage --html"
            echo "  $0 --memcheck --parallel 4"
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
print_header "ЗАПУСК ТЕСТОВ BPE TOKENIZER"

print_info "Директория тестов: $BUILD_DIR"
print_info "Директория отчетов: $REPORTS_DIR"
print_info "Временная метка: $TIMESTAMP"
print_info "Тип сборки: $BUILD_TYPE"

if [ -n "$TEST_FILTER" ]; then
    print_info "Фильтр: $TEST_FILTER"
fi

if [ $RUN_COVERAGE -eq 1 ]; then
    print_info "Режим: с покрытием кода"
fi

if [ $RUN_MEMCHECK -eq 1 ]; then
    print_info "Режим: проверка памяти (Valgrind)"
fi

if [ $PARALLEL_JOBS -gt 1 ]; then
    print_info "Параллельных потоков: $PARALLEL_JOBS"
fi

if [ $REPEAT_COUNT -gt 1 ]; then
    print_info "Повторов: $REPEAT_COUNT"
fi

# ======================================================================
# Сборка проекта
# ======================================================================
if [ $NO_BUILD -eq 0 ]; then
    if [ ! -d "$BUILD_DIR" ] || [ $CLEAN_BUILD -eq 1 ]; then
        print_info "Запуск сборки..."
        
        BUILD_OPTS=""
        if [ $CLEAN_BUILD -eq 1 ]; then
            BUILD_OPTS="$BUILD_OPTS --clean"
        fi
        
        if [ $RUN_COVERAGE -eq 1 ]; then
            BUILD_OPTS="$BUILD_OPTS --coverage"
        fi
        
        if [ $VERBOSE -eq 1 ]; then
            BUILD_OPTS="$BUILD_OPTS --verbose"
        fi
        
        "$SCRIPT_DIR/build.sh" "$BUILD_TYPE" $BUILD_OPTS
    else
        print_info "Директория сборки существует. Используем --clean для полной пересборки"
    fi
else
    print_info "Сборка пропущена (--no-build)"
fi

cd "$BUILD_DIR"

# ======================================================================
# Сборка с флагами покрытия если нужно (альтернативный подход)
# ======================================================================
if [ $RUN_COVERAGE -eq 1 ] && [ $NO_BUILD -eq 0 ]; then
    # Проверка наличия gcov
    if ! command -v gcov &> /dev/null; then
        print_error "gcov не найден. Установите gcc"
        echo "  sudo apt install gcc    # для Ubuntu/Debian"
        echo "  brew install gcc        # для macOS"
        exit 1
    fi
    
    # Убеждаемся, что сборка с флагами покрытия
    if [ ! -f "CMakeCache.txt" ] || ! grep -q "CMAKE_CXX_FLAGS:.*--coverage" CMakeCache.txt; then
        print_warning "Сборка без флагов покрытия. Пересборка..."
        
        print_command "cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_CXX_FLAGS=\"--coverage -fprofile-arcs -ftest-coverage\" -DBUILD_TESTING=ON"
        cmake .. \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DCMAKE_CXX_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
            -DBUILD_TESTING=ON
        
        print_command "make clean && make -j$(nproc)"
        make clean
        make -j$(nproc)
        
        print_success "Пересборка завершена!"
    fi
fi

# ======================================================================
# Запуск тестов
# ======================================================================
print_header "ЗАПУСК ТЕСТОВ"

# Формируем команду ctest
CTEST_CMD="ctest"

if [ -n "$TEST_FILTER" ]; then
    CTEST_CMD="$CTEST_CMD -R \"$TEST_FILTER\""
fi

if [ $PARALLEL_JOBS -gt 1 ]; then
    CTEST_CMD="$CTEST_CMD -j$PARALLEL_JOBS"
fi

if [ $REPEAT_COUNT -gt 1 ]; then
    CTEST_CMD="$CTEST_CMD --repeat-until-fail $REPEAT_COUNT"
fi

if [ $VERBOSE -eq 1 ]; then
    CTEST_CMD="$CTEST_CMD --verbose"
else
    CTEST_CMD="$CTEST_CMD --output-on-failure"
fi

if [ $GEN_XML -eq 1 ]; then
    CTEST_CMD="$CTEST_CMD --output-junit $REPORTS_DIR/junit_${TIMESTAMP}.xml"
fi

# Сохраняем вывод в лог
TEST_LOG="$REPORTS_DIR/ctest_${TIMESTAMP}.log"

# Запуск с Valgrind если нужно
if [ $RUN_MEMCHECK -eq 1 ]; then
    print_header "ПРОВЕРКА ПАМЯТИ (VALGRIND)"
    
    if ! command -v valgrind &> /dev/null; then
        print_error "Valgrind не найден. Установите valgrind"
        echo "  sudo apt install valgrind    # для Ubuntu/Debian"
        echo "  brew install valgrind        # для macOS"
        exit 1
    fi
    
    # Находим все тестовые исполняемые файлы
    TEST_EXES=$(find . -type f -executable -name "test_*" -o -name "*_test" -o -name "*_tests" | grep -v "\.sh$" || true)
    
    if [ -z "$TEST_EXES" ]; then
        print_warning "Тестовые исполняемые файлы не найдены!"
    else
        for test_exe in $TEST_EXES; do
            test_name=$(basename "$test_exe")
            print_info "Проверка $test_name..."
            
            valgrind_log="$REPORTS_DIR/valgrind_${test_name}_${TIMESTAMP}.log"
            
            print_command "valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --log-file=$valgrind_log $test_exe ${TEST_FILTER:+--gtest_filter=$TEST_FILTER}"
            valgrind --leak-check=full \
                --show-leak-kinds=all \
                --track-origins=yes \
                --verbose \
                --log-file="$valgrind_log" \
                "$test_exe" ${TEST_FILTER:+"--gtest_filter=$TEST_FILTER"} 2>&1 | tee -a "$REPORTS_DIR/valgrind_output_${TIMESTAMP}.log" || true
            
            # Проверяем наличие утечек
            if [ -f "$valgrind_log" ]; then
                if grep -q "definitely lost:" "$valgrind_log"; then
                    LOST=$(grep "definitely lost:" "$valgrind_log" | grep -o "[0-9,]\+ bytes" | head -1 | tr -d ',')
                    if [ -n "$LOST" ] && [ "$LOST" != "0" ]; then
                        print_error "Обнаружены утечки памяти в $test_name: $LOST bytes"
                    else
                        print_success "Утечек не обнаружено в $test_name"
                    fi
                fi
            fi
        done
        
        print_success "Проверка памяти завершена!"
        echo "  Отчеты: $REPORTS_DIR/valgrind_*"
    fi
else
    # Обычный запуск тестов
    print_info "Запуск: $CTEST_CMD"
    print_command "$CTEST_CMD | tee $TEST_LOG"
    eval $CTEST_CMD 2>&1 | tee "$TEST_LOG"
    
    # Сохраняем код возврата
    TEST_RESULT=${PIPESTATUS[0]}
    
    if [ $TEST_RESULT -eq 0 ]; then
        print_success "Все тесты пройдены успешно!"
    else
        print_error "Некоторые тесты не пройдены. Код возврата: $TEST_RESULT"
    fi
    
    # Извлекаем статистику из лога
    PASSED=$(grep -c "^[0-9]\+/[0-9]\+ Test.*Passed" "$TEST_LOG" || echo 0)
    FAILED=$(grep -c "^[0-9]\+/[0-9]\+ Test.*Failed" "$TEST_LOG" || echo 0)
    TOTAL=$((PASSED + FAILED))
    
    if [ $TOTAL -gt 0 ]; then
        echo ""
        echo "  Статистика:"
        echo "    Всего:  $TOTAL"
        echo "    Успешно: $PASSED"
        echo "    Провалы: $FAILED"
    fi
fi

# ======================================================================
# Генерация Markdown отчета
# ======================================================================
if [ $GEN_MARKDOWN -eq 1 ]; then
    print_header "ГЕНЕРАЦИЯ MARKDOWN ОТЧЕТА"
    
    MD_REPORT="$REPORTS_DIR/test_report_${TIMESTAMP}.md"
    
    {
        echo "# BPE Tokenizer Test Report"
        echo ""
        echo "**Generated:** $(date)"
        echo "**Build Type:** $BUILD_TYPE"
        echo "**Filter:** ${TEST_FILTER:-all}"
        echo ""
        echo "## Summary"
        echo ""
        
        if [ $TOTAL -gt 0 ]; then
            PASS_RATE=$(echo "scale=2; $PASSED * 100 / $TOTAL" | bc)
            echo "| Metric | Value |"
            echo "|--------|-------|"
            echo "| Total Tests | $TOTAL |"
            echo "| Passed | $PASSED |"
            echo "| Failed | $FAILED |"
            echo "| Pass Rate | ${PASS_RATE}% |"
        else
            echo "No test results available."
        fi
        
        if [ $RUN_MEMCHECK -eq 1 ]; then
            echo ""
            echo "## Memory Check Results"
            echo ""
            for log in "$REPORTS_DIR"/valgrind_*.log; do
                if [ -f "$log" ]; then
                    name=$(basename "$log" .log)
                    echo "### $name"
                    echo '```'
                    grep -E "definitely lost|indirectly lost|possibly lost" "$log" || echo "No leaks detected"
                    echo '```'
                    echo ""
                fi
            done
        fi
        
        echo ""
        echo "## Test Log"
        echo '```'
        tail -n 50 "$TEST_LOG" 2>/dev/null || echo "Log file not found"
        echo '```'
        
    } > "$MD_REPORT"
    
    print_success "Markdown отчет сгенерирован: $MD_REPORT"
fi

# ======================================================================
# Анализ покрытия
# ======================================================================
if [ $RUN_COVERAGE -eq 1 ]; then
    print_header "АНАЛИЗ ПОКРЫТИЯ КОДА"
    
    # Проверка наличия lcov
    if ! command -v lcov &> /dev/null; then
        print_warning "lcov не найден. Установите lcov для генерации отчетов"
        echo "  sudo apt install lcov    # для Ubuntu/Debian"
        echo "  brew install lcov        # для macOS"
    else
        COVERAGE_DIR="$REPORTS_DIR/coverage_${TIMESTAMP}"
        mkdir -p "$COVERAGE_DIR"
        
        print_info "Сбор данных покрытия..."
        
        # Сбор данных покрытия
        print_command "lcov --capture --directory . --output-file $COVERAGE_DIR/coverage.info --rc lcov_branch_coverage=1"
        lcov --capture \
            --directory . \
            --output-file "$COVERAGE_DIR/coverage.info" \
            --rc lcov_branch_coverage=1 \
            --exclude '/usr/*' \
            --exclude '*/tests/*' \
            --exclude '*/googletest/*' \
            --exclude '*/benchmarks/*' \
            --exclude '*/build/*' \
            --exclude '*/CMakeFiles/*' \
            2>/dev/null || true
        
        if [ -f "$COVERAGE_DIR/coverage.info" ]; then
            # Фильтрация для удаления системных и тестовых файлов
            lcov --remove "$COVERAGE_DIR/coverage.info" \
                '/usr/*' '*/tests/*' '*/googletest/*' '*/benchmarks/*' '*/build/*' '*/CMakeFiles/*' \
                -o "$COVERAGE_DIR/coverage_filtered.info" \
                --rc lcov_branch_coverage=1 2>/dev/null || true
            
            # Генерация HTML отчета
            print_command "genhtml $COVERAGE_DIR/coverage_filtered.info --output-directory $COVERAGE_DIR/html"
            genhtml "$COVERAGE_DIR/coverage_filtered.info" \
                --branch-coverage \
                --output-directory "$COVERAGE_DIR/html" \
                --title "BPE Tokenizer Coverage Report" \
                --legend \
                --demangle-cpp \
                2>/dev/null || true
            
            if [ -d "$COVERAGE_DIR/html" ]; then
                print_success "Отчет о покрытии сгенерирован!"
                echo "  HTML: $COVERAGE_DIR/html/index.html"
                
                # Показываем статистику
                echo ""
                echo "  Статистика покрытия:"
                lcov --summary "$COVERAGE_DIR/coverage_filtered.info" 2>&1 | \
                    grep -E "lines|functions|branches" | \
                    sed 's/^/    /' || true
                
                # Открываем в браузере
                if [ $OPEN_REPORT -eq 1 ]; then
                    if command -v firefox &> /dev/null; then
                        firefox "$COVERAGE_DIR/html/index.html" &
                    elif command -v google-chrome &> /dev/null; then
                        google-chrome "$COVERAGE_DIR/html/index.html" &
                    elif command -v xdg-open &> /dev/null; then
                        xdg-open "$COVERAGE_DIR/html/index.html" &
                    fi
                fi
            else
                print_warning "Не удалось сгенерировать HTML отчет о покрытии"
            fi
        else
            print_warning "Не удалось собрать данные о покрытии"
        fi
    fi
fi

# ======================================================================
# Генерация HTML отчета
# ======================================================================
if [ $GEN_HTML -eq 1 ]; then
    print_header "ГЕНЕРАЦИЯ HTML ОТЧЕТА"
    
    HTML_REPORT="$REPORTS_DIR/test_report_${TIMESTAMP}.html"
    
    # Собираем информацию о тестах из CTest
    if [ -f "$BUILD_DIR/Testing/TAG" ]; then
        TEST_DIR=$(cat "$BUILD_DIR/Testing/TAG" | head -1)
        TEST_FILE="$BUILD_DIR/Testing/$TEST_DIR/Test.xml"
        
        if [ -f "$TEST_FILE" ]; then
            print_info "Генерация отчета из $TEST_FILE"
            
            # Python скрипт для парсинга XML и генерации HTML
            python3 -c "
import xml.etree.ElementTree as ET
import html
import os
import sys
from datetime import datetime

def parse_test_results(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f'Ошибка парсинга XML: {e}')
        return []
    
    tests = []
    for test in root.findall('.//Test'):
        name = test.get('name', 'unknown')
        
        # Статус теста
        status_elem = test.find('Results/TestStatus')
        status = status_elem.text if status_elem is not None else 'Unknown'
        
        # Время выполнения
        time_elem = test.find('Results/NamedMeasurement/Value')
        time_val = time_elem.text if time_elem is not None else '0'
        
        # Вывод (stdout/stderr)
        output_elem = test.find('Results/Measurement/Value')
        output = output_elem.text if output_elem is not None else ''
        if len(output) > 500:
            output = output[:500] + '...'
        
        tests.append({
            'name': name,
            'status': status,
            'time': time_val,
            'output': output
        })
    
    return tests

def generate_html(tests, output_file, timestamp, passed, failed, total):
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>BPE Tokenizer Test Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{ 
            color: white; 
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 0;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        .card h3 {{
            margin: 0 0 10px;
            color: #4CAF50;
        }}
        .card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: #4CAF50;
            width: {pass_rate:.1f}%;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        th, td {{ 
            border: none; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #4CAF50; 
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{ 
            background-color: #f8f9fa; 
        }}
        tr:hover {{
            background-color: #e9ecef;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-pass {{
            background: #d4edda;
            color: #155724;
        }}
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 20px;
            text-align: right;
        }}
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            padding: 20px;
        }}
        pre {{
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 11px;
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class='container'>
        <h1>🧪 BPE Tokenizer Test Report</h1>
        
        <div class='summary-cards'>
            <div class='card'>
                <h3>📊 Всего</h3>
                <div class='value'>{total}</div>
            </div>
            <div class='card'>
                <h3>✅ Успешно</h3>
                <div class='value' style='color: #4CAF50;'>{passed}</div>
            </div>
            <div class='card'>
                <h3>❌ Провалы</h3>
                <div class='value' style='color: #e74c3c;'>{failed}</div>
            </div>
            <div class='card'>
                <h3>📈 Успешность</h3>
                <div class='value' style='color: #f39c12;'>{pass_rate:.1f}%</div>
            </div>
        </div>
        
        <div class='progress-bar'>
            <div class='progress-fill'></div>
        </div>
        
        <div class='timestamp'>
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Build Type:</strong> $BUILD_TYPE<br>
            <strong>Filter:</strong> {filter}
        </div>
        
        <h2>📋 Detailed Results</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Time (s)</th>
                <th>Output</th>
            </tr>
'''
    
    for test in tests:
        status_class = 'status-pass' if test['status'].lower() == 'passed' else 'status-fail'
        status_display = '✓ PASSED' if test['status'].lower() == 'passed' else '✗ FAILED'
        
        html_content += f'''
            <tr>
                <td>{html.escape(test['name'])}</td>
                <td><span class='status-badge {status_class}'>{status_display}</span></td>
                <td>{test['time']}</td>
                <td><pre>{html.escape(test['output'])}</pre></td>
            </tr>
'''
    
    html_content += '''
        </table>
        
        <div class='footer'>
            <p>Generated by BPE Tokenizer Test Suite • ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        </div>
    </div>
</body>
</html>
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f'HTML отчет сгенерирован: {output_file}')

# Основной вызов
filter_str = '$TEST_FILTER' if '$TEST_FILTER' else 'all'
tests = parse_test_results('$TEST_FILE')
if tests:
    generate_html(tests, '$HTML_REPORT', '$TIMESTAMP', $PASSED, $FAILED, $TOTAL)
else:
    print('Нет данных для отчета')
    sys.exit(1)
"
            
            if [ $? -eq 0 ]; then
                print_success "HTML отчет сгенерирован: $HTML_REPORT"
                
                # Открываем в браузере
                if [ $OPEN_REPORT -eq 1 ]; then
                    if command -v firefox &> /dev/null; then
                        firefox "$HTML_REPORT" &
                    elif command -v google-chrome &> /dev/null; then
                        google-chrome "$HTML_REPORT" &
                    elif command -v xdg-open &> /dev/null; then
                        xdg-open "$HTML_REPORT" &
                    fi
                fi
            fi
        else
            print_warning "Файл с результатами тестов не найден: $TEST_FILE"
        fi
    else
        print_warning "Тесты не запускались или не создали TAG файл"
    fi
fi

# ======================================================================
# Итог
# ======================================================================
print_header "ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!"

if [ $RUN_MEMCHECK -eq 0 ] && [ $RUN_COVERAGE -eq 0 ] && [ $TEST_RESULT -eq 0 ]; then
    print_success "Все тесты пройдены успешно!"
elif [ $TEST_RESULT -ne 0 ]; then
    print_error "Некоторые тесты не пройдены. Проверьте вывод выше."
fi

print_info "Отчеты сохранены в: $REPORTS_DIR"

echo ""
echo "Сгенерированные файлы:"
ls -lt "$REPORTS_DIR" | head -10 | awk '{printf "  %s %s %s\n", $6, $7, $9}'

echo ""
print_success "Готово!"
exit $TEST_RESULT