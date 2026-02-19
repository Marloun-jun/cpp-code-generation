#!/bin/bash
# ======================================================================
# run_tests.sh - Запуск тестов BPE токенизатора
# ======================================================================
#
# @file run_tests.sh
# @brief Запуск всех модульных тестов и анализ покрытия
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @usage ./run_tests.sh [options]
#   --filter FILTER   Фильтр для запуска конкретных тестов (например: *Tokenizer*)
#   --coverage        Запустить с анализом покрытия кода (требует gcov/lcov)
#   --parallel N      Запустить тесты в N параллельных потоков
#   --repeat N        Повторить каждый тест N раз
#   --verbose         Подробный вывод
#   --xml             Сгенерировать JUnit XML отчет
#   --html            Сгенерировать HTML отчет о тестах
#   --memcheck        Запустить с Valgrind для проверки утечек памяти
#   --help            Показать справку
#
# @example
#   ./run_tests.sh --filter *Tokenizer* --verbose
#   ./run_tests.sh --coverage --html
#   ./run_tests.sh --memcheck --parallel 4
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
TEST_FILTER=""
RUN_COVERAGE=0
PARALLEL_JOBS=1
REPEAT_COUNT=1
VERBOSE=0
GEN_XML=0
GEN_HTML=0
RUN_MEMCHECK=0
BUILD_DIR="$PROJECT_ROOT/cpp/build"
REPORTS_DIR="$PROJECT_ROOT/reports/tests"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$REPORTS_DIR"

# Обработка аргументов
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
        --memcheck)
            RUN_MEMCHECK=1
            shift
            ;;
        --help)
            echo "Использование: $0 [options]"
            echo ""
            echo "Опции:"
            echo "  --filter FILTER   Фильтр тестов (например: *Tokenizer*)"
            echo "  --coverage        Анализ покрытия кода (gcov/lcov)"
            echo "  --parallel N      Параллельный запуск N потоков"
            echo "  --repeat N        Повторить тесты N раз"
            echo "  --verbose         Подробный вывод"
            echo "  --xml             JUnit XML отчет"
            echo "  --html            HTML отчет о тестах"
            echo "  --memcheck        Проверка утечек памяти (Valgrind)"
            echo "  --help            Показать справку"
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
print_header "🧪 ЗАПУСК ТЕСТОВ BPE TOKENIZER"

print_info "Директория тестов: $BUILD_DIR"
print_info "Директория отчетов: $REPORTS_DIR"

if [ -n "$TEST_FILTER" ]; then
    print_info "Фильтр: $TEST_FILTER"
fi

if [ $RUN_COVERAGE -eq 1 ]; then
    print_info "Режим: с покрытием кода"
fi

if [ $RUN_MEMCHECK -eq 1 ]; then
    print_info "Режим: проверка памяти (Valgrind)"
fi

# Проверка наличия сборки
if [ ! -d "$BUILD_DIR" ]; then
    print_warning "Директория сборки не найдена. Запуск сборки..."
    "$SCRIPT_DIR/build.sh" Debug
fi

cd "$BUILD_DIR"

# ======================================================================
# Сборка с флагами покрытия если нужно
# ======================================================================
if [ $RUN_COVERAGE -eq 1 ]; then
    print_info "Пересборка с флагами покрытия..."
    
    # Проверка наличия gcov
    if ! command -v gcov &> /dev/null; then
        print_error "gcov не найден. Установите gcc"
        exit 1
    fi
    
    # Пересборка с флагами покрытия
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage -fprofile-arcs -ftest-coverage" \
        -DBUILD_TESTING=ON
    
    make clean
    make -j$(nproc)
fi

# ======================================================================
# Запуск тестов
# ======================================================================
print_header "🏃 ЗАПУСК ТЕСТОВ"

# Формируем команду ctest
CTEST_CMD="ctest"

if [ -n "$TEST_FILTER" ]; then
    CTEST_CMD="$CTEST_CMD -R \"$TEST_FILTER\""
fi

if [ $PARALLEL_JOBS -gt 1 ]; then
    CTEST_CMD="$CTEST_CMD -j$PARALLEL_JOBS"
fi

if [ $REPEAT_COUNT -gt 1 ]; then
    CTEST_CMD="$CTEST_CMD --repeat until-fail:$REPEAT_COUNT"
fi

if [ $VERBOSE -eq 1 ]; then
    CTEST_CMD="$CTEST_CMD --verbose"
else
    CTEST_CMD="$CTEST_CMD --output-on-failure"
fi

if [ $GEN_XML -eq 1 ]; then
    CTEST_CMD="$CTEST_CMD --output-junit $REPORTS_DIR/junit_${TIMESTAMP}.xml"
fi

# Запуск с Valgrind если нужно
if [ $RUN_MEMCHECK -eq 1 ]; then
    print_info "Запуск с Valgrind memcheck..."
    
    if ! command -v valgrind &> /dev/null; then
        print_error "Valgrind не найден. Установите valgrind"
        exit 1
    fi
    
    # Находим все тестовые исполняемые файлы
    TEST_EXES=$(find . -type f -executable -name "test_*" -o -name "*_tests")
    
    for test_exe in $TEST_EXES; do
        test_name=$(basename "$test_exe")
        print_info "Проверка $test_name..."
        
        valgrind --leak-check=full \
            --show-leak-kinds=all \
            --track-origins=yes \
            --verbose \
            --log-file="$REPORTS_DIR/valgrind_${test_name}_${TIMESTAMP}.log" \
            "$test_exe" --gtest_filter="$TEST_FILTER" 2>&1 | tee -a "$REPORTS_DIR/valgrind_output_${TIMESTAMP}.log"
    done
    
    print_success "Проверка памяти завершена"
    echo "  Отчеты: $REPORTS_DIR/valgrind_*"
else
    # Обычный запуск тестов
    print_info "Запуск: $CTEST_CMD"
    eval $CTEST_CMD
    
    # Сохраняем код возврата
    TEST_RESULT=$?
fi

# ======================================================================
# Анализ покрытия
# ======================================================================
if [ $RUN_COVERAGE -eq 1 ]; then
    print_header "📊 АНАЛИЗ ПОКРЫТИЯ КОДА"
    
    # Проверка наличия lcov
    if ! command -v lcov &> /dev/null; then
        print_warning "lcov не найден. Установите lcov для генерации отчетов"
        print_info "sudo apt install lcov"
    else
        COVERAGE_DIR="$REPORTS_DIR/coverage_${TIMESTAMP}"
        mkdir -p "$COVERAGE_DIR"
        
        print_info "Сбор данных покрытия..."
        
        # Сбор данных покрытия
        lcov --capture \
            --directory . \
            --output-file "$COVERAGE_DIR/coverage.info" \
            --rc lcov_branch_coverage=1 \
            --exclude '/usr/*' \
            --exclude '*/tests/*' \
            --exclude '*/googletest/*' \
            --exclude '*/benchmarks/*'
        
        # Генерация HTML отчета
        genhtml "$COVERAGE_DIR/coverage.info" \
            --branch-coverage \
            --output-directory "$COVERAGE_DIR/html" \
            --title "BPE Tokenizer Coverage Report" \
            --legend \
            --demangle-cpp
        
        print_success "Отчет о покрытии сгенерирован"
        echo "  HTML: $COVERAGE_DIR/html/index.html"
        
        # Показываем статистику
        echo ""
        echo "📈 Статистика покрытия:"
        lcov --summary "$COVERAGE_DIR/coverage.info" | grep -E "lines|functions|branches" | sed 's/^/  /'
        
        if command -v firefox &> /dev/null; then
            firefox "$COVERAGE_DIR/html/index.html" &
        fi
    fi
fi

# ======================================================================
# Генерация HTML отчета
# ======================================================================
if [ $GEN_HTML -eq 1 ]; then
    print_header "📄 ГЕНЕРАЦИЯ HTML ОТЧЕТА"
    
    HTML_REPORT="$REPORTS_DIR/test_report_${TIMESTAMP}.html"
    
    # Собираем информацию о тестах
    if [ -f "$BUILD_DIR/Testing/TAG" ]; then
        TEST_DIR=$(cat "$BUILD_DIR/Testing/TAG")
        TEST_FILE="$BUILD_DIR/Testing/$TEST_DIR/Test.xml"
        
        if [ -f "$TEST_FILE" ]; then
            python3 -c "
import xml.etree.ElementTree as ET
import html

def parse_test_results(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    tests = []
    for test in root.findall('.//Test'):
        name = test.get('name', 'unknown')
        status = test.find('Results/TestStatus').text if test.find('Results/TestStatus') is not None else 'Unknown'
        time = test.find('Results/NamedMeasurement/Value').text if test.find('Results/NamedMeasurement/Value') is not None else '0'
        
        tests.append({
            'name': name,
            'status': status,
            'time': time
        })
    
    return tests

def generate_html(tests, output_file):
    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>BPE Tokenizer Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .pass {{ color: green; }}
        .fail {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>🧪 BPE Tokenizer Test Report</h1>
    <p class='timestamp'>Generated: $TIMESTAMP</p>
    
    <div class='summary'>
        <h2>Summary</h2>
        <ul>
'''
    
    passed = sum(1 for t in tests if t['status'] == 'passed')
    failed = sum(1 for t in tests if t['status'] != 'passed')
    total = len(tests)
    
    html_content += f'''
            <li>Total tests: {total}</li>
            <li class='pass'>Passed: {passed}</li>
            <li class='fail'>Failed: {failed}</li>
            <li>Pass rate: {(passed/total*100):.1f}%</li>
        </ul>
    </div>
    
    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Time (s)</th>
        </tr>
'''
    
    for test in tests:
        status_class = 'pass' if test['status'] == 'passed' else 'fail'
        html_content += f'''
        <tr>
            <td>{html.escape(test['name'])}</td>
            <td class='{status_class}'>{test['status']}</td>
            <td>{test['time']}</td>
        </tr>
'''
    
    html_content += '''
    </table>
</body>
</html>
'''
    
    with open(output_file, 'w') as f:
        f.write(html_content)

tests = parse_test_results('$TEST_FILE')
generate_html(tests, '$HTML_REPORT')
print(f'✅ HTML отчет сгенерирован: $HTML_REPORT')
"
            
            if command -v firefox &> /dev/null; then
                firefox "$HTML_REPORT" &
            fi
        fi
    fi
fi

# ======================================================================
# Итог
# ======================================================================
print_header "✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО"

if [ $RUN_MEMCHECK -eq 0 ]; then
    if [ $TEST_RESULT -eq 0 ]; then
        print_success "Все тесты пройдены успешно!"
    else
        print_error "Некоторые тесты не пройдены. Проверьте вывод выше."
    fi
fi

print_info "Отчеты сохранены в: $REPORTS_DIR"

echo ""
echo "📁 Сгенерированные файлы:"
ls -lt "$REPORTS_DIR" | head -10 | awk '{printf "  %s %s\n", $5, $9}'

echo ""
print_success "Готово!"
exit ${TEST_RESULT:-0}