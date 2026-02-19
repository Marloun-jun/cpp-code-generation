#!/bin/bash
# ======================================================================
# build.sh - Скрипт сборки BPE токенизатора
# ======================================================================
#
# @file build.sh
# @brief Универсальный скрипт для сборки проекта с различными опциями
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @usage ./build.sh [options] [build_type]
#   build_type: Debug, Release, RelWithDebInfo (по умолчанию: Release)
#
#   Options:
#     --clean     - Очистить перед сборкой
#     --test      - Запустить тесты после сборки
#     --benchmark - Запустить бенчмарки после сборки
#     --install   - Установить после сборки
#     --verbose   - Подробный вывод
#     --help      - Показать эту справку
#
# @example
#   ./build.sh                    # Release сборка
#   ./build.sh Debug --clean       # Debug с очисткой
#   ./build.sh --test --benchmark  # Release + тесты + бенчмарки
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
BUILD_TYPE="Release"
CLEAN_BUILD=0
RUN_TESTS=0
RUN_BENCHMARKS=0
INSTALL_AFTER_BUILD=0
VERBOSE=0
GENERATOR="Unix Makefiles"
BUILD_DIR="$PROJECT_ROOT/build"
INSTALL_DIR="$PROJECT_ROOT/install"

# Обработка аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --test)
            RUN_TESTS=1
            shift
            ;;
        --benchmark)
            RUN_BENCHMARKS=1
            shift
            ;;
        --install)
            INSTALL_AFTER_BUILD=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --help)
            echo "Использование: $0 [options] [build_type]"
            echo ""
            echo "Тип сборки:"
            echo "  Debug          - Отладочная сборка"
            echo "  Release        - Оптимизированная сборка (по умолчанию)"
            echo "  RelWithDebInfo - Оптимизированная с отладочной информацией"
            echo ""
            echo "Опции:"
            echo "  --clean        - Очистить перед сборкой"
            echo "  --test         - Запустить тесты после сборки"
            echo "  --benchmark    - Запустить бенчмарки после сборки"
            echo "  --install      - Установить после сборки"
            echo "  --verbose      - Подробный вывод CMake"
            echo "  --help         - Показать эту справку"
            echo ""
            echo "Примеры:"
            echo "  $0                         # Release сборка"
            echo "  $0 Debug --clean            # Debug с очисткой"
            echo "  $0 --test --benchmark       # Release + тесты + бенчмарки"
            exit 0
            ;;
        Debug|Release|RelWithDebInfo)
            BUILD_TYPE="$1"
            shift
            ;;
        *)
            print_error "Неизвестная опция: $1"
            echo "Используйте --help для справки"
            exit 1
            ;;
    esac
done

# ======================================================================
# Начало сборки
# ======================================================================
print_header "🔧 BPE TOKENIZER - СБОРКА ПРОЕКТА"

print_info "Директория проекта: $PROJECT_ROOT"
print_info "Тип сборки: $BUILD_TYPE"
print_info "Директория сборки: $BUILD_DIR"

# Проверка наличия необходимых инструментов
print_info "Проверка инструментов..."

if ! command -v cmake &> /dev/null; then
    print_error "CMake не найден!"
    echo "Установите CMake: sudo apt install cmake"
    exit 1
fi
print_success "CMake найден: $(cmake --version | head -n1)"

if ! command -v make &> /dev/null; then
    print_error "Make не найден!"
    echo "Установите build-essential: sudo apt install build-essential"
    exit 1
fi
print_success "Make найден"

if ! command -v g++ &> /dev/null; then
    print_warning "g++ не найден, проверьте компилятор"
else
    print_success "g++ найден: $(g++ --version | head -n1)"
fi

# Очистка если нужно
if [ $CLEAN_BUILD -eq 1 ]; then
    print_info "Очистка директории сборки..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Создаем директорию сборки
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ======================================================================
# Конфигурация CMake
# ======================================================================
print_header "📦 КОНФИГУРАЦИЯ CMAKE"

CMAKE_OPTIONS="
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    -DBUILD_TESTING=ON
    -DBUILD_BENCHMARKS=ON
    -DBUILD_EXAMPLES=ON
    -DBUILD_PYTHON_BINDINGS=ON
    -DUSE_OPENMP=ON
    -DUSE_SIMD=ON
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
"

if [ $VERBOSE -eq 1 ]; then
    CMAKE_OPTIONS="$CMAKE_OPTIONS --log-level=VERBOSE"
fi

print_info "Запуск CMake с опциями:"
echo "$CMAKE_OPTIONS" | sed 's/    -/  -/g'

# Запуск CMake
cmake .. $CMAKE_OPTIONS

if [ $? -ne 0 ]; then
    print_error "Ошибка конфигурации CMake!"
    exit 1
fi

print_success "Конфигурация завершена"

# ======================================================================
# Сборка
# ======================================================================
print_header "🔨 СБОРКА"

# Определяем количество ядер для параллельной сборки
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    CORES=$(nproc)
fi

print_info "Сборка с использованием $CORES ядер..."

if [ $VERBOSE -eq 1 ]; then
    make VERBOSE=1 -j$CORES
else
    make -j$CORES
fi

if [ $? -ne 0 ]; then
    print_error "Ошибка сборки!"
    exit 1
fi

print_success "Сборка завершена"

# ======================================================================
# Запуск тестов
# ======================================================================
if [ $RUN_TESTS -eq 1 ]; then
    print_header "🧪 ЗАПУСК ТЕСТОВ"
    
    print_info "Запуск модульных тестов..."
    ctest --output-on-failure -j$CORES
    
    if [ $? -ne 0 ]; then
        print_error "Некоторые тесты не пройдены!"
    else
        print_success "Все тесты пройдены!"
    fi
fi

# ======================================================================
# Запуск бенчмарков
# ======================================================================
if [ $RUN_BENCHMARKS -eq 1 ]; then
    print_header "📊 ЗАПУСК БЕНЧМАРКОВ"
    
    if [ -f "./benchmarks/bpe_benchmarks" ]; then
        print_info "Запуск бенчмарков..."
        ./benchmarks/bpe_benchmarks --benchmark_out=benchmark_results.json --benchmark_out_format=json
        
        if [ $? -eq 0 ]; then
            print_success "Бенчмарки завершены. Результаты в benchmark_results.json"
        fi
    else
        print_warning "Бенчмарки не найдены. Соберите с -DBUILD_BENCHMARKS=ON"
    fi
fi

# ======================================================================
# Установка
# ======================================================================
if [ $INSTALL_AFTER_BUILD -eq 1 ]; then
    print_header "📦 УСТАНОВКА"
    
    print_info "Установка в $INSTALL_DIR..."
    make install
    
    if [ $? -eq 0 ]; then
        print_success "Установка завершена"
        
        echo ""
        echo "Установленные файлы:"
        ls -la "$INSTALL_DIR"
    fi
fi

# ======================================================================
# Информация о результатах
# ======================================================================
print_header "✅ СБОРКА ЗАВЕРШЕНА"

print_success "Бинарные файлы в: $BUILD_DIR"

echo ""
echo "📁 Основные компоненты:"
echo "  $(ls -lh examples/simple_example 2>/dev/null || echo '⚠️ simple_example не найден')"
echo "  $(ls -lh examples/batch_example 2>/dev/null || echo '⚠️ batch_example не найден')"
echo "  $(ls -lh tests/bpe_tests 2>/dev/null || echo '⚠️ bpe_tests не найден')"
echo "  $(ls -lh benchmarks/bpe_benchmarks 2>/dev/null || echo '⚠️ bpe_benchmarks не найден')"

echo ""
echo "📊 Размеры файлов:"
du -sh "$BUILD_DIR" 2>/dev/null || echo "  Не удалось определить размер"

echo ""
echo "🚀 Быстрые команды:"
echo "  ./build.sh --test          # Сборка + тесты"
echo "  ./build.sh --benchmark      # Сборка + бенчмарки"
echo "  ./build.sh Debug --clean    # Отладка с очисткой"
echo ""

# ======================================================================
# Конец
# ======================================================================
exit 0