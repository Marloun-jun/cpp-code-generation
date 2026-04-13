#!/bin/bash
# ============================================================================
# build.sh - Универсальный скрипт сборки BPE токенизатора
# ============================================================================
#
# @file build.sh
# @brief Комплексный скрипт для сборки, тестирования и установки проекта
#
# @author Евгений П.
# @date 2026
# @version 3.5.0
#
# @details Этот скрипт автоматизирует процесс сборки проекта с CMake,
#          предоставляя удобный интерфейс для различных сценариев:
#
#          **Основные возможности:**
#          ┌─────────────────────┬───────────────────────────────────┐
#          │ Типы сборки         │ Debug, Release, RelWithDebInfo    │
#          │ Очистка             │ Полное удаление директории сборки │
#          │ Тестирование        │ Запуск модульных тестов           │
#          │ Бенчмарки           │ Запуск тестов производительности  │
#          │ Установка           │ Копирование файлов в install/     │
#          │ Параллельная сборка │ Автоматическое определение ядер   │
#          │ Генераторы          │ Поддержка Make и Ninja            │
#          └─────────────────────┴───────────────────────────────────┘
#
#          **Процесс работы:**
#          1. Проверка наличия необходимых инструментов (cmake, make, g++)
#          2. Конфигурация CMake с выбранными опциями
#          3. Параллельная сборка проекта
#          4. Опциональный запуск тестов/бенчмарков
#          5. Опциональная установка
#
# @usage ./build.sh [options] [build_type]
#   build_type: Debug, Release, RelWithDebInfo, MinSizeRel (по умолчанию: Release)
#
#   Options:
#     --clean     - Очистить перед сборкой
#     --test      - Запустить тесты после сборки
#     --benchmark - Запустить бенчмарки после сборки
#     --install   - Установить после сборки
#     --verbose   - Подробный вывод
#     --generator - Генератор CMake (make или ninja)
#     --help      - Показать эту справку
#
# @example
#   ./build.sh                       # Release сборка
#   ./build.sh Debug --clean         # Debug с очисткой
#   ./build.sh --test --benchmark    # Release + тесты + бенчмарки
#   ./build.sh --generator ninja     # Сборка с Ninja
#
# ============================================================================

# ============================================================================
# Настройки безопасности
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

# Информационное сообщение (синий)
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Сообщение об успехе (зеленый)
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Предупреждение (желтый)
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Ошибка (красный)
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Заголовок раздела (пурпурный + голубой)
print_header() {
    echo -e "\n${MAGENTA}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${MAGENTA}============================================================${NC}\n"
}

# Команда (голубой)
print_command() {
    echo -e "${CYAN}  > $1${NC}"
}

# ============================================================================
# Функции для проверки инструментов
# ============================================================================

check_command() {
    local cmd=$1
    local name=$2
    local install_hint=$3
    
    if ! command -v "$cmd" &> /dev/null; then
        print_error "$name не найден!"
        echo -e "$install_hint"
        return 1
    else
        print_success "$name найден: $($cmd --version | head -n1)"
        return 0
    fi
}

# ============================================================================
# Парсинг аргументов командной строки
# ============================================================================

# Определение путей (скрипт может быть запущен из любой директории)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"    # Переходим в корень проекта

# Значения по умолчанию
BUILD_TYPE="Release"                   # Тип сборки по умолчанию
CLEAN_BUILD=0                          # Флаг очистки
RUN_TESTS=0                            # Флаг запуска тестов
RUN_BENCHMARKS=0                       # Флаг запуска бенчмарков
INSTALL_AFTER_BUILD=0                  # Флаг установки
VERBOSE=0                              # Флаг подробного вывода
GENERATOR="Unix Makefiles"             # Генератор CMake
BUILD_DIR="$PROJECT_ROOT/build"        # Директория сборки
INSTALL_DIR="$PROJECT_ROOT/install"    # Директория установки

# Обработка аргументов командной строки
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
        --generator)
            shift
            if [[ $# -gt 0 ]]; then
                case $1 in
                    make|Make)
                        GENERATOR="Unix Makefiles"
                        ;;
                    ninja|Ninja)
                        GENERATOR="Ninja"
                        ;;
                    *)
                        print_error "Неизвестный генератор: $1"
                        echo "Доступные генераторы: make, ninja"
                        exit 1
                        ;;
                esac
                shift
            else
                print_error "Не указан генератор после --generator"
                exit 1
            fi
            ;;
        --help)
            # Вывод справки
            echo "Использование: $0 [options] [build_type]"
            echo ""
            echo "Тип сборки:"
            echo "  Debug          - Отладочная сборка (без оптимизаций, с символами)"
            echo "  Release        - Оптимизированная сборка (по умолчанию)"
            echo "  RelWithDebInfo - Оптимизированная с отладочной информацией"
            echo "  MinSizeRel     - Оптимизированная для минимального размера"
            echo ""
            echo "Опции:"
            echo "  --clean         - Очистить директорию сборки перед сборкой"
            echo "  --test          - Запустить модульные тесты после сборки"
            echo "  --benchmark     - Запустить бенчмарки после сборки"
            echo "  --install       - Установить собранные файлы"
            echo "  --verbose       - Подробный вывод CMake и make"
            echo "  --generator GEN - Генератор CMake (make или ninja)"
            echo "  --help          - Показать эту справку"
            echo ""
            echo "Примеры:"
            echo "  $0                       # Release сборка"
            echo "  $0 Debug --clean         # Debug с очисткой"
            echo "  $0 --test --benchmark    # Release + тесты + бенчмарки"
            echo "  $0 --generator ninja     # Сборка с Ninja"
            exit 0
            ;;
        Debug|Release|RelWithDebInfo|MinSizeRel)
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

# ============================================================================
# Начало сборки
# ============================================================================
print_header "BPE TOKENIZER - СБОРКА ПРОЕКТА"

print_info "Директория проекта:   $PROJECT_ROOT"
print_info "Тип сборки:           $BUILD_TYPE"
print_info "Генератор:            $GENERATOR"
print_info "Директория сборки:    $BUILD_DIR"
print_info "Директория установки: $INSTALL_DIR"

# ============================================================================
# Проверка наличия необходимых инструментов
# ============================================================================
print_info "Проверка инструментов сборки..."

TOOLS_OK=0

# Проверка CMake
check_command "cmake" "CMake" \
    "Установите CMake:\n  Ubuntu/Debian: sudo apt install cmake\n  macOS: brew install cmake\n  Windows: скачайте с https://cmake.org/download/" || TOOLS_OK=1

# Проверка генератора
if [[ "$GENERATOR" == "Unix Makefiles" ]]; then
    check_command "make" "Make" \
        "Установите build-essential:\n  Ubuntu/Debian: sudo apt install build-essential\n  macOS: xcode-select --install" || TOOLS_OK=1
elif [[ "$GENERATOR" == "Ninja" ]]; then
    check_command "ninja" "Ninja" \
        "Установите Ninja:\n  Ubuntu/Debian: sudo apt install ninja-build\n  macOS: brew install ninja\n  Windows: скачайте с https://ninja-build.org/" || TOOLS_OK=1
fi

# Проверка компилятора (g++ или clang++)
if command -v g++ &> /dev/null; then
    print_success "Компилятор найден: $(g++ --version | head -n1)"
elif command -v clang++ &> /dev/null; then
    print_success "Компилятор найден: $(clang++ --version | head -n1)"
else
    print_warning "Ни g++, ни clang++ не найдены. Будет использован компилятор по умолчанию."
fi

if [ $TOOLS_OK -ne 0 ]; then
    print_error "Отсутствуют необходимые инструменты. Установите их и повторите попытку."
    exit 1
fi

# ============================================================================
# Очистка директории сборки (если запрошено)
# ============================================================================
if [ $CLEAN_BUILD -eq 1 ]; then
    print_info "Очистка директории сборки..."
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    print_success "Очистка завершена!"
fi

# Создаем директорию сборки, если её нет
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ============================================================================
# Конфигурация CMake
# ============================================================================
print_header "КОНФИГУРАЦИЯ CMAKE"

# Формирование опций CMake
CMAKE_OPTIONS=(
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DBUILD_TESTING=ON"
    "-DBUILD_BENCHMARKS=ON"
    "-DBUILD_EXAMPLES=ON"
    "-DBUILD_PYTHON_BINDINGS=ON"
    "-DUSE_OPENMP=ON"
    "-DUSE_SIMD=ON"
    "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
    "-G" "$GENERATOR"
)

print_info "Запуск CMake с опциями:"
printf '%s\n' "${CMAKE_OPTIONS[@]}" | sed 's/^/  /'

# Запуск CMake
print_command "cmake .. ${CMAKE_OPTIONS[*]}"
if [ $VERBOSE -eq 1 ]; then
    cmake .. "${CMAKE_OPTIONS[@]}" --log-level=VERBOSE
else
    cmake .. "${CMAKE_OPTIONS[@]}"
fi

if [ $? -ne 0 ]; then
    print_error "Ошибка конфигурации CMake!"
    exit 1
fi

print_success "Конфигурация завершена!"

# ============================================================================
# Сборка проекта
# ============================================================================
print_header "СБОРКА"

# Автоматическое определение количества ядер процессора
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    CORES=$(sysctl -n hw.ncpu)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    CORES=$(nproc 2>/dev/null || echo 2)
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    CORES=$(nproc 2>/dev/null || echo 2)
else
    # Другие системы (fallback)
    CORES=2
fi

print_info "Сборка с использованием $CORES ядер..."

# Сборка с опциями
if [[ "$GENERATOR" == "Unix Makefiles" ]]; then
    if [ $VERBOSE -eq 1 ]; then
        print_command "make VERBOSE=1 -j$CORES"
        make VERBOSE=1 -j$CORES
    else
        print_command "make -j$CORES"
        make -j$CORES
    fi
elif [[ "$GENERATOR" == "Ninja" ]]; then
    if [ $VERBOSE -eq 1 ]; then
        print_command "ninja -v -j$CORES"
        ninja -v -j$CORES
    else
        print_command "ninja -j$CORES"
        ninja -j$CORES
    fi
fi

if [ $? -ne 0 ]; then
    print_error "Ошибка сборки!"
    exit 1
fi

print_success "Сборка завершена!"

# ============================================================================
# Запуск модульных тестов (если запрошено)
# ============================================================================
if [ $RUN_TESTS -eq 1 ]; then
    print_header "ЗАПУСК ТЕСТОВ"
    
    print_info "Запуск модульных тестов..."
    print_command "ctest --output-on-failure -j$CORES"
    
    # Запуск тестов с выводом результатов
    ctest --output-on-failure -j$CORES
    
    TEST_RESULT=$?
    if [ $TEST_RESULT -ne 0 ]; then
        print_error "Некоторые тесты не пройдены! (код: $TEST_RESULT)"
    else
        print_success "Все тесты пройдены успешно!"
    fi
    
    # Дополнительная информация о тестах
    if [ -f "Testing/TAG" ]; then
        TEST_DIR=$(cat Testing/TAG | head -n1)
        if [ -f "Testing/$TEST_DIR/Test.xml" ]; then
            echo ""
            echo "Подробный отчет: Testing/$TEST_DIR/Test.xml"
        fi
    fi
fi

# ============================================================================
# Запуск бенчмарков (если запрошено)
# ============================================================================
if [ $RUN_BENCHMARKS -eq 1 ]; then
    print_header "ЗАПУСК БЕНЧМАРКОВ"
    
    # Создаем директорию для результатов
    RESULTS_DIR="benchmark_results"
    mkdir -p "$RESULTS_DIR"
    
    # Проверка наличия бенчмарков
    BENCHMARKS_FOUND=0
    
    # Список возможных бенчмарков
    BENCHMARK_LIST=(
        "./benchmarks/bench_tokenizer"
        "./benchmarks/bench_fast_tokenizer"
        "./benchmarks/bench_comparison"
    )
    
    for bench in "${BENCHMARK_LIST[@]}"; do
        if [ -f "$bench" ]; then
            BENCHMARKS_FOUND=1
            bench_name=$(basename "$bench")
            print_info "Запуск бенчмарка: $bench_name..."
            
            # Имя файла с результатами (с временной меткой)
            timestamp=$(date +%Y%m%d_%H%M%S)
            benchmark_file="$RESULTS_DIR/${bench_name}_${timestamp}.json"
            benchmark_csv="$RESULTS_DIR/${bench_name}_${timestamp}.csv"
            
            print_command "$bench --benchmark_out=$benchmark_file --benchmark_out_format=json"
            "$bench" --benchmark_out="$benchmark_file" --benchmark_out_format=json
            
            if [ $? -eq 0 ]; then
                print_success "Бенчмарк завершен."
                echo "Результаты сохранены в: $benchmark_file"
                
                # Конвертируем в CSV для удобства
                print_command "$bench --benchmark_out=$benchmark_csv --benchmark_out_format=csv"
                "$bench" --benchmark_out="$benchmark_csv" --benchmark_out_format=csv 2>/dev/null || true
                
                # Показываем краткую статистику
                if [ -f "$benchmark_file" ]; then
                    echo ""
                    echo "Первые результаты:"
                    head -n 5 "$benchmark_file" 2>/dev/null | sed 's/^/  /' || echo "  (файл пуст)"
                    echo "  ..."
                fi
            else
                print_error "Ошибка при запуске бенчмарка!"
            fi
            echo ""
        fi
    done
    
    if [ $BENCHMARKS_FOUND -eq 0 ]; then
        print_warning "Бенчмарки не найдены. Соберите проект с -DBUILD_BENCHMARKS=ON"
        echo "Ожидаемые пути:"
        for bench in "${BENCHMARK_LIST[@]}"; do
            echo "  $bench"
        done
    else
        echo "Все результаты бенчмарков сохранены в: $RESULTS_DIR/"
    fi
fi

# ============================================================================
# Установка проекта (если запрошено)
# ============================================================================
if [ $INSTALL_AFTER_BUILD -eq 1 ]; then
    print_header "УСТАНОВКА"
    
    print_info "Установка в $INSTALL_DIR..."
    
    if [[ "$GENERATOR" == "Unix Makefiles" ]]; then
        print_command "make install"
        make install
    elif [[ "$GENERATOR" == "Ninja" ]]; then
        print_command "ninja install"
        ninja install
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Установка завершена!"
        
        # Показываем установленные файлы
        echo ""
        echo "Установленные файлы:"
        if [ -d "$INSTALL_DIR" ]; then
            ls -la "$INSTALL_DIR" | sed 's/^/  /'
            
            # Показываем структуру
            echo ""
            echo "Структура установки:"
            find "$INSTALL_DIR" -type d | sed "s|$INSTALL_DIR/||" | sed 's/^/  ├-- /'
        fi
    else
        print_error "Ошибка при установке!"
    fi
fi

# ============================================================================
# Итоговая информация
# ============================================================================
print_header "СБОРКА ЗАВЕРШЕНА!"

print_success "Бинарные файлы находятся в: $BUILD_DIR"

echo ""
echo "Основные компоненты:"

# Проверка наличия основных исполняемых файлов
check_file() {
    local path="$1"
    local desc="$2"
    if [ -f "$path" ]; then
        local size=$(du -h "$path" 2>/dev/null | cut -f1)
        echo "  $desc: ${CYAN}$(basename "$path")${NC} (${size})"
    else
        echo "  $desc: ${YELLOW}не найден${NC}"
    fi
}

check_file "examples/simple_example"      "Простой пример"
check_file "examples/batch_example"       "Пакетная обработка"
check_file "examples/fast_tokenizer_demo" "Демо FastTokenizer"
check_file "examples/train_example"       "Пример обучения"

# Проверка тестов (может быть несколько)
TEST_FOUND=0
for test in tests/test_*; do
    if [ -f "$test" ] && [ -x "$test" ]; then
        TEST_FOUND=1
        break
    fi
done
if [ $TEST_FOUND -eq 1 ]; then
    echo "Модульные тесты: ${CYAN}найдены${NC}"
else
    echo "Модульные тесты: ${YELLOW}не найдены${NC}"
fi

# Проверка бенчмарков
BENCH_FOUND=0
for bench in benchmarks/bench_*; do
    if [ -f "$bench" ] && [ -x "$bench" ]; then
        BENCH_FOUND=1
        break
    fi
done
if [ $BENCH_FOUND -eq 1 ]; then
    echo "Бенчмарки: ${CYAN}найдены${NC}"
else
    echo "Бенчмарки: ${YELLOW}не найдены${NC}"
fi

echo ""
echo "Размер директории сборки:"
if [ -d "$BUILD_DIR" ]; then
    du -sh "$BUILD_DIR" 2>/dev/null | sed 's/^/  /' || echo "  Не удалось определить размер!"
fi

echo ""
echo "Быстрые команды для дальнейшей работы:"
echo "  ${CYAN}./build.sh --test${NC}                   # Сборка + тесты"
echo "  ${CYAN}./build.sh --benchmark${NC}              # Сборка + бенчмарки"
echo "  ${CYAN}./build.sh Debug --clean${NC}            # Отладка с очисткой"
echo "  ${CYAN}cd build && ctest -V${NC}                # Подробный запуск тестов"
echo "  ${CYAN}cd build && ninja run_benchmarks${NC}    # Запуск всех бенчмарков (Ninja)"

echo ""
print_success "Скрипт завершен успешно!"

# ============================================================================
# Конец скрипта
# ============================================================================
exit 0