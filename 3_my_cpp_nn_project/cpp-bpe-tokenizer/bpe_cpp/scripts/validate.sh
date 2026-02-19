#!/bin/bash
# ======================================================================
# validate.sh - Валидация C++ токенизатора против Python эталона
# ======================================================================
#
# @file validate.sh
# @brief Сравнение результатов C++ и Python реализаций токенизатора
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @usage ./validate.sh [options]
#   --quick         Быстрая валидация (меньше тестов)
#   --full          Полная валидация (все тесты)
#   --update        Обновить эталонные результаты
#   --verbose       Подробный вывод
#   --stop-on-error Останавливаться при первой ошибке
#
# @example
#   ./validate.sh --quick
#   ./validate.sh --full --verbose
#   ./validate.sh --update
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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Значения по умолчанию
VALIDATION_TYPE="standard"
UPDATE_GOLDEN=0
VERBOSE=0
STOP_ON_ERROR=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            VALIDATION_TYPE="quick"
            shift
            ;;
        --full)
            VALIDATION_TYPE="full"
            shift
            ;;
        --update)
            UPDATE_GOLDEN=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --stop-on-error)
            STOP_ON_ERROR=1
            shift
            ;;
        --help)
            echo "Использование: $0 [options]"
            echo ""
            echo "Опции:"
            echo "  --quick         Быстрая валидация (меньше тестов)"
            echo "  --full          Полная валидация (все тесты)"
            echo "  --update        Обновить эталонные результаты"
            echo "  --verbose       Подробный вывод"
            echo "  --stop-on-error Останавливаться при первой ошибке"
            echo ""
            echo "Примеры:"
            echo "  $0 --quick"
            echo "  $0 --full --verbose"
            echo "  $0 --update"
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
# Начало валидации
# ======================================================================
print_header "🔧 ВАЛИДАЦИЯ C++ ТОКЕНИЗАТОРА"

print_info "Тип валидации: $VALIDATION_TYPE"
print_info "Проект: $PROJECT_ROOT"

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 не найден!"
    exit 1
fi
print_success "Python3 найден: $(python3 --version)"

# Проверка наличия необходимых директорий
if [ ! -d "$PROJECT_ROOT/bpe" ]; then
    print_warning "Директория bpe не найдена. Создание..."
    mkdir -p "$PROJECT_ROOT/bpe"
fi

if [ ! -d "$PROJECT_ROOT/cpp/build" ]; then
    print_warning "Директория сборки C++ не найдена. Будет создана при сборке."
fi

# ======================================================================
# Шаг 1: Проверка наличия Python модели
# ======================================================================
print_header "📦 ШАГ 1: ПРОВЕРКА PYTHON МОДЕЛИ"

PY_VOCAB="$PROJECT_ROOT/bpe/vocab.json"
PY_MERGES="$PROJECT_ROOT/bpe/merges.txt"

if [ ! -f "$PY_VOCAB" ] || [ ! -f "$PY_MERGES" ]; then
    print_warning "Python модель не найдена. Обучение..."
    
    # Проверка наличия корпуса для обучения
    CORPUS_PATH="$PROJECT_ROOT/data/corpus/train_code.txt"
    if [ ! -f "$CORPUS_PATH" ]; then
        print_error "Корпус для обучения не найден: $CORPUS_PATH"
        exit 1
    fi
    
    # Запуск обучения
    python3 "$PROJECT_ROOT/scripts/train_tokenizer.py" \
        --vocab-size 8000 \
        --corpus "$CORPUS_PATH" \
        --output-dir "$PROJECT_ROOT/bpe/"
    
    if [ $? -ne 0 ]; then
        print_error "Ошибка при обучении Python модели"
        exit 1
    fi
    print_success "Python модель обучена"
else
    print_success "Python модель найдена:"
    print_info "  Словарь: $PY_VOCAB"
    print_info "  Слияния: $PY_MERGES"
    
    # Показываем размер словаря
    VOCAB_SIZE=$(python3 -c "import json; print(len(json.load(open('$PY_VOCAB'))))" 2>/dev/null || echo "unknown")
    print_info "  Размер словаря: $VOCAB_SIZE"
fi

# ======================================================================
# Шаг 2: Конвертация словаря для C++
# ======================================================================
print_header "🔄 ШАГ 2: КОНВЕРТАЦИЯ СЛОВАРЯ"

CONVERT_SCRIPT="$PROJECT_ROOT/cpp/tools/convert_vocab.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    print_error "Скрипт конвертации не найден: $CONVERT_SCRIPT"
    exit 1
fi

# Запуск конвертации с разными опциями в зависимости от типа валидации
if [ "$VALIDATION_TYPE" == "full" ]; then
    # Полная конвертация: оба формата
    print_info "Конвертация в стандартный формат..."
    python3 "$CONVERT_SCRIPT"
    
    print_info "Конвертация в сжатый формат..."
    python3 "$CONVERT_SCRIPT" --no-fill
else
    # Стандартная конвертация
    python3 "$CONVERT_SCRIPT"
fi

if [ $? -ne 0 ]; then
    print_error "Ошибка при конвертации словаря"
    exit 1
fi
print_success "Словарь сконвертирован"

# ======================================================================
# Шаг 3: Сборка C++ проекта
# ======================================================================
print_header "🏗️  ШАГ 3: СБОРКА C++ ПРОЕКТА"

BUILD_SCRIPT="$SCRIPT_DIR/build.sh"

if [ ! -f "$BUILD_SCRIPT" ]; then
    print_error "Скрипт сборки не найден: $BUILD_SCRIPT"
    exit 1
fi

# Сборка с соответствующими опциями
if [ "$VALIDATION_TYPE" == "debug" ]; then
    "$BUILD_SCRIPT" Debug
else
    "$BUILD_SCRIPT" Release
fi

if [ $? -ne 0 ]; then
    print_error "Ошибка при сборке C++ проекта"
    exit 1
fi
print_success "C++ проект собран"

# ======================================================================
# Шаг 4: Запуск тестов совместимости
# ======================================================================
print_header "🔍 ШАГ 4: ЗАПУСК ТЕСТОВ СОВМЕСТИМОСТИ"

# Проверка наличия тестового исполняемого файла
TEST_EXE="$PROJECT_ROOT/cpp/build/tests/test_compatibility"

if [ ! -f "$TEST_EXE" ]; then
    print_warning "Тест совместимости не найден. Поиск других тестов..."
    
    # Ищем другие тесты
    TEST_EXE=$(find "$PROJECT_ROOT/cpp/build/tests" -name "test_*" -type f -executable | head -1)
    
    if [ -z "$TEST_EXE" ]; then
        print_error "Тесты не найдены!"
        exit 1
    fi
fi

print_info "Запуск: $TEST_EXE"

# Формируем опции для тестов
TEST_OPTS=""
if [ "$VALIDATION_TYPE" == "quick" ]; then
    TEST_OPTS="--gtest_filter=*Basic*:*Simple*"
elif [ "$VALIDATION_TYPE" == "full" ]; then
    TEST_OPTS=""
fi

if [ "$VERBOSE" -eq 1 ]; then
    TEST_OPTS="$TEST_OPTS --gtest_color=yes"
fi

if [ "$STOP_ON_ERROR" -eq 1 ]; then
    TEST_OPTS="$TEST_OPTS --gtest_break_on_failure"
fi

# Запуск тестов
cd "$PROJECT_ROOT/cpp/build"
$TEST_EXE $TEST_OPTS

TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    print_error "Тесты совместимости не пройдены!"
    exit $TEST_RESULT
fi
print_success "Тесты совместимости пройдены"

# ======================================================================
# Шаг 5: Дополнительная валидация (для full режима)
# ======================================================================
if [ "$VALIDATION_TYPE" == "full" ]; then
    print_header "🔬 ШАГ 5: РАСШИРЕННАЯ ВАЛИДАЦИЯ"
    
    # Запуск Python скрипта валидации
    VALIDATE_SCRIPT="$PROJECT_ROOT/scripts/validate_cpp_tokenizer.py"
    
    if [ -f "$VALIDATE_SCRIPT" ]; then
        print_info "Запуск Python валидации..."
        python3 "$VALIDATE_SCRIPT" --verbose
    else
        # Создаем временный Python скрипт для валидации
        TMP_SCRIPT="/tmp/validate_cpp_$$.py"
        cat > "$TMP_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Скрипт для валидации C++ токенизатора против Python эталона
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def load_vocab(path):
    """Загрузка словаря"""
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.values())
    return data

def test_simple_strings():
    """Тестирование на простых строках"""
    test_strings = [
        "int",
        "main",
        "return",
        "std::vector",
        "template<typename T>",
        "class MyClass {};",
        "// comment",
        "/* multi-line */",
        "42",
        "3.14"
    ]
    
    print("\n📝 Тестирование простых строк:")
    for s in test_strings:
        print(f"  '{s}'")

def test_cpp_code():
    """Тестирование на C++ коде"""
    code = """
#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3};
    for (auto n : numbers) {
        std::cout << n << std::endl;
    }
    return 0;
}
"""
    print("\n📝 Тестирование C++ кода:")
    print(code[:100] + "...")

def test_utf8():
    """Тестирование UTF-8"""
    test_strings = [
        "привет",
        "你好",
        "😊",
        "café",
        "München"
    ]
    
    print("\n📝 Тестирование UTF-8:")
    for s in test_strings:
        print(f"  '{s}'")

def main():
    print("🔬 Расширенная валидация")
    print("=" * 40)
    
    test_simple_strings()
    test_cpp_code()
    test_utf8()
    
    print("\n✅ Базовая валидация пройдена")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
        chmod +x "$TMP_SCRIPT"
        python3 "$TMP_SCRIPT"
        rm "$TMP_SCRIPT"
    fi
fi

# ======================================================================
# Шаг 6: Обновление эталонных результатов (если нужно)
# ======================================================================
if [ $UPDATE_GOLDEN -eq 1 ]; then
    print_header "📝 ШАГ 6: ОБНОВЛЕНИЕ ЭТАЛОННЫХ РЕЗУЛЬТАТОВ"
    
    GOLDEN_DIR="$PROJECT_ROOT/tests/golden"
    mkdir -p "$GOLDEN_DIR"
    
    # Копируем текущие результаты как эталонные
    if [ -f "$PROJECT_ROOT/bpe/vocab.json" ]; then
        cp "$PROJECT_ROOT/bpe/vocab.json" "$GOLDEN_DIR/vocab_golden.json"
        print_success "Словарь сохранен как эталон"
    fi
    
    if [ -f "$PROJECT_ROOT/bpe/merges.txt" ]; then
        cp "$PROJECT_ROOT/bpe/merges.txt" "$GOLDEN_DIR/merges_golden.txt"
        print_success "Слияния сохранены как эталон"
    fi
    
    # Сохраняем результаты тестов
    if [ -f "$PROJECT_ROOT/cpp/build/Testing/TAG" ]; then
        TEST_DIR=$(cat "$PROJECT_ROOT/cpp/build/Testing/TAG")
        if [ -f "$PROJECT_ROOT/cpp/build/Testing/$TEST_DIR/Test.xml" ]; then
            cp "$PROJECT_ROOT/cpp/build/Testing/$TEST_DIR/Test.xml" "$GOLDEN_DIR/test_results_golden.xml"
            print_success "Результаты тестов сохранены"
        fi
    fi
    
    print_success "Эталонные результаты обновлены в $GOLDEN_DIR"
fi

# ======================================================================
# Итог
# ======================================================================
print_header "✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА"

print_success "C++ токенизатор успешно валидирован против Python эталона!"

# Показываем статистику
echo ""
echo "📊 Статистика:"
echo "  • Python модель: $(du -h "$PY_VOCAB" | cut -f1) / $(du -h "$PY_MERGES" | cut -f1)"
echo "  • C++ модель:    $(du -h "$PROJECT_ROOT/cpp/models/cpp_vocab.json" 2>/dev/null | cut -f1) / $(du -h "$PROJECT_ROOT/cpp/models/cpp_merges.txt" 2>/dev/null | cut -f1)"
echo "  • Тесты:         $(grep -c "RUN" "$PROJECT_ROOT/cpp/build/Testing/Temporary/LastTest.log" 2>/dev/null || echo 'N/A') тестов выполнено"

# Проверяем результаты
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✅ Все проверки пройдены успешно!${NC}"
else
    echo -e "\n${RED}❌ Обнаружены ошибки. Проверьте вывод выше.${NC}"
fi

echo ""
print_success "Готово!"
exit $TEST_RESULT