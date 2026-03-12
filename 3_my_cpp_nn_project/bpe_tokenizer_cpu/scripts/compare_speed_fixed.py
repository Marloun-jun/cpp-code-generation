#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# compare_speed_fixed.py - Сравнение скорости Python и C++ токенизаторов
# ======================================================================
#
# @file compare_speed_fixed.py
# @brief Утилита для быстрого сравнения скорости работы Python и C++ реализаций
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот скрипт выполняет быстрое тестирование производительности
#          Python реализации BPE токенизатора и предоставляет инструкции
#          для запуска эквивалентных C++ бенчмарков.
#
#          **Измеряемые метрики для Python:**
#          - Среднее время encode (мс)
#          - Среднее количество токенов на текст
#          - Скорость обработки (байт/сек)
#          - Пропускная способность (KB/сек)
#
#          **Для C++ предоставляются:**
#          - Путь к собранному бенчмарку
#          - Инструкции по сборке, если бенчмарк не найден
#          - Рекомендация запустить полное сравнение
#
# @note Использует модель bpe_8000 из bpe_python/models/
#
# @usage python compare_speed_fixed.py
#
# @example
#   python compare_speed_fixed.py
#   # Результаты для Python:
#   #   • Среднее время: 1.234 ms
#   #   • Среднее токенов: 45
#   #   • Скорость: 12345 байт/сек
#
# ======================================================================

import sys
import time

from pathlib import Path
from typing import Tuple

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()           # scripts/compare_speed_fixed.py
SCRIPTS_DIR = CURRENT_FILE.parent                  # scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent                  # bpe_tokenizer/
BPE_PYTHON_DIR = PROJECT_ROOT / 'bpe_python'       # bpe_python/

# Добавляем путь для импорта токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

print(f"📁 Корень проекта: {PROJECT_ROOT}")
print(f"📁 Python BPE директория: {BPE_PYTHON_DIR}")

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
    print("✅ Импорт BPETokenizer успешен")
except ImportError as e:
    print(f"❌ Ошибка импорта BPETokenizer: {e}")
    print(f"\n📋 Проверьте наличие файла tokenizer.py в {BPE_PYTHON_DIR}")
    print("   Файлы в директории:")
    for f in BPE_PYTHON_DIR.iterdir():
        if f.suffix == '.py':
            print(f"   - {f.name}")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела для красивого форматирования вывода.
    
    Args:
        title: Заголовок
        width: Ширина линии
    
    Example:
        >>> print_header("ТЕСТИРОВАНИЕ PYTHON")
        ============================================================
                          ТЕСТИРОВАНИЕ PYTHON                     
        ============================================================
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


# ======================================================================
# ТЕСТОВЫЙ ТЕКСТ (C++ КОД)
# ======================================================================

TEST_TEXT = """#include <iostream>
#include <vector>

class Test {
public:
    Test(const std::string& name) : name_(name) {}
    void process() {
        for (int i = 0; i < 10; ++i) {
            data_.push_back(i);
        }
    }
private:
    std::string name_;
    std::vector<int> data_;
};

int main() {
    Test t("example");
    t.process();
    return 0;
}
"""


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ ТЕСТИРОВАНИЯ
# ======================================================================

def test_python_tokenizer(
    text: str,
    vocab_path: str,
    merges_path: str,
    iterations: int = 50
) -> Tuple[int, float]:
    """
    Тестирование Python токенизатора.
    
    Args:
        text: Тестовый текст
        vocab_path: Относительный путь к файлу словаря (относительно BPE_PYTHON_DIR)
        merges_path: Относительный путь к файлу слияний
        iterations: Количество итераций для усреднения
        
    Returns:
        Tuple[int, float]: (среднее количество токенов, среднее время в мс)
    
    **Процесс:**
    1. Формирование полных путей к файлам модели
    2. Проверка существования файлов
    3. Инициализация токенизатора
    4. Прогрев (3 итерации на части текста)
    5. Измерение на iterations итерациях
    6. Расчет средних значений
    """
    print(f"\n🐍 Загрузка Python токенизатора...")
    
    # Формируем полные пути
    vocab_full = BPE_PYTHON_DIR / vocab_path
    merges_full = BPE_PYTHON_DIR / merges_path
    
    print(f"     Vocab: {vocab_full}")
    print(f"     Merges: {merges_full}")
    
    # Проверяем существование файлов
    if not vocab_full.exists():
        print(f"❌ Файл словаря не найден: {vocab_full}")
        return 0, 0
    
    if not merges_full.exists():
        print(f"❌ Файл слияний не найден: {merges_full}")
        return 0, 0
    
    # Инициализируем токенизатор
    tokenizer = BPETokenizer(32000, byte_level=True)
    tokenizer.load(str(vocab_full), str(merges_full))
    
    print(f"     ✓ Токенизатор загружен")
    print(f"     📊 Размер словаря: {len(tokenizer.vocab)}")
    
    # Прогрев
    print(f"     🔥 Прогрев ({min(3, iterations)} итераций)...")
    for i in range(3):
        tokenizer.encode(text[:100])
    
    # Измерение
    print(f"     ⏱️  Измерение ({iterations} итераций)...")
    start = time.time()
    total_tokens = 0
    
    for i in range(iterations):
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
        
        if (i + 1) % 10 == 0:
            print(f"        Прогресс: {i + 1}/{iterations}")
    
    total_time = (time.time() - start) * 1000  # ms
    
    avg_tokens = total_tokens // iterations
    avg_time = total_time / iterations
    
    print(f"✅ Готово!")
    
    return avg_tokens, avg_time


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция.

    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    print_header("⚡ СРАВНЕНИЕ PYTHON И C++ ТОКЕНИЗАТОРОВ")
    
    # Информация о путях
    print(f"\n📁 Директория проекта: {PROJECT_ROOT}")
    print(f"📁 Python BPE директория: {BPE_PYTHON_DIR}")
    
    # Информация о тестовом тексте
    text_size = len(TEST_TEXT)
    print(f"\n📝 Тестовый текст:")
    print(f"   • Размер: {text_size} байт")
    print(f"   • Первые 100 символов:")
    preview = TEST_TEXT[:100].replace('\n', ' ')
    print(f"     {preview}...")
    
    # ======================================================================
    # ТЕСТИРОВАНИЕ PYTHON
    # ======================================================================
    
    print(f"\n{'─' * 60}")
    print("🐍 ТЕСТИРОВАНИЕ PYTHON")
    print(f"{'─' * 60}")
    
    # ИСПРАВЛЕНО: правильные пути к модели 8000
    py_tokens, py_time = test_python_tokenizer(
        TEST_TEXT,
        "models/bpe_8000/vocab.json",    # обновленный путь
        "models/bpe_8000/merges.txt",     # обновленный путь
        iterations=50
    )
    
    if py_time > 0:
        speed = text_size / py_time * 1000  # байт/сек
        
        print(f"\n📊 РЕЗУЛЬТАТЫ (PYTHON):")
        print(f"   • ⏱️  Среднее время: {py_time:.3f} ms")
        print(f"   • 🔢 Среднее токенов: {py_tokens}")
        print(f"   • 📈 Скорость: {speed:.0f} байт/сек")
        print(f"   • 💾 Пропускная способность: {speed / 1024:.2f} KB/сек")
    else:
        print("\n❌ Не удалось выполнить тестирование Python")
        return 1
    
    # ======================================================================
    # ИНСТРУКЦИИ ДЛЯ C++
    # ======================================================================
    
    print(f"\n{'─' * 60}")
    print("⚡ ТЕСТИРОВАНИЕ C++")
    print(f"{'─' * 60}")
    
    # ИСПРАВЛЕНО: путь к C++ бенчмарку
    cpp_benchmark = PROJECT_ROOT / 'bpe_cpp' / 'build' / 'benchmarks' / 'bench_fast_tokenizer'
    
    print("\nДля сравнения с C++ выполните следующие команды:")
    
    if cpp_benchmark.exists():
        print(f"\n✅ C++ бенчмарк найден: {cpp_benchmark}")
        print("\n   Запуск C++ бенчмарка:")
        print(f"   {cpp_benchmark}")
        print("\n   Или запустите полное сравнение всех реализаций:")
        print("   python scripts/benchmark_all.py")
    else:
        print(f"\n⚠️  C++ бенчмарк не найден по пути:")
        print(f"   {cpp_benchmark}")
        print("\n   Сначала соберите C++ проект:")
        print("\n   # Переход в директорию C++ части")
        print(f"   cd {PROJECT_ROOT / 'bpe_cpp'}")
        print("\n   # Создание директории сборки")
        print("   mkdir -p build && cd build")
        print("\n   # Конфигурация и сборка")
        print("   cmake .. -DBUILD_BENCHMARKS=ON")
        print("   make -j$(nproc)")
        print("\n   # После сборки бенчмарк будет доступен по пути:")
        print("   build/benchmarks/bench_fast_tokenizer")
    
    print(f"\n{'─' * 60}")
    print("\n💡 Совет: Для детального сравнения всех трех реализаций")
    print("   (HuggingFace, Python, C++) используйте:")
    print("   python scripts/benchmark_all.py")
    
    print(f"\n{'─' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())