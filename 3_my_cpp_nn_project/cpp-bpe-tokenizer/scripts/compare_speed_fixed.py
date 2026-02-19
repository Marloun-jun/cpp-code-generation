#!/usr/bin/env python3
# ======================================================================
# compare_speed_fixed.py - Сравнение скорости Python и C++ токенизаторов
# ======================================================================
#
# @file compare_speed_fixed.py
# @brief Утилита для сравнения скорости работы Python и C++ реализаций BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Измеряет производительность Python реализации BPE токенизатора
#          и предоставляет инструкции для запуска C++ бенчмарков.
#          Измеряемые метрики:
#          - Среднее время encode (мс)
#          - Среднее количество токенов
#          - Скорость обработки (байт/сек)
#
# @usage python compare_speed_fixed.py
#
# @example
#   python compare_speed_fixed.py
#   # Результаты для Python:
#   #   Среднее время: 1.234 ms
#   #   Среднее токенов: 45
#   #   Скорость: 12345 байт/сек
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
PROJECT_ROOT = SCRIPTS_DIR.parent                  # cpp-bpe-tokenizer/
BPE_PYTHON_DIR = PROJECT_ROOT / 'bpe'              # bpe/

# Добавляем путь для импорта токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
    print("Импорт BPETokenizer успешен")
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}")
    print(f"\nПроверьте наличие файла tokenizer.py в {BPE_PYTHON_DIR}")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела.
    
    Args:
        title: Заголовок
        width: Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


# ======================================================================
# ТЕСТОВЫЙ ТЕКСТ
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
        vocab_path: Путь к файлу словаря
        merges_path: Путь к файлу слияний
        iterations: Количество итераций для усреднения
        
    Returns:
        Tuple[int, float]: (среднее количество токенов, среднее время в мс)
    """
    print(f"Загрузка Python токенизатора...")
    
    # Формируем полные пути
    vocab_full = BPE_PYTHON_DIR / vocab_path
    merges_full = BPE_PYTHON_DIR / merges_path
    
    print(f"     Vocab: {vocab_full}")
    print(f"     Merges: {merges_full}")
    
    # Проверяем существование файлов
    if not vocab_full.exists():
        print(f"Файл словаря не найден: {vocab_full}")
        return 0, 0
    
    if not merges_full.exists():
        print(f"Файл слияний не найден: {merges_full}")
        return 0, 0
    
    # Инициализируем токенизатор
    tokenizer = BPETokenizer(32000, byte_level=True)
    tokenizer.load(str(vocab_full), str(merges_full))
    
    print(f"     Токенизатор загружен")
    print(f"     Размер словаря: {len(tokenizer.vocab)}")
    
    # Прогрев
    print(f"     Прогрев ({min(3, iterations)} итераций)...")
    for i in range(3):
        tokenizer.encode(text[:100])
    
    # Измерение
    print(f"     Измерение ({iterations} итераций)...")
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
    
    print(f"Готово!")
    
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
    print_header("СРАВНЕНИЕ PYTHON И C++ ТОКЕНИЗАТОРОВ")
    
    # Информация о путях
    print(f"\nДиректория проекта: {PROJECT_ROOT}")
    print(f"Python BPE директория: {BPE_PYTHON_DIR}")
    
    # Информация о тестовом тексте
    text_size = len(TEST_TEXT)
    print(f"\nТестовый текст:")
    print(f"   Размер: {text_size} байт")
    print(f"   Первые 100 символов:")
    print(f"   {TEST_TEXT[:100].replace(chr(10), ' ')}...")
    
    # Тестируем Python
    print(f"\n{'─' * 50}")
    print("ТЕСТИРОВАНИЕ PYTHON")
    print(f"{'─' * 50}")
    
    py_tokens, py_time = test_python_tokenizer(
        TEST_TEXT,
        "vocab.json",
        "merges.txt",
        iterations=50
    )
    
    if py_time > 0:
        speed = text_size / py_time * 1000  # байт/сек
        
        print(f"\nРЕЗУЛЬТАТЫ (PYTHON):")
        print(f"   • Среднее время: {py_time:.3f} ms")
        print(f"   • Среднее токенов: {py_tokens}")
        print(f"   • Скорость: {speed:.0f} байт/сек")
        print(f"   • Пропускная способность: {speed / 1024:.2f} KB/сек")
    else:
        print("\nНе удалось выполнить тестирование Python")
        return 1
    
    # Инструкции для C++
    print(f"\n{'─' * 50}")
    print("⚡ ТЕСТИРОВАНИЕ C++")
    print(f"{'─' * 50}")
    
    print("\nДля сравнения с C++ выполните следующие команды:")
    print("\n   cd ~/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/cpp/build")
    print("   ./benchmarks/bench_fast_tokenizer")
    
    cpp_benchmark = PROJECT_ROOT / 'cpp' / 'build' / 'benchmarks' / 'bench_fast_tokenizer'
    if cpp_benchmark.exists():
        print(f"\nC++ бенчмарк найден: {cpp_benchmark}")
        print("\nИли запустите полное сравнение:")
        print("python ../scripts/compare_performance.py")
    else:
        print(f"\n !!! C++ бенчмарк не найден по пути: {cpp_benchmark}")
        print("   Сначала соберите C++ проект:")
        print("   cd ~/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/cpp")
        print("   mkdir -p build && cd build")
        print("   cmake .. && make -j$(nproc)")
    
    print(f"\n{'─' * 50}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())