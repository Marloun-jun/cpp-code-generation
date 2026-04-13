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
# @version 3.2.0
#
# @details Этот скрипт выполняет быстрое тестирование производительности
#          Python реализации BPE токенизатора и предоставляет инструкции
#          для запуска эквивалентных C++ бенчмарков.
#
#          **Измеряемые метрики для Python:**
#          - Среднее время encode (мс)
#          - Среднее количество токенов на текст
#          - Скорость обработки (байт/сек)
#          - Пропускная способность (КБ/сек)
#
#          **Для C++ предоставляются:**
#          - Путь к собранному бенчмарку
#          - Инструкции по сборке, если бенчмарк не найден
#          - Рекомендация запустить полное сравнение
#
# @note Использует модель bpe_10000 из bpe_python/models/
#
# @usage python compare_speed_fixed.py
#
# ======================================================================

import sys
import time

from pathlib import Path
from typing import Tuple, Optional

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()         # scripts/compare_speed_fixed.py
SCRIPTS_DIR = CURRENT_FILE.parent               # scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent               # bpe_tokenizer_cpu/
BPE_PYTHON_DIR = PROJECT_ROOT / 'bpe_python'    # bpe_python/
BPE_CPP_DIR = PROJECT_ROOT / 'bpe_cpp'          # bpe_cpp/

# Добавляем путь для импорта токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

print(f"Корень проекта:        {PROJECT_ROOT}")
print(f"Python BPE директория: {BPE_PYTHON_DIR}")
print(f"C++ BPE директория:    {BPE_CPP_DIR}")

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
    print("Импорт BPETokenizer успешен")
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}!")
    print(f"\nПроверьте наличие файла tokenizer.py в {BPE_PYTHON_DIR}")
    print("Файлы в директории:")
    for f in BPE_PYTHON_DIR.iterdir():
        if f.suffix == '.py':
            print(f"- {f.name}")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """Вывести заголовок раздела."""
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


def find_cpp_benchmark() -> Optional[Path]:
    """
    Поиск собранного C++ бенчмарка в стандартных местах.
    
    Returns:
        Optional[Path]: Путь к бенчмарку или None
    """
    candidates = [
        BPE_CPP_DIR / 'build' / 'bench_fast_tokenizer',
        BPE_CPP_DIR / 'build' / 'benchmarks' / 'bench_fast_tokenizer',
        BPE_CPP_DIR / 'build' / 'bin' / 'bench_fast_tokenizer',
        PROJECT_ROOT / 'build' / 'bench_fast_tokenizer',
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


# ======================================================================
# ТЕСТОВЫЙ ТЕКСТ (C++ КОД)
# ======================================================================

TEST_TEXT = """#include <iostream>
#include <vector>
#include <string>

class Test {
public:
    Test(const std::string& name) : name_(name) {}
    
    void process() {
        for (int i = 0; i < 10; ++i) {
            data_.push_back(i * i);
        }
    }
    
    void print() const {
        std::cout << "Test: " << name_ << std::endl;
        for (auto val : data_) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

private:
    std::string name_;
    std::vector<int> data_;
};

int main() {
    Test t("example");
    t.process();
    t.print();
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
    vocab_size: int = 10000,
    iterations: int = 50
) -> Tuple[int, float]:
    """
    Тестирование Python токенизатора.
    """
    print(f"\nЗагрузка Python токенизатора...")
    
    # Формируем полные пути
    vocab_full = BPE_PYTHON_DIR / vocab_path
    merges_full = BPE_PYTHON_DIR / merges_path
    
    print(f"    Vocab: {vocab_full}")
    print(f"    Merges: {merges_full}")
    
    # Проверяем существование файлов
    if not vocab_full.exists():
        print(f"Файл словаря не найден: {vocab_full}!")
        return 0, 0
    
    if not merges_full.exists():
        print(f"Файл слияний не найден: {merges_full}!")
        return 0, 0
    
    try:
        # Создаем и загружаем токенизатор
        tokenizer = BPETokenizer(vocab_size=vocab_size, byte_level=True)
        tokenizer.load(str(vocab_full), str(merges_full))
        
        print(f"Токенизатор загружен")
        
        # ДИАГНОСТИКА
        print(f"Размер словаря:     {len(tokenizer.vocab)}")
        
        # Проверяем наличие атрибутов
        if hasattr(tokenizer, 'token_to_id'):
            print(f"Размер token_to_id: {len(tokenizer.token_to_id)}")
        elif hasattr(tokenizer, 'token2id'):
            print(f"Размер token2id:    {len(tokenizer.token2id)}")
        
        if hasattr(tokenizer, 'id_to_token'):
            print(f"Пример токена ID 0: {tokenizer.id_to_token[0]}")
            print(f"Пример токена ID 1: {tokenizer.id_to_token[1]}")
            print(f"Пример токена ID 2: {tokenizer.id_to_token[2]}")
        elif hasattr(tokenizer, 'id2token'):
            print(f"Пример токена ID 0: {tokenizer.id2token[0]}")
            print(f"Пример токена ID 1: {tokenizer.id2token[1]}")
        # ============================================
        
        # Прогрев
        print(f"Прогрев ({min(3, iterations)} итераций)...")
        for i in range(3):
            tokenizer.encode(text[:100])
        
        # Измерение
        print(f"Измерение ({iterations} итераций)...")
        start = time.time()
        total_tokens = 0
        
        for i in range(iterations):
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
            
            if (i + 1) % 10 == 0:
                print(f"Прогресс: {i + 1}/{iterations}")
        
        total_time = (time.time() - start) * 1000    # мс
        
        avg_tokens = total_tokens // iterations
        avg_time = total_time / iterations
        
        print(f"Готово!")
        
        return avg_tokens, avg_time
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}!")
        import traceback
        traceback.print_exc()
        return 0, 0

# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """Основная функция."""
    print_header("СРАВНЕНИЕ PYTHON И C++ ТОКЕНИЗАТОРОВ")
    
    # Информация о путях
    print(f"\nДиректория проекта:    {PROJECT_ROOT}")
    print(f"Python BPE директория: {BPE_PYTHON_DIR}")
    print(f"C++ BPE директория:    {BPE_CPP_DIR}")
    
    # Информация о тестовом тексте
    text_size = len(TEST_TEXT)
    print(f"\nТестовый текст:")
    print(f"- Размер: {text_size} байт")
    print(f"- Первые 100 символов:")
    preview = TEST_TEXT[:100].replace('\n', ' ')
    print(f"    {preview}...")
    
    # ======================================================================
    # ТЕСТИРОВАНИЕ PYTHON
    # ======================================================================
    
    print(f"\n{'─' * 60}")
    print("ТЕСТИРОВАНИЕ PYTHON")
    print(f"{'─' * 60}")
    
    # vocab_size=10000
    py_tokens, py_time = test_python_tokenizer(
        TEST_TEXT,
        "models/bpe_10000/vocab.json",
        "models/bpe_10000/merges.txt",
        vocab_size=10000,
        iterations=50
    )
    
    if py_time > 0:
        speed = text_size / py_time * 1000    # байт/сек
        speed_kb = speed / 1024               # КБ/сек
        
        print(f"\nРЕЗУЛЬТАТЫ (PYTHON):")
        print(f"- Среднее время:          {py_time:.3f} мс")
        print(f"- Среднее токенов:        {py_tokens}")
        print(f"- Скорость:               {speed:.0f} байт/сек ({speed_kb:.2f} КБ/сек)")
        print(f"- Пропускная способность: {speed_kb:.2f} КБ/сек")
    else:
        print("\nНе удалось выполнить тестирование Python!")
        return 1
    
    # ======================================================================
    # ИНСТРУКЦИИ ДЛЯ C++
    # ======================================================================
    
    print(f"\n{'─' * 60}")
    print("ТЕСТИРОВАНИЕ C++")
    print(f"{'─' * 60}")
    
    cpp_benchmark = find_cpp_benchmark()
    
    print("\nДля сравнения с C++ выполните следующие команды:")
    
    if cpp_benchmark:
        print(f"\nC++ бенчмарк найден: {cpp_benchmark}")
        print("\nЗапуск C++ бенчмарка:")
        print(f"    {cpp_benchmark}")
        print("\nИли запустите с параметрами:")
        print(f"    {cpp_benchmark} --iterations 1000")
        
        size_kb = cpp_benchmark.stat().st_size / 1024
        print(f"\nРазмер исполняемого файла: {size_kb:.2f} КБ")
    else:
        print(f"\nC++ бенчмарк не найден!")
        print("\nСоберите C++ проект с бенчмарками:")
        print(f"\n    cd {BPE_CPP_DIR}")
        print("    mkdir -p build && cd build")
        print("    cmake .. -DBUILD_BENCHMARKS=ON")
        print("    make -j$(nproc) bench_fast_tokenizer")
    
    # ======================================================================
    # ИНСТРУКЦИИ ДЛЯ ПОЛНОГО СРАВНЕНИЯ
    # ======================================================================
    
    print(f"\n{'─' * 60}")
    print("ПОЛНОЕ СРАВНЕНИЕ")
    print(f"{'─' * 60}")
    
    benchmark_all = SCRIPTS_DIR / 'benchmark_all.py'
    
    if benchmark_all.exists():
        print(f"\nСкрипт полного сравнения найден: {benchmark_all}")
        print("\nЗапустите для сравнения всех трёх реализаций:")
        print(f"    python {benchmark_all}")
        print("\nЭто создаст файл benchmark_results.json с детальными")
        print(" результатами сравнения HuggingFace, Python и C++ реализаций.")
    else:
        print(f"\nСкрипт полного сравнения не найден!")
    
    print(f"\n{'─' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())