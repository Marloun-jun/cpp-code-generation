#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# compare_performance.py - Сравнение производительности Python и C++
# ======================================================================
#
# @file compare_performance.py
# @brief Сравнение скорости и точности Python и C++ реализаций BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Этот скрипт выполняет всестороннее сравнение производительности
#          двух реализаций токенизатора - Python (базовая) и C++ (оптимизированная).
#
#          **Измеряемые метрики:**
#          - Время выполнения (минимальное, максимальное, среднее, медиана)
#          - Стандартное отклонение (стабильность)
#          - Количество токенов (проверка корректности)
#          - Ускорение C++ относительно Python
#
# @usage ./compare_performance.py [options]
#   --iterations N    Количество итераций для усреднения (по умолч. 10)
#   --size N          Размер тестового текста (КБ, по умолч. 100)
#   --warmup N        Количество прогревных итераций (по умолч. 3)
#   --model-size N    Размер модели: 8000, 10000, 12000 (по умолч. 8000)
#   --output FILE     Сохранить результаты в JSON
#   --plot            Показать график (требуется matplotlib)
#   --verbose         Подробный вывод
#   --no-cleanup      Не удалять временные файлы (для отладки)
#
# @example
#   ./compare_performance.py --iterations 20 --size 1000
#   ./compare_performance.py --plot --output results.json --model-size 8000
#
# ======================================================================

import sys
import os
import time
import json
import subprocess
import argparse
import statistics
import tempfile
import re
import platform
from pathlib import Path
from datetime import datetime

# ======================================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================================

# Скрипт находится в: bpe_tokenizer/bpe_cpp/scripts/compare_performance.py
SCRIPT_DIR = Path(__file__).parent.absolute()
BPE_CPP_DIR = SCRIPT_DIR.parent
BPE_TOKENIZER_ROOT = BPE_CPP_DIR.parent

# Пути к C++ части
CPP_BUILD_DIR = BPE_CPP_DIR / 'build'
CPP_EXAMPLES_DIR = CPP_BUILD_DIR / 'examples'
CPP_MODELS_DIR = BPE_CPP_DIR / 'models'

# Пути к Python части
BPE_PYTHON_DIR = BPE_TOKENIZER_ROOT / 'bpe_python'
PYTHON_MODELS_DIR = BPE_PYTHON_DIR / 'models'
PYTHON_SRC_DIR = BPE_PYTHON_DIR

# Добавляем пути для импорта Python модуля
sys.path.insert(0, str(BPE_PYTHON_DIR))
sys.path.insert(0, str(PYTHON_SRC_DIR))

# ======================================================================
# Диагностика путей (для отладки)
# ======================================================================

def print_path_diagnostics(verbose=False):
    """Выводит диагностическую информацию о путях"""
    
    print("\n" + "============================================================")
    print("ПРОВЕРКА ПУТЕЙ")
    print("============================================================")
    
    paths = {
        'SCRIPT_DIR': SCRIPT_DIR,
        'BPE_CPP_DIR': BPE_CPP_DIR,
        'BPE_TOKENIZER_ROOT': BPE_TOKENIZER_ROOT,
        'BPE_PYTHON_DIR': BPE_PYTHON_DIR,
        'CPP_MODELS_DIR': CPP_MODELS_DIR,
        'PYTHON_MODELS_DIR': PYTHON_MODELS_DIR,
        'CPP_EXAMPLES_DIR': CPP_EXAMPLES_DIR,
        'CPP_BUILD_DIR': CPP_BUILD_DIR
    }
    
    for name, path in paths.items():
        exists = "✓" if path.exists() else "✗"
        print(f"{name:<20} {exists} {path}")
    
    if verbose:
        print("\nСодержимое bpe_python:")
        if BPE_PYTHON_DIR.exists():
            for f in BPE_PYTHON_DIR.glob("*.py"):
                print(f"  {f.name}")
        else:
            print("  Папка не существует")

# ======================================================================
# Импорт Python токенизатора
# ======================================================================

def import_python_tokenizer():
    """Импортирует Python токенизатор с обработкой ошибок"""
    
    try:
        from tokenizer import BPETokenizer
        print(f"✓ Импорт из tokenizer.py успешен!")
        return BPETokenizer
    except ImportError as e:
        print(f"✗ Ошибка импорта: {e}")
        print("\nПроверка файлов в bpe_python:")
        if BPE_PYTHON_DIR.exists():
            py_files = list(BPE_PYTHON_DIR.glob("*.py"))
            if py_files:
                for f in py_files:
                    print(f"  Файл: {f.name}")
            else:
                print(f"  Python файлы не найдены в {BPE_PYTHON_DIR}")
        else:
            print(f"  Папка {BPE_PYTHON_DIR} не существует!")
        return None

# ======================================================================
# Поиск файлов модели
# ======================================================================

def find_model_files(model_size=8000, verbose=False):
    """
    Ищет файлы модели указанного размера.
    Сначала ищет в C++ моделях, потом в Python моделях.
    
    Args:
        model_size (int):    Размер модели (8000, 10000, 12000)
        verbose (bool):      Подробный вывод
        
    Returns:
        tuple:    (vocab_path, merges_path) или (None, None) если не найдено
    """
    
    candidates = [
        # C++ модели (приоритет 1)
        {
            'vocab': CPP_MODELS_DIR / f'bpe_{model_size}' / 'cpp_vocab.json',
            'merges': CPP_MODELS_DIR / f'bpe_{model_size}' / 'cpp_merges.txt',
            'desc': f'C++ models/bpe_{model_size}/'
        },
        # Python модели (приоритет 2)
        {
            'vocab': PYTHON_MODELS_DIR / f'bpe_{model_size}' / 'vocab.json',
            'merges': PYTHON_MODELS_DIR / f'bpe_{model_size}' / 'merges.txt',
            'desc': f'Python models/bpe_{model_size}/'
        }
    ]
    
    print(f"\nПоиск модели размером {model_size}...")
    
    for candidate in candidates:
        vocab_exists = candidate['vocab'].exists()
        merges_exists = candidate['merges'].exists()
        
        if vocab_exists and merges_exists:
            print(f"✓ Найдено в: {candidate['desc']}")
            if verbose:
                print(f"  - словарь: {candidate['vocab']}")
                print(f"  - слияния: {candidate['merges']}")
            return str(candidate['vocab']), str(candidate['merges'])
        elif vocab_exists or merges_exists:
            print(f"⚠ Частично найдено в: {candidate['desc']}")
            if vocab_exists and verbose:
                print(f"  ✓ словарь: {candidate['vocab']}")
            elif not vocab_exists:
                print(f"  ✗ словарь отсутствует!")
            if merges_exists and verbose:
                print(f"  ✓ слияния: {candidate['merges']}")
            elif not merges_exists:
                print(f"  ✗ слияния отсутствуют!")
    
    print(f"✗ Модель {model_size} не найдена!")
    if verbose:
        print(f"  Искали в:")
        print(f"  - {CPP_MODELS_DIR}/bpe_{model_size}/")
        print(f"  - {PYTHON_MODELS_DIR}/bpe_{model_size}/")
    
    return None, None

# ======================================================================
# Генерация тестового текста
# ======================================================================

def generate_test_code(size_kb=100, verbose=False):
    """
    Генерирует тестовый C++ код указанного размера.
    
    Args:
        size_kb (int):    Желаемый размер в килобайтах
        verbose (bool):   Подробный вывод
        
    Returns:
        str:    Сгенерированный C++ код для тестирования
    """
    
    base_code = """#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <chrono>

namespace benchmark {
    
template<typename T>
class Vector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
public:
    Vector() : data_(nullptr), size_(0), capacity_(0) {}
    
    ~Vector() { delete[] data_; }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        data_[size_++] = value;
    }
    
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            T* new_data = new T[new_capacity];
            for (size_t i = 0; i < size_; ++i) {
                new_data[i] = data_[i];
            }
            delete[] data_;
            data_ = new_data;
            capacity_ = new_capacity;
        }
    }
    
    size_t size() const { return size_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

class TestClass {
public:
    TestClass(const std::string& name, int value) 
        : name_(name), value_(value) {}
    
    void process() {
        for (int i = 0; i < value_; ++i) {
            data_.push_back(i * i);
        }
    }
    
    double calculate() const {
        double sum = 0;
        for (size_t i = 0; i < data_.size(); ++i) {
            sum += data_[i];
        }
        return sum / data_.size();
    }
    
private:
    std::string name_;
    int value_;
    std::vector<int> data_;
};

} // namespace benchmark

int main() {
    using namespace benchmark;
    
    // Создание объектов
    TestClass obj("example", 100);
    obj.process();
    
    Vector<int> vec;
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i);
    }
    
    // Алгоритмы
    std::vector<int> std_vec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        std_vec[i] = vec[i];
    }
    std::sort(std_vec.begin(), std_vec.end());
    
    std::cout << "Processed " << std_vec.size() << " elements" << std::endl;
    std::cout << "Average: " << obj.calculate() << std::endl;
    
    return 0;
}
"""
    
    # Повторяем код до нужного размера
    target_bytes = size_kb * 1024
    result = base_code
    multiplier = 1
    
    current_size = len(result.encode('utf-8'))
    if verbose:
        print(f"\nГенерация тестового текста:")
        print(f"  Целевой размер: {target_bytes} байт ({size_kb} КБ)")
        print(f"  Базовый размер: {current_size} байт ({current_size/1024:.1f} КБ)")
    
    while len(result.encode('utf-8')) < target_bytes:
        multiplier += 1
        # Изменяем имена классов для уникальности
        modified_base = base_code.replace("class TestClass", f"class TestClass{multiplier}")
        modified_base = modified_base.replace("Vector<int> vec", f"Vector<int> vec{multiplier}")
        result += f"\n\n// Additional block {multiplier}\n" + modified_base
        current_size = len(result.encode('utf-8'))
        if verbose and multiplier % 5 == 0:
            print(f"  После добавления блока {multiplier}: {current_size} байт ({current_size/1024:.1f} КБ)")
    
    # Обрезаем до точного размера (если нужно)
    if current_size > target_bytes:
        result = result[:target_bytes]
        current_size = len(result.encode('utf-8'))
        if verbose:
            print(f"  Обрезано до точного размера: {current_size} байт ({current_size/1024:.1f} КБ)")
    
    if verbose:
        print(f"  Итоговый размер: {current_size} байт ({current_size/1024:.1f} КБ)")
        print(f"  Строк в коде: {result.count(chr(10)) + 1}")
    
    return result

# ======================================================================
# Тестирование Python токенизатора
# ======================================================================

def test_python_tokenizer(text, vocab_path, merges_path, iterations=10, warmup=3, 
                          model_size=8000, verbose=False):
    """
    Тестирование производительности Python токенизатора.
    
    Args:
        text (str):           Тестовый текст для токенизации
        vocab_path (str):     Путь к файлу словаря
        merges_path (str):    Путь к файлу слияний
        iterations (int):     Количество измерительных итераций
        warmup (int):         Количество прогревных итераций
        model_size (int):     Размер модели для создания токенизатора
        verbose (bool):       Подробный вывод
        
    Returns:
        dict:    Статистика времени и количества токенов, или None при ошибке
    """
    
    # Импортируем токенизатор
    BPETokenizer = import_python_tokenizer()
    if BPETokenizer is None:
        return None
    
    print(f"\nЗагрузка Python токенизатора (модель {model_size})...")
    
    if not os.path.exists(vocab_path):
        print(f"✗ Файл не найден: {vocab_path}!")
        return None
    
    if not os.path.exists(merges_path):
        print(f"✗ Файл не найден: {merges_path}!")
        return None
    
    try:
        # Инициализация токенизатора с правильным размером
        tokenizer = BPETokenizer(model_size, byte_level=True)
        tokenizer.load(vocab_path, merges_path)
        actual_vocab_size = tokenizer.vocab_size()
        print(f"✓ Токенизатор загружен. Реальный размер словаря: {actual_vocab_size}")
        
        if actual_vocab_size != model_size:
            print(f"⚠ Размер словаря ({actual_vocab_size}) отличается от ожидаемого ({model_size})")
            
    except Exception as e:
        print(f"✗ Ошибка загрузки токенизатора: {e}!")
        return None
    
    # Прогрев (warm-up)
    print(f"Прогрев ({warmup} итераций)...")
    for i in range(warmup):
        sample_text = text[:min(1000, len(text))]
        tokens = tokenizer.encode(sample_text)
        if i == 0 and verbose:
            print(f"  Пробная токенизация: {len(tokens)} токенов из {len(sample_text)} символов")
    
    # Измерение
    print(f"Измерение ({iterations} итераций)...")
    times = []
    token_counts = []
    
    for i in range(iterations):
        start = time.perf_counter()
        tokens = tokenizer.encode(text)
        end = time.perf_counter()
        
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        token_counts.append(len(tokens))
        
        if verbose or (i + 1) % 5 == 0 or iterations <= 5:
            print(f"  Итерация {i+1:2d}/{iterations}: {elapsed_ms:7.2f} мс, {len(tokens):5d} токенов")
    
    # Проверка консистентности количества токенов
    unique_counts = set(token_counts)
    if len(unique_counts) != 1:
        print(f"⚠ Количество токенов варьируется! От {min(token_counts)} до {max(token_counts)}")
        token_count = max(set(token_counts), key=token_counts.count)  # мода
    else:
        token_count = token_counts[0]
    
    # Статистика
    stats = {
        'min_time': min(times),
        'max_time': max(times),
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'stdev_time': statistics.stdev(times) if len(times) > 1 else 0,
        'token_count': token_count,
        'all_times': times
    }
    
    return stats

# ======================================================================
# Тестирование C++ токенизатора
# ======================================================================

def find_cpp_demo(verbose=False):
    """Находит исполняемый файл C++ демо"""
    
    candidates = [
        CPP_EXAMPLES_DIR / 'fast_tokenizer_demo',
        CPP_EXAMPLES_DIR / 'simple_example',
        CPP_BUILD_DIR / 'examples' / 'fast_tokenizer_demo',
        BPE_CPP_DIR / 'build' / 'examples' / 'fast_tokenizer_demo'
    ]
    
    # Добавляем .exe для Windows
    if platform.system() == 'Windows':
        candidates = [c.with_suffix('.exe') for c in candidates]
    
    if verbose:
        print(f"\nПоиск C++ демо...")
    
    for candidate in candidates:
        if candidate.exists():
            if verbose:
                print(f"✓ Найдено C++ демо: {candidate}")
            return str(candidate)
    
    if verbose:
        print(f"✗ C++ демо не найдено. Искали в:")
        for c in candidates:
            print(f"     {c}")
    
    return None

def test_cpp_tokenizer(text, iterations=10, warmup=3, verbose=False, no_cleanup=False):
    """
    Тестирование производительности C++ токенизатора.
    
    Args:
        text (str):          Тестовый текст для токенизации
        iterations (int):    Количество измерительных итераций
        warmup (int):        Количество прогревных итераций
        verbose (bool):      Подробный вывод
        no_cleanup (bool):   Не удалять временные файлы
        
    Returns:
        dict:    Статистика времени и количества токенов, или None при ошибке
    """
    
    cpp_demo = find_cpp_demo(verbose)
    if not cpp_demo:
        return None
    
    # Создаем временный файл с текстом
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                     delete=False, encoding='utf-8') as f:
        f.write(text)
        temp_file = f.name
    
    if verbose:
        print(f"\nВременный файл: {temp_file} ({len(text)} байт)")
    
    try:
        # Определяем, как передавать текст
        help_result = subprocess.run(
            [cpp_demo, '--help'],
            capture_output=True,
            text=True,
            timeout=3
        )
        
        use_stdin = '--text' not in help_result.stdout and '--file' not in help_result.stdout
        
        if verbose:
            print(f"  Режим передачи текста: {'stdin' if use_stdin else 'аргумент'}")
        
        # Прогрев (warm-up)
        print(f"Прогрев ({warmup} итераций)...")
        for i in range(warmup):
            sample_text = text[:min(1000, len(text))]
            if use_stdin:
                # Передаём текст через stdin
                result = subprocess.run(
                    [cpp_demo],
                    input=sample_text,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                # Передаём через аргумент --text или --file
                result = subprocess.run(
                    [cpp_demo, '--text', sample_text],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            if i == 0 and verbose:
                token_count = extract_token_count(result.stdout)
                print(f"  Пробная токенизация: {token_count} токенов")
        
        # Измерение
        print(f"Измерение ({iterations} итераций)...")
        times = []
        token_counts = []
        
        for i in range(iterations):
            start = time.perf_counter()
            
            if use_stdin:
                result = subprocess.run(
                    [cpp_demo],
                    input=text,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                result = subprocess.run(
                    [cpp_demo, '--text', text],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
            
            token_count = extract_token_count(result.stdout)
            token_counts.append(token_count)
            
            if verbose or (i + 1) % 5 == 0 or iterations <= 5:
                print(f"  Итерация {i+1:2d}/{iterations}: {elapsed_ms:7.2f} мс, {token_count:5d} токенов")
            
            # Небольшая пауза между итерациями
            time.sleep(0.1)
        
        # Статистика
        stats = {
            'min_time': min(times),
            'max_time': max(times),
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'stdev_time': statistics.stdev(times) if len(times) > 1 else 0,
            'token_count': token_counts[0] if token_counts else 0,
            'all_times': times
        }
        
        return stats
        
    except subprocess.TimeoutExpired:
        print(f"✗ Таймаут при выполнении C++ демо!")
        return None
    except Exception as e:
        print(f"✗ Ошибка при выполнении C++ демо: {e}!")
        return None
    finally:
        # Удаляем временный файл
        if not no_cleanup:
            try:
                os.unlink(temp_file)
                if verbose:
                    print(f"  Временный файл удален")
            except:
                pass
        else:
            print(f"  Временный файл сохранен: {temp_file}")

def extract_token_count(output):
    """Извлекает количество токенов из вывода C++ программы"""
    
    # Ищем разные варианты
    patterns = [
        r'(\d+)\s+токенов',
        r'(\d+)\s+токена',
        r'(\d+)\s+token',
        r'tokens:\s*(\d+)',
        r'Tokens:\s*(\d+)',
        r'токенов:\s*(\d+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 0

# ======================================================================
# Вывод результатов
# ======================================================================

def print_results(py_stats, cpp_stats, args):
    """Выводит отформатированные результаты"""
    
    print("\n" + "============================================================")
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("============================================================")
    
    if py_stats:
        print(f"\n🐍 PYTHON (модель {args.model_size}, {args.size} КБ, {args.iterations} итераций):")
        print(f"  {'Среднее время:':<20} {py_stats['mean_time']:8.2f} мс")
        print(f"  {'Медиана:':<20} {py_stats['median_time']:8.2f} мс")
        print(f"  {'Мин / Макс:':<20} {py_stats['min_time']:6.2f} / {py_stats['max_time']:6.2f} мс")
        print(f"  {'Стд. отклонение:':<20} {py_stats['stdev_time']:8.2f} мс")
        print(f"  {'Токенов:':<20} {py_stats['token_count']:8d}")
    
    if cpp_stats:
        print(f"\n⚡ C++ (модель {args.model_size}, {args.size} КБ, {args.iterations} итераций):")
        print(f"  {'Среднее время:':<20} {cpp_stats['mean_time']:8.2f} мс")
        print(f"  {'Медиана:':<20} {cpp_stats['median_time']:8.2f} мс")
        print(f"  {'Мин / Макс:':<20} {cpp_stats['min_time']:6.2f} / {cpp_stats['max_time']:6.2f} мс")
        print(f"  {'Стд. отклонение:':<20} {cpp_stats['stdev_time']:8.2f} мс")
        print(f"  {'Токенов:':<20} {cpp_stats['token_count']:8d}")
    
    # Сравнение
    if py_stats and cpp_stats:
        speedup = py_stats['mean_time'] / cpp_stats['mean_time']
        print(f"\n{'=' * 70}")
        print(f"УСКОРЕНИЕ: C++ быстрее Python в {speedup:.2f} раз")
        
        if py_stats['token_count'] == cpp_stats['token_count']:
            print(f"✓ Количество токенов совпадает: {py_stats['token_count']}")
        else:
            print(f"⚠ Количество токенов РАЗЛИЧАЕТСЯ:")
            print(f"  Python: {py_stats['token_count']}")
            print(f"  C++:    {cpp_stats['token_count']}")

# ======================================================================
# Сохранение результатов в JSON
# ======================================================================

def save_results(py_stats, cpp_stats, args, output_file):
    """Сохраняет результаты в JSON файл"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        },
        'config': {
            'model_size': args.model_size,
            'size_kb': args.size,
            'iterations': args.iterations,
            'warmup': args.warmup
        },
        'speedup': py_stats['mean_time'] / cpp_stats['mean_time'] if py_stats and cpp_stats else None
    }
    
    if py_stats:
        results['python'] = {
            'min_time_ms': py_stats['min_time'],
            'max_time_ms': py_stats['max_time'],
            'mean_time_ms': py_stats['mean_time'],
            'median_time_ms': py_stats['median_time'],
            'stdev_time_ms': py_stats['stdev_time'],
            'token_count': py_stats['token_count']
        }
    
    if cpp_stats:
        results['cpp'] = {
            'min_time_ms': cpp_stats['min_time'],
            'max_time_ms': cpp_stats['max_time'],
            'mean_time_ms': cpp_stats['mean_time'],
            'median_time_ms': cpp_stats['median_time'],
            'stdev_time_ms': cpp_stats['stdev_time'],
            'token_count': cpp_stats['token_count']
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в {output_file}")

# ======================================================================
# Построение графиков
# ======================================================================

def plot_results(py_stats, cpp_stats, args):
    """Строит графики сравнения производительности"""
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"⚠ matplotlib не установлен, график не создан: {e}")
        print("  Установите: pip install matplotlib numpy")
        return
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    speedup = py_stats['mean_time'] / cpp_stats['mean_time']
    
    # График 1: Boxplot распределения времени
    axes[0].boxplot([py_stats['all_times'], cpp_stats['all_times']], 
                   labels=['Python', 'C++'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#ff9999', color='black'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   medianprops=dict(color='red', linewidth=2),
                   flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))
    
    axes[0].set_ylabel('Время (мс)')
    axes[0].set_title(f'Распределение времени выполнения\nМодель {args.model_size}')
    axes[0].grid(True, alpha=0.3)
    
    # Добавляем средние значения
    axes[0].scatter([1, 2], [py_stats['mean_time'], cpp_stats['mean_time']], 
                  color='darkred', s=100, zorder=5, marker='D', 
                  label='Среднее')
    axes[0].legend()
    
    # График 2: Сравнение средних с погрешностью
    x_pos = np.arange(2)
    means = [py_stats['mean_time'], cpp_stats['mean_time']]
    errors = [py_stats['stdev_time'], cpp_stats['stdev_time']]
    
    bars = axes[1].bar(x_pos, means, yerr=errors, 
                     capsize=10, color=['#ff9999', '#66b3ff'],
                     edgecolor='black', linewidth=1, alpha=0.8)
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Python', 'C++'])
    axes[1].set_ylabel('Среднее время (мс)')
    axes[1].set_title(f'Сравнение производительности\nУскорение: {speedup:.2f}x')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.1,
                   f'{mean:.1f} мс', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Сохраняем график
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f'performance_comparison_{args.model_size}_{timestamp}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"График сохранен в {plot_file}")
    
    # Показываем интерактивно
    plt.show()

# ======================================================================
# Основная функция
# ======================================================================

def main():
    """Точка входа в программу"""
    
    parser = argparse.ArgumentParser(
        description='Сравнение производительности Python и C++ BPE токенизаторов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s --iterations 20 --size 1000
  %(prog)s --plot --output results.json --model-size 8000
  %(prog)s --verbose --size 500 --iterations 5
        """
    )
    parser.add_argument('--iterations', type=int, default=10, 
                       help='Количество итераций для усреднения (по умолч. 10)')
    parser.add_argument('--size', type=int, default=100, 
                       help='Размер тестового текста в КБ (по умолч. 100)')
    parser.add_argument('--warmup', type=int, default=3, 
                       help='Количество прогревных итераций (по умолч. 3)')
    parser.add_argument('--model-size', type=int, default=8000, choices=[8000, 10000, 12000],
                       help='Размер модели: 8000, 10000, 12000 (по умолч. 8000)')
    parser.add_argument('--output', type=str, 
                       help='Сохранить результаты в JSON файл')
    parser.add_argument('--plot', action='store_true', 
                       help='Показать график (требуется matplotlib)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Подробный вывод')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Не удалять временные файлы (для отладки)')
    parser.add_argument('--diagnostics', action='store_true',
                       help='Показать диагностику путей и выйти')
    
    args = parser.parse_args()
    
    if args.diagnostics:
        print_path_diagnostics(True)
        return 0
    
    print("============================================================")
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ PYTHON VS C++")
    print("============================================================")
    
    if args.verbose:
        print_path_diagnostics()
    
    # Вывод параметров тестирования
    print(f"\nПараметры тестирования:")
    print(f"  Размер текста: {args.size} КБ")
    print(f"  Итераций: {args.iterations}")
    print(f"  Прогрев: {args.warmup}")
    print(f"  Модель: {args.model_size}")
    
    # Поиск файлов модели
    vocab_path, merges_path = find_model_files(args.model_size, args.verbose)
    
    if not vocab_path or not merges_path:
        print(f"\n✗ Не удалось найти модель {args.model_size}!")
        return 1
    
    # Генерация тестового текста
    test_text = generate_test_code(args.size, args.verbose)
    text_bytes = len(test_text.encode('utf-8'))
    text_lines = test_text.count('\n') + 1
    
    print(f"\nТестовый текст:")
    print(f"  Размер: {text_bytes} байт ({text_bytes/1024:.1f} КБ)")
    print(f"  Строк: {text_lines}")
    
    if args.verbose:
        print("\nПервые 300 символов:")
        print("-" * 50)
        print(test_text[:300])
        print("-" * 50)
    
    # Тестирование Python
    print("\n" + "============================================================")
    print("ТЕСТИРОВАНИЕ PYTHON")
    print("============================================================")
    
    py_stats = test_python_tokenizer(
        test_text, vocab_path, merges_path,
        iterations=args.iterations,
        warmup=args.warmup,
        model_size=args.model_size,
        verbose=args.verbose
    )
    
    # Тестирование C++
    print("\n" + "============================================================")
    print("ТЕСТИРОВАНИЕ C++")
    print("============================================================")
    
    cpp_stats = test_cpp_tokenizer(
        test_text,
        iterations=args.iterations,
        warmup=args.warmup,
        verbose=args.verbose,
        no_cleanup=args.no_cleanup
    )
    
    # Вывод результатов
    print_results(py_stats, cpp_stats, args)
    
    # Сохранение результатов в JSON
    if args.output and py_stats and cpp_stats:
        save_results(py_stats, cpp_stats, args, args.output)
    
    # Построение графиков
    if args.plot and py_stats and cpp_stats:
        plot_results(py_stats, cpp_stats, args)
    
    print("\n" + "============================================================")
    print("Тестирование завершено!")
    print("============================================================")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())