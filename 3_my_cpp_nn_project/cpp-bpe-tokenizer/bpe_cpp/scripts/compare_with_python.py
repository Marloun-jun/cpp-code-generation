#!/usr/bin/env python3
# ======================================================================
# compare_performance.py - Сравнение производительности Python и C++
# ======================================================================
#
# @file compare_performance.py
# @brief Сравнение скорости и точности Python и C++ реализаций
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @usage ./compare_performance.py [options]
#   --iterations N    Количество итераций для усреднения (по умолч. 10)
#   --size N         Размер тестового текста (KB, по умолч. 100)
#   --warmup N       Количество прогревных итераций (по умолч. 3)
#   --output FILE    Сохранить результаты в JSON
#   --plot           Показать график (требуется matplotlib)
#   --verbose        Подробный вывод
#
# @example
#   ./compare_performance.py --iterations 20 --size 1000
#   ./compare_performance.py --plot --output results.json
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

# ======================================================================
# Настройка путей
# ======================================================================

# Добавляем пути для импорта Python модуля
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BPE_DIR = os.path.join(PROJECT_ROOT, 'bpe')
CPP_BUILD_DIR = os.path.join(PROJECT_ROOT, 'cpp', 'build')

sys.path.insert(0, BPE_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ======================================================================
# Импорт Python токенизатора
# ======================================================================

print(f"Путь для Python: {sys.path}")
print(f"BPE каталог: {BPE_DIR}")

try:
    # Пробуем разные варианты импорта
    import_names = [
        ('bpe.tokenizer', 'BPETokenizer'),
        ('tokenizer', 'BPETokenizer'),
        ('bpe.bpe_tokenizer', 'BPETokenizer')
    ]
    
    for module_name, class_name in import_names:
        try:
            module = __import__(module_name, fromlist=[class_name])
            BPETokenizer = getattr(module, class_name)
            print(f"Импорт через {module_name}.{class_name} успешен")
            break
        except (ImportError, AttributeError):
            continue
    else:
        raise ImportError("Не удалось импортировать BPETokenizer")
        
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("\nПроверка структуры папок:")
    print(f"  BPE_DIR = {BPE_DIR}")
    print("  Файлы в папке bpe:")
    for f in os.listdir(BPE_DIR):
        print(f"    {f}")
    sys.exit(1)

# ======================================================================
# Генерация тестового текста
# ======================================================================

def generate_test_code(size_kb=100):
    """Генерирует тестовый C++ код указанного размера"""
    
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

namespace benchmark {{
    
template<typename T>
class Vector {{
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    
public:
    Vector() : data_(nullptr), size_(0), capacity_(0) {{}}
    
    ~Vector() {{ delete[] data_; }}
    
    void push_back(const T& value) {{
        if (size_ >= capacity_) {{
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }}
        data_[size_++] = value;
    }}
    
    void reserve(size_t new_capacity) {{
        if (new_capacity > capacity_) {{
            T* new_data = new T[new_capacity];
            for (size_t i = 0; i < size_; ++i) {{
                new_data[i] = data_[i];
            }}
            delete[] data_;
            data_ = new_data;
            capacity_ = new_capacity;
        }}
    }}
    
    size_t size() const {{ return size_; }}
    T& operator[](size_t i) {{ return data_[i]; }}
}};

class TestClass {{
public:
    TestClass(const std::string& name, int value) 
        : name_(name), value_(value) {{}}
    
    void process() {{
        for (int i = 0; i < value_; ++i) {{
            data_.push_back(i * i);
        }}
    }}
    
    double calculate() const {{
        double sum = 0;
        for (size_t i = 0; i < data_.size(); ++i) {{
            sum += data_[i];
        }}
        return sum / data_.size();
    }}
    
private:
    std::string name_;
    int value_;
    std::vector<int> data_;
}};

}} // namespace benchmark

int main() {{
    using namespace benchmark;
    
    // Создание объектов
    TestClass obj("example", 100);
    obj.process();
    
    Vector<int> vec;
    for (int i = 0; i < 1000; ++i) {{
        vec.push_back(i);
    }}
    
    // Алгоритмы
    std::vector<int> std_vec(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {{
        std_vec[i] = vec[i];
    }}
    std::sort(std_vec.begin(), std_vec.end());
    
    std::cout << "Processed " << std_vec.size() << " elements" << std::endl;
    std::cout << "Average: " << obj.calculate() << std::endl;
    
    return 0;
}}
"""
    
    # Повторяем код до нужного размера
    target_bytes = size_kb * 1024
    result = base_code
    multiplier = 1
    
    while len(result.encode('utf-8')) < target_bytes:
        multiplier += 1
        result = base_code.replace("class TestClass", f"class TestClass{multiplier}")
        result += f"\n\n// Additional block {multiplier}\n" + base_code
    
    # Обрезаем до точного размера
    result = result[:target_bytes]
    
    return result

# ======================================================================
# Тестирование Python токенизатора
# ======================================================================

def test_python_tokenizer(text, vocab_path, merges_path, iterations=10, warmup=3):
    """Тестирование производительности Python токенизатора"""
    
    print(f"Загрузка Python токенизатора...")
    
    if not os.path.exists(vocab_path):
        print(f"Файл не найден: {vocab_path}")
        return None
    
    # Инициализация
    tokenizer = BPETokenizer(32000, byte_level=True)
    tokenizer.load(vocab_path, merges_path)
    
    # Прогрев
    print(f"Прогрев ({warmup} итераций)...")
    for i in range(warmup):
        tokens = tokenizer.encode(text[:1000])
    
    # Измерение
    print(f"Измерение ({iterations} итераций)...")
    times = []
    token_counts = []
    
    for i in range(iterations):
        start = time.perf_counter()
        tokens = tokenizer.encode(text)
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
        token_counts.append(len(tokens))
        
        if (i + 1) % 5 == 0:
            print(f"    Итерация {i+1}/{iterations}: {times[-1]:.2f} ms")
    
    # Статистика
    stats = {
        'min_time': min(times),
        'max_time': max(times),
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'stdev_time': statistics.stdev(times) if len(times) > 1 else 0,
        'token_count': token_counts[0],  # Предполагаем, что всегда одинаково
        'all_times': times
    }
    
    return stats

# ======================================================================
# Тестирование C++ токенизатора
# ======================================================================

def test_cpp_tokenizer(text, iterations=10, warmup=3):
    """Тестирование производительности C++ токенизатора"""
    
    # Поиск C++ демо
    cpp_demo = None
    candidates = [
        os.path.join(CPP_BUILD_DIR, 'examples', 'fast_tokenizer_demo'),
        os.path.join(CPP_BUILD_DIR, 'examples', 'simple_example'),
        os.path.join(PROJECT_ROOT, 'cpp', 'build', 'examples', 'fast_tokenizer_demo')
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            cpp_demo = candidate
            print(f"Найдено C++ демо: {cpp_demo}")
            break
    
    if not cpp_demo:
        print(f"C++ демо не найдено. Искали в:")
        for c in candidates:
            print(f"     {c}")
        return None
    
    # Создаем временный файл с текстом
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text)
        temp_file = f.name
    
    try:
        # Прогрев
        print(f"Прогрев ({warmup} итераций)...")
        for i in range(warmup):
            result = subprocess.run(
                [cpp_demo, temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
        
        # Измерение
        print(f"Измерение ({iterations} итераций)...")
        times = []
        token_counts = []
        
        for i in range(iterations):
            start = time.perf_counter()
            result = subprocess.run(
                [cpp_demo, temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # ms
            
            # Подсчет токенов из вывода
            tokens_count = 0
            for line in result.stdout.split('\n'):
                if 'Tokens:' in line or 'токенов:' in line:
                    try:
                        tokens_count = int(line.split(':')[1].strip())
                        break
                    except:
                        pass
            
            token_counts.append(tokens_count)
            
            if (i + 1) % 5 == 0:
                print(f"    Итерация {i+1}/{iterations}: {times[-1]:.2f} ms")
        
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
        
    finally:
        # Удаляем временный файл
        os.unlink(temp_file)

# ======================================================================
# Основная функция
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Сравнение Python и C++ BPE токенизаторов')
    parser.add_argument('--iterations', type=int, default=10, help='Количество итераций')
    parser.add_argument('--size', type=int, default=100, help='Размер тестового текста в KB')
    parser.add_argument('--warmup', type=int, default=3, help='Количество прогревных итераций')
    parser.add_argument('--output', type=str, help='Сохранить результаты в JSON файл')
    parser.add_argument('--plot', action='store_true', help='Показать график (требуется matplotlib)')
    parser.add_argument('--verbose', action='store_true', help='Подробный вывод')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ PYTHON VS C++")
    print("=" * 60)
    
    # Генерация тестового текста
    print(f"\nГенерация тестового текста ({args.size} KB)...")
    test_text = generate_test_code(args.size)
    print(f"   Размер: {len(test_text)} байт")
    print(f"   Строк: {test_text.count(chr(10)) + 1}")
    
    if args.verbose:
        print("\n   Первые 200 символов:")
        print("-" * 40)
        print(test_text[:200])
        print("-" * 40)
    
    # Пути к файлам модели
    py_vocab = os.path.join(BPE_DIR, 'vocab.json')
    py_merges = os.path.join(BPE_DIR, 'merges.txt')
    
    print(f"\nПути к файлам:")
    print(f"   Python словарь: {py_vocab}")
    print(f"   Python слияния: {py_merges}")
    print(f"   C++ бинарные: {CPP_BUILD_DIR}")
    
    # Проверка наличия файлов
    if not os.path.exists(py_vocab):
        print(f"\nФайл словаря не найден: {py_vocab}")
        print("   Сначала обучите токенизатор или скопируйте файлы.")
        return 1
    
    # Тестирование Python
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ PYTHON")
    print("=" * 60)
    
    py_stats = test_python_tokenizer(
        test_text, py_vocab, py_merges,
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Тестирование C++
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ C++")
    print("=" * 60)
    
    cpp_stats = test_cpp_tokenizer(
        test_text,
        iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    if py_stats:
        print(f"\nPython ({args.size} KB, {args.iterations} итераций):")
        print(f"   Среднее время: {py_stats['mean_time']:.2f} ms")
        print(f"   Медиана:       {py_stats['median_time']:.2f} ms")
        print(f"   Мин/Макс:      {py_stats['min_time']:.2f} / {py_stats['max_time']:.2f} ms")
        print(f"   Стд. откл.:    {py_stats['stdev_time']:.2f} ms")
        print(f"   Токенов:       {py_stats['token_count']}")
    
    if cpp_stats:
        print(f"\nC++ ({args.size} KB, {args.iterations} итераций):")
        print(f"   Среднее время: {cpp_stats['mean_time']:.2f} ms")
        print(f"   Медиана:       {cpp_stats['median_time']:.2f} ms")
        print(f"   Мин/Макс:      {cpp_stats['min_time']:.2f} / {cpp_stats['max_time']:.2f} ms")
        print(f"   Стд. откл.:    {cpp_stats['stdev_time']:.2f} ms")
        print(f"   Токенов:       {cpp_stats['token_count']}")
    
    # Сравнение
    if py_stats and cpp_stats:
        speedup = py_stats['mean_time'] / cpp_stats['mean_time']
        print(f"\nУСКОРЕНИЕ:")
        print(f"   C++ быстрее Python в {speedup:.2f} раз")
        
        if py_stats['token_count'] == cpp_stats['token_count']:
            print(f"Количество токенов совпадает: {py_stats['token_count']}")
        else:
            print(f"Количество токенов разное:")
            print(f"   Python: {py_stats['token_count']}")
            print(f"   C++:    {cpp_stats['token_count']}")
    
    # Сохранение результатов
    if args.output:
        results = {
            'config': {
                'size_kb': args.size,
                'iterations': args.iterations,
                'warmup': args.warmup
            },
            'python': py_stats,
            'cpp': cpp_stats,
            'speedup': py_stats['mean_time'] / cpp_stats['mean_time'] if py_stats and cpp_stats else None
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nРезультаты сохранены в {args.output}")
    
    # График
    if args.plot and py_stats and cpp_stats:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.boxplot([py_stats['all_times'], cpp_stats['all_times']], 
                       labels=['Python', 'C++'])
            plt.ylabel('Время (ms)')
            plt.title('Сравнение времени выполнения')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(['Python', 'C++'], 
                          [py_stats['mean_time'], cpp_stats['mean_time']],
                          yerr=[py_stats['stdev_time'], cpp_stats['stdev_time']],
                          capsize=10)
            plt.ylabel('Среднее время (ms)')
            plt.title(f'Ускорение: {speedup:.2f}x')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('performance_comparison.png', dpi=150)
            plt.show()
            
            print("График сохранен в performance_comparison.png")
            
        except ImportError:
            print("!!! matplotlib не установлен, график не создан")
    
    print("\n" + "=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())