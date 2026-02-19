#!/usr/bin/env python3
# ======================================================================
# test_compare_models.py - Расширенное сравнение трех BPE моделей
# ======================================================================
#
# @file test_compare_models.py
# @brief Расширенное сравнение трех BPE моделей (8000, 10000, 12000)
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Детальный анализ и сравнение моделей с разным размером словаря:
#          - Точность кодирования/декодирования по категориям
#          - Скорость работы (encode/decode)
#          - Степень сжатия текста
#          - Анализ состава словарей
#          - Визуализация результатов
#
# @usage python test_compare_models.py [--quick] [--plot-only]
#
# @example
#   python test_compare_models.py
#   python test_compare_models.py --quick
#   python test_compare_models.py --plot-only
#
# ======================================================================

import sys
import time
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Optional, Any

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()           # tests/test_compare_models.py
TESTS_DIR = CURRENT_FILE.parent                    # tests/
BPE_PYTHON_DIR = TESTS_DIR.parent                  # bpe_python/
PROJECT_ROOT = BPE_PYTHON_DIR.parent               # cpp-bpe-tokenizer/

# Добавляем путь для импорта токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}")
    print(f"Убедитесь, что файл tokenizer.py существует в {BPE_PYTHON_DIR}")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def get_project_paths() -> Dict[str, Path]:
    """
    Получить пути проекта.
    
    Returns:
        Dict[str, Path]: Словарь с путями проекта
    """
    return {
        "project_root": PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "tests_dir": TESTS_DIR,
        "models_dir": BPE_PYTHON_DIR / 'models',
        "output_dir": TESTS_DIR / 'three_model_results'
    }


def print_header(title: str, width: int = 80) -> None:
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
# КЛАСС ДЛЯ СРАВНЕНИЯ ТРЕХ МОДЕЛЕЙ
# ======================================================================

class ThreeModelComparison:
    """
    Класс для сравнения трех BPE моделей с разным размером словаря.
    
    Проводит комплексный анализ моделей по множеству метрик:
    - Точность (roundtrip тесты)
    - Скорость encode/decode
    - Степень сжатия
    - Состав словарей
    """
    
    def __init__(self, models_dir: Path, model_sizes: List[int] = None):
        """
        Инициализация компаратора.
        
        Args:
            models_dir: Директория с моделями
            model_sizes: Список размеров моделей для сравнения
        """
        self.models_dir = Path(models_dir)
        self.model_sizes = model_sizes or [8000, 10000, 12000]
        self.models: Dict[int, BPETokenizer] = {}
        self.results: Dict[str, Dict] = {
            'accuracy': {},
            'speed': {},
            'compression': {},
            'vocabulary': {},
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_sizes': self.model_sizes,
                'test_categories': {}
            }
        }
        
        # Цветовая схема для моделей
        self.colors = {
            8000: '#2E86AB',   # Синий
            10000: '#A23B72',  # Фиолетовый
            12000: '#F18F01'   # Оранжевый
        }
        
    # ======================================================================
    # ЗАГРУЗКА МОДЕЛЕЙ
    # ======================================================================
    
    def load_models(self) -> bool:
        """
        Загрузить все модели.
        
        Returns:
            bool: True если все модели загружены успешно
        """
        print_header("ЗАГРУЗКА МОДЕЛЕЙ")
        
        success = True
        for size in self.model_sizes:
            model_path = self.models_dir / f'bpe_{size}'
            vocab_path = model_path / 'vocab.json'
            merges_path = model_path / 'merges.txt'
            
            if not vocab_path.exists() or not merges_path.exists():
                print(f"Модель {size} не найдена в {model_path}")
                success = False
                continue
                
            print(f"Загрузка модели bpe_{size}...")
            try:
                tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
                self.models[size] = tokenizer
                print(f"Загружено: {len(tokenizer.vocab)} токенов")
            except Exception as e:
                print(f"Ошибка: {e}")
                success = False
        
        return success
    
    # ======================================================================
    # ТЕСТОВЫЙ НАБОР
    # ======================================================================
    
    def get_extended_test_set(self) -> Dict[str, List[str]]:
        """
        Получить расширенный набор тестов по категориям.
        
        Returns:
            Dict[str, List[str]]: Категории с примерами кода
        """
        test_set = {
            "Препроцессор": [
                "#include <iostream>",
                "#include <vector>",
                "#include <algorithm>",
                "#include <memory>",
                "#include <thread>",
                "#include <mutex>",
                "#include <condition_variable>",
                "#include <filesystem>",
                "#include <regex>",
                "#include <random>",
                "#define VERSION \"1.0.0\"",
                "#define MAX_SIZE 1024",
                "#define DEBUG true",
                "#pragma once",
                "#ifdef _WIN32",
                "#ifndef HEADER_H",
                "#error \"Unsupported platform\"",
                "#line 42 \"file.cpp\"",
            ],
            
            "Функции и main": [
                "int main() { return 0; }",
                "int main(int argc, char** argv) {",
                "int main(int argc, char* argv[]) {",
                "void printHello() {",
                "static int counter = 0;",
                "inline int square(int x) { return x * x; }",
                "constexpr int factorial(int n) {",
                "virtual void process() override;",
                "virtual void draw() const override final;",
                "auto lambda = [](int x) { return x * 2; };",
            ],
            
            "Классы и структуры": [
                "class MyClass {",
                "struct Point { int x, y; };",
                "template<typename T> class Vector {",
                "template<class T, size_t N> class Array {",
                "class Derived : public Base {",
                "class Singleton { private: static Singleton* instance; };",
                "struct ListNode { int val; ListNode* next; };",
                "interface Drawable { virtual void draw() = 0; };",
                "enum class Color { RED, GREEN, BLUE };",
                "enum Status { OK = 200, ERROR = 404 };",
            ],
            
            "Стандартная библиотека": [
                "std::vector<int> vec = {1, 2, 3};",
                "std::map<std::string, int> scores;",
                "std::unordered_map<int, std::string> dict;",
                "std::set<double> unique_values;",
                "std::queue<Task> taskQueue;",
                "std::stack<char> symbolStack;",
                "std::deque<int> doubleEnded;",
                "std::pair<int, std::string> myPair;",
                "std::tuple<int, double, char> myTuple;",
                "std::optional<std::string> maybeString;",
                "std::variant<int, float, std::string> var;",
                "std::any anything;",
                "std::shared_ptr<int> sp = std::make_shared<int>(42);",
                "std::unique_ptr<int[]> up = std::make_unique<int[]>(10);",
                "std::weak_ptr<int> wp = sp;",
            ],
            
            "Алгоритмы и итераторы": [
                "std::sort(vec.begin(), vec.end());",
                "std::find_if(vec.begin(), vec.end(), pred);",
                "std::transform(src.begin(), src.end(), dst.begin(), func);",
                "std::accumulate(vec.begin(), vec.end(), 0);",
                "std::copy(src.begin(), src.end(), dst.begin());",
                "std::remove_if(vec.begin(), vec.end(), isOdd);",
                "std::unique(vec.begin(), vec.end());",
                "std::reverse(vec.begin(), vec.end());",
                "std::rotate(vec.begin(), vec.begin() + 3, vec.end());",
                "std::partition(vec.begin(), vec.end(), pred);",
            ],
            
            "Потоки и файлы": [
                "std::ifstream file(\"data.txt\");",
                "std::ofstream log(\"log.txt\", std::ios::app);",
                "std::stringstream ss;",
                "std::cout << \"Hello, world!\" << std::endl;",
                "std::cerr << \"Error occurred\" << std::endl;",
                "std::clog << \"Log message\" << std::endl;",
                "std::fstream fs(\"file.bin\", std::ios::in | std::ios::out);",
                "std::getline(std::cin, line);",
            ],
            
            "Шаблоны и метапрограммирование": [
                "template<typename T, typename U>",
                "template<size_t N> struct Factorial {",
                "template<typename... Args>",
                "template<typename T> concept Integral = std::is_integral_v<T>;",
                "template<typename T> requires std::is_arithmetic_v<T>",
                "template<template<typename> class Container>",
                "typename std::enable_if_t<std::is_integral_v<T>>",
                "decltype(auto) forward(Args&&... args)",
                "static_assert(sizeof(T) > 4, \"Type too small\");",
                "using value_type = typename Container::value_type;",
            ],
            
            "Умные указатели и память": [
                "auto ptr = std::make_unique<int>(42);",
                "auto sptr = std::make_shared<double>(3.14);",
                "std::weak_ptr<int> wptr = sptr;",
                "ptr.reset(new int(100));",
                "if (sptr.use_count() > 1) {",
                "int* raw = new int[10];",
                "delete[] raw;",
                "std::malloc(sizeof(int) * 10);",
                "std::free(ptr);",
                "void* operator new(size_t size);",
            ],
            
            "Многопоточность": [
                "std::thread t(threadFunction);",
                "std::mutex mtx;",
                "std::lock_guard<std::mutex> lock(mtx);",
                "std::unique_lock<std::mutex> ulock(mtx);",
                "std::condition_variable cv;",
                "cv.wait(lock, []{ return ready; });",
                "std::atomic<int> counter{0};",
                "auto future = std::async(std::launch::async, task);",
                "std::promise<int> promise;",
                "std::packaged_task<int()> task(work);",
            ],
            
            "Регулярные выражения": [
                "std::regex pattern(\"\\\\d+\");",
                "std::smatch matches;",
                "std::regex_match(str, matches, pattern);",
                "std::regex_search(str, matches, pattern);",
                "std::regex_replace(str, pattern, \"replacement\");",
                "std::regex_constants::icase",
                "auto words_begin = std::sregex_iterator(str.begin(), str.end(), pattern);",
            ],
            
            "Исключения и ошибки": [
                "try {",
                "catch (const std::exception& e) {",
                "catch (...) {",
                "throw std::runtime_error(\"Error message\");",
                "throw;",
                "std::terminate();",
                "noexcept(false)",
                "std::nested_exception",
            ],
            
            "Комментарии": [
                "// Однострочный комментарий",
                "/* Многострочный комментарий */",
                "/** Doxygen комментарий */",
                "// TODO: реализовать функцию",
                "// FIXME: исправить баг",
                "// NOTE: важное замечание",
                "/// Тройной слеш комментарий",
                "/*! Другой стиль Doxygen */",
                "// Это комментарий на русском языке",
                "// 这是中文评论",
                "// これは日本語のコメントです",
                "// Это смесь English и русского",
            ],
            
            "Смешанный код": [
                "int main() { // главная функция",
                "std::cout << \"Привет, мир!\" << std::endl; // вывод",
                "auto result = сумма(a, b); // вычисление суммы",
                "// Сложный алгоритм:\nfor (int i = 0; i < n; ++i) {",
                "class MyClass { // класс с русским комментарием",
                "throw std::runtime_error(\"Ошибка: \" + message); // исключение",
                "/*\n * Многострочный комментарий\n * с русским текстом\n */",
                "/// \\param name имя пользователя",
            ],
            
            "Сложные конструкции C++17/20": [
                "std::unordered_map<std::string, std::vector<int>> map;",
                "std::function<int(int)> func = [](int x) { return x * x; };",
                "auto result = co_await task;",
                "template<typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;",
                "std::apply([](auto&&... args) { (print(args), ...); }, tuple);",
                "std::visit([](auto&& arg) { std::cout << arg; }, variant);",
                "std::conditional_t<is_integral_v<T>, int, double>",
                "std::chrono::high_resolution_clock::now();",
            ],
            
            "Реальные фрагменты": [
                "class Singleton { private: Singleton() = default; };",
                "struct Person { std::string name; int age; };",
                "void bubbleSort(int arr[], int n) {",
                "    for (int i = 0; i < n - 1; ++i)",
                "        for (int j = 0; j < n - i - 1; ++j)",
                "            if (arr[j] > arr[j + 1])",
                "                std::swap(arr[j], arr[j + 1]);",
                "}",
                "",
                "class ThreadPool {",
                "public:",
                "    template<class F>",
                "    auto enqueue(F&& f) -> std::future<decltype(f())> {",
                "        auto task = std::make_shared<std::packaged_task<decltype(f())()>>(",
                "            std::forward<F>(f)",
                "        );",
                "        auto res = task->get_future();",
                "        queue_.push([task]() { (*task)(); });",
                "        return res;",
                "    }",
                "private:",
                "    std::queue<std::function<void()>> queue_;",
                "};",
            ]
        }
        
        # Сохраняем информацию о категориях в метаданные
        self.results['metadata']['test_categories'] = {
            name: len(tests) for name, tests in test_set.items()
        }
        
        return test_set
    
    # ======================================================================
    # ТЕСТ ТОЧНОСТИ
    # ======================================================================
    
    def test_accuracy(self, test_categories: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Тестирование точности roundtrip по категориям.
        
        Args:
            test_categories: Категории с тестовыми текстами
            
        Returns:
            Dict[int, Dict]: Результаты тестов точности
        """
        print_header("ТЕСТ ТОЧНОСТИ ПО КАТЕГОРИЯМ")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\nМодель bpe_{size}:")
            category_results = {}
            total_perfect = 0
            total_tests = 0
            
            for category, texts in test_categories.items():
                perfect = 0
                total = len(texts)
                
                for text in texts:
                    try:
                        encoded = tokenizer.encode(text)
                        decoded = tokenizer.decode(encoded)
                        if text == decoded:
                            perfect += 1
                    except Exception as e:
                        print(f" !!! Ошибка при обработке '{text[:30]}...': {e}")
                
                accuracy = (perfect / total) * 100 if total > 0 else 0
                category_results[category] = {
                    'perfect': perfect,
                    'total': total,
                    'accuracy': accuracy
                }
                
                total_perfect += perfect
                total_tests += total
                
                # Выводим результат с цветом
                color = '[OK]' if accuracy > 90 else '[!!!]' if accuracy > 70 else '[BAD]'
                print(f"  {color} {category[:30]:<30} {accuracy:>5.1f}% ({perfect}/{total})")
            
            overall_accuracy = (total_perfect / total_tests) * 100 if total_tests > 0 else 0
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'perfect': total_perfect,
                    'total': total_tests,
                    'accuracy': overall_accuracy
                }
            }
            
            print(f"\nОбщая точность: {overall_accuracy:.1f}% ({total_perfect}/{total_tests})")
        
        self.results['accuracy'] = results
        return results
    
    # ======================================================================
    # ТЕСТ СКОРОСТИ
    # ======================================================================
    
    def test_speed(self, test_categories: Dict[str, List[str]], iterations: int = 50) -> Dict[int, Dict]:
        """
        Тестирование скорости encode/decode.
        
        Args:
            test_categories: Категории с тестовыми текстами
            iterations: Количество итераций для усреднения
            
        Returns:
            Dict[int, Dict]: Результаты тестов скорости
        """
        print_header("ТЕСТ СКОРОСТИ")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\n⚡ Модель bpe_{size}:")
            category_results = {}
            
            total_encode_time = 0
            total_decode_time = 0
            total_ops = 0
            
            for category, texts in test_categories.items():
                # Прогрев
                for _ in range(5):
                    for text in texts:
                        tokenizer.encode(text)
                
                # Encode тест
                start = time.perf_counter()
                all_encoded = []
                for _ in range(iterations):
                    for text in texts:
                        all_encoded.append(tokenizer.encode(text))
                encode_time = time.perf_counter() - start
                
                # Decode тест
                start = time.perf_counter()
                for encoded in all_encoded:
                    tokenizer.decode(encoded)
                decode_time = time.perf_counter() - start
                
                ops = len(texts) * iterations
                total_ops += ops
                total_encode_time += encode_time
                total_decode_time += decode_time
                
                encode_per_text = (encode_time / ops) * 1000  # мс
                decode_per_text = (decode_time / ops) * 1000  # мс
                
                category_results[category] = {
                    'encode_time': encode_time,
                    'decode_time': decode_time,
                    'encode_per_text': encode_per_text,
                    'decode_per_text': decode_per_text,
                    'ops': ops
                }
                
                print(f"  {category[:30]:<30} "
                      f"encode: {encode_per_text:>5.3f} мс, "
                      f"decode: {decode_per_text:>5.3f} мс")
            
            # Общие результаты
            avg_encode = (total_encode_time / total_ops) * 1000
            avg_decode = (total_decode_time / total_ops) * 1000
            
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'encode_time': total_encode_time,
                    'decode_time': total_decode_time,
                    'encode_per_text': avg_encode,
                    'decode_per_text': avg_decode,
                    'total_ops': total_ops
                }
            }
            
            print(f"\nСреднее: encode {avg_encode:.3f} мс, decode {avg_decode:.3f} мс")
        
        self.results['speed'] = results
        return results
    
    # ======================================================================
    # ТЕСТ СЖАТИЯ
    # ======================================================================
    
    def test_compression(self, test_categories: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Тестирование степени сжатия.
        
        Args:
            test_categories: Категории с тестовыми текстами
            
        Returns:
            Dict[int, Dict]: Результаты тестов сжатия
        """
        print_header("ТЕСТ СЖАТИЯ")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\nМодель bpe_{size}:")
            category_results = {}
            
            total_chars = 0
            total_tokens = 0
            
            for category, texts in test_categories.items():
                chars = 0
                tokens = 0
                
                for text in texts:
                    encoded = tokenizer.encode(text)
                    chars += len(text)
                    tokens += len(encoded)
                
                total_chars += chars
                total_tokens += tokens
                
                ratio = chars / tokens if tokens > 0 else 0
                category_results[category] = {
                    'chars': chars,
                    'tokens': tokens,
                    'ratio': ratio
                }
                
                print(f"  {category[:30]:<30} {ratio:>5.2f} символов/токен")
            
            overall_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'chars': total_chars,
                    'tokens': total_tokens,
                    'ratio': overall_ratio
                }
            }
            print(f"\nСреднее сжатие: {overall_ratio:.2f} символов/токен")
        
        self.results['compression'] = results
        return results
    
    # ======================================================================
    # АНАЛИЗ СЛОВАРЕЙ
    # ======================================================================
    
    def analyze_vocabulary(self) -> Dict[int, Dict]:
        """
        Анализ состава словарей.
        
        Returns:
            Dict[int, Dict]: Статистика по словарям
        """
        print_header("АНАЛИЗ СЛОВАРЕЙ")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            # Статистика по длинам токенов
            token_lengths = [len(token) for token in tokenizer.vocab.values()]
            
            # Типы токенов
            ascii_tokens = 0
            unicode_tokens = 0
            special_tokens = 0
            single_char = 0
            multi_char = 0
            
            for token in tokenizer.vocab.values():
                if token in tokenizer.special_tokens:
                    special_tokens += 1
                elif all(ord(c) < 128 for c in token):
                    ascii_tokens += 1
                else:
                    unicode_tokens += 1
                
                if len(token) == 1:
                    single_char += 1
                else:
                    multi_char += 1
            
            # Статистика
            results[size] = {
                'vocab_size': len(tokenizer.vocab),
                'special_tokens': special_tokens,
                'ascii_tokens': ascii_tokens,
                'unicode_tokens': unicode_tokens,
                'single_char': single_char,
                'multi_char': multi_char,
                'token_lengths': {
                    'min': min(token_lengths),
                    'max': max(token_lengths),
                    'avg': sum(token_lengths) / len(token_lengths),
                    'median': sorted(token_lengths)[len(token_lengths) // 2]
                }
            }
            
            print(f"\nМодель bpe_{size}:")
            print(f"  Размер: {results[size]['vocab_size']} токенов")
            print(f"  ASCII: {results[size]['ascii_tokens']} ({results[size]['ascii_tokens']*100/len(tokenizer.vocab):.1f}%)")
            print(f"  Unicode: {results[size]['unicode_tokens']} ({results[size]['unicode_tokens']*100/len(tokenizer.vocab):.1f}%)")
            print(f"  Спец. токены: {results[size]['special_tokens']}")
            print(f"  Односимвольных: {results[size]['single_char']}")
            print(f"  Многосимвольных: {results[size]['multi_char']}")
            print(f"  Средняя длина: {results[size]['token_lengths']['avg']:.2f}")
        
        self.results['vocabulary'] = results
        return results
    
    # ======================================================================
    # ГЕНЕРАЦИЯ ОТЧЕТА
    # ======================================================================
    
    def generate_report(self) -> str:
        """
        Сгенерировать текстовый отчет.
        
        Returns:
            str: Текстовый отчет с результатами
        """
        lines = []
        lines.append("=" * 100)
        lines.append("СРАВНЕНИЕ ТРЕХ BPE МОДЕЛЕЙ (8000, 10000, 12000)".center(100))
        lines.append("=" * 100)
        lines.append("")
        
        # Информация о тестах
        lines.append("ИНФОРМАЦИЯ О ТЕСТАХ")
        lines.append("-" * 40)
        total_tests = sum(self.results['metadata']['test_categories'].values())
        lines.append(f"Всего тестов: {total_tests}")
        lines.append(f"Категорий: {len(self.results['metadata']['test_categories'])}")
        lines.append(f"Дата тестирования: {self.results['metadata']['timestamp']}")
        lines.append("")
        
        # Общая точность
        lines.append("ОБЩАЯ ТОЧНОСТЬ")
        lines.append("-" * 60)
        for size in self.model_sizes:
            acc = self.results['accuracy'][size]['overall']['accuracy']
            perfect = self.results['accuracy'][size]['overall']['perfect']
            total = self.results['accuracy'][size]['overall']['total']
            lines.append(f"bpe_{size:5}: {acc:5.1f}% ({perfect}/{total})")
        lines.append("")
        
        # Точность по категориям
        lines.append("ТОЧНОСТЬ ПО КАТЕГОРИЯМ (%)")
        lines.append("-" * 80)
        header = f"{'Категория':<30}"
        for size in self.model_sizes:
            header += f" bpe_{size:>6}"
        header += " разброс"
        lines.append(header)
        lines.append("-" * 80)
        
        categories = list(self.results['accuracy'][self.model_sizes[0]]['by_category'].keys())
        for cat in categories:
            line = f"{cat[:28]:<30}"
            accuracies = []
            for size in self.model_sizes:
                acc = self.results['accuracy'][size]['by_category'][cat]['accuracy']
                accuracies.append(acc)
                line += f" {acc:>6.1f}"
            spread = max(accuracies) - min(accuracies)
            line += f" {spread:>7.1f}"
            lines.append(line)
        lines.append("")
        
        # Сжатие
        lines.append("СТЕПЕНЬ СЖАТИЯ (символов/токен)")
        lines.append("-" * 80)
        header = f"{'Категория':<30}"
        for size in self.model_sizes:
            header += f" bpe_{size:>6}"
        header += " улучшение"
        lines.append(header)
        lines.append("-" * 80)
        
        base_ratios = {}
        for cat in categories:
            line = f"{cat[:28]:<30}"
            ratios = []
            for size in self.model_sizes:
                ratio = self.results['compression'][size]['by_category'][cat]['ratio']
                ratios.append(ratio)
                line += f" {ratio:>6.2f}"
                if size == self.model_sizes[0]:
                    base_ratios[cat] = ratio
            improvement = ((ratios[-1] - base_ratios[cat]) / base_ratios[cat]) * 100
            line += f" {improvement:>+7.1f}%"
            lines.append(line)
        lines.append("")
        
        # Скорость
        lines.append("⚡ СКОРОСТЬ РАБОТЫ")
        lines.append("-" * 60)
        header = f"{'Модель':<10} {'Encode (мс)':>12} {'Decode (мс)':>12} {'Всего операций':>15}"
        lines.append(header)
        lines.append("-" * 60)
        
        for size in self.model_sizes:
            enc = self.results['speed'][size]['overall']['encode_per_text']
            dec = self.results['speed'][size]['overall']['decode_per_text']
            ops = self.results['speed'][size]['overall']['total_ops']
            lines.append(f"bpe_{size:<6} {enc:>11.3f} {dec:>11.3f} {ops:>15}")
        lines.append("")
        
        # Характеристики словарей
        lines.append("ХАРАКТЕРИСТИКИ СЛОВАРЕЙ")
        lines.append("-" * 70)
        header = f"{'Модель':<10} {'Размер':>8} {'ASCII':>8} {'Unicode':>8} {'Спец':>6} {'Ср.длина':>9}"
        lines.append(header)
        lines.append("-" * 70)
        
        for size in self.model_sizes:
            vocab = self.results['vocabulary'][size]
            lines.append(
                f"bpe_{size:<6} {vocab['vocab_size']:>8} "
                f"{vocab['ascii_tokens']:>8} {vocab['unicode_tokens']:>8} "
                f"{vocab['special_tokens']:>6} {vocab['token_lengths']['avg']:>8.2f}"
            )
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    # ======================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ======================================================================
    
    def plot_comparison(self, output_dir: Path):
        """
        Создать графики сравнения.
        
        Args:
            output_dir: Директория для сохранения графиков
        """
        print_header("СОЗДАНИЕ ГРАФИКОВ")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Сравнение трех BPE моделей (8000, 10000, 12000)', 
                        fontsize=16, fontweight='bold')
            
            categories = list(self.results['accuracy'][self.model_sizes[0]]['by_category'].keys())
            # Берем первые 8 категорий для читаемости
            display_cats = categories[:8]
            x = np.arange(len(display_cats))
            width = 0.25
            
            # 1. Точность по категориям
            ax = axes[0, 0]
            for i, size in enumerate(self.model_sizes):
                accuracies = [self.results['accuracy'][size]['by_category'][cat]['accuracy'] 
                             for cat in display_cats]
                bars = ax.bar(x + (i - 1) * width, accuracies, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Точность по категориям', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                              rotation=45, ha='right')
            ax.set_ylabel('Точность (%)')
            ax.legend()
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, axis='y')
            
            # 2. Степень сжатия
            ax = axes[0, 1]
            for i, size in enumerate(self.model_sizes):
                ratios = [self.results['compression'][size]['by_category'][cat]['ratio'] 
                         for cat in display_cats]
                bars = ax.bar(x + (i - 1) * width, ratios, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Степень сжатия (символов/токен)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                              rotation=45, ha='right')
            ax.set_ylabel('Символов на токен')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 3. Скорость encode
            ax = axes[0, 2]
            for i, size in enumerate(self.model_sizes):
                speeds = [self.results['speed'][size]['by_category'][cat]['encode_per_text'] 
                         for cat in display_cats]
                bars = ax.bar(x + (i - 1) * width, speeds, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Скорость encode (мс/текст)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                              rotation=45, ha='right')
            ax.set_ylabel('мс')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. Скорость decode
            ax = axes[1, 0]
            for i, size in enumerate(self.model_sizes):
                speeds = [self.results['speed'][size]['by_category'][cat]['decode_per_text'] 
                         for cat in display_cats]
                bars = ax.bar(x + (i - 1) * width, speeds, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Скорость decode (мс/текст)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                              rotation=45, ha='right')
            ax.set_ylabel('мс')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 5. Распределение токенов по типам
            ax = axes[1, 1]
            x_types = np.arange(3)  # ASCII, Unicode, Специальные
            for i, size in enumerate(self.model_sizes):
                vocab = self.results['vocabulary'][size]
                values = [vocab['ascii_tokens'], vocab['unicode_tokens'], vocab['special_tokens']]
                bars = ax.bar(x_types + (i - 1) * width, values, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                           f'{height}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Состав словарей', fontsize=12, fontweight='bold')
            ax.set_xticks(x_types)
            ax.set_xticklabels(['ASCII', 'Unicode', 'Специальные'])
            ax.set_ylabel('Количество токенов')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 6. Сводное сравнение
            ax = axes[1, 2]
            metrics = ['Точность', 'Сжатие', 'Скорость\nencode', 'Скорость\ndecode', 'Размер\nсловаря']
            x_sum = np.arange(len(metrics))
            
            # Нормализуем метрики для сравнения
            for i, size in enumerate(self.model_sizes):
                # Точность (средняя)
                avg_acc = np.mean([self.results['accuracy'][size]['by_category'][cat]['accuracy'] 
                                  for cat in categories])
                
                # Сжатие
                comp = self.results['compression'][size]['overall']['ratio']
                max_comp = max(self.results['compression'][s]['overall']['ratio'] 
                              for s in self.model_sizes)
                
                # Скорость (инвертирована, т.к. меньше = лучше)
                speed_enc = 1 / self.results['speed'][size]['overall']['encode_per_text']
                speed_dec = 1 / self.results['speed'][size]['overall']['decode_per_text']
                max_speed_enc = max(1 / self.results['speed'][s]['overall']['encode_per_text'] 
                                   for s in self.model_sizes)
                max_speed_dec = max(1 / self.results['speed'][s]['overall']['decode_per_text'] 
                                   for s in self.model_sizes)
                
                # Размер (инвертирован, т.к. меньше = лучше)
                size_norm = 1 / size
                max_size_norm = max(1 / s for s in self.model_sizes)
                
                values = [
                    avg_acc,
                    (comp / max_comp) * 100,
                    (speed_enc / max_speed_enc) * 100,
                    (speed_dec / max_speed_dec) * 100,
                    (size_norm / max_size_norm) * 100
                ]
                
                bars = ax.bar(x_sum + (i - 1) * width, values, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Сводное сравнение (нормализовано)', fontsize=12, fontweight='bold')
            ax.set_xticks(x_sum)
            ax.set_xticklabels(metrics)
            ax.set_ylabel('Относительная оценка (%)')
            ax.legend()
            ax.set_ylim([0, 110])
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Сохраняем графики
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = output_dir / 'three_model_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"  PNG график: {plot_path}")
            
            pdf_path = output_dir / 'three_model_comparison.pdf'
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            print(f"  PDF график: {pdf_path}")
            
            plt.close()
            print(f"  Графики успешно созданы")
            
        except Exception as e:
            print(f"  Ошибка при создании графиков: {e}")
            import traceback
            traceback.print_exc()
    
    # ======================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    def save_results(self, output_dir: Path):
        """
        Сохранить результаты в файлы.
        
        Args:
            output_dir: Директория для сохранения
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Функция для конвертации в JSON-совместимые типы
        def convert_for_json(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_for_json(v) for v in obj]
            return obj
        
        # JSON результаты
        json_path = output_dir / 'three_model_comparison.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_for_json(self.results), f, ensure_ascii=False, indent=2)
        print(f"  JSON результаты: {json_path}")
        
        # Текстовый отчет
        report_path = output_dir / 'three_model_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"  Текстовый отчет: {report_path}")
        
        # Графики
        self.plot_comparison(output_dir)


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(description='Сравнение трех BPE моделей')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Быстрый режим (меньше итераций)')
    parser.add_argument('--plot-only', '-p', action='store_true',
                       help='Только построить графики из сохраненных результатов')
    
    args = parser.parse_args()
    
    # Получаем пути
    paths = get_project_paths()
    
    print_header("СРАВНЕНИЕ ТРЕХ BPE МОДЕЛЕЙ")
    print(f"Директория моделей: {paths['models_dir']}")
    print(f"Директория результатов: {paths['output_dir']}")
    
    # Режим только графиков
    if args.plot_only:
        json_path = paths['output_dir'] / 'three_model_comparison.json'
        if not json_path.exists():
            print(f"Файл с результатами не найден: {json_path}")
            return 1
        
        print(f"\nЗагрузка результатов из {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Создаем временный компаратор для графиков
        comparator = ThreeModelComparison(paths['models_dir'])
        comparator.results = results
        comparator.plot_comparison(paths['output_dir'])
        return 0
    
    # Полный режим
    comparator = ThreeModelComparison(
        models_dir=paths['models_dir'],
        model_sizes=[8000, 10000, 12000]
    )
    
    try:
        # Загружаем модели
        if not comparator.load_models():
            print(f"\nНе удалось загрузить все модели")
            print(f"   Проверьте наличие моделей в: {paths['models_dir']}")
            return 1
        
        print(f"\nЗагружено {len(comparator.models)} моделей")
        
        # Получаем тесты
        test_categories = comparator.get_extended_test_set()
        total_tests = sum(len(v) for v in test_categories.values())
        print(f"Всего тестов: {total_tests} в {len(test_categories)} категориях")
        
        # Запускаем тесты
        iterations = 20 if args.quick else 50
        print(f"Количество итераций: {iterations}")
        
        comparator.test_accuracy(test_categories)
        comparator.test_speed(test_categories, iterations=iterations)
        comparator.test_compression(test_categories)
        comparator.analyze_vocabulary()
        
        # Выводим отчет
        print("\n" + comparator.generate_report())
        
        # Сохраняем результаты
        print_header("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        comparator.save_results(paths['output_dir'])
        
        print_header("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("Все тесты успешно выполнены!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n !!! Тестирование прервано пользователем")
        return 1
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())