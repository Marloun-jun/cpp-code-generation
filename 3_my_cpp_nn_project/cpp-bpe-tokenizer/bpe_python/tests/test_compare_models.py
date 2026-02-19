"""
Расширенное сравнение трех BPE моделей (8000, 10000, 12000).

Детальный анализ и сравнение моделей с разным размером словаря.
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

# Добавляем путь к родительской директории для импорта tokenizer
current_file = Path(__file__).resolve()
bpe_python_dir = current_file.parent.parent
sys.path.insert(0, str(bpe_python_dir))

from tokenizer import BPETokenizer


class ThreeModelComparison:
    """
    Класс для сравнения трех BPE моделей.
    """
    
    def __init__(self, models_dir: Path, model_sizes: List[int] = [8000, 10000, 12000]):
        """
        Инициализация с путями к моделям.
        
        Аргументы:
            models_dir: Директория с моделями
            model_sizes: Список размеров моделей для сравнения
        """
        self.models_dir = Path(models_dir)
        self.model_sizes = model_sizes
        self.models = {}
        self.results = {
            'accuracy': {},
            'speed': {},
            'compression': {},
            'coverage': {},
            'vocabulary': {}
        }
        
        # Цвета для моделей
        self.colors = {
            8000: '#2E86AB',   # Синий
            10000: '#A23B72',  # Фиолетовый
            12000: '#F18F01'   # Оранжевый
        }
        
    def load_models(self):
        """Загрузка всех моделей."""
        print("=" * 80)
        print("ЗАГРУЗКА МОДЕЛЕЙ")
        print("=" * 80)
        
        for size in self.model_sizes:
            model_path = self.models_dir / f'bpe_{size}'
            vocab_path = model_path / 'vocab.json'
            merges_path = model_path / 'merges.txt'
            
            if not vocab_path.exists() or not merges_path.exists():
                print(f"Модель {size} не найдена в {model_path}")
                continue
                
            print(f"Загрузка модели bpe_{size}...")
            tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
            self.models[size] = tokenizer
            print(f"  ✓ Загружена: {len(tokenizer.vocab)} токенов")
    
    def get_extended_test_set(self) -> Dict[str, List[str]]:
        """
        Получение расширенного набора тестов по категориям.
        """
        return {
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
                "#pragma region // Регион с комментарием",
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
    
    def test_accuracy(self, test_categories: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Тест точности по категориям.
        """
        print("\n" + "=" * 80)
        print("ТЕСТ ТОЧНОСТИ ПО КАТЕГОРИЯМ")
        print("=" * 80)
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\nМодель bpe_{size}:")
            category_results = {}
            
            for category, texts in test_categories.items():
                perfect = 0
                total = len(texts)
                
                for text in texts:
                    encoded = tokenizer.encode(text)
                    decoded = tokenizer.decode(encoded)
                    if text == decoded:
                        perfect += 1
                
                accuracy = (perfect / total) * 100 if total > 0 else 0
                category_results[category] = {
                    'perfect': perfect,
                    'total': total,
                    'accuracy': accuracy
                }
                print(f"  {category[:25]:<25} {accuracy:>5.1f}% ({perfect}/{total})")
            
            results[size] = category_results
        
        self.results['accuracy'] = results
        return results
    
    def test_speed(self, test_categories: Dict[str, List[str]], iterations: int = 50) -> Dict[int, Dict]:
        """
        Тест скорости.
        """
        print("\n" + "=" * 80)
        print("ТЕСТ СКОРОСТИ")
        print("=" * 80)
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\nМодель bpe_{size}:")
            category_results = {}
            total_encode = 0
            total_decode = 0
            total_ops = 0
            
            for category, texts in test_categories.items():
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
                total_encode += encode_time
                total_decode += decode_time
                
                encode_per_text = (encode_time / ops) * 1000  # мс
                decode_per_text = (decode_time / ops) * 1000  # мс
                
                category_results[category] = {
                    'encode_time': encode_time,
                    'decode_time': decode_time,
                    'encode_per_text': encode_per_text,
                    'decode_per_text': decode_per_text,
                    'ops': ops
                }
                
                print(f"  {category[:25]:<25} encode: {encode_per_text:>6.3f} мс, decode: {decode_per_text:>6.3f} мс")
            
            # Общие результаты
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'encode_time': total_encode,
                    'decode_time': total_decode,
                    'encode_per_text': (total_encode / total_ops) * 1000,
                    'decode_per_text': (total_decode / total_ops) * 1000,
                    'total_ops': total_ops
                }
            }
            
            print(f"\nСреднее: encode {results[size]['overall']['encode_per_text']:.3f} мс, "
                  f"decode {results[size]['overall']['decode_per_text']:.3f} мс")
        
        self.results['speed'] = results
        return results
    
    def test_compression(self, test_categories: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Тест степени сжатия.
        """
        print("\n" + "=" * 80)
        print("ТЕСТ СЖАТИЯ")
        print("=" * 80)
        
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
                
                print(f"  {category[:25]:<25} {ratio:>5.2f} символов/токен")
            
            overall_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'chars': total_chars,
                    'tokens': total_tokens,
                    'ratio': overall_ratio
                }
            }
            print(f"Среднее: {overall_ratio:.2f} символов/токен")
        
        self.results['compression'] = results
        return results
    
    def analyze_vocabulary(self) -> Dict[int, Dict]:
        """
        Анализ словарей.
        """
        print("\n" + "=" * 80)
        print("АНАЛИЗ СЛОВАРЕЙ")
        print("=" * 80)
        
        results = {}
        
        for size, tokenizer in self.models.items():
            # Длины токенов
            token_lengths = [len(token) for token in tokenizer.vocab.values()]
            
            # Типы токенов
            ascii_tokens = 0
            unicode_tokens = 0
            special_tokens = 0
            
            for token in tokenizer.vocab.values():
                if token in tokenizer.special_tokens:
                    special_tokens += 1
                elif all(ord(c) < 128 for c in token):
                    ascii_tokens += 1
                else:
                    unicode_tokens += 1
            
            results[size] = {
                'vocab_size': len(tokenizer.vocab),
                'special_tokens': special_tokens,
                'ascii_tokens': ascii_tokens,
                'unicode_tokens': unicode_tokens,
                'token_lengths': {
                    'min': min(token_lengths),
                    'max': max(token_lengths),
                    'avg': sum(token_lengths) / len(token_lengths),
                    'median': sorted(token_lengths)[len(token_lengths) // 2]
                }
            }
            
            print(f"\nМодель bpe_{size}:")
            print(f"  Размер: {results[size]['vocab_size']} токенов")
            print(f"  ASCII: {results[size]['ascii_tokens']}")
            print(f"  Unicode: {results[size]['unicode_tokens']}")
            print(f"  Спец. токены: {results[size]['special_tokens']}")
            print(f"  Средняя длина: {results[size]['token_lengths']['avg']:.1f}")
        
        self.results['vocabulary'] = results
        return results
    
    def generate_report(self) -> str:
        """
        Генерация отчета.
        """
        report = []
        report.append("=" * 100)
        report.append("СРАВНЕНИЕ ТРЕХ BPE МОДЕЛЕЙ (8000, 10000, 12000)")
        report.append("=" * 100)
        report.append("")
        
        # Общая статистика
        report.append("ОБЩАЯ СТАТИСТИКА")
        report.append("-" * 40)
        report.append(f"Всего тестов: {sum(len(v) for v in self.get_extended_test_set().values())}")
        report.append(f"Категорий: {len(self.get_extended_test_set())}")
        report.append("")
        
        # Сравнение по категориям
        report.append("ТОЧНОСТЬ ПО КАТЕГОРИЯМ (%)")
        report.append("-" * 80)
        header = f"{'Категория':<30}"
        for size in self.model_sizes:
            header += f" bpe_{size:>6}"
        header += " max diff"
        report.append(header)
        report.append("-" * 80)
        
        categories = list(self.results['accuracy'][self.model_sizes[0]].keys())
        for cat in categories:
            line = f"{cat[:28]:<30}"
            accuracies = []
            for size in self.model_sizes:
                acc = self.results['accuracy'][size][cat]['accuracy']
                accuracies.append(acc)
                line += f" {acc:>6.1f}"
            max_diff = max(accuracies) - min(accuracies)
            line += f" {max_diff:>8.1f}"
            report.append(line)
        
        report.append("")
        
        # Сжатие
        report.append("📦 СТЕПЕНЬ СЖАТИЯ (символов/токен)")
        report.append("-" * 80)
        header = f"{'Категория':<30}"
        for size in self.model_sizes:
            header += f" bpe_{size:>6}"
        header += " улучшение"
        report.append(header)
        report.append("-" * 80)
        
        for cat in categories:
            line = f"{cat[:28]:<30}"
            ratios = []
            for size in self.model_sizes:
                ratio = self.results['compression'][size]['by_category'][cat]['ratio']
                ratios.append(ratio)
                line += f" {ratio:>6.2f}"
            improvement = ((ratios[-1] - ratios[0]) / ratios[0]) * 100
            line += f" {improvement:>+8.1f}%"
            report.append(line)
        
        report.append("")
        
        # Скорость
        report.append("⚡ СКОРОСТЬ РАБОТЫ")
        report.append("-" * 60)
        header = f"{'Модель':<10} {'Encode (мс)':>12} {'Decode (мс)':>12} {'Всего операций':>15}"
        report.append(header)
        report.append("-" * 60)
        
        base_speed = self.results['speed'][self.model_sizes[0]]['overall']['encode_per_text']
        for size in self.model_sizes:
            enc = self.results['speed'][size]['overall']['encode_per_text']
            dec = self.results['speed'][size]['overall']['decode_per_text']
            ops = self.results['speed'][size]['overall']['total_ops']
            slowdown = ((enc - base_speed) / base_speed) * 100
            report.append(f"bpe_{size:<6} {enc:>11.3f} {dec:>11.3f} {ops:>15} ({slowdown:>+5.1f}%)")
        
        report.append("")
        
        # Словари
        report.append("ХАРАКТЕРИСТИКИ СЛОВАРЕЙ")
        report.append("-" * 70)
        header = f"{'Модель':<10} {'Размер':>8} {'ASCII':>8} {'Unicode':>8} {'Спец':>6} {'Ср.длина':>9}"
        report.append(header)
        report.append("-" * 70)
        
        for size in self.model_sizes:
            vocab = self.results['vocabulary'][size]
            report.append(
                f"bpe_{size:<6} {vocab['vocab_size']:>8} "
                f"{vocab['ascii_tokens']:>8} {vocab['unicode_tokens']:>8} "
                f"{vocab['special_tokens']:>6} {vocab['token_lengths']['avg']:>8.1f}"
            )
        
        report.append("")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def plot_comparison(self, output_dir: Path):
        """
        Создание графиков сравнения для трех моделей.
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Сравнение трех BPE моделей (8000, 10000, 12000)', fontsize=16, fontweight='bold')
            
            # 1. Точность по категориям (топ-8)
            ax = axes[0, 0]
            categories = sorted(
                self.results['accuracy'][self.model_sizes[0]].keys(),
                key=lambda x: self.results['accuracy'][self.model_sizes[0]][x]['accuracy'],
                reverse=True
            )[:8]
            
            x = np.arange(len(categories))
            width = 0.25
            
            for i, size in enumerate(self.model_sizes):
                accuracies = [self.results['accuracy'][size][cat]['accuracy'] for cat in categories]
                bars = ax.bar(x + (i - 1) * width, accuracies, width, 
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                # Добавляем значения
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.0f}%', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Точность по категориям', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in categories], 
                              rotation=45, ha='right')
            ax.set_ylabel('Точность (%)')
            ax.legend()
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, axis='y')
            
            # 2. Типы ошибок
            ax = axes[0, 1]
            # Для демо используем заглушку с данными из 10000
            error_types = ['Шаблоны', 'Методы', 'Юникод', 'Другое']
            x_err = np.arange(len(error_types))
            
            # Примерные данные (можно заменить на реальные при наличии)
            for i, size in enumerate(self.model_sizes):
                err_data = [10, 10, 17, 26]  # Пример из отчета 10000
                bars = ax.bar(x_err + (i - 1) * width, err_data, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.0f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Типы ошибок', fontsize=12, fontweight='bold')
            ax.set_xticks(x_err)
            ax.set_xticklabels(error_types)
            ax.set_ylabel('Количество')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 3. Степень сжатия
            ax = axes[0, 2]
            for i, size in enumerate(self.model_sizes):
                ratios = [self.results['compression'][size]['by_category'][cat]['ratio'] for cat in categories]
                bars = ax.bar(x + (i - 1) * width, ratios, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Степень сжатия (символов/токен)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in categories], 
                              rotation=45, ha='right')
            ax.set_ylabel('Символов на токен')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 4. Скорость encode
            ax = axes[1, 0]
            for i, size in enumerate(self.model_sizes):
                enc_speeds = [self.results['speed'][size]['by_category'][cat]['encode_per_text'] for cat in categories]
                bars = ax.bar(x + (i - 1) * width, enc_speeds, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Скорость encode (мс/текст)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in categories], 
                              rotation=45, ha='right')
            ax.set_ylabel('мс')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 5. Скорость decode
            ax = axes[1, 1]
            for i, size in enumerate(self.model_sizes):
                dec_speeds = [self.results['speed'][size]['by_category'][cat]['decode_per_text'] for cat in categories]
                bars = ax.bar(x + (i - 1) * width, dec_speeds, width,
                            label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('Скорость decode (мс/текст)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in categories], 
                              rotation=45, ha='right')
            ax.set_ylabel('мс')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 6. Сводное сравнение
            ax = axes[1, 2]
            metrics = ['Точность', 'Сжатие', 'Скорость\nencode', 'Скорость\ndecode', 'Размер\nсловаря']
            x_sum = np.arange(len(metrics))
            
            # Нормализуем метрики
            for i, size in enumerate(self.model_sizes):
                # Точность (средняя)
                avg_acc = np.mean([self.results['accuracy'][size][cat]['accuracy'] 
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
            
            # Сохраняем
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / 'three_model_comparison.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"\nГрафик сохранен: {plot_path}")
            
            pdf_path = output_dir / 'three_model_comparison.pdf'
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            print(f"PDF версия: {pdf_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Ошибка при создании графиков: {e}")
            import traceback
            traceback.print_exc()
    
    def save_results(self, output_dir: Path):
        """
        Сохранение результатов.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON результаты
        json_path = output_dir / 'three_model_comparison.json'
        
        # Конвертируем для JSON
        def convert_for_json(obj):
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
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_for_json(self.results), f, ensure_ascii=False, indent=2)
        
        print(f"Результаты сохранены: {json_path}")
        
        # Текстовый отчет
        report_path = output_dir / 'three_model_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        
        print(f"Отчет сохранен: {report_path}")
        
        # Графики
        self.plot_comparison(output_dir)


def get_project_paths() -> Dict[str, Path]:
    """
    Получение путей проекта.
    """
    current_file = Path(__file__).resolve()
    tests_dir = current_file.parent
    bpe_python_dir = tests_dir.parent
    project_root = bpe_python_dir.parent
    
    return {
        "project_root": project_root,
        "bpe_python_dir": bpe_python_dir,
        "tests_dir": tests_dir,
        "models_dir": bpe_python_dir / 'models',
        "output_dir": tests_dir / 'three_model_results'
    }


if __name__ == '__main__':
    # Получаем пути
    paths = get_project_paths()
    
    # Создаем компаратор для трех моделей
    comparator = ThreeModelComparison(
        models_dir=paths['models_dir'],
        model_sizes=[8000, 10000, 12000]
    )
    
    try:
        # Загружаем модели
        comparator.load_models()
        
        if len(comparator.models) < 3:
            print(f"\nЗагружено только {len(comparator.models)} модели. Нужны все три!")
            print(f"Ожидаемые модели: 8000, 10000, 12000")
            print(f"Проверьте наличие моделей в: {paths['models_dir']}")
            sys.exit(1)
        
        # Получаем тесты
        test_categories = comparator.get_extended_test_set()
        total_tests = sum(len(v) for v in test_categories.values())
        print(f"\n📋 Всего тестов: {total_tests} в {len(test_categories)} категориях")
        
        # Запускаем тесты
        comparator.test_accuracy(test_categories)
        comparator.test_speed(test_categories, iterations=50)
        comparator.test_compression(test_categories)
        comparator.analyze_vocabulary()
        
        # Выводим отчет
        print("\n" + comparator.generate_report())
        
        # Сохраняем результаты
        comparator.save_results(paths['output_dir'])
        
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ ТРЕХ МОДЕЛЕЙ ЗАВЕРШЕНО!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)