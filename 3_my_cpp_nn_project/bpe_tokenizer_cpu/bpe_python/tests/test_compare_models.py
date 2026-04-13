#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# test_compare_models.py - Расширенное сравнение трех BPE моделей
# ======================================================================
#
# @file test_compare_models.py
# @brief Расширенное сравнение трех BPE моделей (8000, 10000, 12000)
#
# @author Евгений П.
# @date 2026
# @version 3.5.0
#
# @details Детальный анализ и сравнение моделей с разным размером словаря.
#          Этот скрипт проводит комплексное исследование производительности
#          и качества трех основных моделей токенизатора.
#
#          **Измеряемые метрики:**
#
#          1. **Точность (accuracy)**
#             - Процент успешных roundtrip преобразований
#             - Детальный анализ по категориям C++ кода
#             - Список ошибок для отладки
#
#          2. **Скорость работы (performance)**
#             - Время encode (мс на текст)
#             - Время decode (мс на текст)
#             - Пропускная способность (тыс. символов/с)
#             - Без кэширования для чистоты измерений
#
#          3. **Степень сжатия (compression)**
#             - Соотношение символов к токенам
#             - Процент сжатия
#             - Размер в байтах (UTF-8 vs токены)
#
#          4. **Анализ словарей (vocabulary)**
#             - Состав (ASCII, Unicode, специальные токены)
#             - Распределение длин токенов
#             - Самые длинные токены
#             - Частотность символов
#
#          5. **OOV анализ (out-of-vocabulary)**
#             - Покрытие символов
#             - Неизвестные символы по категориям Unicode
#             - Проблемные последовательности
#
#          6. **Глубина сжатия (compression depth)**
#             - Распределение длин токенов в закодированном тексте
#             - Самые частые токены
#             - Диапазоны длин (1-2, 3-5, 6-10, 11+)
#
#          **Категории тестов:**
#          - Препроцессор (#include, #define, #pragma)
#          - Функции и main
#          - Классы и структуры
#          - Стандартная библиотека
#          - Алгоритмы и итераторы
#          - Потоки и файлы
#          - Шаблоны и метапрограммирование
#          - Умные указатели и память
#          - Многопоточность
#          - Регулярные выражения
#          - Исключения и ошибки
#          - Комментарии (включая русские и Unicode)
#          - Сложные конструкции C++17/20
#          - Строковые и числовые литералы
#          - Атрибуты, concepts, coroutines
#          - Unicode и эмодзи
#          - Реальные фрагменты кода
#
# @usage python test_compare_models.py [--quick] [--plot-only] [--full-analysis]
#
# @example
#   python test_compare_models.py                    # Полный анализ
#   python test_compare_models.py --quick            # Быстрый режим (меньше итераций)
#   python test_compare_models.py --full-analysis    # Включая OOV и глубину сжатия
#   python test_compare_models.py --plot-only        # Только графики из сохраненных данных
#   python test_compare_models.py --no-real-files    # Без загрузки реальных файлов
#   python test_compare_models.py --verify-speed     # Доп. проверка скорости
#
# ======================================================================

import sys
import time
import json
import argparse
import random
import hashlib

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()    # tests/test_compare_models.py
TESTS_DIR = CURRENT_FILE.parent            # tests/
BPE_PYTHON_DIR = TESTS_DIR.parent          # bpe_python/
PROJECT_ROOT = BPE_PYTHON_DIR.parent       # bpe_tokenizer_cpu/

# Добавляем путь для импорта токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}!")
    print(f"Убедитесь, что файл tokenizer.py существует в {BPE_PYTHON_DIR}!")
    sys.exit(1)

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def get_project_paths() -> Dict[str, Path]:
    """
    Получить пути проекта.
    
    Returns:
        Dict[str, Path]:
            Словарь с путями проекта со следующими ключами:
            - project_root   - Корневая директория проекта
            - bpe_python_dir - Директория с Python кодом
            - tests_dir      - Директория с тестами
            - models_dir     - Директория с моделями
            - reports_dir    - Директория для сохранения результатов
    """
    return {
        "project_root": PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "tests_dir": TESTS_DIR,
        "models_dir": BPE_PYTHON_DIR / 'models',
        "reports_dir": BPE_PYTHON_DIR / 'reports'
    }

def print_header(title: str, width: int = 80) -> None:
    """
    Вывести заголовок раздела для красивого форматирования вывода.
    
    Args:
        title: Заголовок
        width: Ширина линии
    
    Example:
        >>> print_header("СРАВНЕНИЕ МОДЕЛЕЙ")
        ================================================================================
                                    СРАВНЕНИЕ МОДЕЛЕЙ                                 
        ================================================================================
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

def ensure_reports_dir(path: Path) -> Path:
    """
    Создать директорию для вывода, если её нет.
    
    Args:
        path: Путь к директории
        
    Returns:
        Path: Путь к директории
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

# ======================================================================
# КЛАСС ДЛЯ СРАВНЕНИЯ ТРЕХ МОДЕЛЕЙ
# ======================================================================

class ThreeModelComparison:
    """
    Класс для сравнения трех BPE моделей с разным размером словаря.
    
    Проводит комплексный анализ моделей по множеству метрик:
    - Точность (roundtrip тесты)
    - Скорость encode/decode (без кэширования)
    - Степень сжатия
    - Состав словарей
    - Анализ OOV (out-of-vocabulary)
    - Анализ глубины сжатия
    
    **Цветовая схема для графиков:**
    - 8000  - Синий (#2E86AB)
    - 10000 - Фиолетовый (#A23B72)
    - 12000 - Оранжевый (#F18F01)
    """
    
    def __init__(self, models_dir: Path, model_sizes: List[int] = None):
        """
        Инициализация компаратора.
        
        Args:
            models_dir:  Директория с моделями
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
            'oov_analysis': {},
            'compression_depth': {},
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_sizes': self.model_sizes,
                'test_categories': {},
                'total_tests': 0
            }
        }
        
        # Цветовая схема для моделей
        self.colors = {
            8000: '#2E86AB',     # Синий
            10000: '#A23B72',    # Фиолетовый
            12000: '#F18F01'     # Оранжевый
        }
        
        # Кэш для результатов encode/decode (ОТКЛЮЧЕН для точного измерения скорости)
        self._encode_cache = {}
        self._decode_cache = {}
        self._cache_enabled = False    # Кэш отключен для точных замеров
        
    def _get_cache_key(self, text: str, model_size: int) -> str:
        """Создать ключ для кэша."""
        return hashlib.md5(f"{model_size}:{text}".encode()).hexdigest()
    
    def encode_without_cache(self, tokenizer: BPETokenizer, text: str) -> List[int]:
        """
        Выполнить encode без использования кэша.
        
        Args:
            tokenizer: Токенизатор
            text:      Текст для кодирования
            
        Returns:
            List[int]: Список токенов
        """
        # Принудительно вызываем метод encode без кэширования
        return tokenizer.encode(text)
    
    def decode_without_cache(self, tokenizer: BPETokenizer, tokens: List[int]) -> str:
        """
        Выполнить decode без использования кэша.
        
        Args:
            tokenizer: Токенизатор
            tokens:    Токены для декодирования
            
        Returns:
            str: Декодированный текст
        """
        # Принудительно вызываем метод decode без кэширования
        return tokenizer.decode(tokens)
    
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
                print(f"Модель {size} не найдена в {model_path}!")
                success = False
                continue
                
            print(f"Загрузка модели bpe_{size}...")
            try:
                tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
                self.models[size] = tokenizer
                print(f"Загружено: {len(tokenizer.vocab)} токенов")
            except Exception as e:
                print(f"Ошибка: {e}!")
                success = False
        
        return success
    
    # ======================================================================
    # ЗАГРУЗКА РЕАЛЬНЫХ ФАЙЛОВ
    # ======================================================================
    
    def _load_real_code_files(self, max_files: int = 5, max_lines_per_file: int = 10) -> List[str]:
        """
        Загрузить реальные фрагменты кода из файлов проекта.
        
        Args:
            max_files:          Максимальное количество файлов для загрузки
            max_lines_per_file: Максимальное количество строк из файла
            
        Returns:
            List[str]: Список строк кода
        """
        code_fragments = []
        
        # Поиск .cpp и .h файлов в проекте
        project_root = get_project_paths()['project_root']
        source_files = list(project_root.rglob('*.cpp')) + list(project_root.rglob('*.h'))
        
        # Перемешиваем для разнообразия
        random.shuffle(source_files)
        
        # Берем первые несколько файлов
        for file_path in source_files[:max_files]:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Разбиваем на строки и берем непустые строки кода
                    lines = [line.strip() for line in content.split('\n') 
                            if line.strip() and not line.strip().startswith('//')]
                    # Добавляем первые несколько строк из файла
                    code_fragments.extend(lines[:max_lines_per_file])
                    print(f"Загружено {min(max_lines_per_file, len(lines))} строк из {file_path.name}")
            except Exception as e:
                print(f"Не удалось загрузить {file_path}: {e}!")
        
        return code_fragments[:50]    # Ограничиваем количество
    
    # ======================================================================
    # РАСШИРЕННЫЙ ТЕСТОВЫЙ НАБОР
    # ======================================================================
    
    def get_extended_test_set(self, include_real_files: bool = True) -> Dict[str, List[str]]:
        """
        Получить расширенный набор тестов по категориям.
        
        Args:
            include_real_files: Включить реальные файлы из проекта
            
        Returns:
            Dict[str, List[str]]: Категории с примерами кода
        
        **Категории:**
        - Препроцессор                   - Директивы #include, #define, #pragma
        - Функции и main                 - Объявления функций, main, лямбды
        - Классы и структуры             - class, struct, enum, interface
        - Стандартная библиотека         - Контейнеры, указатели, optional
        - Алгоритмы и итераторы          - sort, find, transform
        - Потоки и файлы                 - ifstream, ofstream, stringstream
        - Шаблоны и метапрограммирование - template, concept, static_assert
        - Умные указатели и память       - unique_ptr, shared_ptr, new/delete
        - Многопоточность                - thread, mutex, future, async
        - Регулярные выражения           - regex, smatch, regex_replace
        - Исключения и ошибки            - try/catch, throw, noexcept
        - Комментарии                    - Однострочные, многострочные, Doxygen
        - Сложные конструкции C++17/20   - fold expressions, coroutines, modules
        - Строковые литералы             - Обычные, raw, wide, UTF-8
        - Числовые литералы              - Десятичные, шестнадцатеричные, двоичные
        - Операторы и выражения          - Арифметика, логика, битовые операции
        - Атрибуты C++11/14/17/20        - nodiscard, maybe_unused, noreturn
        - Concepts C++20                 - integral, same_as, requires
        - Coroutines C++20               - co_await, co_yield, co_return
        - Модули C++20                   - export, import, module
        - Сложные объявления             - Указатели на функции, using, typedef
        - Unicode и эмодзи               - Кириллица, китайский, японский, эмодзи
        - Смешанный код                  - Русские комментарии в коде
        - Реальные фрагменты             - Паттерны Singleton, ThreadPool и др.
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
            
            "Строковые литералы": [
                "std::string s = \"Hello, world!\";",
                "std::string multi = \"Это многострочная\\nстрока с табуляцией\\t\";",
                "R\"raw(Сырая строка с \\\\ спецсимволами)raw\"",
                "u8\"UTF-8 строка с русским текстом\"",
                "L\"Широкая строка\"",
                "\"Смесь English и русского текста в одной строке\"",
                "\"Спецсимволы: \\\" \\n \\t \\\\ \\x41 \\u0410\"",
            ],
            
            "Числовые литералы": [
                "42",
                "3.14159",
                "0x7FFF",
                "0b101010",
                "0777",
                "1.2e-10",
                "1'000'000",     # C++14 digit separators
                "auto val = 42ull;",
                "float f = 3.14f;",
                "long double ld = 3.14L;",
            ],
            
            "Операторы и выражения": [
                "a + b * c",
                "x = (y > z) ? y : z",
                "i += 1",
                "ptr->value = 42",
                "obj.method().field",
                "a & b | c ^ d",
                "~x << 2 >> 1",
                "sp->get()->value",
                "std::move(obj)",
                "std::forward<T>(arg)",
            ],
            
            "Атрибуты C++11/14/17/20": [
                "[[nodiscard]] int func();",
                "[[maybe_unused]] int x;",
                "[[deprecated(\"use new_func\")]]",
                "[[noreturn]] void abort();",
                "[[fallthrough]];",
                "[[likely]] if (x > 0)",
                "[[unlikely]] return;",
                "[[no_unique_address]] T obj;",
            ],
            
            "Concepts C++20": [
                "template<std::integral T>",
                "template<typename T> requires std::is_integral_v<T>",
                "std::same_as<auto, int> auto x = 42;",
                "template<typename T> concept Hashable = requires(T t) { { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>; };",
            ],
            
            "Coroutines C++20": [
                "co_await task;",
                "co_yield value;",
                "co_return result;",
                "task<void> my_coroutine() { co_await something(); }",
                "generator<int> counter() { for (int i = 0; ; ++i) co_yield i; }",
            ],
            
            "Модули C++20": [
                "export module mymodule;",
                "import <iostream>;",
                "export import :partition;",
                "module :private;",
                "export template<typename T> T identity(T t) { return t; }",
            ],
            
            "Сложные объявления": [
                "int (*func_ptr)(int, double);",
                "int (*(*func_ptr_ptr)(int))(double);",
                "int (&arr_ref)[10];",
                "auto (*fp)(int) -> int (*)(double);",
                "template<typename T> using Ptr = T*;",
                "template<typename... Args> using FuncPtr = void(*)(Args...);",
            ],
            
            "Unicode и эмодзи": [
                "// 你好，世界！",
                "// Привет, мир!",
                "// שלום עולם",
                "// 🌍 🌎 🌏",
                "// ⚡ 🚀 💻",
                "// Café München",
                "// 東京特許許可局",
                "std::wstring ws = L\"広域変数\";",
                "u8\"Кириллица в UTF-8\"",
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
        
        # Добавляем реальные файлы, если нужно
        if include_real_files:
            print("\nЗагрузка реальных файлов проекта...")
            real_files = self._load_real_code_files()
            if real_files:
                test_set["Реальные файлы"] = real_files
        
        # Сохраняем информацию о категориях в метаданные
        total_tests = 0
        for name, tests in test_set.items():
            count = len(tests)
            self.results['metadata']['test_categories'][name] = count
            total_tests += count
        
        self.results['metadata']['total_tests'] = total_tests
        
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
                failed_texts = []
                
                for text in texts:
                    try:
                        encoded = self.encode_without_cache(tokenizer, text)
                        decoded = self.decode_without_cache(tokenizer, encoded)
                        if text == decoded:
                            perfect += 1
                        else:
                            failed_texts.append(text[:50] + "..." if len(text) > 50 else text)
                    except Exception as e:
                        print(f"Ошибка при обработке '{text[:30]}...': {e}")
                        failed_texts.append(f"[ERROR] {text[:30]}...")
                
                accuracy = (perfect / total) * 100 if total > 0 else 0
                category_results[category] = {
                    'perfect': perfect,
                    'total': total,
                    'accuracy': accuracy,
                    'failed': failed_texts[:5]    # Сохраняем первые 5 ошибок
                }
                
                total_perfect += perfect
                total_tests += total
                
                # Выводим результат с цветом
                if accuracy > 90:
                    color = 'V'
                elif accuracy > 70:
                    color = '!'
                else:
                    color = 'X'
                
                print(f"{color} {category[:30]:<30} {accuracy:>5.1f}% ({perfect}/{total})")
                
                # Если есть ошибки, показываем примеры
                if failed_texts and accuracy < 100:
                    print(f"Примеры ошибок: {failed_texts[:2]}")
            
            overall_accuracy = (total_perfect / total_tests) * 100 if total_tests > 0 else 0
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'perfect': total_perfect,
                    'total': total_tests,
                    'accuracy': overall_accuracy
                }
            }
            
            print(f"\nОбщая точность: {overall_accuracy:.2f}% ({total_perfect}/{total_tests})")
        
        self.results['accuracy'] = results
        return results
    
    # ======================================================================
    # ТЕСТ СКОРОСТИ
    # ======================================================================
    
    def test_speed(self, test_categories: Dict[str, List[str]], iterations: int = 50) -> Dict[int, Dict]:
        """
        Тестирование скорости encode/decode без кэширования.
        
        Args:
            test_categories: Категории с тестовыми текстами
            iterations:      Количество итераций для усреднения
            
        Returns:
            Dict[int, Dict]: Результаты тестов скорости
        """
        print_header("ТЕСТ СКОРОСТИ (БЕЗ КЭШИРОВАНИЯ)")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\nМодель bpe_{size}:")
            category_results = {}
            
            total_encode_time = 0
            total_decode_time = 0
            total_ops = 0
            
            # Прогрев для JIT и кэшей процессора
            print("Прогрев...")
            warmup_texts = ["int main() { return 0; }", "std::vector<int> v;", "template<typename T>"]
            for _ in range(10):
                for text in warmup_texts:
                    self.encode_without_cache(tokenizer, text)
            
            for category, texts in test_categories.items():
                # Создаем новые списки для каждого замера
                all_encoded = []
                
                # Encode тест
                start = time.perf_counter()
                for _ in range(iterations):
                    for text in texts:
                        encoded = self.encode_without_cache(tokenizer, text)
                        all_encoded.append(encoded)
                encode_time = time.perf_counter() - start
                
                # Decode тест
                start = time.perf_counter()
                for encoded in all_encoded:
                    self.decode_without_cache(tokenizer, encoded)
                decode_time = time.perf_counter() - start
                
                ops = len(texts) * iterations
                total_ops += ops
                total_encode_time += encode_time
                total_decode_time += decode_time
                
                encode_per_text = (encode_time / ops) * 1000    # мс
                decode_per_text = (decode_time / ops) * 1000    # мс
                
                # Подсчет символов для скорости
                total_chars_in_category = sum(len(t) for t in texts)
                encode_speed_chars = (total_chars_in_category * iterations) / encode_time    # Символов/с
                decode_speed_chars = (total_chars_in_category * iterations) / decode_time    # Символов/с
                
                category_results[category] = {
                    'encode_time': encode_time,
                    'decode_time': decode_time,
                    'encode_per_text': encode_per_text,
                    'decode_per_text': decode_per_text,
                    'encode_speed_chars': encode_speed_chars,
                    'decode_speed_chars': decode_speed_chars,
                    'ops': ops
                }
                
                print(f"  {category[:30]:<30} "
                      f"encode: {encode_per_text:>8.5f} мс, "
                      f"decode: {decode_per_text:>8.5f} мс")
            
            # Общие результаты
            avg_encode = (total_encode_time / total_ops) * 1000
            avg_decode = (total_decode_time / total_ops) * 1000
            total_chars = sum(len(t) for cat in test_categories.values() for t in cat)
            
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'encode_time': total_encode_time,
                    'decode_time': total_decode_time,
                    'encode_per_text': avg_encode,
                    'decode_per_text': avg_decode,
                    'encode_speed_chars': (total_chars * iterations) / total_encode_time,
                    'decode_speed_chars': (total_chars * iterations) / total_decode_time,
                    'total_ops': total_ops
                }
            }
            
            print(f"\nСреднее: encode {avg_encode:.5f} мс ({results[size]['overall']['encode_speed_chars']:.0f} симв/с), "
                  f"decode {avg_decode:.5f} мс ({results[size]['overall']['decode_speed_chars']:.0f} симв/с)")
        
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
            total_bytes_utf8 = 0
            total_bytes_tokens = 0
            
            for category, texts in test_categories.items():
                chars = 0
                tokens = 0
                bytes_utf8 = 0
                bytes_tokens = 0
                
                for text in texts:
                    encoded = self.encode_without_cache(tokenizer, text)
                    chars += len(text)
                    tokens += len(encoded)
                    bytes_utf8 += len(text.encode('utf-8'))
                    # Оценка размера в байтах для токенов (int -> 4 байта)
                    bytes_tokens += len(encoded) * 4
                
                total_chars += chars
                total_tokens += tokens
                total_bytes_utf8 += bytes_utf8
                total_bytes_tokens += bytes_tokens
                
                ratio = chars / tokens if tokens > 0 else 0
                compression_percent = (1 - tokens / chars) * 100 if chars > 0 else 0
                bytes_ratio = bytes_utf8 / bytes_tokens if bytes_tokens > 0 else 0
                
                category_results[category] = {
                    'chars': chars,
                    'tokens': tokens,
                    'ratio': ratio,
                    'compression_percent': compression_percent,
                    'bytes_utf8': bytes_utf8,
                    'bytes_tokens': bytes_tokens,
                    'bytes_ratio': bytes_ratio
                }
                
                print(f"- {category[:30]:<30} {ratio:>5.2f} симв/токен ({compression_percent:>5.1f}% сжатие)")
            
            overall_ratio = total_chars / total_tokens if total_tokens > 0 else 0
            overall_compression = (1 - total_tokens / total_chars) * 100 if total_chars > 0 else 0
            overall_bytes_ratio = total_bytes_utf8 / total_bytes_tokens if total_bytes_tokens > 0 else 0
            
            results[size] = {
                'by_category': category_results,
                'overall': {
                    'chars': total_chars,
                    'tokens': total_tokens,
                    'ratio': overall_ratio,
                    'compression_percent': overall_compression,
                    'bytes_utf8': total_bytes_utf8,
                    'bytes_tokens': total_bytes_tokens,
                    'bytes_ratio': overall_bytes_ratio
                }
            }
            print(f"\nСреднее сжатие:  {overall_ratio:.2f} символов/токен ({overall_compression:.1f}%)")
            print(f"Сжатие в байтах: UTF-8: {total_bytes_utf8} -> токены: {total_bytes_tokens} байт (соотношение {overall_bytes_ratio:.2f})")
        
        self.results['compression'] = results
        return results
    
    # ======================================================================
    # АНАЛИЗ СЛОВАРЕЙ
    # ======================================================================

    def analyze_vocabulary(self) -> Dict[int, Dict]:
        """
        Анализ состава словарей с корректным подсчетом специальных токенов.
        
        Returns:
            Dict[int, Dict]: Статистика по словарям
        """
        print_header("АНАЛИЗ СЛОВАРЕЙ")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            # Статистика по длинам токенов
            token_lengths = [len(token) for token in tokenizer.vocab.values()]
            
            # Типы токенов
            special_tokens = 0
            ascii_tokens = 0
            unicode_tokens = 0
            single_char = 0
            multi_char = 0
            
            # Анализ частотности
            char_freq = Counter()
            for token in tokenizer.vocab.values():
                # Проверяем специальные токены (по шаблону <...>)
                if token.startswith('<') and token.endswith('>'):
                    special_tokens += 1
                elif all(ord(c) < 128 for c in token):
                    ascii_tokens += 1
                else:
                    unicode_tokens += 1
                
                if len(token) == 1:
                    single_char += 1
                else:
                    multi_char += 1
                
                # Собираем статистику по символам
                for char in token:
                    char_freq[char] += 1
            
            # Статистика распределения длин
            length_dist = {}
            for length in set(token_lengths):
                count = token_lengths.count(length)
                length_dist[length] = count
            
            # Топ самых длинных токенов
            token_items = list(tokenizer.vocab.items())
            token_items.sort(key=lambda x: len(x[1]), reverse=True)
            longest_tokens = [token for _, token in token_items[:20]]
            
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
                    'median': sorted(token_lengths)[len(token_lengths) // 2],
                    'distribution': length_dist
                },
                'char_frequency': dict(char_freq.most_common(20)),
                'longest_tokens': longest_tokens[:10]
            }
            
            print(f"\nМодель bpe_{size}:")
            print(f"- Размер:          {results[size]['vocab_size']} токенов")
            print(f"- ASCII:           {results[size]['ascii_tokens']} ({results[size]['ascii_tokens']*100/len(tokenizer.vocab):.1f}%)")
            print(f"- Unicode:         {results[size]['unicode_tokens']} ({results[size]['unicode_tokens']*100/len(tokenizer.vocab):.1f}%)")
            print(f"- Спец. токены:    {results[size]['special_tokens']} ({results[size]['special_tokens']*100/len(tokenizer.vocab):.1f}%)")
            print(f"- Односимвольных:  {results[size]['single_char']}")
            print(f"- Многосимвольных: {results[size]['multi_char']}")
            print(f"- Средняя длина:   {results[size]['token_lengths']['avg']:.2f}")
            print(f"- Макс. длина:     {results[size]['token_lengths']['max']}")
            print(f"- Топ-5 символов:  {dict(list(results[size]['char_frequency'].items())[:5])}")
        
        self.results['vocabulary'] = results
        return results

    # ======================================================================
    # АНАЛИЗ OOV (OUT OF VOCABULARY)
    # ======================================================================
    
    def analyze_oov(self, test_categories: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Анализ токенов вне словаря (OOV) и неизвестных символов.
        
        Args:
            test_categories: Категории с тестовыми текстами
            
        Returns:
            Dict[int, Dict]: Результаты анализа OOV
        """
        print_header("АНАЛИЗ OOV (OUT OF VOCABULARY)")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            # Собираем все символы из тестов
            all_chars = set()
            unknown_chars = set()
            unknown_sequences = []
            
            for category, texts in test_categories.items():
                for text in texts:
                    all_chars.update(text)
                    
                    # Пытаемся закодировать и анализируем
                    try:
                        encoded = self.encode_without_cache(tokenizer, text)
                        # Проверяем, все ли символы покрыты
                        for char in text:
                            if char not in str(encoded) and ord(char) > 127:
                                # Символ может быть неизвестным
                                if char not in tokenizer.vocab:
                                    unknown_chars.add(char)
                    except Exception as e:
                        # При ошибке пытаемся найти проблемные символы
                        for char in text:
                            if char not in tokenizer.vocab:
                                unknown_chars.add(char)
                        
                        # Сохраняем проблемную последовательность
                        unknown_sequences.append(text[:50])
            
            # Оценка покрытия
            coverage = (len(all_chars) - len(unknown_chars)) / len(all_chars) * 100 if all_chars else 100
            
            # Группировка неизвестных символов по категориям Unicode
            unknown_by_category = {}
            for char in unknown_chars:
                cat = self._get_unicode_category(char)
                if cat not in unknown_by_category:
                    unknown_by_category[cat] = []
                unknown_by_category[cat].append(char)
            
            results[size] = {
                'total_unique_chars': len(all_chars),
                'unknown_chars': len(unknown_chars),
                'coverage': coverage,
                'unknown_char_list': list(unknown_chars)[:30],
                'unknown_by_category': {k: len(v) for k, v in unknown_by_category.items()},
                'problematic_sequences': unknown_sequences[:10]
            }
            
            print(f"\nМодель bpe_{size}:")
            print(f"Покрытие символов:   {coverage:.2f}% ({len(all_chars) - len(unknown_chars)}/{len(all_chars)})")
            print(f"Неизвестные символы: {len(unknown_chars)}")
            
            if len(unknown_chars) > 0:
                print(f"По категориям Unicode:")
                for cat, count in results[size]['unknown_by_category'].items():
                    print(f"{cat}: {count}")
                
                # Показываем примеры
                example_chars = list(unknown_chars)[:10]
                example_repr = ' '.join([repr(c) for c in example_chars])
                print(f"Примеры: {example_repr}")
            
            if unknown_sequences:
                print(f"Проблемные последовательности: {unknown_sequences[:3]}")
        
        self.results['oov_analysis'] = results
        return results
    
    def _get_unicode_category(self, char: str) -> str:
        """
        Определить категорию Unicode символа.
        
        Args:
            char: Символ для анализа
            
        Returns:
            str: Категория символа
        """
        code = ord(char)
        if code < 128:
            return "ASCII"
        elif 0x0400 <= code <= 0x04FF:
            return "Кириллица"
        elif 0x4E00 <= code <= 0x9FFF:
            return "Китайский"
        elif 0x3040 <= code <= 0x309F:
            return "Хирагана"
        elif 0x30A0 <= code <= 0x30FF:
            return "Катакана"
        elif 0xAC00 <= code <= 0xD7AF:
            return "Корейский"
        elif 0x1F300 <= code <= 0x1F9FF:
            return "Эмодзи"
        else:
            return "Другое Unicode"
    
    # ======================================================================
    # АНАЛИЗ ГЛУБИНЫ СЖАТИЯ
    # ======================================================================
    
    def analyze_compression_depth(self, test_categories: Dict[str, List[str]]) -> Dict[int, Dict]:
        """
        Анализ глубины сжатия - распределение длин токенов в закодированном тексте.
        
        Args:
            test_categories: Категории с тестовыми текстами
            
        Returns:
            Dict[int, Dict]: Результаты анализа глубины сжатия
        """
        print_header("АНАЛИЗ ГЛУБИНЫ СЖАТИЯ")
        
        results = {}
        
        for size, tokenizer in self.models.items():
            print(f"\nМодель bpe_{size}:")
            
            all_token_lengths = []
            token_length_dist = Counter()
            token_frequency = Counter()
            
            for category, texts in test_categories.items():
                for text in texts:
                    try:
                        encoded = self.encode_without_cache(tokenizer, text)
                        for token_id in encoded:
                            token = tokenizer.vocab.get(token_id, f"<UNK:{token_id}>")
                            token_length = len(token)
                            all_token_lengths.append(token_length)
                            token_length_dist[token_length] += 1
                            token_frequency[token] += 1
                    except Exception:
                        pass
            
            if not all_token_lengths:
                print(f"Нет данных для анализа!")
                continue
            
            # Статистика
            avg_token_length = sum(all_token_lengths) / len(all_token_lengths)
            median_token_length = sorted(all_token_lengths)[len(all_token_lengths) // 2]
            
            # Распределение по диапазонам
            length_ranges = {
                '1-2': sum(1 for l in all_token_lengths if l <= 2),
                '3-5': sum(1 for l in all_token_lengths if 3 <= l <= 5),
                '6-10': sum(1 for l in all_token_lengths if 6 <= l <= 10),
                '11+': sum(1 for l in all_token_lengths if l >= 11)
            }
            
            # Топ самых частых токенов
            most_common_tokens = token_frequency.most_common(20)
            
            results[size] = {
                'total_tokens_analyzed': len(all_token_lengths),
                'avg_token_length': avg_token_length,
                'median_token_length': median_token_length,
                'max_token_length': max(all_token_lengths),
                'min_token_length': min(all_token_lengths),
                'length_distribution': dict(token_length_dist.most_common(20)),
                'length_ranges': length_ranges,
                'most_common_tokens': [(token, count) for token, count in most_common_tokens[:10]]
            }
            
            print(f"Всего токенов проанализировано: {len(all_token_lengths)}")
            print(f"Средняя длина токена:           {avg_token_length:.2f}")
            print(f"Медианная длина:                {median_token_length}")
            print(f"Распределение по длинам:")
            for range_name, count in length_ranges.items():
                percent = count / len(all_token_lengths) * 100
                print(f"    {range_name}: {count} ({percent:.1f}%)")
            
            print(f"Топ-5 самых частых токенов:")
            for token, count in most_common_tokens[:5]:
                print(f"'{token}': {count}")
        
        self.results['compression_depth'] = results
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
        lines.append("=" * 60)
        lines.append("СРАВНЕНИЕ ТРЕХ BPE МОДЕЛЕЙ (8000, 10000, 12000)".center(60))
        lines.append("=" * 60)
        lines.append("")
        
        # Информация о тестах
        lines.append("ИНФОРМАЦИЯ О ТЕСТАХ")
        lines.append("-" * 40)
        lines.append(f"Всего тестов:      {self.results['metadata']['total_tests']}")
        lines.append(f"Категорий:         {len(self.results['metadata']['test_categories'])}")
        lines.append(f"Дата тестирования: {self.results['metadata']['timestamp']}")
        lines.append("")
        
        # Общая точность
        lines.append("ОБЩАЯ ТОЧНОСТЬ")
        lines.append("-" * 60)
        for size in self.model_sizes:
            acc = self.results['accuracy'][size]['overall']['accuracy']
            perfect = self.results['accuracy'][size]['overall']['perfect']
            total = self.results['accuracy'][size]['overall']['total']
            lines.append(f"bpe_{size:5}: {acc:6.2f}% ({perfect:4d}/{total:4d})")
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
        lines.append("СТЕПЕНЬ СЖАТИЯ")
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
            improvement = ((ratios[-1] - base_ratios[cat]) / base_ratios[cat]) * 100 if base_ratios[cat] > 0 else 0
            line += f" {improvement:>+7.1f}%"
            lines.append(line)
        lines.append("")
        
        # Общее сжатие
        lines.append("ОБЩЕЕ СЖАТИЕ")
        lines.append("-" * 60)
        for size in self.model_sizes:
            comp = self.results['compression'][size]['overall']
            lines.append(f"bpe_{size:5}: {comp['ratio']:5.2f} симв/токен ({comp['compression_percent']:5.1f}%)")
            lines.append(f"UTF-8: {comp['bytes_utf8']:8d} байт -> токены: {comp['bytes_tokens']:8d} байт (x{comp['bytes_ratio']:.2f})")
        lines.append("")
        
        # Скорость
        lines.append("СКОРОСТЬ РАБОТЫ (БЕЗ КЭШИРОВАНИЯ)")
        lines.append("-" * 80)
        header = f"{'Модель':<10} {'Encode (мс)':>15} {'Decode (мс)':>15} {'Enc (сим/с)':>15} {'Dec (сим/с)':>15}"
        lines.append(header)
        lines.append("-" * 80)
        
        for size in self.model_sizes:
            enc = self.results['speed'][size]['overall']['encode_per_text']
            dec = self.results['speed'][size]['overall']['decode_per_text']
            enc_speed = self.results['speed'][size]['overall']['encode_speed_chars']
            dec_speed = self.results['speed'][size]['overall']['decode_speed_chars']
            lines.append(f"bpe_{size:<6} {enc:>14.6f} {dec:>14.6f} {enc_speed:>14.0f} {dec_speed:>14.0f}")
        lines.append("")
        
        # Характеристики словарей
        lines.append("ХАРАКТЕРИСТИКИ СЛОВАРЕЙ")
        lines.append("-" * 70)
        header = f"{'Модель':<10} {'Размер':>8} {'ASCII':>8} {'Unicode':>8} {'Спец':>6} {'Ср.длина':>9} {'Макс':>6}"
        lines.append(header)
        lines.append("-" * 70)
        
        for size in self.model_sizes:
            vocab = self.results['vocabulary'][size]
            lines.append(
                f"bpe_{size:<6} {vocab['vocab_size']:>8} "
                f"{vocab['ascii_tokens']:>8} {vocab['unicode_tokens']:>8} "
                f"{vocab['special_tokens']:>6} {vocab['token_lengths']['avg']:>8.2f} "
                f"{vocab['token_lengths']['max']:>6}"
            )
        
        # OOV анализ
        if 'oov_analysis' in self.results and self.results['oov_analysis']:
            lines.append("")
            lines.append("АНАЛИЗ OOV (OUT OF VOCABULARY)")
            lines.append("-" * 70)
            
            for size in self.model_sizes:
                oov = self.results['oov_analysis'][size]
                lines.append(f"bpe_{size:5}: Покрытие символов: {oov['coverage']:6.2f}% "
                           f"(неизвестно: {oov['unknown_chars']:3d} из {oov['total_unique_chars']:3d})")
        
        # Глубина сжатия
        if 'compression_depth' in self.results and self.results['compression_depth']:
            lines.append("")
            lines.append("ГЛУБИНА СЖАТИЯ")
            lines.append("-" * 70)
            
            for size in self.model_sizes:
                depth = self.results['compression_depth'][size]
                lines.append(f"bpe_{size:5}: Средняя длина токена: {depth['avg_token_length']:.2f}")
                ranges = depth['length_ranges']
                lines.append(f"Распределение: 1-2: {ranges['1-2']:4d} | 3-5: {ranges['3-5']:4d} | "
                           f"6-10: {ranges['6-10']:4d} | 11+: {ranges['11+']:4d}")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("РЕКОМЕНДАЦИИ".center(100))
        lines.append("=" * 100)
        lines.append("")
        
        # Анализ и рекомендации
        acc_8000 = self.results['accuracy'][8000]['overall']['accuracy']
        acc_10000 = self.results['accuracy'][10000]['overall']['accuracy']
        acc_12000 = self.results['accuracy'][12000]['overall']['accuracy']
        
        comp_8000 = self.results['compression'][8000]['overall']['ratio']
        comp_10000 = self.results['compression'][10000]['overall']['ratio']
        comp_12000 = self.results['compression'][12000]['overall']['ratio']
        
        speed_8000 = self.results['speed'][8000]['overall']['encode_per_text']
        speed_10000 = self.results['speed'][10000]['overall']['encode_per_text']
        speed_12000 = self.results['speed'][12000]['overall']['encode_per_text']
        
        # Находим лучшую модель по разным критериям
        best_accuracy = max([(8000, acc_8000), (10000, acc_10000), (12000, acc_12000)], key=lambda x: x[1])
        best_compression = max([(8000, comp_8000), (10000, comp_10000), (12000, comp_12000)], key=lambda x: x[1])
        best_speed = min([(8000, speed_8000), (10000, speed_10000), (12000, speed_12000)], key=lambda x: x[1])
        
        lines.append(f"По точности: bpe_{best_accuracy[0]} ({best_accuracy[1]:.1f}%)")
        lines.append(f"По сжатию:   bpe_{best_compression[0]} ({best_compression[1]:.2f} симв/токен)")
        lines.append(f"По скорости: bpe_{best_speed[0]} ({best_speed[1]:.6f} мс)")
        lines.append("")
        
        # Компромиссный анализ
        lines.append("Компромиссный анализ:")
        if acc_12000 - acc_8000 > 1:
            lines.append("- Модель 12000 дает заметно лучшую точность")
        elif acc_8000 - acc_12000 > 1:
            lines.append("- Модель 8000 неожиданно лучше по точности (проверьте данные)")
        
        if comp_12000 / comp_8000 > 1.1:
            lines.append("- Модель 12000 сжимает лучше")
        
        if speed_8000 / speed_12000 < 0.9:
            lines.append("- Модель 8000 работает быстрее")
        
        lines.append("")
        lines.append("Итоговая рекомендация:")
        
        # Взвешенная оценка
        score_8000 = acc_8000 * 0.4 + comp_8000 * 10 * 0.3 + (1/speed_8000) * 100 * 0.3
        score_10000 = acc_10000 * 0.4 + comp_10000 * 10 * 0.3 + (1/speed_10000) * 100 * 0.3
        score_12000 = acc_12000 * 0.4 + comp_12000 * 10 * 0.3 + (1/speed_12000) * 100 * 0.3
        
        best_overall = max([(8000, score_8000), (10000, score_10000), (12000, score_12000)], key=lambda x: x[1])
        
        if best_overall[0] == 8000:
            lines.append(f"bpe_8000 - Лучший баланс для production (оценка: {best_overall[1]:.1f})")
        elif best_overall[0] == 10000:
            lines.append(f"bpe_10000 - Оптимальный выбор (оценка: {best_overall[1]:.1f})")
        else:
            lines.append(f"bpe_12000 - Максимальное качество (оценка: {best_overall[1]:.1f})")
        
        lines.append("")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    # ======================================================================
    # ВИЗУАЛИЗАЦИЯ
    # ======================================================================

    def plot_comparison(self, reports_dir: Path):
        """
        Создать графики сравнения с цифрами внутри границ.
        
        Args:
            reports_dir: Директория для сохранения графиков
        """
        print_header("СОЗДАНИЕ ГРАФИКОВ")
        
        try:
            # Проверяем наличие данных для графиков
            required_keys = ['accuracy', 'compression', 'speed', 'vocabulary']
            for key in required_keys:
                if key not in self.results or not self.results[key]:
                    print(f"Нет данных для {key}, графики не будут созданы!")
                    return
            
            # Проверяем установлен ли matplotlib
            try:
                import matplotlib
                print(f"Matplotlib версия:  {matplotlib.__version__}")
                print(f"Matplotlib backend: {matplotlib.get_backend()}")
            except ImportError:
                print("Matplotlib не установлен. Установите: pip install matplotlib")
                return
            
            # Создаем фигуру
            try:
                fig, axes = plt.subplots(3, 3, figsize=(24, 18))
                fig.suptitle('Сравнение трех BPE моделей (8000, 10000, 12000)', 
                            fontsize=16, fontweight='bold')
            except Exception as e:
                print(f"Ошибка создания фигуры: {e}!")
                return
            
            categories = list(self.results['accuracy'][self.model_sizes[0]]['by_category'].keys())
            # Берем первые 8 категорий для читаемости
            display_cats = categories[:8]
            x = np.arange(len(display_cats))
            width = 0.25
            
            try:
                # 1. Точность по категориям
                ax = axes[0, 0]
                for i, size in enumerate(self.model_sizes):
                    accuracies = [self.results['accuracy'][size]['by_category'][cat]['accuracy'] 
                                for cat in display_cats]
                    bars = ax.bar(x + (i - 1) * width, accuracies, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    # Вертикальные цифры внутри границ
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height - 3,
                                f'{height:.0f}%', ha='center', va='top', 
                                fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Точность по категориям', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                                rotation=45, ha='right')
                ax.set_ylabel('Точность (%)')
                ax.legend()
                ax.set_ylim([0, 105])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 1/9: Точность")
                
                # 2. Степень сжатия
                ax = axes[0, 1]
                max_ratio = 0
                for i, size in enumerate(self.model_sizes):
                    ratios = [self.results['compression'][size]['by_category'][cat]['ratio'] 
                            for cat in display_cats]
                    max_ratio = max(max_ratio, max(ratios))
                    bars = ax.bar(x + (i - 1) * width, ratios, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    # Вертикальные цифры внутри границ
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height - 0.1,
                            f'{height:.2f}', ha='center', va='top', 
                            fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Степень сжатия (символов/токен)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                                rotation=45, ha='right')
                ax.set_ylabel('Символов на токен')
                ax.legend()
                ax.set_ylim([0, max_ratio * 1.3])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 2/9: Сжатие")
                
                # 3. Скорость encode
                ax = axes[0, 2]
                max_speed = 0
                for i, size in enumerate(self.model_sizes):
                    speeds = [self.results['speed'][size]['by_category'][cat]['encode_per_text'] 
                            for cat in display_cats]
                    max_speed = max(max_speed, max(speeds))
                    bars = ax.bar(x + (i - 1) * width, speeds, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    # Вертикальные цифры внутри границ
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                                f'{height:.3f}', ha='center', va='center', 
                                fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Скорость encode (мс/текст)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                                rotation=45, ha='right')
                ax.set_ylabel('мс')
                ax.legend()
                ax.set_ylim([0, max_speed * 1.4])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 3/9: Скорость encode")
                
                # 4. Скорость decode
                ax = axes[1, 0]
                max_speed = 0
                for i, size in enumerate(self.model_sizes):
                    speeds = [self.results['speed'][size]['by_category'][cat]['decode_per_text'] 
                            for cat in display_cats]
                    max_speed = max(max_speed, max(speeds))
                    bars = ax.bar(x + (i - 1) * width, speeds, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    # Вертикальные цифры внутри границ
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                                f'{height:.3f}', ha='center', va='center', 
                                fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Скорость decode (мс/текст)', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([c[:12] + '...' if len(c) > 12 else c for c in display_cats], 
                                rotation=45, ha='right')
                ax.set_ylabel('мс')
                ax.legend()
                ax.set_ylim([0, max_speed * 1.4])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 4/9: Скорость decode")
                
                # 5. Распределение токенов по типам
                ax = axes[1, 1]
                x_types = np.arange(3)    # ASCII, Unicode, Специальные
                max_value = 0
                for i, size in enumerate(self.model_sizes):
                    vocab = self.results['vocabulary'][size]
                    values = [vocab['ascii_tokens'], vocab['unicode_tokens'], vocab['special_tokens']]
                    max_value = max(max_value, max(values))
                    bars = ax.bar(x_types + (i - 1) * width, values, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    # Вертикальные цифры внутри границ
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height * 0.8,
                            f'{height}', ha='center', va='center', 
                            fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Состав словарей', fontsize=12, fontweight='bold')
                ax.set_xticks(x_types)
                ax.set_xticklabels(['ASCII', 'Unicode', 'Специальные'])
                ax.set_ylabel('Количество токенов')
                ax.legend()
                ax.set_ylim([0, max_value * 1.2])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 5/9: Состав словарей")
                
                # 6. Распределение длин токенов
                ax = axes[1, 2]
                if 'compression_depth' in self.results and self.results['compression_depth']:
                    for i, size in enumerate(self.model_sizes):
                        if size in self.results['compression_depth']:
                            depth = self.results['compression_depth'][size]
                            ranges = depth['length_ranges']
                            values = [ranges['1-2'], ranges['3-5'], ranges['6-10'], ranges['11+']]
                            x_ranges = np.arange(4)
                            bars = ax.bar(x_ranges + (i - 1) * width/1.5, values, width/1.5,
                                        label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                            
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height * 0.8,
                                    f'{height}', ha='center', va='center', 
                                    fontsize=7, rotation=90, color='white', fontweight='bold')
                    
                    ax.set_title('Распределение длин токенов', fontsize=12, fontweight='bold')
                    ax.set_xticks(x_ranges)
                    ax.set_xticklabels(['1-2', '3-5', '6-10', '11+'])
                    ax.set_ylabel('Количество токенов')
                    ax.legend()
                    ax.grid(True, alpha=0.3, axis='y')
                    print("- График 6/9: Распределение длин")
                else:
                    ax.text(0.5, 0.5, 'Нет данных\n(запустите с --full-analysis)', 
                        ha='center', va='center', fontsize=12)
                    ax.set_title('Распределение длин токенов', fontsize=12)
                
                # 7. OOV анализ
                ax = axes[2, 0]
                if 'oov_analysis' in self.results and self.results['oov_analysis']:
                    x_oov = np.arange(2)    # Покрытие и неизвестные
                    for i, size in enumerate(self.model_sizes):
                        if size in self.results['oov_analysis']:
                            oov = self.results['oov_analysis'][size]
                            coverage = oov['coverage']
                            unknown_pct = (oov['unknown_chars'] / oov['total_unique_chars']) * 100 if oov['total_unique_chars'] > 0 else 0
                            values = [coverage, unknown_pct]
                            bars = ax.bar(x_oov + (i - 1) * width, values, width,
                                        label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                            
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height * 0.8,
                                    f'{height:.1f}%', ha='center', va='center', 
                                    fontsize=8, rotation=90, color='white', fontweight='bold')
                    
                    ax.set_title('OOV анализ', fontsize=12, fontweight='bold')
                    ax.set_xticks(x_oov)
                    ax.set_xticklabels(['Покрытие', 'Неизвестные'])
                    ax.set_ylabel('Процент')
                    ax.legend()
                    ax.set_ylim([0, 105])
                    ax.grid(True, alpha=0.3, axis='y')
                    print("- График 7/9: OOV анализ")
                else:
                    ax.text(0.5, 0.5, 'Нет данных\n(запустите с --full-analysis)', 
                        ha='center', va='center', fontsize=12)
                    ax.set_title('OOV анализ', fontsize=12)
                
                # 8. Сравнение скорости в символ/с
                ax = axes[2, 1]
                x_speed = np.arange(2)    # Encode, Decode
                max_speed_val = 0
                for i, size in enumerate(self.model_sizes):
                    speed = self.results['speed'][size]['overall']
                    values = [speed['encode_speed_chars'] / 1000, speed['decode_speed_chars'] / 1000]    # В тысячах симв/с
                    max_speed_val = max(max_speed_val, max(values))
                    bars = ax.bar(x_speed + (i - 1) * width, values, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height * 0.8,
                            f'{height:.1f}k', ha='center', va='center', 
                            fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Скорость (тыс. символов/с)', fontsize=12, fontweight='bold')
                ax.set_xticks(x_speed)
                ax.set_xticklabels(['Encode', 'Decode'])
                ax.set_ylabel('Тысяч символов/с')
                ax.legend()
                ax.set_ylim([0, max_speed_val * 1.2])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 8/9: Скорость")
                
                # 9. Сводное сравнение
                ax = axes[2, 2]
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
                        (comp / max_comp) * 100 if max_comp > 0 else 0,
                        (speed_enc / max_speed_enc) * 100 if max_speed_enc > 0 else 0,
                        (speed_dec / max_speed_dec) * 100 if max_speed_dec > 0 else 0,
                        (size_norm / max_size_norm) * 100 if max_size_norm > 0 else 0
                    ]
                    
                    bars = ax.bar(x_sum + (i - 1) * width, values, width,
                                label=f'bpe_{size}', color=self.colors[size], alpha=0.8)
                    
                    # Вертикальные цифры внутри границ
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width()/2., height * 0.85,
                                f'{height:.0f}%', ha='center', va='center', 
                                fontsize=8, rotation=90, color='white', fontweight='bold')
                
                ax.set_title('Сводное сравнение (нормализовано)', fontsize=12, fontweight='bold')
                ax.set_xticks(x_sum)
                ax.set_xticklabels(metrics)
                ax.set_ylabel('Относительная оценка (%)')
                ax.legend()
                ax.set_ylim([0, 110])
                ax.grid(True, alpha=0.3, axis='y')
                print("- График 9/9: Сводное сравнение")
                
                # Добавляем отступы для всех графиков
                plt.tight_layout(pad=4.0)
                
                # Сохраняем графики
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                # Пробуем разные форматы
                try:
                    plot_path = reports_dir / 'three_model_comparison.png'
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"PNG график сохранен: {plot_path}")
                    print(f"Размер файла:        {plot_path.stat().st_size / 1024:.1f} КБ")
                except Exception as e:
                    print(f"Ошибка сохранения PNG: {e}!")
                
                try:
                    pdf_path = reports_dir / 'three_model_comparison.pdf'
                    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
                    print(f"PDF график сохранен: {pdf_path}")
                except Exception as e:
                    print(f"Ошибка сохранения PDF: {e}!")
                
                plt.close()
                print(f"Все графики успешно созданы!")
                
            except Exception as e:
                print(f"Ошибка при создании графиков: {e}!")
                import traceback
                traceback.print_exc()
                plt.close()
                
        except Exception as e:
            print(f"Критическая ошибка: {e}!")
            import traceback
            traceback.print_exc()

    # ======================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    def save_results(self, reports_dir: Path):
        """
        Сохранить результаты в файлы.
        
        Args:
            reports_dir: Директория для сохранения
        """
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Функция для конвертации в JSON-совместимые типы
        def convert_for_json(obj: Any) -> Any:
            if isinstance(obj, (np.integer, np.floating, np.number)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, set):
                return list(obj)
            if isinstance(obj, (Counter, dict)):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_for_json(v) for v in obj]
            if hasattr(obj, 'item'):    # Для numpy скаляров
                return float(obj)
            return obj
        
        # JSON результаты
        json_path = reports_dir / 'three_model_comparison.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_for_json(self.results), f, ensure_ascii=False, indent=2)
        print(f"JSON результаты: {json_path}")
        
        # Текстовый отчет
        report_path = reports_dir / 'three_model_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        print(f"Текстовый отчет: {report_path}")
        
        # Сохраняем тестовые данные
        tests_path = reports_dir / 'test_categories.json'
        with open(tests_path, 'w', encoding='utf-8') as f:
            json.dump(self.results['metadata']['test_categories'], f, ensure_ascii=False, indent=2)
        print(f"Информация о тестах: {tests_path}")
        
        # Графики
        self.plot_comparison(reports_dir)


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(
        description='Сравнение трех BPE моделей (8000, 10000, 12000)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Примеры использования:
    python test_compare_models.py                    # Полный анализ
    python test_compare_models.py --quick            # Быстрый режим (меньше итераций)
    python test_compare_models.py --full-analysis    # Включая OOV и глубину сжатия
    python test_compare_models.py --plot-only        # Только графики из сохраненных данных
    python test_compare_models.py --no-real-files    # Без загрузки реальных файлов
    python test_compare_models.py --verify-speed     # Доп. проверка скорости
    """
    )
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Быстрый режим (меньше итераций)')
    parser.add_argument('--plot-only', '-p', action='store_true',
                       help='Только построить графики из сохраненных результатов')
    parser.add_argument('--full-analysis', '-f', action='store_true',
                       help='Полный анализ (включая OOV и глубину сжатия)')
    parser.add_argument('--no-real-files', action='store_true',
                       help='Не загружать реальные файлы')
    parser.add_argument('--no-plots', action='store_true',
                       help='Не создавать графики')
    parser.add_argument('--verify-speed', action='store_true',
                       help='Запустить дополнительную проверку скорости')
    
    args = parser.parse_args()
    
    # Проверяем наличие matplotlib если нужны графики
    if not args.no_plots:
        try:
            import matplotlib
            print(f"Matplotlib найден (версия {matplotlib.__version__})")
        except ImportError:
            print("Matplotlib не установлен. Графики не будут созданы.")
            print("Установите: pip install matplotlib")
            args.no_plots = True
    
    # Получаем пути
    paths = get_project_paths()
    
    print_header("СРАВНЕНИЕ ТРЕХ BPE МОДЕЛЕЙ (БЕЗ КЭШИРОВАНИЯ)")
    print(f"Директория моделей:     {paths['models_dir']}")
    print(f"Директория результатов: {paths['reports_dir']}")
    
    # Режим только графиков
    if args.plot_only:
        json_path = paths['reports_dir'] / 'three_model_comparison.json'
        if not json_path.exists():
            print(f"Файл с результатами не найден: {json_path}!")
            return 1
        
        print(f"\nЗагрузка результатов из {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Создаем временный компаратор для графиков
        comparator = ThreeModelComparison(paths['models_dir'])
        comparator.results = results
        comparator.plot_comparison(paths['reports_dir'])
        return 0
    
    # Полный режим
    comparator = ThreeModelComparison(
        models_dir=paths['models_dir'],
        model_sizes=[8000, 10000, 12000]
    )
    
    try:
        # Загружаем модели
        if not comparator.load_models():
            print(f"\nНе удалось загрузить все модели!")
            print(f"Проверьте наличие моделей в: {paths['models_dir']}")
            return 1
        
        print(f"\nЗагружено {len(comparator.models)} моделей")
        
        # Получаем тесты
        test_categories = comparator.get_extended_test_set(include_real_files=not args.no_real_files)
        total_tests = sum(len(v) for v in test_categories.values())
        print(f"Всего тестов: {total_tests} в {len(test_categories)} категориях")
        
        # Запускаем тесты
        iterations = 5 if args.quick else 20    # Уменьшаем итерации для точности
        print(f"Количество итераций скорости: {iterations} (без кэширования)")
        
        comparator.test_accuracy(test_categories)
        comparator.test_speed(test_categories, iterations=iterations)
        comparator.test_compression(test_categories)
        comparator.analyze_vocabulary()
        
        # Дополнительный анализ если нужно
        if args.full_analysis:
            comparator.analyze_oov(test_categories)
            comparator.analyze_compression_depth(test_categories)
        
        # Выводим отчет
        print("\n" + comparator.generate_report())
        
        # Сохраняем результаты
        print_header("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        comparator.save_results(paths['reports_dir'])
        
        print_header("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print("Все тесты успешно выполнены!")
        print(f"Результаты сохранены в: {paths['reports_dir']}")
        print("\nВНИМАНИЕ: Скорость измерена БЕЗ кэширования для точности!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nТестирование прервано пользователем!")
        return 1
    except Exception as e:
        print(f"\nОшибка: {e}!")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())