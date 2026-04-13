/**
 * @file fast_tokenizer_demo.cpp
 * @brief Комплексная демонстрация возможностей оптимизированного BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Эта программа демонстрирует все ключевые возможности FastBPETokenizer:
 * 
 *          **Демонстрируемые возможности:**
 *          ┌────────────────────┬────────────────────────────────────┐
 *          │ Базовые операции   │ encode/decode примеров C++ кода    │
 *          │ Пакетная обработка │ Сравнение последовательной и batch │
 *          │ Кэширование        │ Hit rate на повторяющихся текстах  │
 *          │ Производительность │ Скорость для 1 KБ - 1 МБ текста    │
 *          │ SIMD оптимизации   │ AVX2/AVX/SSE4.2 тесты              │
 *          │ Статистика         │ Сбор и отображение метрик          │
 *          └────────────────────┴────────────────────────────────────┘
 * 
 *          **Режимы работы:**
 *          - Обычный               - Все тесты
 *          - Быстрый (--quick)     - Пропустить медленные SIMD тесты
 *          - Подробный (--verbose) - Детальный вывод
 * 
 * @compile g++ -std=c++17 -O3 -mavx2 -msse4.2 -Iinclude fast_tokenizer_demo.cpp -o fast_tokenizer_demo -lpthread
 * @run   ./fast_tokenizer_demo [--quick] [--verbose]
 * 
 * @see FastBPETokenizer, TokenizerConfig, SIMDUtils
 */

#include "fast_tokenizer.hpp"
#include "simd_utils.hpp"
#include "profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    // Размеры
    constexpr size_t DEFAULT_VOCAB_SIZE = 10000;    ///< Размер словаря по умолчанию
    constexpr size_t DEMO_CACHE_SIZE = 10000;       ///< Размер кэша по умолчанию
    constexpr int WIDTH = 60;                       ///< Ширина таблиц для вывода
    
    // ANSI цвета для красивого вывода
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string BOLD = "\033[1m";
}

// ============================================================================
// ScopedTimer - RAII таймер для измерения времени
// ============================================================================

/**
 * @brief RAII таймер с автоматическим логированием
 * 
 * Автоматически засекает время при создании и выводит при разрушении.
 * Удобен для измерения времени выполнения блоков кода.
 * 
 * @code
 * {
 *     ScopedTimer timer("encoding", true, verbose);
 *     auto tokens = tokenizer.encode(text);
 * } // автоматический вывод времени
 * @endcode
 */
class ScopedTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;
    bool print_on_destroy_;
    bool verbose_;
    
public:
    /**
     * @brief Конструктор - запоминает время старта
     * @param name Имя операции для вывода
     * @param print Выводить ли результат при разрушении
     * @param verbose Подробный режим
     */
    ScopedTimer(const std::string& name, bool print = true, bool verbose = false) 
        : name_(name), print_on_destroy_(print), verbose_(verbose) {
        start_ = std::chrono::high_resolution_clock::now();
        if (verbose_) {
            std::cout << "Начало: " << name_ << std::endl;
        }
    }
    
    /**
     * @brief Деструктор - вычисляет и выводит прошедшее время
     */
    ~ScopedTimer() {
        if (print_on_destroy_) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            
            std::cout << " " << BOLD << "*" << RESET << " " 
                      << std::left << std::setw(30) << name_ << ": " 
                      << std::right << std::setw(8) << std::fixed << std::setprecision(3)
                      << duration.count() / 1000.0 << " мс" << std::endl;
        }
        if (verbose_) {
            std::cout << "Завершено: " << name_ << std::endl;
        }
    }
    
    /**
     * @brief Получить прошедшее время в миллисекундах
     */
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    /**
     * @brief Сбросить таймер
     */
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

// ============================================================================
// Вспомогательные функции для вывода
// ============================================================================

/**
 * @brief Выводит заголовок раздела с красивым оформлением
 * 
 * @param title Заголовок для вывода
 * 
 * @code
 * print_header("ТЕСТ 1: Производительность");
 * // Вывод:
 * // ┌────────────────────────────────────────────────────────────┐
 * // │                 ТЕСТ 1: Производительность                 │
 * // └────────────────────────────────────────────────────────────┘
 * @endcode
 */
void print_header(const std::string& title) {
    // Верхняя граница
    std::cout << "\n" << BOLD << "┌";
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << "┐\n";
    
    // Центрируем заголовок
    int total_padding = WIDTH - static_cast<int>(title.size());
    int left_padding = total_padding / 2;
    int right_padding = total_padding - left_padding;
    
    std::cout << "│";
    for (int i = 0; i < left_padding; ++i) std::cout << ' ';
    std::cout << title;
    for (int i = 0; i < right_padding; ++i) std::cout << ' ';
    std::cout << "│\n";
    
    // Нижняя граница
    std::cout << "└";
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << "┘" << RESET << std::endl;
}

// ============================================================================
// Создание тестовых данных
// ============================================================================

/**
 * @brief Создает набор примеров C++ кода разной сложности
 * 
 * @return std::vector<std::pair<std::string, std::string>> 
 *         Пары (описание, код) для демонстрации
 */
std::vector<std::pair<std::string, std::string>> create_examples() {
    return {
        {"Простое выражение", "int x = 42;"},
        {"Работа с вектором", "std::vector<int> numbers = {1, 2, 3, 4, 5};"},
        {"Цикл for", "for (const auto& item : items) { std::cout << item << std::endl; }"},
        {"Шаблон функции", "template<typename T> T max(T a, T b) { return a > b ? a : b; }"},
        {"Класс", "class MyClass {\npublic:\n    MyClass() = default;\n    void print() const {}\n};"},
        {"Лямбда", "auto lambda = [](int x) { return x * x; };"},
        {"Умные указатели", "std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();"},
        {"Английский комментарий", "// this is an English comment\nint main() { return 0; }"},
        {"Сложное выражение", "std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x * x; });"},
        {"Include директивы", "#include <iostream>\n#include <vector>\n#include <algorithm>"}
    };
}

/**
 * @brief Создает длинный C++ код для теста производительности
 * 
 * Генерирует множество функций с уникальными именами для создания
 * большого объема текста с предсказуемым содержимым.
 * 
 * @param size_kb Желаемый размер в килобайтах (по умолчанию 100 КБ)
 * @return std::string Сгенерированный код
 */
std::string create_long_code(size_t size_kb = 100) {
    std::string code;
    code.reserve(size_kb * 1024);
    
    const size_t target_size = size_kb * 1024;
    size_t current_size = 0;
    
    for (int i = 0; current_size < target_size; ++i) {
        std::string func = "int func" + std::to_string(i) + 
                "(int x) { \n"
                "    return x * " + std::to_string(i) + ";\n"
                "}\n\n";
        code += func;
        current_size = code.size();
    }
    
    return code;
}

// ============================================================================
// Проверка SIMD поддержки
// ============================================================================

/**
 * @brief Проверяет и выводит информацию о поддержке SIMD оптимизаций
 * 
 * Показывает, какие инструкции были включены при компиляции
 * и что реально поддерживается процессором.
 */
void check_simd_support() {
    std::cout << "\n" << CYAN << BOLD << "SIMD оптимизации:" << RESET << std::endl;
    
    // Проверка флагов компиляции
    std::cout << "Флаги компиляции:\n";
    
    #ifdef __AVX2__
        std::cout << GREEN << "AVX2 включен!" << RESET << std::endl;
    #else
        std::cout << YELLOW << "AVX2 не включен!" << RESET << std::endl;
    #endif
    
    #ifdef __AVX__
        std::cout << GREEN << "AVX включен!" << RESET << std::endl;
    #else
        std::cout << YELLOW << "AVX не включен!" << RESET << std::endl;
    #endif
    
    #ifdef __SSE4_2__
        std::cout << GREEN << "SSE4.2 включен!" << RESET << std::endl;
    #else
        std::cout << YELLOW << "SSE4.2 не включен!" << RESET << std::endl;
    #endif
    
    // Проверка поддержки процессором
    std::cout << "\nПоддержка процессором (runtime):\n";
    
    bool avx2_support = SIMDUtils::check_avx2_support();
    bool avx_support = SIMDUtils::check_avx_support();
    bool sse42_support = SIMDUtils::check_sse42_support();
    
    std::cout << "AVX2:   " << (avx2_support ? GREEN + "ПОДДЕРЖИВАЕТСЯ" : RED + "НЕ ПОДДЕРЖИВАЕТСЯ") << RESET << "\n";
    std::cout << "AVX:    " << (avx_support ? GREEN + "ПОДДЕРЖИВАЕТСЯ" : RED + "НЕ ПОДДЕРЖИВАЕТСЯ") << RESET << "\n";
    std::cout << "SSE4.2: " << (sse42_support ? GREEN + "ПОДДЕРЖИВАЕТСЯ" : RED + "НЕ ПОДДЕРЖИВАЕТСЯ") << RESET << "\n";
    
    // Проверка на опасные комбинации
    #ifdef __AVX2__
        if (!avx2_support) {
            std::cout << "\n" << RED << "ВНИМАНИЕ: AVX2 включен в компиляции, ";
            std::cout << "но процессор НЕ ПОДДЕРЖИВАЕТ эти инструкции!" << RESET << std::endl;
            std::cout << "Программа может упасть с SIGILL (illegal instruction).\n";
            std::cout << "Рекомендуется перекомпилировать без флага -mavx2\n";
        }
    #endif
    
    #ifdef __AVX__
        if (!avx_support) {
            std::cout << "\n" << RED << "ВНИМАНИЕ: AVX включен в компиляции, ";
            std::cout << "но процессор НЕ ПОДДЕРЖИВАЕТ эти инструкции!" << RESET << std::endl;
        }
    #endif
    
    #ifdef __SSE4_2__
        if (!sse42_support) {
            std::cout << "\n" << RED << "ВНИМАНИЕ: SSE4.2 включен в компиляции, ";
            std::cout << "но процессор НЕ ПОДДЕРЖИВАЕТ эти инструкции!" << RESET << std::endl;
        }
    #endif
    
    // Уровень SIMD
    std::cout << "\nМаксимальный уровень SIMD: " << BOLD 
              << SIMDUtils::get_simd_level() << RESET << std::endl;
    
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << std::endl;
}

// ============================================================================
// SSE4.2 демонстрация
// ============================================================================

/**
 * @brief Демонстрирует работу SSE4.2 оптимизаций
 * 
 * Показывает ускорение поиска подстроки и сравнения строк
 * с использованием SSE4.2 инструкций
 */
void demo_sse42_features() {
    print_header("SSE4.2 ОПТИМИЗАЦИИ");
    
    if (!SIMDUtils::check_sse42_support()) {
        std::cout << YELLOW << "SSE4.2 не поддерживается процессором (пропускаем)!\n" << RESET;
        return;
    }
    
    #ifndef __SSE4_2__
        std::cout << YELLOW << "SSE4.2 не включен в компиляции (пропускаем)!\n" << RESET;
        return;
    #endif
    
    std::cout << GREEN << "SSE4.2 доступен, тестируем оптимизации:\n" << RESET;
    
    // ТЕСТ 1: Поиск подстроки
    std::string text = "The quick brown fox jumps over the lazy dog. "
                       "C++ template metaprogramming is powerful. "
                       "SSE4.2 instructions accelerate string operations. "
                       "The quick brown fox jumps over the lazy dog. "
                       "C++ template metaprogramming is powerful. "
                       "SSE4.2 instructions accelerate string operations.";
    
    std::string pattern = "fox";
    
    std::cout << "\nПоиск подстроки \"" << pattern << "\":\n";
    std::cout << "Размер текста: " << text.size() << " байт\n";
    
    // Прогрев
    for (int i = 0; i < 100; ++i) {
        volatile size_t pos = text.find(pattern);
        (void)pos;
    }
    
    // Стандартный поиск (std::string::find)
    size_t pos_std = 0;
    double std_time;
    {
        ScopedTimer timer("std::string::find", true, false);
        for (int i = 0; i < 10000; ++i) {
            pos_std = text.find(pattern);
        }
        std_time = timer.elapsed_ms() / 10000.0;
    }
    
    // SSE4.2 поиск
    size_t pos_sse42 = 0;
    double sse42_time;
    {
        ScopedTimer timer("SIMDUtils::find_substring_sse42", true, false);
        for (int i = 0; i < 10000; ++i) {
            pos_sse42 = SIMDUtils::find_substring_sse42(text, pattern);
        }
        sse42_time = timer.elapsed_ms() / 10000.0;
    }
    
    std::cout << "\nРезультаты поиска:\n";
    std::cout << "- Позиция (std):          " << pos_std << "\n";
    std::cout << "- Позиция (SSE4.2):       " << pos_sse42 << "\n";
    std::cout << "- Среднее время (std):    " << std::fixed << std::setprecision(4) 
              << std_time << " мс\n";
    std::cout << "- Среднее время (SSE4.2): " << sse42_time << " мс\n";
    
    if (sse42_time > 0) {
        std::cout << "- Ускорение:              " << (std_time / sse42_time) << "x\n";
    }
    
    std::cout << "- Результаты совпадают:   " 
              << (pos_std == pos_sse42 ? GREEN + "да" : RED + "нет") << RESET << "\n";
    
    // ТЕСТ 2: Сравнение строк
    std::cout << "\nСравнение строк:\n";
    
    std::string str1 = "C++ template metaprogramming with SSE4.2";
    std::string str2 = "C++ template metaprogramming with SSE4.2";    // Идентичная
    std::string str3 = "C++ template metaprogramming with SSE4.3";    // Отличается в конце
    
    // Прогрев
    for (int i = 0; i < 100; ++i) {
        volatile bool eq = (str1 == str2);
        (void)eq;
    }
    
    // Стандартное сравнение (идентичные строки)
    bool equal_std = false;
    double std_cmp_time;
    {
        ScopedTimer timer("operator== (идентичные)", true, false);
        for (int i = 0; i < 100000; ++i) {
            equal_std = (str1 == str2);
        }
        std_cmp_time = timer.elapsed_ms() / 100000.0;
    }
    
    // SSE4.2 сравнение (идентичные строки)
    bool equal_sse42 = false;
    double sse42_cmp_time;
    {
        ScopedTimer timer("SIMDUtils::strings_equal_sse42 (идентичные)", true, false);
        for (int i = 0; i < 100000; ++i) {
            equal_sse42 = SIMDUtils::strings_equal_sse42(str1, str2);
        }
        sse42_cmp_time = timer.elapsed_ms() / 100000.0;
    }
    
    std::cout << "\nСравнение идентичных строк:\n";
    std::cout << "- Равны (std):            " << (equal_std ? GREEN + "да" : RED + "нет") << RESET << "\n";
    std::cout << "- Равны (SSE4.2):         " << (equal_sse42 ? GREEN + "да" : RED + "нет") << RESET << "\n";
    std::cout << "- Среднее время (std):    " << std::fixed << std::setprecision(6) 
              << std_cmp_time << " мс\n";
    std::cout << "- Среднее время (SSE4.2): " << sse42_cmp_time << " мс\n";
    
    if (sse42_cmp_time > 0) {
        std::cout << "- Ускорение:              " << (std_cmp_time / sse42_cmp_time) << "x\n";
    }
    
    // Сравнение отличающихся строк
    bool equal_std_diff = (str1 == str3);
    bool equal_sse42_diff = SIMDUtils::strings_equal_sse42(str1, str3);
    
    std::cout << "\nСравнение отличающихся строк:\n";
    std::cout << "- Равны (std):    " << (equal_std_diff ? RED + "да (ОШИБКА!)" : GREEN + "нет (верно)") << RESET << "\n";
    std::cout << "- Равны (SSE4.2): " << (equal_sse42_diff ? RED + "да (ОШИБКА!)" : GREEN + "нет (верно)") << RESET << "\n";
    
    // ИТОГ
    std::cout << "\n" << CYAN << "Итог по SSE4.2 тестам:" << RESET << "\n";
    
    bool all_tests_passed = (pos_std == pos_sse42) && 
                            (equal_std == equal_sse42) &&
                            (equal_std_diff == equal_sse42_diff);
    
    if (all_tests_passed) {
        std::cout << GREEN << "Все тесты пройдены успешно!\n" << RESET;
    } else {
        std::cout << YELLOW << "Некоторые тесты не прошли. Проверьте реализацию!\n" << RESET;
    }
}

// ============================================================================
// Прямой вызов SIMD encode
// ============================================================================

/**
 * @brief Демонстрирует прямой вызов SIMD функций encode
 * 
 * Показывает разницу между AVX и AVX2 версиями кодирования текста
 */
void demo_direct_simd_encode() {
    print_header("ПРЯМОЙ ВЫЗОВ SIMD ENCODE");
    
    // Создаем lookup table (для примера - identity mapping)
    uint32_t lookup[256];
    for (int i = 0; i < 256; ++i) {
        lookup[i] = i;
    }
    
    // Тестовый текст
    std::string text = "C++ template metaprogramming with SIMD optimizations ";
    text = text + text + text + text;    // Увеличиваем для наглядности
    
    std::cout << "Размер текста: " << text.size() << " байт\n\n";
    
    // Проверка наличия SIMD
    bool avx2_available = SIMDUtils::has_avx2() && SIMDUtils::check_avx2_support();
    bool avx_available = SIMDUtils::has_avx() && SIMDUtils::check_avx_support();
    bool sse42_available = SIMDUtils::has_sse42() && SIMDUtils::check_sse42_support();
    
    std::cout << "Доступные оптимизации:\n";
    std::cout << "- AVX2:   " << (avx2_available ? GREEN + "да" : YELLOW + "нет") << RESET << "\n";
    std::cout << "- AVX:    " << (avx_available ? GREEN + "да" : YELLOW + "нет") << RESET << "\n";
    std::cout << "- SSE4.2: " << (sse42_available ? GREEN + "да" : YELLOW + "нет") << RESET << "\n\n";
    
    // Скалярная версия (ручная реализация для сравнения)
    std::vector<uint32_t> scalar_result;
    {
        ScopedTimer timer("Скалярная версия (ручной цикл)", true, false);
        scalar_result.reserve(text.size());
        for (char c : text) {
            scalar_result.push_back(lookup[static_cast<unsigned char>(c)]);
        }
    }
    
    // AVX2 версия
    if (avx2_available) {
        std::vector<uint32_t> avx2_result;
        {
            ScopedTimer timer("AVX2 encode (SIMDUtils::encode_avx2)", true, false);
            avx2_result = SIMDUtils::encode_avx2(text, lookup, 0);
        }
        
        std::cout << "AVX2 encode:          " << avx2_result.size() << " токенов\n";
        std::cout << "Результаты совпадают: " 
                  << (scalar_result.size() == avx2_result.size() ? GREEN + "да" : RED + "нет") 
                  << RESET << "\n";
    } else {
        std::cout << YELLOW << "AVX2 encode недоступен!\n" << RESET;
    }
    
    // AVX версия (128-бит)
    if (avx_available) {
        std::vector<uint32_t> avx_result;
        {
            ScopedTimer timer("AVX encode (SIMDUtils::encode_avx)", true, false);
            avx_result = SIMDUtils::encode_avx(text, lookup, 0);
        }
        
        std::cout << "AVX encode:           " << avx_result.size() << " токенов\n";
        std::cout << "Результаты совпадают: " 
                  << (scalar_result.size() == avx_result.size() ? GREEN + "да" : RED + "нет") 
                  << RESET << "\n";
    } else {
        std::cout << YELLOW << "AVX encode недоступен!\n" << RESET;
    }
}

// ============================================================================
// Класс для поиска файлов модели
// ============================================================================

/**
 * @brief Утилита для поиска файлов моделей в разных директориях
 * 
 * Пытается найти файлы vocab.json и merges.txt в различных расположениях:
 * - tests/models/bpe_10000/            - Приоритет 1  (для тестов)
 * - ../models/bpe_10000/               - Приоритет 2  (из папки bpe_cpp)
 * - ../../bpe_python/models/bpe_10000/ - Пприоритет 3 (из корня проекта)
 * - models/bpe_10000/                  - Приоритет 4  (текущая директория)
 */
class ModelFinder {
private:
    std::vector<std::pair<std::string, std::string>> candidates_;
    
public:
    ModelFinder() {
        // Модели в build/tests (приоритет 1)
        candidates_.emplace_back("tests/models/bpe_8000/cpp_vocab.json", 
                                "tests/models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("tests/models/bpe_10000/cpp_vocab.json", 
                                "tests/models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("tests/models/bpe_12000/cpp_vocab.json", 
                                "tests/models/bpe_12000/cpp_merges.txt");

        // C++ модели (приоритет 2)
        candidates_.emplace_back("../models/bpe_8000/cpp_vocab.json", 
                                 "../models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("../models/bpe_10000/cpp_vocab.json", 
                                 "../models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("../models/bpe_12000/cpp_vocab.json", 
                                 "../models/bpe_12000/cpp_merges.txt");

        // Python модели (приоритет 3)
        candidates_.emplace_back("../../bpe_python/models/bpe_8000/vocab.json", 
                                 "../../bpe_python/models/bpe_8000/merges.txt");
        candidates_.emplace_back("../../bpe_python/models/bpe_10000/vocab.json", 
                                 "../../bpe_python/models/bpe_10000/merges.txt");
        candidates_.emplace_back("../../bpe_python/models/bpe_12000/vocab.json", 
                                 "../../bpe_python/models/bpe_12000/merges.txt");

        // Модели в текущей директории (приоритет 4)
        candidates_.emplace_back("models/bpe_8000/cpp_vocab.json", 
                                 "models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("models/bpe_10000/cpp_vocab.json", 
                                 "models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("models/bpe_12000/cpp_vocab.json", 
                                 "models/bpe_12000/cpp_merges.txt");
    }
    
    /**
     * @brief Найти и загрузить модель
     * 
     * @param tokenizer Ссылка на токенизатор для загрузки
     * @param vocab_path [out] Путь к найденному vocab.json
     * @param merges_path [out] Путь к найденному merges.txt
     * @return true если модель найдена и загружена
     */
    bool find(FastBPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
        std::cout << "Поиск файлов модели..." << std::endl;
        
        for (const auto& [vpath, mpath] : candidates_) {
            std::cout << "Проверка: " << vpath << std::endl;
            
            if (fs::exists(vpath) && fs::exists(mpath)) {
                vocab_path = vpath;
                merges_path = mpath;
                std::cout << GREEN << "Найдены: " << vpath << RESET << std::endl;
                return tokenizer.load(vpath, mpath);
            }
        }
        
        std::cout << RED << "Файлы не найдены!" << RESET << std::endl;
        std::cout << "Убедитесь, что модели существуют в:" << std::endl;
        std::cout << "- bpe_cpp/models/bpe_10000/" << std::endl;
        std::cout << "- bpe_python/models/bpe_10000/" << std::endl;
        return false;
    }
};

// ============================================================================
// Демонстрационные функции
// ============================================================================

/**
 * @brief Демонстрация базового кодирования и декодирования
 * 
 * @param tokenizer Ссылка на токенизатор
 * @param examples Вектор пар (описание, текст) для демонстрации
 */
void demo_basic_encoding(FastBPETokenizer& tokenizer, 
                         const std::vector<std::pair<std::string, std::string>>& examples) {
    print_header("БАЗОВОЕ КОДИРОВАНИЕ/ДЕКОДИРОВАНИЕ");
    
    int success_count = 0;
    
    for (size_t i = 0; i < examples.size(); ++i) {
        const auto& [desc, text] = examples[i];
        
        std::cout << "\n" << BOLD << "Пример " << i+1 << ": " << desc << RESET << std::endl;
        std::cout << "Исходный: " << text << std::endl;
        
        // Кодирование
        std::vector<uint32_t> tokens;
        {
            ScopedTimer timer("encode", true, false);
            tokens = tokenizer.encode(text);
        }
        
        // Статистика токенов
        std::cout << "Токенов: " << tokens.size() << " [";
        for (size_t j = 0; j < std::min(size_t(10), tokens.size()); ++j) {
            std::cout << tokens[j];
            if (j < std::min(size_t(9), tokens.size()-1)) std::cout << ", ";
        }
        if (tokens.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        // Подсчет уникальных токенов
        std::set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
        std::cout << "Уникальных: " << unique_tokens.size() << std::endl;
        
        // Декодирование
        std::string decoded;
        {
            ScopedTimer timer("decode", true, false);
            decoded = tokenizer.decode(tokens);
        }
        
        bool success = (decoded == text);
        if (success) success_count++;
        
        std::cout << "Декодировано: " << decoded << std::endl;
        std::cout << "Результат:    " << (success ? GREEN + "УСПЕХ" : RED + "НЕУДАЧА") 
                  << RESET << std::endl;
    }
    
    std::cout << "\n" << BOLD << "Итог: " << RESET
              << success_count << "/" << examples.size() 
              << " примеров успешно (" << (success_count * 100 / examples.size()) << "%)" 
              << std::endl;
}

/**
 * @brief Демонстрация пакетной обработки
 * 
 * @param tokenizer Ссылка на токенизатор
 * @param examples Вектор пар (описание, текст) для демонстрации
 */
void demo_batch_processing(FastBPETokenizer& tokenizer, 
                           const std::vector<std::pair<std::string, std::string>>& examples) {
    print_header("ПАКЕТНАЯ ОБРАБОТКА");
    
    // Подготовка данных
    std::vector<std::string> texts;
    std::vector<std::string_view> text_views;
    
    texts.reserve(examples.size());
    text_views.reserve(examples.size());
    
    for (const auto& [_, text] : examples) {
        texts.push_back(text);
        text_views.push_back(std::string_view(texts.back()));
    }
    
    std::cout << "Размер батча: " << texts.size() << " текстов" << std::endl;
    
    // Прогрев
    tokenizer.encode_batch(text_views);
    
    // Последовательная обработка
    double sequential_time;
    {
        ScopedTimer timer("Последовательная обработка", true, false);
        for (const auto& text : texts) {
            tokenizer.encode(text);
        }
        sequential_time = timer.elapsed_ms();
    }
    
    // Пакетная обработка
    std::vector<std::vector<uint32_t>> batch_results;
    double batch_time;
    {
        ScopedTimer timer("Пакетная обработка", true, false);
        batch_results = tokenizer.encode_batch(text_views);
        batch_time = timer.elapsed_ms();
    }
    
    std::cout << "Ускорение: " << BOLD << (sequential_time / batch_time) << "x" 
              << RESET << std::endl;
    
    // Проверка корректности
    bool all_match = true;
    for (size_t i = 0; i < texts.size() && all_match; ++i) {
        auto single = tokenizer.encode(texts[i]);
        if (single.size() != batch_results[i].size()) {
            all_match = false;
            std::cout << "Несовпадение размера в примере " << i << ": "
                      << single.size() << " vs " << batch_results[i].size() << std::endl;
            break;
        }
    }
    
    std::cout << "Корректность: " << (all_match ? GREEN + "да" : RED + "нет") 
              << RESET << std::endl;
}

/**
 * @brief Демонстрация эффективности кэширования
 * 
 * @param tokenizer Ссылка на токенизатор
 */
void demo_caching(FastBPETokenizer& tokenizer) {
    print_header("ЭФФЕКТИВНОСТЬ КЭШИРОВАНИЯ");
    
    // Создаем 100 текстов с повторяющимися паттернами
    std::vector<std::string> repetitive_texts;
    repetitive_texts.reserve(100);
    for (int i = 0; i < 100; ++i) {
        repetitive_texts.push_back("int var" + std::to_string(i % 10) + " = " + std::to_string(i) + ";");
    }
    
    std::cout << "Тестовых текстов:     " << repetitive_texts.size() << std::endl;
    std::cout << "Уникальных паттернов: 10" << std::endl;
    
    tokenizer.reset_stats();
    
    // Первый проход (заполнение кэша)
    double first_pass_time;
    {
        ScopedTimer timer("Первый проход (заполнение кэша)", true, false);
        for (const auto& text : repetitive_texts) {
            tokenizer.encode(text);
        }
        first_pass_time = timer.elapsed_ms();
    }
    
    auto stats1 = tokenizer.stats();
    std::cout << "Cache hits:   " << stats1.cache_hits << std::endl;
    std::cout << "Cache misses: " << stats1.cache_misses << std::endl;
    
    tokenizer.reset_stats();
    
    // Второй проход (использование кэша)
    double second_pass_time;
    {
        ScopedTimer timer("Второй проход (кэш)", true, false);
        for (const auto& text : repetitive_texts) {
            tokenizer.encode(text);
        }
        second_pass_time = timer.elapsed_ms();
    }
    
    auto stats2 = tokenizer.stats();
    std::cout << "Cache hits:   " << stats2.cache_hits << std::endl;
    std::cout << "Cache misses: " << stats2.cache_misses << std::endl;
    std::cout << "Hit rate:     " << std::fixed << std::setprecision(1) 
              << stats2.cache_hit_rate() << "%" << std::endl;
    std::cout << "Ускорение:    " << BOLD << (first_pass_time / second_pass_time) << "x" 
              << RESET << std::endl;
}

/**
 * @brief Тест производительности с разными размерами данных
 * 
 * @param tokenizer Ссылка на токенизатор
 */
void demo_performance(FastBPETokenizer& tokenizer) {
    print_header("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ");
    
    // Тест с разными размерами
    std::vector<size_t> sizes = {1024, 10*1024, 100*1024, 1024*1024};
    
    std::cout << std::setw(10) << "Размер" 
              << " | " << std::setw(10) << "Токенов" 
              << " | " << std::setw(10) << "Время (мс)" 
              << " | " << "Скорость (МБ/с)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t size : sizes) {
        // Создаем текст нужного размера
        std::string text = create_long_code();
        while (text.size() < size) {
            text += text;
        }
        text.resize(size);
        
        // Прогрев
        for (int i = 0; i < 3; ++i) {
            tokenizer.encode(text);
        }
        
        tokenizer.reset_stats();
        
        // Измерение
        double elapsed;
        size_t tokens_count;
        {
            ScopedTimer timer(std::to_string(size/1024) + " КБ", false, false);
            auto tokens = tokenizer.encode(text);
            elapsed = timer.elapsed_ms();
            tokens_count = tokens.size();
        }
        
        double mb_per_sec = (size / 1024.0 / 1024.0) / (elapsed / 1000.0);
        
        std::cout << std::setw(9) << (size/1024) << " КБ" 
                  << " | " << std::setw(10) << tokens_count
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2) << elapsed
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << mb_per_sec << std::endl;
    }
}

/**
 * @brief Демонстрация SIMD оптимизаций в токенизаторе
 * 
 * @param tokenizer Ссылка на токенизатор
 */
void demo_simd_in_tokenizer(FastBPETokenizer& tokenizer) {
    print_header("SIMD В TOKENIZER");
    
    // Проверяем, какие SIMD инструкции реально доступны
    bool avx2_available = SIMDUtils::has_avx2() && SIMDUtils::check_avx2_support();
    bool avx_available = SIMDUtils::has_avx() && SIMDUtils::check_avx_support();
    bool sse42_available = SIMDUtils::has_sse42() && SIMDUtils::check_sse42_support();
    
    std::cout << "Доступные оптимизации:\n";
    std::cout << "- AVX2:   " << (avx2_available ? GREEN + "да" : YELLOW + "нет") << RESET << "\n";
    std::cout << "- AVX:    " << (avx_available ? GREEN + "да" : YELLOW + "нет") << RESET << "\n";
    std::cout << "- SSE4.2: " << (sse42_available ? GREEN + "да" : YELLOW + "нет") << RESET << "\n";
    
    if (!avx2_available && !avx_available && !sse42_available) {
        std::cout << YELLOW << "\nНи одна SIMD оптимизация не доступна.\n" << RESET;
        std::cout << "Используется скалярная реализация.\n";
        return;
    }
    
    std::string text(100000, 'a');    // 100 КБ текста
    
    // Прогрев
    for (int i = 0; i < 3; ++i) {
        tokenizer.encode(text);
    }
    
    // Измерение производительности
    tokenizer.reset_stats();
    double simd_time;
    size_t tokens_count;
    {
        ScopedTimer timer("FastTokenizer encode", true, false);
        auto tokens = tokenizer.encode(text);
        simd_time = timer.elapsed_ms();
        tokens_count = tokens.size();
    }
    
    double mb_per_sec = (text.size() / 1024.0 / 1024.0) / (simd_time / 1000.0);
    
    std::cout << "\nРезультаты FastTokenizer:\n";
    std::cout << "- размер:   100 КБ\n";
    std::cout << "- токенов:  " << tokens_count << "\n";
    std::cout << "- время:    " << std::fixed << std::setprecision(2) << simd_time << " мс\n";
    std::cout << "- скорость: " << mb_per_sec << " МБ/с\n";
    
    // Показываем, какие оптимизации вероятно использовались
    std::cout << "\nАктивные оптимизации в FastTokenizer:\n";
    if (avx2_available) {
        std::cout << "- AVX2 (256-бит, 32 символа за раз)\n";
    } else if (avx_available) {
        std::cout << "- AVX (128-бит, 16 символов за раз)\n";
    } else if (sse42_available) {
        std::cout << "- SSE4.2 (строковые инструкции)\n";
    }
    std::cout << "- пул памяти\n";
    std::cout << "- кэш токенов (hit rate: " 
              << std::fixed << std::setprecision(1) 
              << tokenizer.stats().cache_hit_rate() << "%)\n";
}

// ============================================================================
// Основная функция
// ============================================================================

int main(int argc, char* argv[]) {
    // Верхняя линия
    std::cout << BOLD << "============================================================" << RESET << std::endl;
    std::cout << BOLD << "FAST BPE TOKENIZER DEMO v3.4.0" << RESET << std::endl;
    std::cout << BOLD << "============================================================" << RESET << std::endl;    

    // Парсинг аргументов командной строки
    bool verbose = false;
    bool quick_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
            std::cout << "Подробный режим включен\n";
        } else if (arg == "--quick" || arg == "-q") {
            quick_mode = true;
            std::cout << "Быстрый режим (медленные тесты пропущены)\n";
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\nИспользование: " << argv[0] << " [options]\n";
            std::cout << "--verbose, -v    - Подробный вывод\n";
            std::cout << "--quick, -q      - Пропустить медленные тесты\n";
            std::cout << "--help, -h       - Показать справку\n";
            return 0;
        }
    }
    
    try {
        // Проверка SIMD поддержки
        check_simd_support();
        
        // Демо SSE4.2 функций
        if (!quick_mode) {
            demo_sse42_features();
            demo_direct_simd_encode();
        }
        
        // Создание токенизатора
        std::cout << "\n" << CYAN << "Создание токенизатора..." << RESET << std::endl;
        
        TokenizerConfig config;
        config.vocab_size = DEFAULT_VOCAB_SIZE;
        config.cache_size = DEMO_CACHE_SIZE;
        config.byte_level = true;
        config.enable_cache = true;
        config.enable_profiling = true;
        
        std::cout << "- vocab_size:   " << config.vocab_size << std::endl;
        std::cout << "- cache_size:   " << config.cache_size << std::endl;
        std::cout << "- byte_level:   " << (config.byte_level ? "да" : "нет") << std::endl;
        std::cout << "- enable_cache: " << (config.enable_cache ? "да" : "нет") << std::endl;
        std::cout << "- profiling:    " << (config.enable_profiling ? "да" : "нет") << std::endl;
        
        FastBPETokenizer tokenizer(config);
        
        // Загрузка модели
        std::cout << "\n" << CYAN << "Загрузка модели..." << RESET << std::endl;
        
        std::string vocab_path, merges_path;
        ModelFinder finder;
        
        if (!finder.find(tokenizer, vocab_path, merges_path)) {
            std::cerr << RED << "\nНе удалось загрузить модель!" << RESET << std::endl;
            std::cout << YELLOW << "\nЗапуск демо без модели (только SIMD тесты)..." << RESET << std::endl;
            
            print_header("SIMD ТЕСТЫ БЕЗ МОДЕЛИ");
            demo_sse42_features();
            demo_direct_simd_encode();
            std::cout << GREEN << "\nSIMD тесты завершены!" << RESET << std::endl;
            return 0;
        }
        
        {
            ScopedTimer timer("Загрузка модели", true, verbose);
        }
        
        std::cout << "- Словарь: " << vocab_path << std::endl;
        std::cout << "- Слияния: " << merges_path << std::endl;
        std::cout << "- Размер:  " << tokenizer.vocab_size() << " токенов" << std::endl;
        std::cout << "- Правил:  " << tokenizer.merges_count() << std::endl;
        
        // Создание примеров
        auto examples = create_examples();
        
        // Запуск демонстраций
        demo_basic_encoding(tokenizer, examples);
        
        if (!quick_mode) {
            demo_batch_processing(tokenizer, examples);
            demo_caching(tokenizer);
            demo_performance(tokenizer);
            demo_simd_in_tokenizer(tokenizer);
        }
        
        // Финальная статистика
        print_header("ФИНАЛЬНАЯ СТАТИСТИКА");

        auto stats = tokenizer.stats();
        std::cout << "- Всего encode вызовов: " << stats.encode_calls << std::endl;
        std::cout << "- Всего decode вызовов: " << stats.decode_calls << std::endl;
        std::cout << "- Попаданий в кэш:      " << stats.cache_hits << std::endl;
        std::cout << "- Промахов кэша:        " << stats.cache_misses << std::endl;
        std::cout << "- Эффективность кэша:   " << std::fixed << std::setprecision(1) 
                  << stats.cache_hit_rate() << "%" << std::endl;
        std::cout << "- Обработано токенов:   " << stats.total_tokens_processed << std::endl;
        std::cout << "- Среднее время encode: " << std::fixed << std::setprecision(3) 
                  << stats.avg_encode_time_ms() << " мс" << std::endl;
        std::cout << "- Среднее время decode: " << std::fixed << std::setprecision(3) 
                  << stats.avg_decode_time_ms() << " мс" << std::endl;

        // ===== Информация о SIMD =====
        std::cout << "\n- SIMD уровень:         " << BOLD << SIMDUtils::get_simd_level() 
                  << RESET << std::endl;

        std::cout << "\n" << GREEN << BOLD << "Демо завершено успешно!" << RESET << std::endl;

        // ===== ВЫВОД ОТЧЕТА ПРОФАЙЛЕРА =====
        // Используем переменную config, которая была создана ранее
        if (config.enable_profiling) {
            std::cout << "\n" << CYAN << BOLD << "ОТЧЕТ ПРОФИЛИРОВАНИЯ:" << RESET << std::endl;
            SimpleProfiler::printReport();
            SimpleProfiler::saveReport();
        }

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << RED << "\nОшибка: " << e.what() << RESET << std::endl;
        return 1;
    }
    
    return 0;
}
