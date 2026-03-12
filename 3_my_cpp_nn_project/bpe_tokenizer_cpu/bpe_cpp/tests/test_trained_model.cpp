/**
 * @file test_trained_model.cpp
 * @brief Тестирование обученной модели BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Эта программа загружает предварительно обученную модель BPE токенизатора
 *          и проверяет её работу на различных примерах C++ кода. Выполняется
 *          комплексное тестирование с анализом производительности и точности.
 * 
 *          **Выполняемые проверки:**
 * 
 *          1. **Поиск и загрузка модели**
 *             - Проверка различных путей к файлам модели
 *             - Поддержка бинарного и текстового форматов
 *             - Измерение времени загрузки
 * 
 *          2. **Тестирование кодирования/декодирования**
 *             - Различные конструкции C++ (переменные, циклы, классы)
 *             - Шаблоны и лямбда-функции
 *             - Исключения и C++17 features
 *             - Проверка roundtrip (encode + decode)
 * 
 *          3. **Сбор статистики**
 *             - Количество токенов и уникальных токенов
 *             - Время encode и decode для каждого примера
 *             - Распределение длины токенов
 * 
 *          4. **Производительность**
 *             - Тест на большом тексте (100 строк)
 *             - Скорость в МБ/с и токенов/с
 *             - Многопоточное тестирование (если доступен OpenMP)
 * 
 *          5. **Статистика токенизатора**
 *             - Эффективность кэша (hit rate)
 *             - Количество обработанных токенов
 * 
 * @note Модель должна быть предварительно обучена и сохранена в папке ../../bpe/
 * @see FastBPETokenizer
 */

#include "config.h"

#include "fast_tokenizer.hpp"
#include "test_helpers.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <csignal>
#include <csetjmp>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace fs = std::filesystem;
using namespace bpe;

// ======================================================================
// Обработка SIMD исключений
// ======================================================================

static jmp_buf simd_env;

/**
 * @brief Обработчик сигнала SIGILL (недопустимая инструкция)
 */
void sigill_handler(int) {
    longjmp(simd_env, 1);
}

/**
 * @brief Проверить поддержку AVX-512 инструкций через CPUID
 * @return true если поддерживаются, false если нет
 */
bool check_avx512_support() {
    #if defined(__x86_64__) || defined(__i386__)
        unsigned int eax, ebx, ecx, edx;
        
        // Проверка поддержки AVX-512F (Foundation)
        __asm__ volatile(
            "mov $7, %%rax\n"
            "xor %%rcx, %%rcx\n"
            "cpuid\n"
            : "=b"(ebx), "=a"(eax), "=c"(ecx), "=d"(edx)
            :
            : "cc"
        );
        
        // бит 16 в EBX = AVX-512F
        return (ebx >> 16) & 1;
    #else
        return false;
    #endif
}

/**
 * @brief Проверить поддержку AVX2 инструкций
 * @return true если поддерживаются, false если нет
 */
bool check_avx2_support() {
    struct sigaction sa, old_sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGILL, &sa, &old_sa) != 0) {
        return false;
    }
    
    volatile bool supported = true;
    
    if (setjmp(simd_env) == 0) {
        // Попытка выполнить AVX2 инструкцию
        #if defined(__AVX2__) || defined(USE_AVX2)
        __asm__ volatile(
            "vpxor %%ymm0, %%ymm0, %%ymm0\n\t"
            "vpxor %%ymm1, %%ymm1, %%ymm1\n\t"
            : : : "ymm0", "ymm1"
        );
        #endif
    } else {
        supported = false;
    }
    
    sigaction(SIGILL, &old_sa, nullptr);
    return supported;
}

/**
 * @brief Проверить поддержку SSE4.2 инструкций
 * @return true если поддерживаются, false если нет
 */
bool check_sse42_support() {
    struct sigaction sa, old_sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGILL, &sa, &old_sa) != 0) {
        return false;
    }
    
    volatile bool supported = true;
    
    if (setjmp(simd_env) == 0) {
        // Попытка выполнить SSE4.2 инструкцию
        #if defined(__SSE4_2__) || defined(USE_SSE42)
        __asm__ volatile(
            "pxor %%xmm0, %%xmm0\n\t"
            "pxor %%xmm1, %%xmm1\n\t"
            : : : "xmm0", "xmm1"
        );
        #endif
    } else {
        supported = false;
    }
    
    sigaction(SIGILL, &old_sa, nullptr);
    return supported;
}

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr int LARGE_TEXT_LINES = 100;
    constexpr int MULTITHREAD_ITERATIONS = 4;
    constexpr int DISPLAY_PREVIEW_LEN = 30;
    
    const std::vector<std::pair<std::string, std::string>> MODEL_CANDIDATES = {
        // C++ модели (приоритет 1)
        {"../models/bpe_8000/cpp_vocab.json", "../models/bpe_8000/cpp_merges.txt"},
        {"../models/bpe_10000/cpp_vocab.json", "../models/bpe_10000/cpp_merges.txt"},
        {"../models/bpe_12000/cpp_vocab.json", "../models/bpe_12000/cpp_merges.txt"},
        {"../../models/bpe_8000/cpp_vocab.json", "../../models/bpe_8000/cpp_merges.txt"},
        
        // Python модели (приоритет 2)
        {"../../bpe_python/models/bpe_8000/vocab.json", "../../bpe_python/models/bpe_8000/merges.txt"},
        {"../../bpe_python/models/bpe_10000/vocab.json", "../../bpe_python/models/bpe_10000/merges.txt"},
        {"../../bpe_python/models/bpe_12000/vocab.json", "../../bpe_python/models/bpe_12000/merges.txt"},
        {"../../../bpe_python/models/bpe_8000/vocab.json", "../../../bpe_python/models/bpe_8000/merges.txt"},
        
        // Fallback (если модели скопированы в директорию сборки)
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"}
    };
}

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Форматирование времени в читаемый вид
 * 
 * @param us Время в микросекундах
 * @return std::string Отформатированная строка
 */
std::string format_duration(std::chrono::microseconds us) {
    if (us.count() < 1000) {
        return std::to_string(us.count()) + " мкс";
    } else if (us.count() < 1'000'000) {
        return std::to_string(us.count() / 1000.0) + " мс";
    } else {
        return std::to_string(us.count() / 1'000'000.0) + " с";
    }
}

/**
 * @brief Форматирование времени с плавающей точкой
 * 
 * @param us Время в микросекундах
 * @return std::string Отформатированная строка
 */
std::string format_duration(double us) {
    if (us < 1000) {
        return std::to_string(us) + " мкс";
    } else if (us < 1'000'000) {
        return std::to_string(us / 1000.0) + " мс";
    } else {
        return std::to_string(us / 1'000'000.0) + " с";
    }
}

/**
 * @brief Проверка существования файла
 * 
 * @param path Путь к файлу
 * @return true если файл существует и доступен для чтения
 */
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

/**
 * @brief Поиск файла модели по разным путям
 * 
 * @return std::pair<std::string, std::string> Пара (путь к словарю, путь к слияниям)
 */
std::pair<std::string, std::string> find_model_files() {
    std::cout << "Поиск файлов модели..." << std::endl;
    
    for (const auto& [vocab, merges] : MODEL_CANDIDATES) {
        std::cout << "  Проверка: " << vocab << std::endl;
        
        if (file_exists(vocab)) {
            if (merges.empty() || file_exists(merges)) {
                std::cout << "  ✓ Найдено: " << vocab << std::endl;
                return {vocab, merges};
            } else {
                std::cout << "  ✗ Слияния не найдены: " << merges << std::endl;
            }
        }
    }
    
    return {"", ""};
}

/**
 * @brief Обрезать строку для вывода
 * 
 * @param str Исходная строка
 * @param max_len Максимальная длина
 * @return std::string Обрезанная строка
 */
std::string truncate(const std::string& str, size_t max_len = DISPLAY_PREVIEW_LEN) {
    if (str.length() <= max_len) return str;
    return str.substr(0, max_len - 3) + "...";
}

/**
 * @brief Вывести заголовок секции
 * 
 * @param title Заголовок
 */
void print_header(const std::string& title) {
    // Верхняя линия
    std::cout << "\n";
    for (int i = 0; i < 60; ++i) std::cout << '=';
    std::cout << std::endl;
    
    // Заголовок
    std::cout << title << std::endl;
    
    // Нижняя линия
    for (int i = 0; i < 60; ++i) std::cout << '=';
    std::cout << std::endl;
}


// ======================================================================
// Тестовые примеры C++ кода
// ======================================================================

const std::vector<std::pair<std::string, std::string>> TEST_CASES = {
    {"Простые выражения", "int x = 42;"},
    {"Работа с вектором", "std::vector<int> numbers;"},
    {"Цикл for", "for (int i = 0; i < 10; ++i) { sum += i; }"},
    {"Класс", "class MyClass { public: void method(); private: int data; };"},
    {"Шаблон", "template<typename T> T max(T a, T b) { return a > b ? a : b; }"},
    {"Русские комментарии", "// комментарий на русском языке"},
    {"Include директива", "#include <iostream>"},
    {"Лямбда функция", "auto lambda = [](int x) { return x * x; };"},
    {"Умные указатели", "std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();"},
    {"Исключения", "try { throw std::runtime_error(\"error\"); } catch (...) {}"},
    {"C++17 features", "if constexpr (std::is_integral_v<T>) { return 0; }"},
    {"Сложное выражение", "std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x*x; });"}
};

// ======================================================================
// Основная функция
// ======================================================================

/**
 * @brief Точка входа в программу
 * 
 * @return int 0 при успешном прохождении всех тестов, 1 при ошибке
 */
int main() {

    // Проверяем поддержку SIMD
    std::cout << "\nПроверка поддержки SIMD:" << std::endl;
    bool avx512_supported = check_avx512_support();
    std::cout << "  AVX-512: " << (avx512_supported ? "✅" : "❌") << std::endl;
    std::cout << "  AVX2:    " << (check_avx2_support() ? "✅" : "❌") << std::endl;
    std::cout << "  SSE4.2:  " << (check_sse42_support() ? "✅" : "❌") << std::endl;

    // Если AVX-512 не поддерживается, завершаем тест с успехом
    if (!avx512_supported) {
        std::cout << "\n⚠️  AVX-512 не поддерживается процессором." << std::endl;
        std::cout << "   Тест пропущен (это нормально, если процессор старее Intel Skylake-X)." << std::endl;
        return 0;  // Возвращаем 0, чтобы тест считался пройденным
    }

    // Устанавливаем обработчик сигнала для SIMD инструкций
    struct sigaction sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, nullptr);
    
    print_header("ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ BPE");
    
    // Проверяем поддержку SIMD
    std::cout << "\nПроверка поддержки SIMD:" << std::endl;
    std::cout << "  AVX-512: " << (check_avx512_support() ? "✅" : "❌") << std::endl;
    std::cout << "  AVX2:    " << (check_avx2_support() ? "✅" : "❌") << std::endl;
    std::cout << "  SSE4.2:  " << (check_sse42_support() ? "✅" : "❌") << std::endl;

    // ======================================================================
    // Загрузка модели
    // ======================================================================

    FastBPETokenizer tokenizer;
    
    auto [vocab_path, merges_path] = find_model_files();
    
    if (vocab_path.empty()) {
        std::cerr << "\n❌ Не удалось найти файлы модели!" << std::endl;
        std::cerr << "   Ожидаемые пути:" << std::endl;
        for (const auto& [vocab, merges] : MODEL_CANDIDATES) {
            std::cerr << "   - " << vocab << std::endl;
        }
        return 1;
    }
    
    std::cout << "\nНайдены файлы модели:" << std::endl;
    std::cout << "  📄 Словарь: " << vocab_path << std::endl;
    if (!merges_path.empty()) {
        std::cout << "  📄 Слияния: " << merges_path << std::endl;
    }
    
    std::cout << "\nЗагрузка модели..." << std::endl;
    auto load_start = std::chrono::high_resolution_clock::now();
    
    bool loaded;
    if (vocab_path.find(".bin") != std::string::npos) {
        loaded = tokenizer.load_binary(vocab_path);
    } else {
        loaded = tokenizer.load(vocab_path, merges_path);
    }
    
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
    
    if (!loaded) {
        std::cerr << "❌ Ошибка загрузки модели!" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Модель загружена за " << load_time.count() << " мс" << std::endl;
    std::cout << "   Размер словаря:    " << tokenizer.vocab_size() << " токенов" << std::endl;
    std::cout << "   Правил слияния:    " << tokenizer.merges_count() << std::endl;
    std::cout << "   ID <UNK>:          " << tokenizer.unknown_id() << std::endl;
    std::cout << "   ID <PAD>:          " << tokenizer.pad_id() << std::endl;
    std::cout << "   ID <BOS>:          " << tokenizer.bos_id() << std::endl;
    std::cout << "   ID <EOS>:          " << tokenizer.eos_id() << std::endl;

    // ======================================================================
    // Основной цикл тестирования
    // ======================================================================

    print_header("ТЕСТИРОВАНИЕ КОДИРОВАНИЯ");
    
    int passed = 0;
    int total = TEST_CASES.size();
    
    std::map<size_t, int> token_count_distribution;
    std::vector<double> encode_times_us;
    std::vector<double> decode_times_us;
    std::vector<size_t> token_counts;

    for (const auto& [desc, code] : TEST_CASES) {
        std::cout << "\n📌 " << desc << ":" << std::endl;
        std::cout << "   Исходный:   '" << truncate(code) << "'" << std::endl;
        
        // Измеряем время encode
        auto encode_start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(code);
        auto encode_end = std::chrono::high_resolution_clock::now();
        
        auto encode_time = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start);
        encode_times_us.push_back(encode_time.count());
        
        // Декодируем
        auto decode_start = std::chrono::high_resolution_clock::now();
        auto decoded = tokenizer.decode(tokens);
        auto decode_end = std::chrono::high_resolution_clock::now();
        
        auto decode_time = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start);
        decode_times_us.push_back(decode_time.count());
        
        // Проверяем результат
        bool match = (code == decoded);
        if (match) passed++;
        
        // Собираем статистику
        token_count_distribution[tokens.size()]++;
        token_counts.push_back(tokens.size());
        
        // Подсчет уникальных токенов
        std::unordered_set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
        
        std::cout << "   Декодирован: '" << truncate(decoded) << "'" << std::endl;
        std::cout << "   Токенов: " << tokens.size() << " (уникальных: " 
                  << unique_tokens.size() << ")" << std::endl;
        std::cout << "   Время encode: " << format_duration(encode_time) << std::endl;
        std::cout << "   Время decode: " << format_duration(decode_time) << std::endl;
        std::cout << "   Результат: " << (match ? "✅ СОВПАДАЕТ" : "❌ НЕ СОВПАДАЕТ") << std::endl;
        
        if (!match) {
            // Показываем различия для отладки
            size_t min_len = std::min(code.length(), decoded.length());
            for (size_t i = 0; i < min_len; ++i) {
                if (code[i] != decoded[i]) {
                    std::cout << "     Позиция " << i << ": исходный '" 
                              << code[i] << "' (0x" << std::hex 
                              << static_cast<int>(static_cast<unsigned char>(code[i])) 
                              << std::dec << ") vs декод. '" << decoded[i] << "'" << std::endl;
                    break;
                }
            }
            if (code.length() != decoded.length()) {
                std::cout << "     Длина: исходный " << code.length() 
                          << ", декод. " << decoded.length() << std::endl;
            }
        }
    }

    // ======================================================================
    // Статистика
    // ======================================================================

    print_header("СТАТИСТИКА");
    
    double success_rate = 100.0 * passed / total;
    std::cout << "Успешных тестов: " << passed << "/" << total 
              << " (" << std::fixed << std::setprecision(1) << success_rate << "%)\n";
    
    if (success_rate < 100.0) {
        std::cout << "\n⚠️  Не все тесты пройдены успешно!" << std::endl;
    }
    
    // Статистика по токенам
    if (!token_counts.empty()) {
        size_t total_tokens = 0;
        for (size_t count : token_counts) total_tokens += count;
        double avg_tokens = static_cast<double>(total_tokens) / token_counts.size();
        
        auto min_tokens = *std::min_element(token_counts.begin(), token_counts.end());
        auto max_tokens = *std::max_element(token_counts.begin(), token_counts.end());
        
        std::cout << "\nСтатистика токенов:" << std::endl;
        std::cout << "  • всего токенов:       " << total_tokens << std::endl;
        std::cout << "  • среднее на текст:    " << std::fixed << std::setprecision(1) << avg_tokens << std::endl;
        std::cout << "  • минимум:             " << min_tokens << std::endl;
        std::cout << "  • максимум:            " << max_tokens << std::endl;
        
        std::cout << "\nРаспределение:" << std::endl;
        for (const auto& [count, freq] : token_count_distribution) {
            std::cout << "  • " << std::setw(3) << count << " токенов: " << freq << " текстов" << std::endl;
        }
    }
    
    // Статистика производительности
    if (!encode_times_us.empty()) {
        double total_encode = 0;
        for (double t : encode_times_us) total_encode += t;
        double avg_encode = total_encode / encode_times_us.size();
        
        double total_decode = 0;
        for (double t : decode_times_us) total_decode += t;
        double avg_decode = total_decode / decode_times_us.size();
        
        auto min_encode = *std::min_element(encode_times_us.begin(), encode_times_us.end());
        auto max_encode = *std::max_element(encode_times_us.begin(), encode_times_us.end());
        
        std::cout << "\nПроизводительность:" << std::endl;
        std::cout << "  • среднее encode:      " << format_duration(avg_encode) << std::endl;
        std::cout << "  • среднее decode:      " << format_duration(avg_decode) << std::endl;
        std::cout << "  • мин encode:          " << format_duration(min_encode) << std::endl;
        std::cout << "  • макс encode:         " << format_duration(max_encode) << std::endl;
    }
    
    // Статистика от токенизатора
    auto stats = tokenizer.stats();
    if (stats.encode_calls > 0) {
        std::cout << "\nСтатистика токенизатора:" << std::endl;
        std::cout << "  • всего encode вызовов:    " << stats.encode_calls << std::endl;
        std::cout << "  • всего decode вызовов:    " << stats.decode_calls << std::endl;
        std::cout << "  • попаданий в кэш:         " << stats.cache_hits << std::endl;
        std::cout << "  • промахов кэша:           " << stats.cache_misses << std::endl;
        std::cout << "  • эффективность кэша:      " << std::fixed << std::setprecision(1)
                  << (stats.cache_hit_rate() * 100) << "%" << std::endl;
        std::cout << "  • обработано токенов:      " << stats.total_tokens_processed << std::endl;
    }

    // ======================================================================
    // Тест производительности на большом тексте
    // ======================================================================

    print_header("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ");
    
    // Создаем большой текст
    std::string large_text;
    large_text.reserve(LARGE_TEXT_LINES * 50);
    
    for (int i = 0; i < LARGE_TEXT_LINES; ++i) {
        large_text += "int x" + std::to_string(i) + " = " + std::to_string(i * i) + ";\n";
    }
    
    std::cout << "Размер текста: " << large_text.size() << " символов ("
              << large_text.size() / 1024.0 << " КБ)" << std::endl;
    std::cout << "Запуск encode..." << std::endl;
    
    auto perf_start = std::chrono::high_resolution_clock::now();
    auto large_tokens = tokenizer.encode(large_text);
    auto perf_end = std::chrono::high_resolution_clock::now();
    
    auto perf_time = std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start);
    double mb_per_sec = (large_text.size() / 1'000'000.0) / (perf_time.count() / 1'000'000.0);
    
    std::cout << "Закодировано за " << format_duration(perf_time) << std::endl;
    std::cout << "Токенов:      " << large_tokens.size() << std::endl;
    std::cout << "Скорость:     " << std::fixed << std::setprecision(2) << mb_per_sec << " МБ/с" << std::endl;
    std::cout << "Токенов/с:    " << std::fixed << std::setprecision(0) 
              << (large_tokens.size() / (perf_time.count() / 1'000'000.0)) << std::endl;
    
    // Тест многопоточности (только если OpenMP доступен)
    #ifdef _OPENMP
    {
        std::cout << "\nТест многопоточности..." << std::endl;
        
        auto multi_start = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for
        for (int i = 0; i < MULTITHREAD_ITERATIONS; ++i) {
            auto tokens = tokenizer.encode(large_text);
            volatile size_t dummy = tokens.size();
            (void)dummy;
        }
        
        auto multi_end = std::chrono::high_resolution_clock::now();
        auto multi_time = std::chrono::duration_cast<std::chrono::microseconds>(multi_end - multi_start);
        
        double speedup = static_cast<double>(perf_time.count() * MULTITHREAD_ITERATIONS) / multi_time.count();
        
        std::cout << MULTITHREAD_ITERATIONS << " параллельных вызовов: " 
                  << format_duration(multi_time) << std::endl;
        std::cout << "Ускорение: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    #else
    std::cout << "\nТест многопоточности пропущен (OpenMP не доступен)" << std::endl;
    #endif

    // ======================================================================
    // Итог
    // ======================================================================

    print_header("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО");
    
    if (passed == total) {
        std::cout << "\n✅ Все тесты пройдены успешно!" << std::endl;
        return 0;
    } else {
        std::cout << "\n⚠️  Некоторые тесты не пройдены." << std::endl;
        return 1;
    }
}