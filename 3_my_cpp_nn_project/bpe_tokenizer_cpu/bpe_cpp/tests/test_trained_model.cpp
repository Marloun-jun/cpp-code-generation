/**
 * @file test_trained_model.cpp
 * @brief Тестирование обученной модели BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.6.0
 * 
 * @details Комплексное тестирование предварительно обученной модели:
 *          - Проверка кодирования/декодирования для разных языков
 *          - Анализ производительности
 *          - Валидация UTF-8 корректности
 *          - Статистика по токенам
 * 
 *          **Поддерживаемые языки:**
 *          - Английский (ASCII)
 *          - Русский (кириллица)
 *          - Эмодзи (🔥, 😂, 🚀)
 *          - Смешанные тексты
 * 
 * @note Все тесты должны проходить с 100% успехом
 * @see FastBPETokenizer
 */

#include "config.h"

#include "fast_tokenizer.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace fs = std::filesystem;
using namespace bpe;

// ============================================================================
// Обработка SIMD исключений
// ============================================================================

static jmp_buf simd_env;

void sigill_handler(int) {
    longjmp(simd_env, 1);
}

bool check_avx512_support() {
    #if defined(__x86_64__) || defined(__i386__)
        unsigned int eax, ebx, ecx, edx;
        __asm__ volatile(
            "mov $7, %%rax\n"
            "xor %%rcx, %%rcx\n"
            "cpuid\n"
            : "=b"(ebx), "=a"(eax), "=c"(ecx), "=d"(edx)
            :
            : "cc"
        );
        return (ebx >> 16) & 1;
    #else
        return false;
    #endif
}

bool check_avx2_support() {
    struct sigaction sa, old_sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGILL, &sa, &old_sa) != 0) return false;
    
    volatile bool supported = true;
    if (setjmp(simd_env) == 0) {
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

bool check_sse42_support() {
    struct sigaction sa, old_sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    
    if (sigaction(SIGILL, &sa, &old_sa) != 0) return false;
    
    volatile bool supported = true;
    if (setjmp(simd_env) == 0) {
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

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    constexpr int LARGE_TEXT_LINES = 100;
    constexpr int MULTITHREAD_ITERATIONS = 4;
    constexpr int DISPLAY_PREVIEW_LEN = 30;
    
    const std::vector<std::pair<std::string, std::string>> MODEL_CANDIDATES = {
        {"../models/bpe_8000/cpp_vocab.json", "../models/bpe_8000/cpp_merges.txt"},
        {"../models/bpe_10000/cpp_vocab.json", "../models/bpe_10000/cpp_merges.txt"},
        {"../models/bpe_12000/cpp_vocab.json", "../models/bpe_12000/cpp_merges.txt"},
        {"../../models/bpe_8000/cpp_vocab.json", "../../models/bpe_8000/cpp_merges.txt"},
    };
    
    // Цвета для вывода
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ============================================================================
// Вспомогательные функции для UTF-8 диагностики
// ============================================================================

bool is_valid_utf8(const std::string& str) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    size_t len = str.length();
    size_t i = 0;
    
    while (i < len) {
        unsigned char c = bytes[i];
        
        if (c <= 0x7F) {
            i++;
            continue;
        }
        
        int char_len;
        if ((c & 0xE0) == 0xC0) {
            char_len = 2;
            if (c < 0xC2) return false;
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4;
            if (c > 0xF4) return false;
        } else {
            return false;
        }
        
        if (i + char_len > len) return false;
        
        for (int j = 1; j < char_len; ++j) {
            if ((bytes[i + j] & 0xC0) != 0x80) return false;
        }
        
        i += char_len;
    }
    
    return true;
}

void print_bytes_hex(const std::string& str, size_t max_bytes = 20) {
    std::cout << "[";
    for (size_t i = 0; i < std::min(max_bytes, str.length()); ++i) {
        if (i > 0) std::cout << " ";
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(static_cast<unsigned char>(str[i]));
    }
    std::cout << std::dec;
    if (str.length() > max_bytes) std::cout << " ...";
    std::cout << "]";
}

// ============================================================================
// Вспомогательные функции (форматирование, поиск файлов)
// ============================================================================

std::string format_duration(std::chrono::microseconds us) {
    if (us.count() < 1000) {
        return std::to_string(us.count()) + " мкс";
    } else if (us.count() < 1'000'000) {
        return std::to_string(us.count() / 1000.0) + " мс";
    } else {
        return std::to_string(us.count() / 1'000'000.0) + " с";
    }
}

std::string format_duration(double us) {
    if (us < 1000) {
        return std::to_string(us) + " мкс";
    } else if (us < 1'000'000) {
        return std::to_string(us / 1000.0) + " мс";
    } else {
        return std::to_string(us / 1'000'000.0) + " с";
    }
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::pair<std::string, std::string> find_model_files() {
    std::cout << CYAN << "Поиск файлов модели..." << RESET << std::endl;
    
    for (const auto& [vocab, merges] : MODEL_CANDIDATES) {
        std::cout << "Проверка: " << vocab << std::endl;
        
        if (file_exists(vocab)) {
            if (merges.empty() || file_exists(merges)) {
                std::cout << GREEN << "Найдено:  " << vocab << RESET << std::endl;
                return {vocab, merges};
            } else {
                std::cout << YELLOW << "Слияния не найдены: " << merges << RESET << std::endl;
            }
        }
    }
    
    return {"", ""};
}

std::string truncate(const std::string& str, size_t max_len = DISPLAY_PREVIEW_LEN) {
    if (str.length() <= max_len) return str;
    return str.substr(0, max_len - 3) + "...";
}

void print_header(const std::string& title) {
    std::cout << "\n" << BOLD;
    for (int i = 0; i < 60; ++i) std::cout << '=';
    std::cout << RESET << std::endl;
    std::cout << BOLD << title << RESET << std::endl;
    std::cout << BOLD;
    for (int i = 0; i < 60; ++i) std::cout << '=';
    std::cout << RESET << std::endl;
}

// ============================================================================
// Тестовые примеры
// ============================================================================

const std::vector<std::pair<std::string, std::string>> TEST_CASES = {
    {"Простые выражения", "int x = 42;"},
    {"Работа с вектором", "std::vector<int> numbers;"},
    {"Цикл for", "for (int i = 0; i < 10; ++i) { sum += i; }"},
    {"Класс", "class MyClass { public: void method(); private: int data; };"},
    {"Шаблон", "template<typename T> T max(T a, T b) { return a > b ? a : b; }"},
    {"Русские комментарии", "// комментарий на русском языке"},
    {"Эмодзи", "🔥, 😂, 🚀"},
    {"Include директива", "#include <iostream>"},
    {"Лямбда функция", "auto lambda = [](int x) { return x * x; };"},
    {"Умные указатели", "std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();"},
    {"Исключения", "try { throw std::runtime_error(\"error\"); } catch (...) {}"},
    {"C++17 features", "if constexpr (std::is_integral_v<T>) { return 0; }"},
    {"Сложное выражение", "std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x*x; });"}
};

// ============================================================================
// Основная функция
// ============================================================================

int main() {
    // Инициализация и проверка SIMD
    struct sigaction sa;
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, nullptr);
    
    print_header("ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ BPE");
    
    std::cout << "\n" << CYAN << "Проверка поддержки SIMD:" << RESET << std::endl;
    std::cout << "- AVX-512: " << (check_avx512_support() ? GREEN + "да" : RED + "нет") << RESET << std::endl;
    std::cout << "- AVX2:    " << (check_avx2_support() ? GREEN + "да" : RED + "нет") << RESET << std::endl;
    std::cout << "- SSE4.2:  " << (check_sse42_support() ? GREEN + "да" : RED + "нет") << RESET << std::endl;

    // Загрузка модели
    FastBPETokenizer tokenizer;
    
    auto [vocab_path, merges_path] = find_model_files();
    
    if (vocab_path.empty()) {
        std::cerr << RED << "\nНе удалось найти файлы модели!" << RESET << std::endl;
        return 1;
    }
    
    std::cout << "\n" << CYAN << "Найдены файлы модели:" << RESET << std::endl;
    std::cout << "Словарь: " << vocab_path << std::endl;
    if (!merges_path.empty()) {
        std::cout << "Слияния: " << merges_path << std::endl;
    }
    
    std::cout << "\n" << CYAN << "Загрузка модели..." << RESET << std::endl;
    auto load_start = std::chrono::high_resolution_clock::now();
    
    bool loaded = tokenizer.load(vocab_path, merges_path);
    
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
    
    if (!loaded) {
        std::cerr << RED << "Ошибка загрузки модели!" << RESET << std::endl;
        return 1;
    }
    
    std::cout << GREEN << "Модель загружена за " << load_time.count() << " мс" << RESET << std::endl;
    std::cout << "- Размер словаря: " << tokenizer.vocab_size() << " токенов" << std::endl;
    std::cout << "- Правил слияния: " << tokenizer.merges_count() << std::endl;
    std::cout << "- ID <UNK>:       " << tokenizer.unknown_id() << std::endl;
    std::cout << "- ID <PAD>:       " << tokenizer.pad_id() << std::endl;
    std::cout << "- ID <BOS>:       " << tokenizer.bos_id() << std::endl;
    std::cout << "- ID <EOS>:       " << tokenizer.eos_id() << std::endl;

    // Основной цикл тестирования
    print_header("ТЕСТИРОВАНИЕ КОДИРОВАНИЯ");
    
    int passed = 0;
    int total = TEST_CASES.size();
    
    std::map<size_t, int> token_count_distribution;
    std::vector<double> encode_times_us;
    std::vector<double> decode_times_us;
    std::vector<size_t> token_counts;

    for (const auto& [desc, code] : TEST_CASES) {
        std::cout << "\n" << BOLD << desc << ":" << RESET << std::endl;
        std::cout << "Исходный:   '" << truncate(code) << "'" << std::endl;
        
        auto encode_start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(code);
        auto encode_end = std::chrono::high_resolution_clock::now();
        
        auto encode_time = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start);
        encode_times_us.push_back(encode_time.count());
        
        auto decode_start = std::chrono::high_resolution_clock::now();
        auto decoded = tokenizer.decode(tokens);
        auto decode_end = std::chrono::high_resolution_clock::now();
        
        auto decode_time = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start);
        decode_times_us.push_back(decode_time.count());
        
        bool match = (code == decoded);
        if (match) passed++;
        
        token_count_distribution[tokens.size()]++;
        token_counts.push_back(tokens.size());
        
        std::unordered_set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
        
        std::cout << "- Декодирован: '" << truncate(decoded) << "'" << std::endl;
        std::cout << "- Токенов:      " << tokens.size() << " (уникальных: " 
                  << unique_tokens.size() << ")" << std::endl;
        std::cout << "- Время encode: " << format_duration(encode_time) << std::endl;
        std::cout << "- Время decode: " << format_duration(decode_time) << std::endl;
        std::cout << "- Результат:    " << (match ? GREEN + "СОВПАДАЕТ" : RED + "НЕ СОВПАДАЕТ") << RESET << std::endl;
        
        if (!match) {
            std::cout << YELLOW << "\nДИАГНОСТИКА UTF-8:" << RESET << std::endl;
            
            bool original_valid = is_valid_utf8(code);
            bool decoded_valid = is_valid_utf8(decoded);
            
            std::cout << "- Исходный байты:  ";
            print_bytes_hex(code, 20);
            std::cout << std::endl;
            
            std::cout << "- Декодированный:  ";
            print_bytes_hex(decoded, 20);
            std::cout << std::endl;
            
            std::cout << "- UTF-8 валидность исходного: " 
                      << (original_valid ? GREEN + "да" : RED + "нет") << RESET << std::endl;
            std::cout << "- UTF-8 валидность декод.:    " 
                      << (decoded_valid ? GREEN + "да" : RED + "нет") << RESET << std::endl;
            
            if (!decoded_valid && original_valid) {
                std::cout << RED << "Декодированный текст не является валидным UTF-8!" << RESET << std::endl;
            }
            
            size_t min_len = std::min(code.length(), decoded.length());
            for (size_t i = 0; i < min_len; ++i) {
                if (code[i] != decoded[i]) {
                    unsigned char orig_byte = static_cast<unsigned char>(code[i]);
                    unsigned char dec_byte = static_cast<unsigned char>(decoded[i]);
                    
                    std::cout << YELLOW << "Позиция " << i << ": "
                              << "0x" << std::hex << std::setw(2) << std::setfill('0') 
                              << static_cast<int>(orig_byte) << " vs 0x" 
                              << static_cast<int>(dec_byte) << std::dec;
                    
                    if (orig_byte < 0x80) std::cout << " ('" << code[i] << "')";
                    if (dec_byte < 0x80) std::cout << " ('" << decoded[i] << "')";
                    
                    std::cout << RESET << std::endl;
                    break;
                }
            }
        }
    }

    // Статистика
    print_header("СТАТИСТИКА");
    
    double success_rate = 100.0 * passed / total;
    std::cout << "Успешных тестов: " << passed << "/" << total 
              << " (" << std::fixed << std::setprecision(1) << success_rate << "%)\n";
    
    if (success_rate < 100.0) {
        std::cout << YELLOW << "\nНе все тесты пройдены успешно!" << RESET << std::endl;
    } else {
        std::cout << GREEN << "\nВсе тесты пройдены успешно!" << RESET << std::endl;
    }
    
    // Статистика по токенам
    if (!token_counts.empty()) {
        size_t total_tokens = 0;
        for (size_t count : token_counts) total_tokens += count;
        double avg_tokens = static_cast<double>(total_tokens) / token_counts.size();
        
        auto min_tokens = *std::min_element(token_counts.begin(), token_counts.end());
        auto max_tokens = *std::max_element(token_counts.begin(), token_counts.end());
        
        std::cout << "\n" << CYAN << "Статистика токенов:" << RESET << std::endl;
        std::cout << "- Всего токенов:    " << total_tokens << std::endl;
        std::cout << "- Среднее на текст: " << std::fixed << std::setprecision(1) << avg_tokens << std::endl;
        std::cout << "- Минимум:          " << min_tokens << std::endl;
        std::cout << "- Максимум:         " << max_tokens << std::endl;
        
        std::cout << "\nРаспределение:" << std::endl;
        for (const auto& [count, freq] : token_count_distribution) {
            std::cout << "- " << std::setw(3) << count << " токенов: " << freq << " текстов" << std::endl;
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
        
        std::cout << "\n" << CYAN << "Производительность:" << RESET << std::endl;
        std::cout << "- Среднее encode: " << format_duration(avg_encode) << std::endl;
        std::cout << "- Среднее decode: " << format_duration(avg_decode) << std::endl;
        std::cout << "- Мин encode:     " << format_duration(min_encode) << std::endl;
        std::cout << "- Макс encode:    " << format_duration(max_encode) << std::endl;
    }
    
    // Статистика токенизатора
    auto stats = tokenizer.stats();
    if (stats.encode_calls > 0) {
        std::cout << "\n" << CYAN << "Статистика токенизатора:" << RESET << std::endl;
        std::cout << "- Всего encode вызовов: " << stats.encode_calls << std::endl;
        std::cout << "- Всего decode вызовов: " << stats.decode_calls << std::endl;
        std::cout << "- Попаданий в кэш:      " << stats.cache_hits << std::endl;
        std::cout << "- Промахов кэша:        " << stats.cache_misses << std::endl;
        std::cout << "- Эффективность кэша:   " << std::fixed << std::setprecision(1)
                  << (stats.cache_hit_rate() * 100) << "%" << std::endl;
        std::cout << "- Обработано токенов:   " << stats.total_tokens_processed << std::endl;
    }

    // Тест производительности на большом тексте
    print_header("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ");
    
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
    std::cout << "Токенов:   " << large_tokens.size() << std::endl;
    std::cout << "Скорость:  " << std::fixed << std::setprecision(2) << mb_per_sec << " МБ/с" << std::endl;
    std::cout << "Токенов/с: " << std::fixed << std::setprecision(0) 
              << (large_tokens.size() / (perf_time.count() / 1'000'000.0)) << std::endl;
    
    #ifdef _OPENMP
    {
        std::cout << "\n" << CYAN << "Тест многопоточности..." << RESET << std::endl;
        
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
    std::cout << YELLOW << "\nТест многопоточности пропущен (OpenMP не доступен)!" << RESET << std::endl;
    #endif

    // Итог
    print_header("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО");
    
    return (passed == total) ? 0 : 1;
}