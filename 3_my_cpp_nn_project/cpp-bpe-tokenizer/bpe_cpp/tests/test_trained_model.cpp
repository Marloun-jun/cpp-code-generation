/**
 * @file test_trained_model.cpp
 * @brief Тестирование обученной модели BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Загружает обученную модель и проверяет её работу на различных
 *          примерах C++ кода. Выводит статистику и результаты сравнения.
 * 
 * @note Модель должна быть предварительно обучена и сохранена в папке ../../bpe/
 * @see FastBPETokenizer
 */

#include "fast_tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <set>

/**
 * @brief Форматирование времени в читаемый вид
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
 * @brief Проверка существования файла
 */
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

/**
 * @brief Поиск файла модели по разным путям
 */
std::pair<std::string, std::string> find_model_files() {
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"../../bpe/model_trained.bin", ""},  // Бинарный
        {"../../bpe/vocab_trained.json", "../../bpe/merges_trained.txt"},
        {"../bpe/vocab_trained.json", "../bpe/merges_trained.txt"},
        {"bpe/vocab_trained.json", "bpe/merges_trained.txt"},
        {"models/cpp_vocab.json", "models/cpp_merges.txt"}
    };
    
    for (const auto& [vocab, merges] : candidates) {
        if (vocab.find(".bin") != std::string::npos) {
            if (file_exists(vocab)) {
                return {vocab, ""};
            }
        } else if (file_exists(vocab) && file_exists(merges)) {
            return {vocab, merges};
        }
    }
    
    return {"", ""};
}

int main() {
    std::cout << "========================================\n";
    std::cout << "ТЕСТИРОВАНИЕ ОБУЧЕННОЙ МОДЕЛИ BPE\n";
    std::cout << "========================================\n\n";

    // Загружаем модель
    bpe::TokenizerConfig config;
    config.byte_level = true;
    config.enable_cache = true;
    config.enable_profiling = true;  // Включаем сбор статистики
    
    bpe::FastBPETokenizer tokenizer(config);
    
    std::cout << "Поиск файлов модели...\n";
    auto [vocab_path, merges_path] = find_model_files();
    
    if (vocab_path.empty()) {
        std::cerr << "Не удалось найти файлы модели!\n";
        std::cerr << "   Ожидаемые пути:\n";
        std::cerr << "   - ../../bpe/model_trained.bin\n";
        std::cerr << "   - ../../bpe/vocab_trained.json + merges_trained.txt\n";
        return 1;
    }
    
    std::cout << "Найдены файлы модели:\n";
    std::cout << "   Словарь: " << vocab_path << "\n";
    if (!merges_path.empty()) {
        std::cout << "   Слияния: " << merges_path << "\n";
    }
    std::cout << std::endl;
    
    std::cout << "Загрузка модели...\n";
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
        std::cerr << "Ошибка загрузки модели!\n";
        return 1;
    }
    
    std::cout << "Модель загружена за " << load_time.count() << " мс\n";
    std::cout << "Размер словаря: " << tokenizer.vocab_size() << " токенов\n";
    std::cout << "Правил слияния: " << tokenizer.merges_count() << "\n";
    std::cout << "ID <UNK>: " << tokenizer.unknown_id() << "\n";
    std::cout << std::endl;
    
    // Тестовые примеры
    std::vector<std::pair<std::string, std::string>> test_cases = {
        {"Простые выражения", "int x = 42;"},
        {"Работа с вектором", "std::vector<int> numbers;"},
        {"Цикл for", "for (int i = 0; i < 10; ++i)"},
        {"Класс", "class MyClass { public: void method(); };"},
        {"Шаблон", "template<typename T> T max(T a, T b)"},
        {"Русские комментарии", "// комментарий на русском языке"},
        {"Include директива", "#include <iostream>"},
        {"Лямбда функция", "auto result = calculate(42);"},
        {"Умные указатели", "std::unique_ptr<MyClass> ptr;"},
        {"Исключения", "try { throw std::runtime_error(\"error\"); } catch (...) {}"},
        {"C++17 features", "if constexpr (std::is_integral_v<T>)"},
        {"Сложное выражение", "std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x*x; });"}
    };
    
    std::cout << "========================================\n";
    std::cout << "ТЕСТИРОВАНИЕ КОДИРОВАНИЯ\n";
    std::cout << "========================================\n\n";
    
    int passed = 0;
    int total = test_cases.size();
    
    std::map<size_t, int> token_count_distribution;
    std::vector<double> encode_times;
    std::vector<size_t> token_counts;
    
    for (const auto& [desc, code] : test_cases) {
        std::cout << desc << ":\n";
        std::cout << "   Исходный:   '" << code << "'\n";
        
        // Измеряем время encode
        auto encode_start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(code);
        auto encode_end = std::chrono::high_resolution_clock::now();
        
        auto encode_time = std::chrono::duration_cast<std::chrono::microseconds>(encode_end - encode_start);
        encode_times.push_back(encode_time.count());
        
        // Декодируем
        auto decode_start = std::chrono::high_resolution_clock::now();
        auto decoded = tokenizer.decode(tokens);
        auto decode_end = std::chrono::high_resolution_clock::now();
        
        auto decode_time = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start);
        
        // Проверяем результат
        bool match = (code == decoded);
        if (match) passed++;
        
        // Собираем статистику
        token_count_distribution[tokens.size()]++;
        token_counts.push_back(tokens.size());
        
        std::cout << "   Декодирован: '" << decoded << "'\n";
        std::cout << "   Токенов: " << tokens.size() << " (из них уникальных: " 
                  << std::set<int>(tokens.begin(), tokens.end()).size() << ")\n";
        std::cout << "   Время encode: " << format_duration(encode_time) << "\n";
        std::cout << "   Время decode: " << format_duration(decode_time) << "\n";
        std::cout << "   Результат: " << (match ? "СОВПАДАЕТ" : "НЕ СОВПАДАЕТ") << "\n";
        
        if (!match) {
            // Показываем различия
            size_t min_len = std::min(code.length(), decoded.length());
            std::cout << "   Различия:\n";
            for (size_t i = 0; i < min_len; ++i) {
                if (code[i] != decoded[i]) {
                    std::cout << "     Позиция " << i << ": исходный '" << code[i] 
                              << "' (0x" << std::hex << int(code[i]) << std::dec 
                              << ") vs декод. '" << decoded[i] << "'\n";
                }
            }
            if (code.length() != decoded.length()) {
                std::cout << "     Длина: исходный " << code.length() 
                          << ", декод. " << decoded.length() << "\n";
            }
        }
        
        std::cout << std::endl;
    }
    
    // ======================================================================
    // Статистика
    // ======================================================================
    
    std::cout << "========================================\n";
    std::cout << "СТАТИСТИКА\n";
    std::cout << "========================================\n\n";
    
    double success_rate = 100.0 * passed / total;
    std::cout << "Успешных тестов: " << passed << "/" << total 
              << " (" << std::fixed << std::setprecision(1) << success_rate << "%)\n\n";
    
    // Статистика по токенам
    if (!token_counts.empty()) {
        size_t total_tokens = 0;
        for (size_t count : token_counts) total_tokens += count;
        double avg_tokens = static_cast<double>(total_tokens) / token_counts.size();
        
        std::cout << "Статистика токенов:\n";
        std::cout << "   Всего токенов: " << total_tokens << "\n";
        std::cout << "   Среднее на текст: " << std::fixed << std::setprecision(1) << avg_tokens << "\n";
        std::cout << "   Минимум: " << *std::min_element(token_counts.begin(), token_counts.end()) << "\n";
        std::cout << "   Максимум: " << *std::max_element(token_counts.begin(), token_counts.end()) << "\n";
        
        std::cout << "\n   Распределение:\n";
        for (const auto& [count, freq] : token_count_distribution) {
            std::cout << "     " << count << " токенов: " << freq << " текстов\n";
        }
    }
    
    // Статистика производительности
    if (!encode_times.empty()) {
        double total_time = 0;
        for (double time : encode_times) total_time += time;
        double avg_time = total_time / encode_times.size();
        
        std::cout << "\nПроизводительность encode:\n";
        std::cout << "   Среднее время: " << std::fixed << std::setprecision(2) << avg_time << " мкс\n";
        std::cout << "   Минимальное: " << *std::min_element(encode_times.begin(), encode_times.end()) << " мкс\n";
        std::cout << "   Максимальное: " << *std::max_element(encode_times.begin(), encode_times.end()) << " мкс\n";
    }
    
    // Статистика от токенизатора
    auto stats = tokenizer.stats();
    if (stats.encode_calls > 0) {
        std::cout << "\nСтатистика токенизатора:\n";
        std::cout << "   Всего encode вызовов: " << stats.encode_calls << "\n";
        std::cout << "   Всего decode вызовов: " << stats.decode_calls << "\n";
        std::cout << "   Попаданий в кэш: " << stats.cache_hits << "\n";
        std::cout << "   Промахов кэша: " << stats.cache_misses << "\n";
        std::cout << "   Эффективность кэша: " << std::fixed << std::setprecision(1)
                  << (stats.cache_hit_rate() * 100) << "%\n";
        std::cout << "   Обработано токенов: " << stats.total_tokens_processed << "\n";
    }
    
    // ======================================================================
    // Тест производительности на большом тексте
    // ======================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ\n";
    std::cout << "========================================\n\n";
    
    // Создаем большой текст
    std::string large_text;
    for (int i = 0; i < 100; ++i) {
        large_text += "int x" + std::to_string(i) + " = " + std::to_string(i * i) + ";\n";
    }
    
    std::cout << "Размер текста: " << large_text.size() << " символов\n";
    std::cout << "Запуск encode...\n";
    
    auto perf_start = std::chrono::high_resolution_clock::now();
    auto large_tokens = tokenizer.encode(large_text);
    auto perf_end = std::chrono::high_resolution_clock::now();
    
    auto perf_time = std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start);
    double mb_per_sec = (large_text.size() / 1'000'000.0) / (perf_time.count() / 1'000'000.0);
    
    std::cout << "Закодировано за " << format_duration(perf_time) << "\n";
    std::cout << "   Токенов: " << large_tokens.size() << "\n";
    std::cout << "   Скорость: " << std::fixed << std::setprecision(2) << mb_per_sec << " МБ/сек\n";
    std::cout << "   Токенов/сек: " << std::fixed << std::setprecision(0) 
              << (large_tokens.size() / (perf_time.count() / 1'000'000.0)) << "\n";
    
    // Тест многопоточности
    std::cout << "\nТест многопоточности...\n";
    
    auto multi_start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        auto tokens = tokenizer.encode(large_text);
        volatile size_t dummy = tokens.size();
        (void)dummy;
    }
    
    auto multi_end = std::chrono::high_resolution_clock::now();
    auto multi_time = std::chrono::duration_cast<std::chrono::microseconds>(multi_end - multi_start);
    
    std::cout << "   4 параллельных вызова: " << format_duration(multi_time) << "\n";
    
    // ======================================================================
    // Итог
    // ======================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "ТЕСТИРОВАНИЕ ЗАВЕРШЕНО\n";
    std::cout << "========================================\n";
    
    return (passed == total) ? 0 : 1;
}