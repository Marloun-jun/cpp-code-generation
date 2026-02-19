/**
 * @file batch_example.cpp
 * @brief Пример пакетной обработки текстов с BPE токенизатором
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Демонстрация эффективности пакетной обработки:
 *          - Сравнение одиночной и пакетной обработки
 *          - Статистика по каждому примеру
 *          - Проверка корректности результатов
 *          - Измерение производительности
 * 
 * @compile g++ -std=c++17 batch_example.cpp -o batch_example -lbpe_tokenizer
 * @run ./batch_example
 */

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"  // Добавляем быструю версию для сравнения
#include "utils.hpp"

#include <iostream>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <set>

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Создать тестовый батч из разных примеров C++ кода
 */
std::vector<std::string> create_test_batch() {
    return {
        // Простые выражения
        "int x = 42;",
        "float y = 3.14f;",
        "char c = 'A';",
        "bool flag = true;",
        "auto result = x + y;",
        
        // STL контейнеры
        "std::vector<int> numbers = {1, 2, 3};",
        "std::map<std::string, int> ages;",
        "std::set<double> values;",
        "std::unordered_map<int, std::string> dict;",
        "std::array<int, 10> arr;",
        
        // Управляющие конструкции
        "if (condition) { do_something(); }",
        "for (int i = 0; i < 10; ++i) { sum += i; }",
        "while (running) { process(); }",
        "switch (value) { case 1: break; default: break; }",
        "try { throw std::runtime_error(\"error\"); } catch (...) {}",
        
        // Шаблоны и классы
        "template<typename T> T max(T a, T b) { return (a > b) ? a : b; }",
        "class MyClass { public: void method(); private: int data_; };",
        "struct Point { int x, y; };",
        "enum Color { RED, GREEN, BLUE };",
        "namespace my_namespace { class MyClass {}; }",
        
        // Функции и лямбды
        "void func(int a, float b) { return a + b; }",
        "auto lambda = [](int x) { return x * x; };",
        "int* ptr = nullptr;",
        "constexpr int MAX_SIZE = 100;",
        "static_assert(sizeof(int) == 4, \"int must be 4 bytes\");",
        
        // Строки и комментарии
        "\"string literal with \\\"escapes\\\"\"",
        "'c'",
        "R\"(raw string)\"",
        "// single line comment",
        "/* multi-line\n comment */",
        
        // Include директивы
        "#include <iostream>",
        "#include \"myheader.hpp\"",
        "#include <vector>",
        "#include <algorithm>",
        "#include <memory>",
        
        // Сложные выражения
        "std::cout << \"Hello, \" << name << \"!\" << std::endl;",
        "auto result = std::accumulate(v.begin(), v.end(), 0);",
        "std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();",
        
        // Русские комментарии
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        
        // Эмодзи (для byte-level режима)
        "// 🔥 C++ code with emoji 😊",
        "// 🚀 performance test",
        "// 📊 statistics"
    };
}

/**
 * @brief Создать батч повторяющихся элементов для теста кэша
 */
std::vector<std::string> create_repetitive_batch(size_t size) {
    std::vector<std::string> batch;
    batch.reserve(size);
    
    std::vector<std::string> patterns = {
        "int x = 42;",
        "std::vector<int> v;",
        "for (int i = 0; i < 10; ++i) {}",
        "class MyClass {};"
    };
    
    for (size_t i = 0; i < size; ++i) {
        batch.push_back(patterns[i % patterns.size()] + " // " + std::to_string(i));
    }
    
    return batch;
}

/**
 * @brief Вывести статистику по токенам
 */
void print_token_stats(const std::vector<std::vector<token_id_t>>& results,
                       const std::vector<std::string>& batch) {
    
    std::cout << "\n📊 Статистика по токенам:" << std::endl;
    std::cout << std::setw(4) << "ID" << " | " 
              << std::setw(30) << "Пример" << " | "
              << std::setw(8) << "Токены" << " | "
              << std::setw(10) << "Уникальных" << " | "
              << "Сжатие" << std::endl;
    std::cout << std::string(75, '-') << std::endl;
    
    size_t total_tokens = 0;
    size_t total_chars = 0;
    
    for (size_t i = 0; i < batch.size(); ++i) {
        std::string display = batch[i];
        if (display.size() > 28) {
            display = display.substr(0, 25) + "...";
        }
        
        // Подсчет уникальных токенов
        std::set<token_id_t> unique_tokens(results[i].begin(), results[i].end());
        
        size_t num_tokens = results[i].size();
        size_t num_chars = batch[i].size();
        double compression = 100.0 * (1.0 - static_cast<double>(num_tokens) / num_chars);
        
        std::cout << std::setw(4) << i << " | " 
                  << std::setw(30) << display << " | "
                  << std::setw(8) << num_tokens << " | "
                  << std::setw(10) << unique_tokens.size() << " | "
                  << std::fixed << std::setprecision(1) << std::setw(6)
                  << compression << "%" << std::endl;
        
        total_tokens += num_tokens;
        total_chars += num_chars;
    }
    
    std::cout << std::string(75, '-') << std::endl;
    std::cout << std::setw(4) << "TOTAL" << " | " 
              << std::setw(30) << batch.size() << " примеров | "
              << std::setw(8) << total_tokens << " | "
              << std::setw(10) << "-" << " | "
              << std::fixed << std::setprecision(1) << std::setw(6)
              << (100.0 * (1.0 - static_cast<double>(total_tokens) / total_chars)) 
              << "%" << std::endl;
}

// ======================================================================
// Основная функция
// ======================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "📦 ПАКЕТНАЯ ОБРАБОТКА BPE ТОКЕНИЗАТОРОМ\n";
    std::cout << "========================================\n" << std::endl;
    
    try {
        // ======================================================================
        // Инициализация токенизатора
        // ======================================================================
        
        std::cout << "🔧 Инициализация токенизатора..." << std::endl;
        
        BPETokenizer tokenizer;
        tokenizer.set_byte_level(true);
        
        // Пробуем загрузить модель из разных мест
        std::vector<std::pair<std::string, std::string>> model_paths = {
            {"models/cpp_vocab.json", "models/cpp_merges.txt"},
            {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
            {"../../models/cpp_vocab.json", "../../models/cpp_merges.txt"},
            {"vocab.json", "merges.txt"}
        };
        
        bool loaded = false;
        for (const auto& [vocab_path, merges_path] : model_paths) {
            if (tokenizer.load_from_files(vocab_path, merges_path)) {
                std::cout << "  ✅ Загружена модель: " << vocab_path << std::endl;
                loaded = true;
                break;
            }
        }
        
        if (!loaded) {
            std::cerr << "❌ Не удалось загрузить модель!" << std::endl;
            return 1;
        }
        
        std::cout << "  📚 Размер словаря: " << tokenizer.vocab_size() << std::endl;
        std::cout << "  🔗 Правил слияния: " << tokenizer.merges_count() << std::endl;
        
        // ======================================================================
        // Создание тестового батча
        // ======================================================================
        
        auto batch = create_test_batch();
        std::cout << "\n📋 Создан тестовый батч из " << batch.size() << " примеров" << std::endl;
        
        // ======================================================================
        // Тест 1: Сравнение одиночной и пакетной обработки
        // ======================================================================
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ТЕСТ 1: Сравнение производительности" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        utils::Timer timer;
        
        // Одиночная обработка
        std::vector<std::vector<token_id_t>> single_results;
        single_results.reserve(batch.size());
        
        timer.reset();
        for (const auto& text : batch) {
            single_results.push_back(tokenizer.encode(text));
        }
        double single_time = timer.elapsed();
        
        // Пакетная обработка
        std::vector<std::vector<token_id_t>> batch_results;
        
        timer.reset();
        batch_results = tokenizer.encode_batch(batch);
        double batch_time = timer.elapsed();
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Одиночная обработка: " << single_time * 1000 << " ms" << std::endl;
        std::cout << "  Пакетная обработка:  " << batch_time * 1000 << " ms" << std::endl;
        std::cout << "  Ускорение:           " << (single_time / batch_time) << "x" << std::endl;
        
        // Проверка корректности
        bool all_match = true;
        for (size_t i = 0; i < batch.size(); ++i) {
            if (single_results[i] != batch_results[i]) {
                all_match = false;
                std::cout << "  ❌ Несовпадение в примере " << i << std::endl;
                
                // Показываем различия
                std::cout << "     Одиночная: ";
                for (size_t j = 0; j < std::min(size_t(5), single_results[i].size()); ++j) {
                    std::cout << single_results[i][j] << " ";
                }
                std::cout << std::endl;
                
                std::cout << "     Пакетная:  ";
                for (size_t j = 0; j < std::min(size_t(5), batch_results[i].size()); ++j) {
                    std::cout << batch_results[i][j] << " ";
                }
                std::cout << std::endl;
                break;
            }
        }
        std::cout << "  Корректность: " << (all_match ? "✅" : "❌") << std::endl;
        
        // ======================================================================
        // Тест 2: Статистика по токенам
        // ======================================================================
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ТЕСТ 2: Статистика по примерам" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        print_token_stats(batch_results, batch);
        
        // ======================================================================
        // Тест 3: Эффективность кэширования
        // ======================================================================
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ТЕСТ 3: Эффективность кэширования" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        auto repetitive_batch = create_repetitive_batch(100);
        std::cout << "  Батч с повторениями: " << repetitive_batch.size() << " примеров" << std::endl;
        
        // Первый проход (заполнение кэша)
        timer.reset();
        auto first_pass = tokenizer.encode_batch(repetitive_batch);
        double first_pass_time = timer.elapsed();
        
        // Второй проход (должен быть быстрее из-за кэша)
        timer.reset();
        auto second_pass = tokenizer.encode_batch(repetitive_batch);
        double second_pass_time = timer.elapsed();
        
        std::cout << "  Первый проход:  " << first_pass_time * 1000 << " ms" << std::endl;
        std::cout << "  Второй проход:  " << second_pass_time * 1000 << " ms" << std::endl;
        std::cout << "  Ускорение:      " << (first_pass_time / second_pass_time) << "x" << std::endl;
        
        // ======================================================================
        // Тест 4: Разные размеры батча
        // ======================================================================
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ТЕСТ 4: Зависимость от размера батча" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::vector<size_t> batch_sizes = {1, 2, 5, 10, 20, 50, 100};
        std::cout << std::setw(10) << "Размер" << " | "
                  << std::setw(12) << "Время (ms)" << " | "
                  << std::setw(12) << "На элемент" << " | "
                  << std::setw(10) << "Ускорение" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        double base_time = 0;
        
        for (size_t bs : batch_sizes) {
            auto test_batch = create_repetitive_batch(bs);
            
            timer.reset();
            auto results = tokenizer.encode_batch(test_batch);
            double elapsed = timer.elapsed() * 1000; // ms
            
            double per_item = elapsed / bs;
            
            if (bs == 1) {
                base_time = elapsed;
                std::cout << std::setw(10) << bs << " | "
                          << std::setw(12) << std::fixed << std::setprecision(3) << elapsed << " | "
                          << std::setw(12) << per_item << " | "
                          << std::setw(10) << "1.0x" << std::endl;
            } else {
                double speedup = base_time / per_item;
                std::cout << std::setw(10) << bs << " | "
                          << std::setw(12) << std::fixed << std::setprecision(3) << elapsed << " | "
                          << std::setw(12) << per_item << " | "
                          << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            }
        }
        
        // ======================================================================
        // Тест 5: Сравнение с быстрым токенизатором (если доступен)
        // ======================================================================
        
        #ifdef USE_FAST_TOKENIZER
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ТЕСТ 5: Сравнение с FastTokenizer" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        TokenizerConfig fast_config;
        fast_config.byte_level = true;
        fast_config.cache_size = 10000;
        
        FastBPETokenizer fast_tokenizer(fast_config);
        
        // Загружаем ту же модель
        bool fast_loaded = false;
        for (const auto& [vocab_path, merges_path] : model_paths) {
            if (fast_tokenizer.load(vocab_path, merges_path)) {
                fast_loaded = true;
                break;
            }
        }
        
        if (fast_loaded) {
            // Сравниваем производительность
            timer.reset();
            auto fast_results = fast_tokenizer.encode_batch(batch);
            double fast_time = timer.elapsed() * 1000;
            
            std::cout << "  BPETokenizer:    " << batch_time << " ms" << std::endl;
            std::cout << "  FastTokenizer:   " << fast_time << " ms" << std::endl;
            std::cout << "  Ускорение:       " << (batch_time / fast_time) << "x" << std::endl;
            
            // Проверяем корректность
            bool fast_match = (batch_results.size() == fast_results.size());
            if (fast_match) {
                for (size_t i = 0; i < std::min(batch_results.size(), fast_results.size()); ++i) {
                    if (batch_results[i].size() != fast_results[i].size()) {
                        fast_match = false;
                        break;
                    }
                }
            }
            std::cout << "  Корректность:    " << (fast_match ? "✅" : "❌") << std::endl;
        }
        #endif
        
        // ======================================================================
        // Итог
        // ======================================================================
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Сводка
        std::cout << "\n📊 Сводка:" << std::endl;
        std::cout << "  • Всего примеров: " << batch.size() << std::endl;
        std::cout << "  • Всего токенов: " << std::accumulate(batch_results.begin(), batch_results.end(), 0,
            [](size_t sum, const auto& tokens) { return sum + tokens.size(); }) << std::endl;
        std::cout << "  • Среднее сжатие: " << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - static_cast<double>(
                      std::accumulate(batch_results.begin(), batch_results.end(), 0,
                          [](size_t sum, const auto& tokens) { return sum + tokens.size(); })) /
                      std::accumulate(batch.begin(), batch.end(), 0,
                          [](size_t sum, const auto& text) { return sum + text.size(); }))) << "%" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}