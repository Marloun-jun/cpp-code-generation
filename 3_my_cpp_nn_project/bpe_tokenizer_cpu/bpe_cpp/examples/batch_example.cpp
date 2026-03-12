/**
 * @file batch_example.cpp
 * @brief Демонстрация пакетной обработки текстов с BPE токенизатором
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Этот пример показывает преимущества пакетной обработки (batch processing)
 *          при работе с BPE токенизатором. Основные демонстрируемые концепции:
 * 
 *          - Пакетная vs одиночная обработка - сравнение производительности
 *          - Статистика токенизации для разных примеров C++ кода
 *          - Эффективность кэширования при повторяющихся паттернах
 *          - Масштабирование с ростом размера батча
 *          - Сравнение с оптимизированной версией FastTokenizer
 * 
 *          Тестовые примеры включают различные конструкции C++:
 *          - Базовые выражения и операторы
 *          - STL контейнеры и алгоритмы
 *          - Управляющие конструкции (if, for, while, switch)
 *          - Шаблоны, классы, структуры, перечисления
 *          - Функции, лямбды, указатели
 *          - Строковые литералы и комментарии (включая русские и эмодзи)
 *          - Preprocessor директивы
 * 
 * @compile g++ -std=c++17 -O2 batch_example.cpp -o batch_example -lbpe_tokenizer
 * @run   ./batch_example
 * 
 * @see BPETokenizer
 * @see FastBPETokenizer
 */

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <filesystem>
#include <iostream>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <set>
#include <numeric>
#include <chrono>
#include <thread>
#include <string>

namespace fs = std::filesystem;
using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr size_t DEFAULT_VOCAB_SIZE = 8000;
    constexpr size_t FAST_CACHE_SIZE = 10000;
    constexpr int REPETITIVE_BATCH_SIZE = 100;
    constexpr int WIDTH = 80;
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string BOLD = "\033[1m";
}

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Выводит заголовок раздела
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

/**
 * @brief Создает тестовый батч из разнообразных примеров C++ кода
 * 
 * @return std::vector<std::string> Вектор с тестовыми примерами
 */
std::vector<std::string> create_test_batch() {
    return {
        // ===== Простые выражения =====
        "int x = 42;",
        "float y = 3.14f;",
        "char c = 'A';",
        "bool flag = true;",
        "auto result = x + y;",
        
        // ===== STL контейнеры =====
        "std::vector<int> numbers = {1, 2, 3, 4, 5};",
        "std::map<std::string, int> ages;",
        "std::set<double> values;",
        "std::unordered_map<int, std::string> dict;",
        "std::array<int, 10> arr;",
        "std::pair<int, std::string> p = {1, \"one\"};",
        "std::optional<int> opt;",
        
        // ===== Управляющие конструкции =====
        "if (condition) { do_something(); }",
        "for (int i = 0; i < 10; ++i) { sum += i; }",
        "while (running) { process(); }",
        "do { count++; } while (count < 10);",
        "switch (value) { case 1: break; default: break; }",
        "try { throw std::runtime_error(\"error\"); } catch (const std::exception& e) {}",
        
        // ===== Шаблоны и классы =====
        "template<typename T> T max(T a, T b) { return (a > b) ? a : b; }",
        "class MyClass { public: void method(); private: int data_; };",
        "struct Point { int x, y; };",
        "enum Color { RED, GREEN, BLUE };",
        "namespace my_namespace { class MyClass {}; }",
        "template<typename T, typename U> struct Pair { T first; U second; };",
        
        // ===== Функции и лямбды =====
        "int add(int a, int b) { return a + b; }",
        "auto lambda = [](int x) { return x * x; };",
        "int* ptr = nullptr;",
        "constexpr int MAX_SIZE = 100;",
        "static_assert(sizeof(int) == 4, \"int must be 4 bytes\");",
        "void (*func_ptr)(int) = &function;",
        
        // ===== Строки и комментарии =====
        "\"string literal with \\\"escapes\\\"\"",
        "'c'",
        "R\"(raw string)\"",
        "// single line comment",
        "/* multi-line\n comment */",
        "std::string s = \"Hello\";",
        
        // ===== Include директивы =====
        "#include <iostream>",
        "#include \"myheader.hpp\"",
        "#include <vector>",
        "#include <algorithm>",
        "#include <memory>",
        "#include <thread>",
        
        // ===== Сложные выражения =====
        "std::cout << \"Hello, \" << name << \"!\" << std::endl;",
        "auto result = std::accumulate(v.begin(), v.end(), 0);",
        "std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();",
        "std::sort(vec.begin(), vec.end(), std::greater<int>());",
        
        // ===== Русские комментарии (тест Unicode) =====
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        "// Тестирование кириллицы в коде",
        
        // ===== Эмодзи (для byte-level режима) =====
        "// 🔥 C++ code with emoji 😊",
        "// 🚀 performance test",
        "// 📊 statistics",
        "// ⚡ fast processing"
    };
}

/**
 * @brief Создает батч с повторяющимися паттернами для тестирования кэша
 * 
 * @param size Желаемое количество примеров в батче
 * @return std::vector<std::string> Вектор с повторяющимися примерами
 */
std::vector<std::string> create_repetitive_batch(size_t size) {
    std::vector<std::string> batch;
    batch.reserve(size);
    
    // Базовые паттерны для повторения
    std::vector<std::string> patterns = {
        "int x = 42;",
        "std::vector<int> v;",
        "for (int i = 0; i < 10; ++i) { sum += i; }",
        "class MyClass { int data; };"
    };
    
    for (size_t i = 0; i < size; ++i) {
        batch.push_back(patterns[i % patterns.size()] + " // ID: " + std::to_string(i));
    }
    
    return batch;
}

/**
 * @brief Выводит детальную статистику по токенизации каждого примера
 * 
 * @param results Вектор результатов токенизации
 * @param batch Исходные текстовые примеры
 */
void print_token_stats(const std::vector<std::vector<token_id_t>>& results,
                       const std::vector<std::string>& batch) {
    
std::cout << "\n" << CYAN << BOLD << "📊 Статистика по токенам:" << RESET << std::endl;

// Линия-разделитель через цикл
for (int i = 0; i < WIDTH; ++i) std::cout << '-';
std::cout << "\n";

std::cout << std::setw(4) << "ID" << " │ " 
          << std::setw(32) << "Пример" << " │ "
          << std::setw(8) << "Токены" << " │ "
          << std::setw(10) << "Уникальных" << " │ "
          << "Сжатие" << std::endl;

// Линия-разделитель
for (int i = 0; i < WIDTH; ++i) std::cout << '-';
std::cout << std::endl;

    size_t total_tokens = 0;
    size_t total_chars = 0;
    
    for (size_t i = 0; i < batch.size(); ++i) {
        // Обрезаем длинные примеры для читаемости
        std::string display = batch[i];
        if (display.size() > 30) {
            display = display.substr(0, 27) + "...";
        }
        
        // Подсчет уникальных токенов
        std::set<token_id_t> unique_tokens(results[i].begin(), results[i].end());
        
        size_t num_tokens = results[i].size();
        size_t num_chars = batch[i].size();
        double compression = 100.0 * (1.0 - static_cast<double>(num_tokens) / num_chars);
        
        std::cout << std::setw(4) << i << " │ " 
                  << std::setw(32) << display << " │ "
                  << std::setw(8) << num_tokens << " │ "
                  << std::setw(10) << unique_tokens.size() << " │ "
                  << std::fixed << std::setprecision(1) << std::setw(6)
                  << compression << "%" << std::endl;
        
        total_tokens += num_tokens;
        total_chars += num_chars;
    }
    
    // Верхняя линия
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << "\n";

    double avg_compression = 100.0 * (1.0 - static_cast<double>(total_tokens) / total_chars);

    char buffer[64];
    snprintf(buffer, sizeof(buffer), "ВСЕГО: %zu примеров", batch.size());

    std::cout << std::setw(4) << "Σ" << " │ " 
            << std::setw(32) << buffer << " │ "
            << std::setw(8) << total_tokens << " │ "
            << std::setw(10) << "-" << " │ "
            << std::fixed << std::setprecision(1) << std::setw(6)
            << avg_compression << "%" << std::endl;

    // Нижняя линия
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << std::endl;
}

/**
 * @brief Класс для поиска и загрузки моделей
 */
class ModelLoader {
private:
    std::vector<std::pair<std::string, std::string>> paths_;
    
public:
    ModelLoader() {
        // C++ модели (в папке models/)
        paths_.emplace_back("../models/bpe_8000/cpp_vocab.json", 
                            "../models/bpe_8000/cpp_merges.txt");
        paths_.emplace_back("../models/bpe_10000/cpp_vocab.json", 
                            "../models/bpe_10000/cpp_merges.txt");
        paths_.emplace_back("../models/bpe_12000/cpp_vocab.json", 
                            "../models/bpe_12000/cpp_merges.txt");
        
        // Python модели (из корня проекта)
        paths_.emplace_back("../../bpe_python/models/bpe_8000/vocab.json", 
                            "../../bpe_python/models/bpe_8000/merges.txt");
        paths_.emplace_back("../../bpe_python/models/bpe_10000/vocab.json", 
                            "../../bpe_python/models/bpe_10000/merges.txt");
        paths_.emplace_back("../../bpe_python/models/bpe_12000/vocab.json", 
                            "../../bpe_python/models/bpe_12000/merges.txt");
        
        // Модели в текущей директории
        paths_.emplace_back("models/bpe_8000/cpp_vocab.json", 
                            "models/bpe_8000/cpp_merges.txt");
        paths_.emplace_back("models/bpe_10000/cpp_vocab.json", 
                            "models/bpe_10000/cpp_merges.txt");
        paths_.emplace_back("models/bpe_12000/cpp_vocab.json", 
                            "models/bpe_12000/cpp_merges.txt");
        
        // Модели в родительской директории
        paths_.emplace_back("bpe_python/models/bpe_8000/vocab.json", 
                            "bpe_python/models/bpe_8000/merges.txt");
    }
    
    bool load_basic_tokenizer(BPETokenizer& tokenizer) {
        for (const auto& [vocab, merges] : paths_) {
            std::cout << "  Пробуем: " << vocab << std::endl;
            
            if (fs::exists(vocab) && fs::exists(merges)) {
                if (tokenizer.load_from_files(vocab, merges)) {
                    std::cout << GREEN << "  ✓ Загружено: " << vocab << RESET << std::endl;
                    return true;
                }
            }
        }
        return false;
    }
    
    bool load_fast_tokenizer(FastBPETokenizer& tokenizer) {
        for (const auto& [vocab, merges] : paths_) {
            if (fs::exists(vocab) && fs::exists(merges)) {
                if (tokenizer.load(vocab, merges)) {
                    std::cout << GREEN << "  ✓ Загружено: " << vocab << RESET << std::endl;
                    return true;
                }
            }
        }
        return false;
    }
};

// ======================================================================
// Основная функция
// ======================================================================

/**
 * @brief Точка входа в программу
 * 
 * @return int 0 при успешном выполнении, 1 при ошибке
 */
int main() {
    try {
        print_header("ПАКЕТНАЯ ОБРАБОТКА BPE ТОКЕНИЗАТОРОМ");
        
        // ======================================================================
        // 1. Инициализация токенизатора
        // ======================================================================

        std::cout << YELLOW << "\n🔧 Инициализация токенизатора..." << RESET << std::endl;

        BPETokenizer tokenizer;
        tokenizer.set_byte_level(true);    // Включаем byte-level режим для Unicode

        ModelLoader loader;
        if (!loader.load_basic_tokenizer(tokenizer)) {
            std::cerr << "\n❌ Не удалось загрузить модель!" << std::endl;
            std::cerr << "   Убедитесь, что файлы моделей находятся в директории:" << std::endl;
            std::cerr << "   - bpe_cpp/models/bpe_8000/cpp_vocab.json" << std::endl;
            std::cerr << "   - bpe_cpp/models/bpe_8000/cpp_merges.txt" << std::endl;
            std::cerr << "   или в bpe_python/models/bpe_8000/" << std::endl;
            return 1;
        }
        
        std::cout << "📊 Размер словаря: " << tokenizer.vocab_size() << std::endl;
        std::cout << "📊 Правил слияния: " << tokenizer.merges_count() << std::endl;
        
        // ======================================================================
        // 2. Создание тестового батча
        // ======================================================================
        
        auto batch = create_test_batch();
        std::cout << "\n📦 Создан тестовый батч из " << batch.size() << " примеров" << std::endl;
        
        // ======================================================================
        // 3. Тест 1: Сравнение одиночной и пакетной обработки
        // ======================================================================
        
        print_header("ТЕСТ 1: Сравнение производительности");
        
        utils::Timer timer;
        
        // Одиночная обработка (последовательное кодирование)
        std::vector<std::vector<token_id_t>> single_results;
        single_results.reserve(batch.size());
        
        timer.reset();
        for (const auto& text : batch) {
            single_results.push_back(tokenizer.encode(text));
        }
        double single_time = timer.elapsed();
        
        // Пакетная обработка (один вызов для всех текстов)
        std::vector<std::vector<token_id_t>> batch_results;
        
        timer.reset();
        batch_results = tokenizer.encode_batch(batch);
        double batch_time = timer.elapsed();
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "📈 Одиночная обработка: " << CYAN << single_time * 1000 << " мс" << RESET << std::endl;
        std::cout << "📈 Пакетная обработка:  " << GREEN << batch_time * 1000 << " мс" << RESET << std::endl;
        std::cout << "⚡ Ускорение:           " << BOLD << (single_time / batch_time) << "x" << RESET << std::endl;
        
        // Проверка корректности (результаты должны совпадать)
        bool all_match = true;
        for (size_t i = 0; i < batch.size(); ++i) {
            if (single_results[i] != batch_results[i]) {
                all_match = false;
                std::cout << "\n❌ Несовпадение в примере " << i << std::endl;
                
                std::cout << "   Одиночная: ";
                for (size_t j = 0; j < std::min(size_t(10), single_results[i].size()); ++j) {
                    std::cout << single_results[i][j] << " ";
                }
                std::cout << (single_results[i].size() > 10 ? "..." : "") << std::endl;
                
                std::cout << "   Пакетная:  ";
                for (size_t j = 0; j < std::min(size_t(10), batch_results[i].size()); ++j) {
                    std::cout << batch_results[i][j] << " ";
                }
                std::cout << (batch_results[i].size() > 10 ? "..." : "") << std::endl;
                break;
            }
        }
        std::cout << "✓ Корректность: " << (all_match ? GREEN + "OK" : "ОШИБКА") << RESET << std::endl;
        
        // ======================================================================
        // 4. Тест 2: Статистика по токенам
        // ======================================================================
        
        print_header("ТЕСТ 2: Статистика по примерам");
        print_token_stats(batch_results, batch);
        
        // ======================================================================
        // 5. Тест 3: Эффективность кэширования
        // ======================================================================
        
        print_header("ТЕСТ 3: Эффективность кэширования");
        
        auto repetitive_batch = create_repetitive_batch(REPETITIVE_BATCH_SIZE);
        std::cout << "📦 Батч с повторениями: " << repetitive_batch.size() << " примеров" << std::endl;
        std::cout << "📊 Паттернов: 4 (повторяются циклически)" << std::endl;
        
        // Первый проход (заполнение кэша)
        timer.reset();
        auto first_pass = tokenizer.encode_batch(repetitive_batch);
        double first_pass_time = timer.elapsed();
        
        // Второй проход (должен быть быстрее из-за попаданий в кэш)
        timer.reset();
        auto second_pass = tokenizer.encode_batch(repetitive_batch);
        double second_pass_time = timer.elapsed();
        
        std::cout << "🔄 Первый проход:  " << CYAN << first_pass_time * 1000 << " мс" << RESET << std::endl;
        std::cout << "🔄 Второй проход:  " << GREEN << second_pass_time * 1000 << " мс" << RESET << std::endl;
        std::cout << "⚡ Ускорение:      " << BOLD << (first_pass_time / second_pass_time) << "x" << RESET << std::endl;
        
        // ======================================================================
        // 6. Тест 4: Зависимость от размера батча
        // ======================================================================
        
        print_header("ТЕСТ 4: Зависимость от размера батча");
        
        std::vector<size_t> batch_sizes = {1, 2, 5, 10, 20, 50, 100};
        std::cout << std::setw(10) << "Размер" << " │ "
                  << std::setw(14) << "Время (мс)" << " │ "
                  << std::setw(12) << "На элемент" << " │ "
                  << std::setw(10) << "Ускорение" << std::endl;
        std::cout << "------------------------------------------------------------\n";
        
        double base_time = 0;
        
        for (size_t bs : batch_sizes) {
            auto test_batch = create_repetitive_batch(bs);
            
            timer.reset();
            auto results = tokenizer.encode_batch(test_batch);
            double elapsed = timer.elapsed() * 1000;    // Конвертируем в мс
            
            double per_item = elapsed / bs;
            
            if (bs == 1) {
                base_time = elapsed;
                std::cout << std::setw(10) << bs << " │ "
                          << std::setw(12) << std::fixed << std::setprecision(3) << elapsed << " │ "
                          << std::setw(12) << per_item << " │ "
                          << std::setw(10) << "1.0x" << std::endl;
            } else {
                double speedup = base_time / per_item;
                std::cout << std::setw(10) << bs << " │ "
                          << std::setw(12) << std::fixed << std::setprecision(3) << elapsed << " │ "
                          << std::setw(12) << per_item << " │ "
                          << std::setw(10) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            }
        }
        
        // ======================================================================
        // 7. Тест 5: Сравнение с быстрым токенизатором
        // ======================================================================

        print_header("ТЕСТ 5: Сравнение с FastTokenizer");

        TokenizerConfig fast_config;
        fast_config.byte_level = true;
        fast_config.enable_cache = true;
        fast_config.cache_size = FAST_CACHE_SIZE;

        FastBPETokenizer fast_tokenizer(fast_config);

        if (loader.load_fast_tokenizer(fast_tokenizer)) {
            std::cout << "FastTokenizer загружен!" << std::endl;
            
            // Создаем вектор string_view для передачи в FastTokenizer
            std::vector<std::string_view> batch_views;
            batch_views.reserve(batch.size());
            
            for (const auto& text : batch) {
                batch_views.push_back(std::string_view(text));
            }
            
            // Прогрев
            fast_tokenizer.encode_batch(batch_views);
            
            // Сравниваем производительность на том же батче
            timer.reset();
            auto fast_results = fast_tokenizer.encode_batch(batch_views);
            double fast_time = timer.elapsed() * 1000;
            
            std::cout << "📊 BPETokenizer:    " << CYAN << batch_time << " мс" << RESET << std::endl;
            std::cout << "📊 FastTokenizer:   " << GREEN << fast_time << " мс" << RESET << std::endl;
            std::cout << "⚡ Ускорение:        " << BOLD << (batch_time / fast_time) << "x" << RESET << std::endl;
            
            // Проверяем корректность
            bool fast_match = (batch_results.size() == fast_results.size());
            if (fast_match) {
                for (size_t i = 0; i < batch_results.size(); ++i) {
                    if (batch_results[i].size() != fast_results[i].size()) {
                        fast_match = false;
                        std::cout << "  Несовпадение размера в примере " << i << ": "
                                << batch_results[i].size() << " vs " << fast_results[i].size() << std::endl;
                        break;
                    }
                }
            }
            std::cout << "✓ Корректность:    " << (fast_match ? GREEN + "OK" : "ОШИБКА") << RESET << std::endl;
            
            // Показываем статистику FastTokenizer
            auto fast_stats = fast_tokenizer.stats();
            std::cout << "📊 Cache hits:      " << fast_stats.cache_hits << std::endl;
            std::cout << "📊 Cache misses:    " << fast_stats.cache_misses << std::endl;
            std::cout << "📊 Hit rate:        " << std::fixed << std::setprecision(1)
                    << fast_stats.cache_hit_rate() << "%" << std::endl;
        } else {
            std::cout << YELLOW << "⚠ FastTokenizer не загружен (пропускаем)" << RESET << std::endl;
        }
        
        // ======================================================================
        // 8. Итог
        // ======================================================================
        
        print_header("ИТОГИ");
        
        // Сводная статистика
        size_t total_tokens = std::accumulate(batch_results.begin(), batch_results.end(), 0,
            [](size_t sum, const auto& tokens) { return sum + tokens.size(); });
        
        size_t total_chars = std::accumulate(batch.begin(), batch.end(), 0,
            [](size_t sum, const auto& text) { return sum + text.size(); });
        
        double avg_compression = 100.0 * (1.0 - static_cast<double>(total_tokens) / total_chars);
        
        std::cout << "\n📊 Сводка по тестированию:" << std::endl;
        std::cout << "  • Всего примеров:    " << batch.size() << std::endl;
        std::cout << "  • Всего символов:    " << total_chars << std::endl;
        std::cout << "  • Всего токенов:     " << total_tokens << std::endl;
        std::cout << "  • Среднее сжатие:    " << std::fixed << std::setprecision(1)
                  << avg_compression << "%" << std::endl;
        std::cout << "  • Токенов в среднем: " << std::fixed << std::setprecision(1)
                  << static_cast<double>(total_tokens) / batch.size() << " на пример" << std::endl;
        std::cout << "  • Ускорение батча:   " << (single_time / batch_time) << "x" << std::endl;
        
        std::cout << "\n" << GREEN << BOLD << "✓ Программа успешно завершена!" << RESET << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Ошибка: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Неизвестная ошибка" << std::endl;
        return 1;
    }
    
    return 0;
}