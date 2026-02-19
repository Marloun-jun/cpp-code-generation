/**
 * @file fast_tokenizer_demo.cpp
 * @brief Демонстрация возможностей оптимизированного BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 2.0.0
 * 
 * @details Демонстрация всех ключевых возможностей FastTokenizer:
 *          - Загрузка модели
 *          - Кодирование/декодирование примеров C++ кода
 *          - Пакетная обработка
 *          - Тест производительности
 *          - Статистика и профилирование
 *          - Сравнение с SIMD и без
 * 
 * @compile g++ -std=c++17 -O3 -mavx2 -Iinclude fast_tokenizer_demo.cpp -o fast_tokenizer_demo
 * @run ./fast_tokenizer_demo [--simd] [--no-cache] [--verbose]
 */

#include "fast_tokenizer.hpp"
#include "simd_utils.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <set>

using namespace bpe;

// ======================================================================
// Класс для измерения времени с автоматическим логированием
// ======================================================================

class ScopedTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;
    bool print_on_destroy_;
    
public:
    ScopedTimer(const std::string& name, bool print = true) 
        : name_(name), print_on_destroy_(print) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~ScopedTimer() {
        if (print_on_destroy_) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            std::cout << "  ⏱️  " << std::left << std::setw(30) << name_ << ": " 
                      << std::right << std::setw(8) << std::fixed << std::setprecision(3)
                      << duration.count() / 1000.0 << " ms" << std::endl;
        }
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Создать примеры C++ кода разной сложности
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
        {"Русские комментарии", "// это комментарий на русском языке\nint main() { return 0; }"},
        {"Сложное выражение", "std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x * x; });"},
        {"Include директивы", "#include <iostream>\n#include <vector>\n#include <algorithm>"}
    };
}

/**
 * @brief Создать длинный C++ код для теста производительности
 */
std::string create_long_code(size_t size_kb = 100) {
    std::string code;
    code.reserve(size_kb * 1024);
    
    for (int i = 0; i < 1000 && code.size() < size_kb * 1024; ++i) {
        code += "int func" + std::to_string(i) + 
                "(int x" + std::to_string(i) + ") { \n"
                "    return x" + std::to_string(i) + " * " + std::to_string(i) + ";\n"
                "}\n\n";
    }
    
    return code;
}

/**
 * @brief Проверить доступность SIMD оптимизаций
 */
void check_simd_support() {
    std::cout << "\n🔧 SIMD оптимизации:\n";
    
    #ifdef USE_AVX2
        std::cout << "  ✅ AVX2: ВКЛЮЧЕН (компиляция с -mavx2)\n";
    #else
        std::cout << "  ❌ AVX2: ОТКЛЮЧЕН\n";
    #endif
    
    #ifdef USE_SSE42
        std::cout << "  ✅ SSE4.2: ВКЛЮЧЕН\n";
    #else
        std::cout << "  ❌ SSE4.2: ОТКЛЮЧЕН\n";
    #endif
    
    // Проверка во время выполнения
    if (SIMDUtils::check_avx2_support()) {
        std::cout << "  ✅ AVX2 поддерживается процессором\n";
    } else {
        std::cout << "  ❌ AVX2 НЕ поддерживается процессором\n";
    }
    
    std::cout << std::endl;
}

/**
 * @brief Поиск файлов модели
 */
bool find_model_files(FastBPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"../../bpe/vocab_complete.json", "../../bpe/merges.txt"},
        {"../../bpe/vocab.json", "../../bpe/merges.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"vocab.json", "merges.txt"}
    };
    
    for (const auto& [vpath, mpath] : candidates) {
        std::ifstream vfile(vpath);
        std::ifstream mfile(mpath);
        if (vfile.good() && mfile.good()) {
            vocab_path = vpath;
            merges_path = mpath;
            return tokenizer.load(vpath, mpath);
        }
    }
    
    return false;
}

// ======================================================================
// Демонстрационные функции
// ======================================================================

/**
 * @brief Демонстрация базового кодирования/декодирования
 */
void demo_basic_encoding(FastBPETokenizer& tokenizer, const std::vector<std::pair<std::string, std::string>>& examples) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "🔤 БАЗОВОЕ КОДИРОВАНИЕ/ДЕКОДИРОВАНИЕ\n";
    std::cout << std::string(60, '=') << "\n";
    
    int success_count = 0;
    
    for (size_t i = 0; i < examples.size(); ++i) {
        const auto& [desc, text] = examples[i];
        
        std::cout << "\n📝 Пример " << i+1 << ": " << desc << "\n";
        std::cout << "   Исходный: " << text << "\n";
        
        // Кодирование
        std::vector<uint32_t> tokens;
        {
            ScopedTimer timer("encode", false);
            tokens = tokenizer.encode(text);
        }
        
        // Статистика токенов
        std::cout << "   Токенов: " << tokens.size() << " [";
        for (size_t j = 0; j < std::min(size_t(10), tokens.size()); ++j) {
            std::cout << tokens[j];
            if (j < std::min(size_t(9), tokens.size()-1)) std::cout << ", ";
        }
        if (tokens.size() > 10) std::cout << ", ...";
        std::cout << "]\n";
        
        // Подсчет уникальных токенов
        std::set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
        std::cout << "   Уникальных: " << unique_tokens.size() << "\n";
        
        // Декодирование
        std::string decoded;
        {
            ScopedTimer timer("decode", false);
            decoded = tokenizer.decode(tokens);
        }
        
        bool success = (decoded == text);
        if (success) success_count++;
        
        std::cout << "   Декодировано: " << decoded << "\n";
        std::cout << "   Результат: " << (success ? "✅ УСПЕХ" : "❌ НЕУДАЧА") << "\n";
    }
    
    std::cout << "\n📊 Итог: " << success_count << "/" << examples.size() 
              << " примеров успешно (" << (success_count * 100 / examples.size()) << "%)\n";
}

/**
 * @brief Демонстрация пакетной обработки
 */
void demo_batch_processing(FastBPETokenizer& tokenizer, const std::vector<std::pair<std::string, std::string>>& examples) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "📦 ПАКЕТНАЯ ОБРАБОТКА\n";
    std::cout << std::string(60, '=') << "\n";
    
    // Подготовка данных
    std::vector<std::string> texts;
    std::vector<std::string_view> text_views;
    for (const auto& [_, text] : examples) {
        texts.push_back(text);
        text_views.push_back(texts.back());
    }
    
    std::cout << "   Размер батча: " << texts.size() << " текстов\n";
    
    // Последовательная обработка
    {
        ScopedTimer timer("Последовательная обработка");
        for (const auto& text : texts) {
            tokenizer.encode(text);
        }
    }
    
    // Пакетная обработка
    std::vector<std::vector<uint32_t>> batch_results;
    {
        ScopedTimer timer("Пакетная обработка");
        batch_results = tokenizer.encode_batch(text_views);
    }
    
    // Проверка корректности
    bool all_match = true;
    for (size_t i = 0; i < texts.size(); ++i) {
        auto single = tokenizer.encode(texts[i]);
        if (single.size() != batch_results[i].size()) {
            all_match = false;
            std::cout << "   ❌ Несовпадение в тексте " << i << "\n";
            break;
        }
    }
    
    std::cout << "   Корректность: " << (all_match ? "✅" : "❌") << "\n";
}

/**
 * @brief Демонстрация кэширования
 */
void demo_caching(FastBPETokenizer& tokenizer) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "💾 ЭФФЕКТИВНОСТЬ КЭШИРОВАНИЯ\n";
    std::cout << std::string(60, '=') << "\n";
    
    std::vector<std::string> repetitive_texts;
    for (int i = 0; i < 100; ++i) {
        repetitive_texts.push_back("int var" + std::to_string(i % 10) + " = " + std::to_string(i) + ";");
    }
    
    std::cout << "   Тестовых текстов: " << repetitive_texts.size() << "\n";
    std::cout << "   Уникальных паттернов: 10\n";
    
    tokenizer.reset_stats();
    
    // Первый проход (заполнение кэша)
    {
        ScopedTimer timer("Первый проход (заполнение кэша)");
        for (const auto& text : repetitive_texts) {
            tokenizer.encode(text);
        }
    }
    
    auto stats1 = tokenizer.stats();
    std::cout << "   Cache hits: " << stats1.cache_hits << "\n";
    std::cout << "   Cache misses: " << stats1.cache_misses << "\n";
    
    tokenizer.reset_stats();
    
    // Второй проход (использование кэша)
    {
        ScopedTimer timer("Второй проход (кэш)");
        for (const auto& text : repetitive_texts) {
            tokenizer.encode(text);
        }
    }
    
    auto stats2 = tokenizer.stats();
    std::cout << "   Cache hits: " << stats2.cache_hits << "\n";
    std::cout << "   Cache misses: " << stats2.cache_misses << "\n";
    std::cout << "   Hit rate: " << (stats2.cache_hit_rate() * 100) << "%\n";
}

/**
 * @brief Тест производительности
 */
void demo_performance(FastBPETokenizer& tokenizer) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "🚀 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ\n";
    std::cout << std::string(60, '=') << "\n";
    
    // Тест с разными размерами
    std::vector<size_t> sizes = {1024, 10*1024, 100*1024, 1024*1024};
    
    for (size_t size : sizes) {
        std::string text(size, 'a');  // Простой текст для измерения скорости
        
        // Прогрев
        for (int i = 0; i < 3; ++i) {
            tokenizer.encode(text);
        }
        
        tokenizer.reset_stats();
        
        ScopedTimer timer(std::to_string(size/1024) + " KB");
        auto tokens = tokenizer.encode(text);
        
        auto stats = tokenizer.stats();
        double mb_per_sec = (size / 1024.0 / 1024.0) / (stats.avg_encode_time_ms() / 1000.0);
        
        std::cout << "   Размер: " << (size/1024) << " KB, "
                  << "Токенов: " << tokens.size() << ", "
                  << "Скорость: " << std::fixed << std::setprecision(2) << mb_per_sec << " MB/s\n";
    }
}

/**
 * @brief Демонстрация SIMD оптимизаций
 */
void demo_simd(FastBPETokenizer& tokenizer) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "⚡ SIMD ОПТИМИЗАЦИИ\n";
    std::cout << std::string(60, '=') << "\n";
    
    if (!SIMDUtils::has_avx2()) {
        std::cout << "   AVX2 не поддерживается процессором\n";
        return;
    }
    
    std::string text(100000, 'a');  // 100KB текста
    
    // Прогрев
    for (int i = 0; i < 5; ++i) {
        tokenizer.encode(text);
    }
    
    tokenizer.reset_stats();
    
    // Измерение с SIMD
    double simd_time;
    {
        ScopedTimer timer("С SIMD", false);
        auto tokens = tokenizer.encode(text);
        simd_time = tokenizer.stats().avg_encode_time_ms();
    }
    
    // TODO: Сравнение без SIMD (если есть возможность отключить)
    std::cout << "   Время с SIMD: " << std::fixed << std::setprecision(2) << simd_time << " ms\n";
    std::cout << "   Скорость: " << (text.size() / 1024.0 / 1024.0) / (simd_time / 1000.0) << " MB/s\n";
}

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "🚀 FAST BPE TOKENIZER DEMO\n";
    std::cout << "========================================\n";
    
    // Парсинг аргументов командной строки
    bool verbose = false;
    bool skip_slow_tests = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--quick" || arg == "-q") {
            skip_slow_tests = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --verbose, -v    Подробный вывод\n";
            std::cout << "  --quick, -q      Пропустить медленные тесты\n";
            std::cout << "  --help, -h       Показать справку\n";
            return 0;
        }
    }
    
    try {
        // Проверка SIMD
        check_simd_support();
        
        // Создание токенизатора
        TokenizerConfig config;
        config.vocab_size = 32000;
        config.cache_size = 10000;
        config.byte_level = true;
        config.enable_cache = true;
        config.enable_profiling = true;
        
        std::cout << "📦 Создание токенизатора:\n";
        std::cout << "  • vocab_size: " << config.vocab_size << "\n";
        std::cout << "  • cache_size: " << config.cache_size << "\n";
        std::cout << "  • byte_level: " << (config.byte_level ? "да" : "нет") << "\n";
        std::cout << "  • enable_cache: " << (config.enable_cache ? "да" : "нет") << "\n";
        std::cout << std::endl;
        
        FastBPETokenizer tokenizer(config);
        
        // Загрузка модели
        std::cout << "📖 Загрузка модели...\n";
        
        std::string vocab_path, merges_path;
        ScopedTimer load_timer("Загрузка модели");
        
        if (!find_model_files(tokenizer, vocab_path, merges_path)) {
            std::cerr << "❌ Не удалось загрузить модель!\n";
            std::cerr << "   Убедитесь, что файлы существуют:\n";
            std::cerr << "   - ../../bpe/vocab_complete.json\n";
            std::cerr << "   - ../../bpe/merges.txt\n";
            return 1;
        }
        
        std::cout << "  ✅ Словарь: " << vocab_path << "\n";
        std::cout << "  ✅ Слияния: " << merges_path << "\n";
        std::cout << "  📚 Размер словаря: " << tokenizer.vocab_size() << " токенов\n";
        std::cout << "  🔗 Правил слияния: " << tokenizer.merges_count() << "\n";
        
        // Создание примеров
        auto examples = create_examples();
        
        // Демонстрации
        demo_basic_encoding(tokenizer, examples);
        
        if (!skip_slow_tests) {
            demo_batch_processing(tokenizer, examples);
            demo_caching(tokenizer);
            demo_performance(tokenizer);
            demo_simd(tokenizer);
        }
        
        // Финальная статистика
        auto stats = tokenizer.stats();
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "📊 ФИНАЛЬНАЯ СТАТИСТИКА\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  • Всего encode вызовов: " << stats.encode_calls << "\n";
        std::cout << "  • Всего decode вызовов: " << stats.decode_calls << "\n";
        std::cout << "  • Попаданий в кэш: " << stats.cache_hits << "\n";
        std::cout << "  • Промахов кэша: " << stats.cache_misses << "\n";
        std::cout << "  • Эффективность кэша: " << std::fixed << std::setprecision(1) 
                  << (stats.cache_hit_rate() * 100) << "%\n";
        std::cout << "  • Обработано токенов: " << stats.total_tokens_processed << "\n";
        std::cout << "  • Среднее время encode: " << std::fixed << std::setprecision(3) 
                  << stats.avg_encode_time_ms() << " ms\n";
        
        std::cout << "\n✅ Демо завершено успешно!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}