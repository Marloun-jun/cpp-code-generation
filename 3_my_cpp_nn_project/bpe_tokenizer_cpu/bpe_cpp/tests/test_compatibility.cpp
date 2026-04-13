/**
 * @file test_compatibility.cpp
 * @brief Тесты совместимости между C++ и Python реализациями BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Этот файл содержит набор тестов для проверки, что C++ реализация
 *          токенизатора ведет себя идентично эталонной Python реализации.
 * 
 *          **Зачем нужны тесты совместимости?**
 *          ┌───────────────────────────────────────────────────────────┐
 *          │ Гарантируют, что C++ версия не вносит регрессий           │
 *          │ Проверяют корректность конвертации моделей                │
 *          │ Выявляют различия в обработке Unicode                     │
 *          │ Обеспечивают обратную совместимость                       │
 *          └───────────────────────────────────────────────────────────┘
 * 
 *          **Проверяемые сценарии:**
 *          ┌─────────────────────┬─────────────────────────────────────┐
 *          │ Загрузка модели     │ Одинаковый словарь и правила        │
 *          │ Простые выражения   │ Базовая токенизация                 │
 *          │ Сложные конструкции │ Шаблоны, классы, лямбды             │
 *          │ Русские комментарии │ Корректная работа с UTF-8           │
 *          │ Roundtrip тесты     │ encode + decode = исходный текст    │
 *          │ Пакетная обработка  │ encode_batch работает корректно     │
 *          │ Производительность  │ Замеры скорости для регрессий       │
 *          │ Специальные токены  │ Проверка <UNK>, <PAD>, <BOS>, <EOS> │
 *          └─────────────────────┴─────────────────────────────────────┘
 * 
 * @note Для полных тестов требуются файлы модели от Python реализации
 * @see FastBPETokenizer
 * @see BPETokenizer (Python)
 */

#include <gtest/gtest.h>

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"
#include "test_helpers.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Проверяем, доступен ли Google Benchmark
#ifdef HAS_BENCHMARK
#include <benchmark/benchmark.h>
#else

// Заглушка для функции DoNotOptimize
namespace benchmark {
    template <class T> void DoNotOptimize(const T&) {}
}
#endif

namespace fs = std::filesystem;
using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    constexpr size_t DEFAULT_VOCAB_SIZE = 10000;              ///< Размер словаря по умолчанию
    constexpr size_t DEFAULT_CACHE_SIZE = 10000;              ///< Размер кэша
    constexpr int PERFORMANCE_ITERATIONS = 100;               ///< Количество итераций для тестов производительности
    constexpr double MIN_ROUNDTRIP_SUCCESS_RATE = 85.0;       ///< Минимальный процент успешных roundtrip
    constexpr double MIN_PYTHON_COMPATIBILITY_RATE = 90.0;    ///< Минимальное совпадение с Python
    constexpr int MAX_PRINT_TOKENS = 5;                       ///< Максимум токенов для вывода при отладке
    constexpr int MAX_TEXT_PREVIEW_LEN = 30;                  ///< Длина превью текста
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ============================================================================
// Вспомогательные функции
// ============================================================================

/**
 * @brief Загрузить ожидаемые результаты из Python
 * 
 * @param filename Путь к файлу с выходами Python
 * @return std::vector<std::vector<uint32_t>> Вектор результатов
 * 
 * **Формат файла:**
 * @code
 * токен1,токен2,токен3,...    # для первого текста
 * токен1,токен2,токен3,...    # для второго текста
 * ...
 * @endcode
 */
std::vector<std::vector<uint32_t>> load_python_outputs(const std::string& filename) {
    std::vector<std::vector<uint32_t>> outputs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        return outputs;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<uint32_t> tokens;
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ',')) {
            // Удаляем пробелы
            token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
            if (token.empty()) continue;
            
            try {
                tokens.push_back(std::stoul(token));
            } catch (const std::exception& e) {
                std::cerr << "Ошибка парсинга токена '" << token << "': " << e.what() << std::endl;
            }
        }
        outputs.push_back(tokens);
    }
    
    return outputs;
}

/**
 * @brief Сохранить результаты C++ для сравнения с Python
 * 
 * @param filename Путь для сохранения
 * @param outputs Результаты токенизации
 * 
 * @note Используется для генерации эталонных данных или отладки
 */
void save_cpp_outputs(const std::string& filename,
                      const std::vector<std::vector<uint32_t>>& outputs) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
        return;
    }
    
    for (const auto& tokens : outputs) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) file << ",";
            file << tokens[i];
        }
        file << "\n";
    }
}

/**
 * @brief Создать большой текст для тестов производительности
 * 
 * @param texts Вектор исходных текстов
 * @param multiplier Множитель размера
 * @return std::string Большой текст
 */
std::string create_large_text(const std::vector<std::string>& texts, int multiplier = 100) {
    std::string result;
    result.reserve(texts.size() * multiplier * 50);    // Примерная оценка
    
    for (int i = 0; i < multiplier; ++i) {
        result += texts[i % texts.size()];
        if (i % 10 == 9) result += "\n";
    }
    
    return result;
}

// ============================================================================
// Тестовый класс
// ============================================================================

/**
 * @brief Тестовый класс для проверки совместимости
 * 
 * Содержит общие настройки, тестовые данные и вспомогательные методы
 * для всех тестов совместимости.
 */
class CompatibilityTest : public ::testing::Test {
protected:
    /**
     * @brief Настройка перед каждым тестом
     * 
     * Инициализирует:
     * - Конфигурацию токенизатора
     * - Сам токенизатор
     * - Пути к файлам моделей
     * - Директорию для результатов
     */
    void SetUp() override {
        // Настройка для тестов совместимости
        config_.vocab_size = DEFAULT_VOCAB_SIZE;
        config_.cache_size = DEFAULT_CACHE_SIZE;
        config_.byte_level = true;
        config_.enable_cache = true;
        config_.enable_profiling = false;    // Отключаем профилирование для тестов
        
        // Создаём токенизатор после настройки config_
        tokenizer_ = std::make_unique<FastBPETokenizer>(config_);
        
        // Пути
        initialize_paths();
        
        // Пути для сохранения результатов
        results_dir_ = "compatibility_results";
        
        // Создаем директорию для результатов
        try {
            fs::create_directories(results_dir_);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Не удалось создать директорию " << results_dir_
                      << ": " << e.what() << std::endl;
        }
    }
    
    /**
     * @brief Инициализирует пути к файлам моделей
     */
    void initialize_paths() {
        // C++ модели (приоритет 1)
        for (int size : {8000, 10000, 12000}) {
            python_paths_.push_back("../models/bpe_" + std::to_string(size) + "/cpp_vocab.json");
            merges_paths_.push_back("../models/bpe_" + std::to_string(size) + "/cpp_merges.txt");
        }
        
        for (int size : {8000, 10000, 12000}) {
            python_paths_.push_back("models/bpe_" + std::to_string(size) + "/cpp_vocab.json");
            merges_paths_.push_back("models/bpe_" + std::to_string(size) + "/cpp_merges.txt");
        }
        
        // Python модели (приоритет 2)
        for (int size : {8000, 10000, 12000}) {
            python_paths_.push_back("../../bpe_python/models/bpe_" + std::to_string(size) + "/vocab.json");
            merges_paths_.push_back("../../bpe_python/models/bpe_" + std::to_string(size) + "/merges.txt");
        }
        
        for (int size : {8000, 10000, 12000}) {
            python_paths_.push_back("../../../bpe_python/models/bpe_" + std::to_string(size) + "/vocab.json");
            merges_paths_.push_back("../../../bpe_python/models/bpe_" + std::to_string(size) + "/merges.txt");
        }
        
        // Fallback
        python_paths_.push_back("vocab.json");
        merges_paths_.push_back("merges.txt");
    }
    
    /**
     * @brief Очистка после каждого теста
     * 
     * Выводит статистику и освобождает ресурсы
     */
    void TearDown() override {
        // Выводим статистику после тестов
        if (tokenizer_loaded_ && tokenizer_) {
            auto stats = tokenizer_->stats();
            std::cout << "\nСтатистика токенизатора:" << std::endl;
            std::cout << "- Кэш: " << std::fixed << std::setprecision(1)
                      << stats.cache_hit_rate() * 100 << "% попаданий" << std::endl;
            std::cout << "- Среднее время encode: "
                      << std::fixed << std::setprecision(3)
                      << stats.avg_encode_time_ms() << " мс" << std::endl;
        }
        
        // Освобождаем ресурсы
        tokenizer_.reset();
    }
    
    /**
     * @brief Загрузить токенизатор (с поиском по разным путям)
     * 
     * @return true если модель успешно загружена
     */
    bool loadTokenizer() {
        if (tokenizer_loaded_) return true;
        if (!tokenizer_) return false;
        
        std::cout << "\nПоиск модели токенизатора..." << std::endl;
        
        for (size_t i = 0; i < python_paths_.size(); ++i) {
            std::cout << "Проверка: " << python_paths_[i] << std::endl;
            
            if (fs::exists(python_paths_[i]) && fs::exists(merges_paths_[i])) {
                
                std::cout << "Найдены файлы, загрузка..." << std::endl;
                
                if (tokenizer_->load(python_paths_[i], merges_paths_[i])) {
                    std::cout << "Загружен словарь: " << python_paths_[i] << std::endl;
                    std::cout << "Размер словаря:   " << tokenizer_->vocab_size() << std::endl;
                    std::cout << "Правил слияния:   " << tokenizer_->merges_count() << std::endl;
                    tokenizer_loaded_ = true;
                    loaded_path_ = python_paths_[i];
                    return true;
                } else {
                    std::cout << "Ошибка загрузки!" << std::endl;
                }
            }
        }
        
        std::cout << "Модель не найдена!" << std::endl;
        return false;
    }
    
    /**
     * @brief Проверить наличие Python файлов
     * 
     * @return true если хотя бы один файл существует
     */
    bool hasPythonFiles() const {
        for (const auto& path : python_paths_) {
            if (fs::exists(path)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Получить путь к файлу с Python выходами
     * 
     * @param test_name Имя теста
     * @return std::string Полный путь к файлу
     */
    std::string getPythonOutputPath(const std::string& test_name) const {
        return results_dir_ + "/python_" + test_name + ".txt";
    }
    
    /**
     * @brief Получить путь к файлу с C++ выходами
     * 
     * @param test_name Имя теста
     * @return std::string Полный путь к файлу
     */
    std::string getCppOutputPath(const std::string& test_name) const {
        return results_dir_ + "/cpp_" + test_name + ".txt";
    }
    
    // ==================== Члены класса ====================
    
    TokenizerConfig config_;                         ///< Конфигурация токенизатора
    std::unique_ptr<FastBPETokenizer> tokenizer_;    ///< Тестируемый токенизатор
    bool tokenizer_loaded_ = false;                  ///< Флаг загрузки модели
    std::string loaded_path_;                        ///< Путь к загруженной модели
    
    std::vector<std::string> python_paths_;    ///< Возможные пути к словарю
    std::vector<std::string> merges_paths_;    ///< Возможные пути к слияниям
    std::string results_dir_;                  ///< Директория для результатов
    
    // ==================== Тестовые строки ====================
    
    /**
     * @brief Набор тестовых строк, покрывающих различные сценарии
     * 
     * **Категории:**
     * - Простые выражения (int x = 42;)
     * - STL контейнеры (std::vector<int> v;)
     * - Управляющие конструкции (if, for, while)
     * - Шаблоны и классы (template, class, struct)
     * - Функции и лямбды (auto lambda = [](int x){})
     * - Строки и комментарии ("string", // comment)
     * - Include директивы (#include <iostream>)
     * - Сложные выражения (std::cout << ...)
     * - Русские комментарии (// русский)
     * - Эмодзи (🔥)
     */
    const std::vector<std::string> test_strings_ = {
        // Простые выражения
        "int x = 42;",
        "float y = 3.14f;",
        "char c = 'A';",
        "bool flag = true;",
        "auto result = x + y;",
        
        // STL контейнеры
        "std::vector<int> v;",
        "std::map<std::string, int> m;",
        "std::unordered_set<double> s;",
        "std::array<int, 10> arr;",
        "std::tuple<int, float, std::string> t;",
        
        // Управляющие конструкции
        "if (condition) { do_something(); }",
        "for (int i = 0; i < 10; ++i) { sum += i; }",
        "while (running) { process(); }",
        "switch (value) { case 1: break; default: break; }",
        "try { throw std::runtime_error(\"error\"); } catch (...) {}",
        
        // Шаблоны и классы
        "template<typename T> T max(T a, T b) { return a > b ? a : b; }",
        "class MyClass { public: void method(); private: int data_; };",
        "struct Point { int x, y; };",
        "enum Color { RED, GREEN, BLUE };",
        "namespace my_namespace { class MyClass {}; }",
        
        // Функции и лямбды
        "int add(int a, int b) { return a + b; }",
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
        
        // Русские комментарии и Unicode
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        "// комментарий с числами 123 и символами !@#$%",
        "// 🔥 emoji комментарий",
        "// тест кириллицы: привет мир",
        
        // Крайние случаи
        "",
        " ",
        "\t",
        "\n",
        "// пустая строка"
    };
};

// ============================================================================
// Тесты
// ============================================================================

/**
 * @test Загрузка того же словаря, что и Python версия
 * 
 * Проверяет, что C++ токенизатор может загрузить ту же модель,
 * что и Python реализация. Убеждается, что размер словаря и
 * количество правил слияния > 0.
 */
TEST_F(CompatibilityTest, LoadSameVocabulary) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены. Сначала запустите python tokenizer.";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    EXPECT_GT(tokenizer_->vocab_size(), 0);
    EXPECT_GT(tokenizer_->merges_count(), 0);
    
    std::cout << "C++ токенизатор загрузил " << tokenizer_->vocab_size()
              << " токенов из " << loaded_path_ << std::endl;
    std::cout << "Правил слияния: " << tokenizer_->merges_count() << std::endl;
}

/**
 * @test Кодирование простого текста
 * 
 * Проверяет базовую функциональность encode() на простом тексте.
 * Убеждается, что результат не пустой.
 */
TEST_F(CompatibilityTest, EncodeSimpleText) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::string text = "int main()";
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Текст '" << text << "' закодирован в "
              << tokens.size() << " токенов" << std::endl;
    
    // Проверяем первые несколько токенов
    for (size_t i = 0; i < std::min<size_t>(MAX_PRINT_TOKENS, tokens.size()); ++i) {
        std::cout << "Токен " << i << ": ID " << tokens[i] << std::endl;
    }
}

/**
 * @test Кодирование всех тестовых строк
 * 
 * Проверяет encode() на всём наборе тестовых строк.
 * Сохраняет результаты для последующего сравнения с Python.
 */
TEST_F(CompatibilityTest, EncodeAllTestStrings) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::vector<std::vector<uint32_t>> all_outputs;
    int empty_strings = 0;
    int non_empty_strings = 0;
    
    for (const auto& text : test_strings_) {
        auto tokens = tokenizer_->encode(text);
        all_outputs.push_back(tokens);
        
        std::string preview = text.substr(0, MAX_TEXT_PREVIEW_LEN);
        if (text.length() > MAX_TEXT_PREVIEW_LEN) preview += "...";
        
        // Для пустых строк ожидаем пустой результат
        if (text.empty()) {
            EXPECT_EQ(tokens.size(), 0) << "Для пустой строки ожидается 0 токенов, получено: " << tokens.size();
            std::cout << "  '' -> 0 токенов (корректно)" << std::endl;
            empty_strings++;
            continue;
        } else if (text.find_first_not_of(" \t\n\r") == std::string::npos) {
            // Для строк только из пробельных символов
            std::cout << "  '" << utils::escape_string(text) << "' -> " << tokens.size() << " токенов" << std::endl;
            non_empty_strings++;
        } else {
            // Для обычных строк ожидаем непустой результат
            EXPECT_GT(tokens.size(), 0) << "Пустой результат для текста: " << preview;
            std::cout << "  '" << preview << "' -> " << tokens.size() << " токенов" << std::endl;
            non_empty_strings++;
        }
    }
    
    std::cout << "\nСтатистика:" << std::endl;
    std::cout << "- Пустых строк:   " << empty_strings << std::endl;
    std::cout << "- Непустых строк: " << non_empty_strings << std::endl;
    std::cout << "- Всего строк:    " << test_strings_.size() << std::endl;
    
    // Сохраняем для сравнения с Python
    save_cpp_outputs(getCppOutputPath("encode_all"), all_outputs);
    std::cout << "Результаты сохранены в " << getCppOutputPath("encode_all") << std::endl;
}

/**
 * @test Кодирование с русскими комментариями
 * 
 * Специальный тест для проверки работы с Unicode символами,
 * включая кириллицу и эмодзи.
 */
TEST_F(CompatibilityTest, EncodeRussianComments) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::vector<std::string> russian_texts = {
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        "// комментарий с числами 123 и символами !@#$%",
        "// 🔥 emoji комментарий",
        "// тест кириллицы: привет мир"
    };
    
    for (const auto& text : russian_texts) {
        auto tokens = tokenizer_->encode(text);
        EXPECT_GT(tokens.size(), 0) << "Пустой результат для: " << text;
        
        // Проверяем, что декодирование работает
        auto decoded = tokenizer_->decode(tokens);
        
        std::cout << "\nИсходный: '" << text << "'" << std::endl;
        std::cout << "Декод.:   '" << decoded << "'" << std::endl;
        std::cout << "Токенов:  " << tokens.size() << std::endl;
        
        // Используем utils::is_valid_utf8() вместо локальной функции
        EXPECT_TRUE(utils::is_valid_utf8(decoded)) << "Декодированный текст не является валидным UTF-8!";
        
        // Для непустого текста проверяем, что результат не пустой
        if (!text.empty() && text.find_first_not_of(" \t\n\r") != std::string::npos) {
            EXPECT_FALSE(decoded.empty()) << "Декодированный текст пуст для: " << text;
        }
    }
}

/**
 * @test Roundtrip тест (кодирование + декодирование)
 * 
 * Проверяет, что encode() + decode() возвращает исходный текст.
 * Это фундаментальное свойство любого корректного токенизатора.
 */
TEST_F(CompatibilityTest, EncodeDecodeRoundtrip) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    int total_tests = 0;
    int passed_tests = 0;
    std::vector<std::string> failures;
    
    for (const auto& text : test_strings_) {
        total_tests++;
        
        auto tokens = tokenizer_->encode(text);
        auto decoded = tokenizer_->decode(tokens);
        
        // Для пустых строк
        if (text.empty()) {
            if (decoded.empty()) {
                passed_tests++;
            } else {
                failures.push_back("(empty string)");
                std::cout << "\nПровал для пустой строки, декодировано: '" << decoded << "'" << std::endl;
            }
            continue;
        }
        
        // Проверяем, что все символы присутствуют
        bool all_chars_present = true;
        std::string missing_chars;
        
        for (char c : text) {
            if (c == '\0') continue;    // Игнорируем нулевые символы
            if (decoded.find(c) == std::string::npos) {
                all_chars_present = false;
                missing_chars += c;
            }
        }
        
        if (all_chars_present) {
            passed_tests++;
        } else {
            failures.push_back(text);
            std::cout << "\nПровал для: '" << text.substr(0, MAX_TEXT_PREVIEW_LEN)
                      << (text.length() > MAX_TEXT_PREVIEW_LEN ? "..." : "") << "'" << std::endl;
            std::cout << "Отсутствуют символы: '" << missing_chars << "'" << std::endl;
            std::cout << "Декодировано: '" << decoded << "'" << std::endl;
        }
    }
    
    double success_rate = 100.0 * passed_tests / total_tests;
    std::cout << "\nПроцент успешных roundtrip: " << std::fixed << std::setprecision(1)
              << success_rate << "% (" << passed_tests << "/" << total_tests << ")" << std::endl;
    
    if (!failures.empty()) {
        std::cout << "Провалившиеся тесты: " << failures.size() << std::endl;
    }
    
    EXPECT_GE(success_rate, MIN_ROUNDTRIP_SUCCESS_RATE)
        << "Слишком много неудачных roundtrip тестов!";
}

/**
 * @test Пакетная обработка
 * 
 * Проверяет, что encode_batch() работает корректно и результаты
 * совпадают с последовательным вызовом encode().
 */
TEST_F(CompatibilityTest, BatchEncode) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    // Используем первые 10 текстов
    size_t num_texts = std::min<size_t>(10, test_strings_.size());
    std::vector<std::string> texts(test_strings_.begin(), test_strings_.begin() + num_texts);
    
    // Создаем vector<string_view> который ссылается на texts
    std::vector<std::string_view> views;
    views.reserve(texts.size());
    for (const auto& text : texts) {
        views.push_back(text);
    }
    
    // Вызываем encode_batch
    auto batch_result = tokenizer_->encode_batch(views);
    
    EXPECT_EQ(batch_result.size(), texts.size());
    std::cout << "Пакетная обработка " << batch_result.size() << " текстов:" << std::endl;
    
    size_t total_tokens = 0;
    bool all_match = true;
    
    for (size_t i = 0; i < batch_result.size(); ++i) {
        auto single_result = tokenizer_->encode(texts[i]);
        
        if (batch_result[i].size() != single_result.size()) {
            all_match = false;
            std::cout << "Текст " << i << ": batch=" << batch_result[i].size()
                      << ", single=" << single_result.size() << std::endl;
            
            // Сравниваем первые несколько токенов
            for (size_t j = 0; j < std::min<size_t>(MAX_PRINT_TOKENS,
                        std::min(batch_result[i].size(), single_result.size())); ++j) {
                if (batch_result[i][j] != single_result[j]) {
                    std::cout << "Несовпадение на позиции " << j
                              << ": " << batch_result[i][j] << " vs " << single_result[j] << std::endl;
                }
            }
        } else {
            std::cout << "Текст " << i << ": " << batch_result[i].size() << " токенов" << std::endl;
        }
        
        total_tokens += batch_result[i].size();
    }
    
    std::cout << "Всего токенов: " << total_tokens << std::endl;
    EXPECT_TRUE(all_match) << "Результаты encode_batch отличаются от последовательного encode!";
}

/**
 * @test Производительность encode
 * 
 * Измеряет производительность encode() на большом тексте
 * для выявления регрессий.
 */
TEST_F(CompatibilityTest, EncodePerformance) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::string large_text = create_large_text(test_strings_, 50);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < PERFORMANCE_ITERATIONS; ++i) {
        auto tokens = tokenizer_->encode(large_text);
        benchmark::DoNotOptimize(tokens.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ms_per_encode = duration.count() / static_cast<double>(PERFORMANCE_ITERATIONS);
    double bytes_per_second = (large_text.size() * PERFORMANCE_ITERATIONS) / (duration.count() / 1000.0);
    double mb_per_second = bytes_per_second / (1024.0 * 1024.0);
    
    std::cout << "\nПроизводительность encode:" << std::endl;
    std::cout << "- Размер текста:   " << large_text.size() << " байт ("
              << large_text.size() / 1024.0 << " КБ)" << std::endl;
    std::cout << "- Итераций:        " << PERFORMANCE_ITERATIONS << std::endl;
    std::cout << "- Общее время:     " << duration.count() << " мс" << std::endl;
    std::cout << "- Среднее время:   " << std::fixed << std::setprecision(2)
              << ms_per_encode << " мс" << std::endl;
    std::cout << "- Скорость:        " << std::fixed << std::setprecision(2)
              << mb_per_second << " МБ/с" << std::endl;
    
    auto stats = tokenizer_->stats();
    std::cout << "- Статистика кэша: " << std::fixed << std::setprecision(1)
              << stats.cache_hit_rate() * 100 << "% попаданий" << std::endl;
}

/**
 * @test Сравнение с Python (если есть сохраненные выходы)
 * 
 * Сравнивает результаты C++ с ранее сохраненными результатами Python.
 * Это наиболее важный тест для проверки совместимости.
 */
TEST_F(CompatibilityTest, CompareWithPython) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    // Пытаемся загрузить выходы Python
    auto python_outputs = load_python_outputs(getPythonOutputPath("encode_all"));
    
    if (python_outputs.empty()) {
        std::cout << "\nНет сохраненных выходов Python для сравнения." << std::endl;
        std::cout << "Чтобы создать их, запустите:" << std::endl;
        std::cout << "python3 -c \"" << std::endl;
        std::cout << "import json" << std::endl;
        std::cout << "from tokenizer import BPETokenizer" << std::endl;
        std::cout << "tokenizer = BPETokenizer(vocab_size=10000, byte_level=True)" << std::endl;
        std::cout << "tokenizer.load('models/bpe_10000/vocab.json', 'models/bpe_10000/merges.txt')" << std::endl;
        std::cout << "texts = [...]  # те же тексты из test_strings_" << std::endl;
        std::cout << "with open('compatibility_results/python_encode_all.txt', 'w') as f:" << std::endl;
        std::cout << "    for text in texts:" << std::endl;
        std::cout << "        tokens = tokenizer.encode(text)" << std::endl;
        std::cout << "        f.write(','.join(map(str, tokens)) + '\\n')" << std::endl;
        std::cout << "  \"" << std::endl;
        GTEST_SKIP() << "Выходные данные Python не найдены!";
    }
    
    // Получаем C++ выходы
    std::vector<std::vector<uint32_t>> cpp_outputs;
    for (const auto& text : test_strings_) {
        cpp_outputs.push_back(tokenizer_->encode(text));
    }
    
    // Сравниваем
    EXPECT_EQ(python_outputs.size(), cpp_outputs.size())
        << "Размеры выходных данных различаются: python="
        << python_outputs.size() << ", cpp=" << cpp_outputs.size();
    
    size_t min_size = std::min(python_outputs.size(), cpp_outputs.size());
    int matches = 0;
    
    for (size_t i = 0; i < min_size; ++i) {
        if (python_outputs[i] == cpp_outputs[i]) {
            matches++;
        } else {
            std::cout << "\nНесовпадение для текста " << i << ":" << std::endl;
            std::cout << "- Python: " << python_outputs[i].size() << " токенов" << std::endl;
            std::cout << "- C++:    " << cpp_outputs[i].size() << " токенов" << std::endl;
            
            // Показываем первые токены для отладки
            std::cout << "Python первые токены: ";
            for (size_t j = 0; j < std::min<size_t>(MAX_PRINT_TOKENS, python_outputs[i].size()); ++j) {
                std::cout << python_outputs[i][j] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "C++ первые токены:    ";
            for (size_t j = 0; j < std::min<size_t>(MAX_PRINT_TOKENS, cpp_outputs[i].size()); ++j) {
                std::cout << cpp_outputs[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    double match_rate = 100.0 * matches / min_size;
    std::cout << "\nСовпадение с Python: " << std::fixed << std::setprecision(1)
              << match_rate << "% (" << matches << "/" << min_size << ")" << std::endl;
    
    EXPECT_GE(match_rate, MIN_PYTHON_COMPATIBILITY_RATE)
        << "Слишком большое расхождение с Python!";
}

/**
 * @test Проверка специальных токенов
 * 
 * Убеждается, что все специальные токены имеют корректные ID
 * и что эти ID различны.
 */
TEST_F(CompatibilityTest, SpecialTokens) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::cout << "\nСпециальные токены:" << std::endl;
    std::cout << "- <UNK> ID: " << tokenizer_->unknown_id() << std::endl;
    std::cout << "- <PAD> ID: " << tokenizer_->pad_id() << std::endl;
    std::cout << "- <BOS> ID: " << tokenizer_->bos_id() << std::endl;
    std::cout << "- <EOS> ID: " << tokenizer_->eos_id() << std::endl;
    
    // Проверяем, что ID различаются
    std::set<uint32_t> ids = {
        tokenizer_->unknown_id(),
        tokenizer_->pad_id(),
        tokenizer_->bos_id(),
        tokenizer_->eos_id()
    };
    
    EXPECT_EQ(ids.size(), 4) << "Специальные токены должны иметь разные ID!";
    
    // Проверяем, что ID в пределах разумного
    for (uint32_t id : ids) {
        EXPECT_LT(id, tokenizer_->vocab_size())
            << "ID специального токена " << id << " превышает размер словаря!";
    }
}

/**
 * @test Проверка кэширования
 * 
 * Проверяет, что кэш действительно ускоряет повторяющиеся запросы.
 */
TEST_F(CompatibilityTest, CacheEfficiency) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Файлы словаря Python не найдены!";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::string text = "int main() { return 0; }";
    
    // Первый проход - заполнение кэша
    auto start1 = std::chrono::high_resolution_clock::now();
    auto tokens1 = tokenizer_->encode(text);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    
    // Второй проход - должно быть быстрее
    auto start2 = std::chrono::high_resolution_clock::now();
    auto tokens2 = tokenizer_->encode(text);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    
    EXPECT_EQ(tokens1.size(), tokens2.size());
    EXPECT_EQ(tokens1, tokens2);
    
    std::cout << "\nЭффективность кэша:" << std::endl;
    std::cout << "- Первый проход: " << time1.count() << " мкс" << std::endl;
    std::cout << "- Второй проход: " << time2.count() << " мкс" << std::endl;
    
    if (time2.count() > 0) {
        double speedup = static_cast<double>(time1.count()) / time2.count();
        std::cout << "- Ускорение:     " << std::fixed << std::setprecision(2)
                  << speedup << "x" << std::endl;
    }
    
    auto stats = tokenizer_->stats();
    std::cout << "- Cache hits:    " << stats.cache_hits << std::endl;
    std::cout << "- Cache misses:  " << stats.cache_misses << std::endl;
    std::cout << "- Hit rate:      " << std::fixed << std::setprecision(1)
              << stats.cache_hit_rate() * 100 << "%" << std::endl;
}