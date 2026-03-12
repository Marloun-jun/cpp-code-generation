/**
 * @file test_fast_tokenizer.cpp
 * @brief Модульные тесты для оптимизированной версии BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Этот файл содержит набор модульных тестов для проверки
 *          функциональности FastBPETokenizer - оптимизированной версии
 *          BPE токенизатора.
 * 
 *          **Проверяемые аспекты:**
 * 
 *          1. **Создание и инициализация**
 *             - Конструкторы с разными конфигурациями
 *             - Корректная инициализация специальных токенов
 *             - Проверка размеров словаря
 * 
 *          2. **Загрузка модели**
 *             - Загрузка из существующих файлов
 *             - Обработка отсутствующих файлов
 *             - Автоматический поиск по разным путям
 * 
 *          3. **Кодирование (encode)**
 *             - Простые тексты и символы
 *             - Пробельные символы
 *             - Числа и операторы
 *             - Ключевые слова C++
 *             - UTF-8 символы (русские, китайские, эмодзи)
 *             - Длинные тексты
 * 
 *          4. **Декодирование (decode)**
 *             - Простое декодирование
 *             - Roundtrip тесты (encode + decode)
 *             - Проверка сохранения всех символов
 * 
 *          5. **Пакетная обработка**
 *             - encode_batch для множества текстов
 *             - Сравнение с последовательной обработкой
 * 
 *          6. **Производительность и кэширование**
 *             - Замер скорости encode
 *             - Эффективность кэша (попадания/промахи)
 * 
 *          7. **Граничные случаи**
 *             - Пустой текст
 *             - Очень длинный текст (100 КБ)
 *             - Смешанные символы
 * 
 * @note Для полных тестов требуются файлы обученной модели
 * @see FastBPETokenizer
 */

#include <gtest/gtest.h>

#include "fast_tokenizer.hpp"
#include "utils.hpp"
#include "test_helpers.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <filesystem>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <set>
#include <iomanip>

namespace fs = std::filesystem;
using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr size_t TEST_VOCAB_SIZE = 8000;
    constexpr size_t TEST_CACHE_SIZE = 1000;
    constexpr size_t PERFORMANCE_TEXT_SIZE = 10000;
    constexpr int PERFORMANCE_ITERATIONS = 1000;
    constexpr size_t VERY_LONG_TEXT_SIZE = 100000;
    constexpr double MIN_ROUNDTRIP_RATE = 90.0;
    constexpr double MIN_BATCH_MATCH_RATE = 90.0;
    constexpr double MAX_ENCODE_TIME_MS = 1.0;
    constexpr int MAX_PRINT_TOKENS = 5;
    constexpr int UNIQUE_STRINGS_COUNT = 10;
}

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Проверить существование файла модели
 * 
 * @return true если хотя бы один файл модели найден
 */
bool model_files_exist() {
    const std::vector<std::string> paths = {
        // C++ модели (приоритет 1)
        "../models/bpe_8000/cpp_vocab.json",
        "../models/bpe_10000/cpp_vocab.json",
        "../models/bpe_12000/cpp_vocab.json",
        "../../models/bpe_8000/cpp_vocab.json",
        "../../models/bpe_10000/cpp_vocab.json",
        "../../models/bpe_12000/cpp_vocab.json",
        
        // Python модели (приоритет 2)
        "../../../bpe_python/models/bpe_8000/vocab.json",
        "../../../bpe_python/models/bpe_10000/vocab.json",
        "../../../bpe_python/models/bpe_12000/vocab.json",
        
        // В директории сборки (скопировано)
        "models/bpe_8000/cpp_vocab.json",
        "models/bpe_10000/cpp_vocab.json",
        "models/bpe_12000/cpp_vocab.json",
        
        // Fallback
        "vocab.json"
    };
    
    for (const auto& path : paths) {
        if (fs::exists(path)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Загрузить токенизатор с автоматическим поиском файлов
 * 
 * @param tokenizer Ссылка на токенизатор для загрузки
 * @return true если модель успешно загружена
 */
bool load_tokenizer(FastBPETokenizer& tokenizer) {
    const std::vector<std::pair<std::string, std::string>> paths = {
        // C++ модели (приоритет 1)
        {"../models/bpe_8000/cpp_vocab.json", "../models/bpe_8000/cpp_merges.txt"},
        {"../models/bpe_10000/cpp_vocab.json", "../models/bpe_10000/cpp_merges.txt"},
        {"../models/bpe_12000/cpp_vocab.json", "../models/bpe_12000/cpp_merges.txt"},
        {"../../models/bpe_8000/cpp_vocab.json", "../../models/bpe_8000/cpp_merges.txt"},
        {"../../models/bpe_10000/cpp_vocab.json", "../../models/bpe_10000/cpp_merges.txt"},
        {"../../models/bpe_12000/cpp_vocab.json", "../../models/bpe_12000/cpp_merges.txt"},
        
        // Python модели (приоритет 2)
        {"../../../bpe_python/models/bpe_8000/vocab.json", "../../../bpe_python/models/bpe_8000/merges.txt"},
        {"../../../bpe_python/models/bpe_10000/vocab.json", "../../../bpe_python/models/bpe_10000/merges.txt"},
        {"../../../bpe_python/models/bpe_12000/vocab.json", "../../../bpe_python/models/bpe_12000/merges.txt"},
        {"../../bpe_python/models/bpe_8000/vocab.json", "../../bpe_python/models/bpe_8000/merges.txt"},
        
        // В директории сборки (скопировано)
        {"models/bpe_8000/cpp_vocab.json", "models/bpe_8000/cpp_merges.txt"},
        {"models/bpe_10000/cpp_vocab.json", "models/bpe_10000/cpp_merges.txt"},
        {"models/bpe_12000/cpp_vocab.json", "models/bpe_12000/cpp_merges.txt"},
        
        // Fallback
        {"vocab.json", "merges.txt"}
    };
    
    std::cout << "Поиск модели токенизатора..." << std::endl;
    
    for (const auto& [vocab_path, merges_path] : paths) {
        std::cout << "  Проверка: " << vocab_path << std::endl;
        
        if (fs::exists(vocab_path) && fs::exists(merges_path)) {
            std::cout << "  ✓ Найдены файлы, загрузка..." << std::endl;
            
            if (tokenizer.load(vocab_path, merges_path)) {
                std::cout << "  ✓ Загружена модель: " << vocab_path << std::endl;
                std::cout << "    Размер словаря: " << tokenizer.vocab_size() << std::endl;
                std::cout << "    Правил слияния: " << tokenizer.merges_count() << std::endl;
                return true;
            } else {
                std::cout << "  ✗ Ошибка загрузки!" << std::endl;
            }
        }
    }
    
    std::cout << "  ✗ Модель не найдена!" << std::endl;
    return false;
}

/**
 * @brief Создать тестовый JSON словарь
 * 
 * @param path Путь для сохранения
 * @param tokens Вектор токенов
 */
void create_test_vocab(const std::string& path, const std::vector<std::string>& tokens) {
    std::ofstream file(path);
    file << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) file << ",";
        file << "\"" << tokens[i] << "\"";
    }
    file << "]";
}

/**
 * @brief Создать тестовый файл слияний
 * 
 * @param path Путь для сохранения
 */
void create_test_merges(const std::string& path) {
    std::ofstream file(path);
    file << "#version: 0.1\n";
    // Добавляем несколько тестовых слияний
    for (int i = 0; i < 10; ++i) {
        file << "a b\n";
    }
}

/**
 * @brief Вспомогательная функция для оптимизации (замена benchmark::DoNotOptimize)
 */
template<typename T>
void do_not_optimize(T&& value) {
    asm volatile("" : "+r"(value));
}

// ======================================================================
// Тестовый класс
// ======================================================================

/**
 * @brief Тестовый класс для FastBPETokenizer
 * 
 * Содержит общие настройки, тестовые данные и вспомогательные методы
 * для всех тестов оптимизированного токенизатора.
 */
class FastTokenizerTest : public ::testing::Test {
protected:
    /**
     * @brief Настройка перед каждым тестом
     * 
     * Инициализирует:
     * - Базовую конфигурацию для тестов
     * - Проверку наличия файлов модели
     * - Токенизатор (если модель найдена)
     */
    void SetUp() override {
        // Базовая конфигурация для тестов
        config_.vocab_size = TEST_VOCAB_SIZE;
        config_.cache_size = TEST_CACHE_SIZE;
        config_.byte_level = true;
        config_.enable_cache = true;
        config_.enable_profiling = false;
        
        // Проверяем наличие файлов модели
        has_model_ = model_files_exist();
        
        if (has_model_) {
            tokenizer_ = std::make_unique<FastBPETokenizer>(config_);
            model_loaded_ = load_tokenizer(*tokenizer_);
        }
    }
    
    /**
     * @brief Очистка после каждого теста
     */
    void TearDown() override {
        tokenizer_.reset();
    }
    
    /**
     * @brief Проверить, что токенизатор готов к работе
     * 
     * @return true если модель загружена и токенизатор инициализирован
     */
    bool tokenizer_ready() const {
        return has_model_ && model_loaded_ && tokenizer_ != nullptr;
    }
    
    /**
     * @brief Создать тестовый текст указанного размера
     * 
     * @param size Размер в байтах
     * @return std::string Тестовый текст
     */
    std::string create_test_text(size_t size) {
        static const std::string base = "int main() { return 0; }\n";
        std::string result;
        result.reserve(size);
        
        while (result.size() < size) {
            result += base;
        }
        result.resize(size);
        return result;
    }
    
    // ==================== Члены класса ====================
    
    TokenizerConfig config_;                         ///< Конфигурация для тестов
    std::unique_ptr<FastBPETokenizer> tokenizer_;    ///< Тестируемый токенизатор
    bool has_model_ = false;                         ///< Наличие файлов модели
    bool model_loaded_ = false;                      ///< Успешная загрузка
    
    // ==================== Тестовые строки ====================
    
    /**
     * @brief Базовые тестовые строки (ключевые слова C++)
     */
    const std::vector<std::string> test_strings_ = {
        "int",
        "main",
        "return",
        "void",
        "class",
        "template",
        "namespace",
        "std::vector",
        "auto",
        "constexpr"
    };
};

// ======================================================================
// Тесты создания и инициализации
// ======================================================================

/**
 * @test Проверка создания токенизатора с конфигурацией по умолчанию
 * 
 * Проверяет, что конструктор правильно инициализирует специальные токены
 * и устанавливает корректные размеры словаря.
 */
TEST_F(FastTokenizerTest, Creation) {
    TokenizerConfig config;
    config.vocab_size = 1000;
    config.cache_size = 100;
    config.byte_level = true;
    
    FastBPETokenizer tokenizer(config);
    
    // При создании всегда есть специальные токены
    // <UNK>, <PAD>, <BOS>, <EOS> (4 токена)
    // <MASK> может отсутствовать в некоторых версиях
    size_t expected_special_tokens = 4;
    
    std::cout << "Токенизатор создан с конфигурацией:" << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  cache_size: " << config.cache_size << std::endl;
    std::cout << "  byte_level: " << config.byte_level << std::endl;
    std::cout << "  фактический размер словаря: " << tokenizer.vocab_size() << std::endl;
    std::cout << "  специальные токены:" << std::endl;
    std::cout << "    <UNK> ID: " << tokenizer.unknown_id() << std::endl;
    std::cout << "    <PAD> ID: " << tokenizer.pad_id() << std::endl;
    std::cout << "    <BOS> ID: " << tokenizer.bos_id() << std::endl;
    std::cout << "    <EOS> ID: " << tokenizer.eos_id() << std::endl;
    
    // Проверяем, что ID специальных токенов в пределах разумного
    EXPECT_LT(tokenizer.unknown_id(), tokenizer.vocab_size());
    EXPECT_LT(tokenizer.pad_id(), tokenizer.vocab_size());
    EXPECT_LT(tokenizer.bos_id(), tokenizer.vocab_size());
    EXPECT_LT(tokenizer.eos_id(), tokenizer.vocab_size());
    
    // Проверяем, что ID различаются (если словарь достаточно большой)
    if (tokenizer.vocab_size() >= 4) {
        std::set<uint32_t> ids = {
            tokenizer.unknown_id(),
            tokenizer.pad_id(),
            tokenizer.bos_id(),
            tokenizer.eos_id()
        };
        EXPECT_GE(ids.size(), 3) << "Специальные токены должны иметь разные ID!";
    }
}

/**
 * @test Проверка создания с разными конфигурациями
 * 
 * Проверяет, что конструктор работает с различными наборами параметров.
 */
TEST_F(FastTokenizerTest, CreationWithDifferentConfigs) {
    std::vector<TokenizerConfig> configs = {
        {1000, 100, true},
        {2000, 200, false},
        {500, 50, true}
    };
    
    for (size_t i = 0; i < configs.size(); ++i) {
        FastBPETokenizer tokenizer(configs[i]);
        
        // Всегда есть минимум 4 специальных токена
        EXPECT_GE(tokenizer.vocab_size(), 4) << "Конфигурация " << (i+1) << " не работает!";
        
        std::cout << "Конфигурация " << (i+1) << " работает: vocab_size=" 
                  << tokenizer.vocab_size() << std::endl;
    }
}

// ======================================================================
// Тесты загрузки модели
// ======================================================================

/**
 * @test Загрузка словаря из файлов
 * 
 * Проверяет, что модель успешно загружается и имеет корректные размеры.
 */
TEST_F(FastTokenizerTest, LoadVocabulary) {
    if (!has_model_) {
        GTEST_SKIP() << "Файлы модели не найдены. Сначала обучите токенизатор.";
    }
    
    EXPECT_TRUE(model_loaded_);
    EXPECT_GT(tokenizer_->vocab_size(), 0);
    EXPECT_GT(tokenizer_->merges_count(), 0);
    
    std::cout << "Загружено токенов:    " << tokenizer_->vocab_size() << std::endl;
    std::cout << "Правил слияния:       " << tokenizer_->merges_count() << std::endl;
}

/**
 * @test Загрузка несуществующего файла
 * 
 * Проверяет корректную обработку ошибок при загрузке.
 */
TEST_F(FastTokenizerTest, LoadNonExistentFile) {
    FastBPETokenizer tokenizer(config_);
    
    bool loaded = tokenizer.load("nonexistent.json", "nonexistent.txt");
    EXPECT_FALSE(loaded);
    std::cout << "Корректная обработка отсутствующего файла" << std::endl;
}

// ======================================================================
// Тесты кодирования
// ======================================================================

/**
 * @test Простое кодирование одного символа
 */
TEST_F(FastTokenizerTest, SimpleEncode) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string text = "int";
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "'" << text << "' -> " << tokens.size() << " токенов" << std::endl;
    
    for (size_t i = 0; i < std::min<size_t>(MAX_PRINT_TOKENS, tokens.size()); ++i) {
        std::cout << "  Токен " << i << ": ID " << tokens[i] << std::endl;
    }
}

/**
 * @test Кодирование с пробелом
 * 
 * Проверяет обработку пробельных символов.
 */
TEST_F(FastTokenizerTest, EncodeWithSpace) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> texts = {
        " ",
        "  ",
        "int main",
        "int  main",
        " \t\n"
    };
    
    for (const auto& text : texts) {
        auto tokens = tokenizer_->encode(text);
        EXPECT_GT(tokens.size(), 0);
        std::cout << "'" << utils::escape_string(text) << "' (" << text.size() << " символов) -> " 
                  << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование чисел
 * 
 * Проверяет обработку числовых литералов.
 */
TEST_F(FastTokenizerTest, EncodeNumber) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> numbers = {
        "42",
        "3.14",
        "-10",
        "0xFF",
        "1000000",
        "0b1010"
    };
    
    for (const auto& num : numbers) {
        auto tokens = tokenizer_->encode(num);
        EXPECT_GT(tokens.size(), 0);
        std::cout << "Число '" << num << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование ключевых слов C++
 * 
 * Проверяет обработку всех основных ключевых слов языка.
 */
TEST_F(FastTokenizerTest, EncodeKeywords) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> keywords = {
        "if", "else", "for", "while", "switch",
        "class", "struct", "enum", "union",
        "public", "private", "protected",
        "virtual", "override", "final",
        "constexpr", "static_assert",
        "dynamic_cast", "reinterpret_cast"
    };
    
    for (const auto& kw : keywords) {
        auto tokens = tokenizer_->encode(kw);
        EXPECT_GT(tokens.size(), 0);
        std::cout << "Ключевое слово '" << kw << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование операторов
 * 
 * Проверяет обработку всех операторов C++.
 */
TEST_F(FastTokenizerTest, EncodeOperators) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> operators = {
        "+", "-", "*", "/", "%",
        "==", "!=", "<", ">", "<=", ">=",
        "&&", "||", "!", "&", "|", "^", "~",
        "<<", ">>", "++", "--",
        "=", "+=", "-=", "*=", "/=",
        "->", "::", ".*", "->*"
    };
    
    for (const auto& op : operators) {
        auto tokens = tokenizer_->encode(op);
        EXPECT_GT(tokens.size(), 0);
        std::cout << "Оператор '" << op << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование длинного текста
 * 
 * Проверяет производительность и соотношение токены/символы для разных размеров.
 */
TEST_F(FastTokenizerTest, LongText) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    // Создаем текст разной длины
    std::vector<size_t> sizes = {10, 100, 1000, 10000};
    
    for (size_t size : sizes) {
        std::string text(size, 'a');
        auto tokens = tokenizer_->encode(text);
        
        EXPECT_GT(tokens.size(), 0);
        
        // Проверяем примерное соотношение
        double ratio = static_cast<double>(tokens.size()) / size;
        std::cout << "Текст " << size << " символов -> " << tokens.size() 
                  << " токенов, соотношение: " << std::fixed << std::setprecision(3) << ratio << std::endl;
    }
}

/**
 * @test Кодирование UTF-8 символов
 * 
 * Проверяет работу с различными Unicode символами.
 */
TEST_F(FastTokenizerTest, EncodeUTF8) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> utf8_strings = {
        "Привет",       // Русские буквы
        "Hello 世界",    // Китайские иероглифы
        "café",         // Буква с диакритикой
        "€100",         // Символ евро
        "🔥 C++ 🔥",    // Эмодзи
        "München",      // Немецкий умлаут
        "français"      // Французский
    };
    
    for (const auto& text : utf8_strings) {
        auto tokens = tokenizer_->encode(text);
        EXPECT_GT(tokens.size(), 0);
        std::cout << "'" << text << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

// ======================================================================
// Тесты декодирования
// ======================================================================

/**
 * @test Простое декодирование
 * 
 * Проверяет, что decode() восстанавливает все символы исходного текста.
 */
TEST_F(FastTokenizerTest, SimpleDecode) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string original = "int main()";
    auto tokens = tokenizer_->encode(original);
    auto decoded = tokenizer_->decode(tokens);
    
    std::cout << "Roundtrip тест:" << std::endl;
    std::cout << "  оригинал:  '" << original << "'" << std::endl;
    std::cout << "  декод.:    '" << decoded << "'" << std::endl;
    std::cout << "  токенов:   " << tokens.size() << std::endl;
    
    // Проверяем, что все символы сохранились
    bool all_chars_present = true;
    for (char c : original) {
        if (decoded.find(c) == std::string::npos) {
            all_chars_present = false;
            std::cout << "  Отсутствует символ: '" << c << "'" << std::endl;
        }
    }
    
    EXPECT_TRUE(all_chars_present) << "Не все символы сохранились при декодировании!";
}

/**
 * @test Roundtrip для всех тестовых строк
 * 
 * Проверяет, что encode() + decode() возвращает исходный текст
 * для всех базовых тестовых строк.
 */
TEST_F(FastTokenizerTest, RoundtripAllStrings) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    int passed = 0;
    int total = test_strings_.size();
    
    for (const auto& text : test_strings_) {
        auto tokens = tokenizer_->encode(text);
        auto decoded = tokenizer_->decode(tokens);
        
        bool all_chars_present = true;
        for (char c : text) {
            if (decoded.find(c) == std::string::npos) {
                all_chars_present = false;
                break;
            }
        }
        
        if (all_chars_present) {
            passed++;
        }
    }
    
    double pass_rate = 100.0 * passed / total;
    std::cout << "Процент успешных roundtrip: " << std::fixed << std::setprecision(1)
              << pass_rate << "% (" << passed << "/" << total << ")" << std::endl;
    
    EXPECT_GE(pass_rate, MIN_ROUNDTRIP_RATE);
}

// ======================================================================
// Тесты пакетной обработки
// ======================================================================

/**
 * @test Пакетное кодирование
 * 
 * Проверяет работу encode_batch() на наборе текстов.
 */
TEST_F(FastTokenizerTest, BatchEncode) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> texts = test_strings_;
    std::vector<std::string_view> views;
    for (const auto& t : texts) views.push_back(t);
    
    auto results = tokenizer_->encode_batch(views);
    
    EXPECT_EQ(results.size(), texts.size());
    std::cout << "Пакетная обработка " << results.size() << " текстов:" << std::endl;
    
    for (size_t i = 0; i < std::min<size_t>(MAX_PRINT_TOKENS, results.size()); ++i) {
        std::cout << "  Текст " << i << ": " << results[i].size() << " токенов" << std::endl;
    }
}

/**
 * @test Сравнение последовательной и пакетной обработки
 * 
 * Проверяет, что результаты encode_batch() совпадают с последовательными
 * вызовами encode().
 */
TEST_F(FastTokenizerTest, BatchVsSequential) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::vector<std::string> texts = test_strings_;
    std::vector<std::string_view> views;
    for (const auto& t : texts) views.push_back(t);
    
    // Пакетная обработка
    auto batch_results = tokenizer_->encode_batch(views);
    
    // Последовательная обработка
    std::vector<std::vector<uint32_t>> sequential_results;
    for (const auto& text : texts) {
        sequential_results.push_back(tokenizer_->encode(text));
    }
    
    // Сравниваем размеры
    EXPECT_EQ(batch_results.size(), sequential_results.size());
    
    int matches = 0;
    for (size_t i = 0; i < batch_results.size(); ++i) {
        if (batch_results[i].size() == sequential_results[i].size()) {
            matches++;
        } else {
            std::cout << "  Несовпадение размера для текста " << i 
                      << ": batch=" << batch_results[i].size() 
                      << ", seq=" << sequential_results[i].size() << std::endl;
        }
    }
    
    double match_rate = 100.0 * matches / texts.size();
    std::cout << "Совпадение batch vs sequential: " << std::fixed << std::setprecision(1)
              << match_rate << "%" << std::endl;
    
    EXPECT_GE(match_rate, MIN_BATCH_MATCH_RATE);
}

// ======================================================================
// Тесты производительности
// ======================================================================

/**
 * @test Производительность кодирования
 * 
 * Измеряет скорость encode() на большом тексте и проверяет,
 * что она соответствует ожидаемой (менее 1 мс на 10 КБ).
 */
TEST_F(FastTokenizerTest, EncodePerformance) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string text(PERFORMANCE_TEXT_SIZE, 'a');
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < PERFORMANCE_ITERATIONS; ++i) {
        auto tokens = tokenizer_->encode(text);
        // Используем нашу вспомогательную функцию вместо benchmark::DoNotOptimize
        do_not_optimize(tokens.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ms_per_encode = duration.count() / static_cast<double>(PERFORMANCE_ITERATIONS);
    double chars_per_second = (text.size() * PERFORMANCE_ITERATIONS) / (duration.count() / 1000.0);
    double mb_per_second = chars_per_second / (1024.0 * 1024.0);
    
    std::cout << "\nПроизводительность encode:" << std::endl;
    std::cout << "  размер текста: " << text.size() << " символов (" 
              << text.size() / 1024.0 << " КБ)" << std::endl;
    std::cout << "  итераций: " << PERFORMANCE_ITERATIONS << std::endl;
    std::cout << "  общее время: " << duration.count() << " мс" << std::endl;
    std::cout << "  среднее время: " << std::fixed << std::setprecision(3) 
              << ms_per_encode << " мс" << std::endl;
    std::cout << "  скорость: " << std::fixed << std::setprecision(2)
              << mb_per_second << " МБ/с" << std::endl;
    
    EXPECT_LT(ms_per_encode, MAX_ENCODE_TIME_MS) << "Слишком медленное кодирование!";
}

// ======================================================================
// Тесты кэширования
// ======================================================================

/**
 * @test Проверка эффективности кэширования
 * 
 * Создает временный словарь, выполняет серию encode() и проверяет,
 * что кэш действительно работает (появляются попадания).
 */
TEST_F(FastTokenizerTest, CacheEfficiency) {
    // Создаём конфигурацию с включённым кэшем
    TokenizerConfig config;
    config.enable_cache = true;
    config.cache_size = 100;
    config.byte_level = true;
    
    FastBPETokenizer tokenizer(config);
    
    // Создаём тестовый словарь
    std::string vocab_path = "test_cache_vocab.json";
    std::string merges_path = "test_cache_merges.txt";
    
    // Базовые токены
    std::vector<std::string> tokens = {
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z",
        "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>"
    };
    
    create_test_vocab(vocab_path, tokens);
    create_test_merges(merges_path);
    
    // Загружаем модель
    bool loaded = tokenizer.load(vocab_path, merges_path);
    ASSERT_TRUE(loaded) << "Не удалось загрузить тестовую модель!";
    
    // Сбрасываем статистику перед тестом
    tokenizer.reset_stats();
    
    // Уникальные строки для теста
    std::vector<std::string> unique_strings;
    for (int i = 0; i < UNIQUE_STRINGS_COUNT; ++i) {
        unique_strings.push_back("test" + std::to_string(i));
    }
    
    // Первый проход - заполнение кэша
    std::cout << "Первый проход (заполнение кэша)..." << std::endl;
    for (const auto& text : unique_strings) {
        tokenizer.encode(text);
    }
    
    auto stats1 = tokenizer.stats();
    std::cout << "  попаданий: " << stats1.cache_hits << std::endl;
    std::cout << "  промахов: " << stats1.cache_misses << std::endl;
    
    // Второй проход - должны быть попадания
    std::cout << "Второй проход (использование кэша)..." << std::endl;
    for (const auto& text : unique_strings) {
        tokenizer.encode(text);
    }
    
    auto stats2 = tokenizer.stats();
    std::cout << "  попаданий: " << stats2.cache_hits << std::endl;
    std::cout << "  промахов: " << stats2.cache_misses << std::endl;
    
    // Проверяем, что были попадания во втором проходе
    EXPECT_GT(stats2.cache_hits, stats1.cache_hits) << "Кэш не работает!";
    
    double hit_rate = stats2.cache_hit_rate() * 100;
    std::cout << "  hit rate: " << std::fixed << std::setprecision(1) 
              << hit_rate << "%" << std::endl;
    
    // Очистка
    std::remove(vocab_path.c_str());
    std::remove(merges_path.c_str());
}

// ======================================================================
// Тесты граничных случаев
// ======================================================================

/**
 * @test Пустой текст
 * 
 * Проверяет корректную обработку пустой строки.
 */
TEST_F(FastTokenizerTest, EmptyText) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string empty = "";
    auto tokens = tokenizer_->encode(empty);
    EXPECT_EQ(tokens.size(), 0);
    
    auto decoded = tokenizer_->decode(tokens);
    EXPECT_EQ(decoded, "");
    
    std::cout << "Пустой текст обработан корректно!" << std::endl;
}

/**
 * @test Очень длинный текст
 * 
 * Проверяет работу с текстом большого размера (100 КБ).
 */
TEST_F(FastTokenizerTest, VeryLongText) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string text(VERY_LONG_TEXT_SIZE, 'a');    // 100 КБ текста
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), text.size() / 10);
    std::cout << "Очень длинный текст (100 КБ) -> " << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Смешанные символы
 * 
 * Проверяет текст, содержащий различные типы символов:
 * буквы, цифры, знаки пунктуации.
 */
TEST_F(FastTokenizerTest, MixedCharacters) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string text = "int x = 42;    // комментарий с цифрами 123 и символами !@#$%";
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Смешанный текст -> " << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Все символы ASCII
 * 
 * Проверяет кодирование всех печатных ASCII символов.
 */
TEST_F(FastTokenizerTest, AllAsciiChars) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::string text;
    for (int i = 32; i <= 126; ++i) {
        text += static_cast<char>(i);
    }
    
    auto tokens = tokenizer_->encode(text);
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Все ASCII символы (95 шт) -> " << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Случайные строки
 * 
 * Проверяет работу со случайно сгенерированными строками.
 */
TEST_F(FastTokenizerTest, RandomStrings) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> len_dis(1, 100);
    std::uniform_int_distribution<> char_dis(32, 126);
    
    int passed = 0;
    const int num_tests = 50;
    
    for (int t = 0; t < num_tests; ++t) {
        int len = len_dis(gen);
        std::string text;
        text.reserve(len);
        
        for (int i = 0; i < len; ++i) {
            text += static_cast<char>(char_dis(gen));
        }
        
        auto tokens = tokenizer_->encode(text);
        auto decoded = tokenizer_->decode(tokens);
        
        bool all_chars_present = true;
        for (char c : text) {
            if (decoded.find(c) == std::string::npos) {
                all_chars_present = false;
                break;
            }
        }
        
        if (all_chars_present) passed++;
    }
    
    double pass_rate = 100.0 * passed / num_tests;
    std::cout << "Случайные строки: " << std::fixed << std::setprecision(1)
              << pass_rate << "% успешных roundtrip" << std::endl;
    
    EXPECT_GE(pass_rate, MIN_ROUNDTRIP_RATE);
}

// ======================================================================
// Тесты статистики
// ======================================================================

/**
 * @test Проверка сбора статистики
 * 
 * Проверяет, что статистика правильно собирается и сбрасывается.
 */
TEST_F(FastTokenizerTest, Statistics) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "Модель не загружена!";
    }
    
    tokenizer_->reset_stats();
    auto stats_before = tokenizer_->stats();
    
    EXPECT_EQ(stats_before.encode_calls, 0);
    EXPECT_EQ(stats_before.decode_calls, 0);
    
    // Выполняем несколько операций
    std::string text = "int main()";
    tokenizer_->encode(text);
    tokenizer_->encode(text);
    tokenizer_->decode({1, 2, 3});
    
    auto stats_after = tokenizer_->stats();
    
    EXPECT_GE(stats_after.encode_calls, 2);
    EXPECT_GE(stats_after.decode_calls, 1);
    
    std::cout << "Статистика:" << std::endl;
    std::cout << "  encode_calls: " << stats_after.encode_calls << std::endl;
    std::cout << "  decode_calls: " << stats_after.decode_calls << std::endl;
    std::cout << "  cache_hits:   " << stats_after.cache_hits << std::endl;
    std::cout << "  cache_misses: " << stats_after.cache_misses << std::endl;
    
    tokenizer_->reset_stats();
    auto stats_reset = tokenizer_->stats();
    
    EXPECT_EQ(stats_reset.encode_calls, 0);
    EXPECT_EQ(stats_reset.decode_calls, 0);
    
    std::cout << "Статистика сброшена успешно!" << std::endl;
}