/**
 * @file test_fast_tokenizer.cpp
 * @brief Модульные тесты для оптимизированной версии BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 2.0.0
 * 
 * @details Набор тестов для проверки функциональности FastBPETokenizer:
 *          - Создание и инициализация
 *          - Загрузка словаря
 *          - Кодирование различных типов входных данных
 *          - Пакетная обработка
 *          - Производительность и кэширование
 *          - Обработка граничных случаев
 * 
 * @see FastBPETokenizer
 */

#include <gtest/gtest.h>
#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <thread>

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Проверить существование файла модели
 */
bool model_files_exist() {
    const std::vector<std::string> paths = {
        "../../bpe/vocab.json",
        "../bpe/vocab.json",
        "bpe/vocab.json",
        "vocab.json"
    };
    
    for (const auto& path : paths) {
        if (std::ifstream(path).good()) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Загрузить токенизатор с автоматическим поиском файлов
 */
bool load_tokenizer(FastBPETokenizer& tokenizer) {
    const std::vector<std::pair<std::string, std::string>> paths = {
        {"../../bpe/vocab.json", "../../bpe/merges.txt"},
        {"../bpe/vocab.json", "../bpe/merges.txt"},
        {"bpe/vocab.json", "bpe/merges.txt"},
        {"vocab.json", "merges.txt"},
        {"models/cpp_vocab.json", "models/cpp_merges.txt"}
    };
    
    for (const auto& [vocab_path, merges_path] : paths) {
        if (std::ifstream(vocab_path).good() && std::ifstream(merges_path).good()) {
            if (tokenizer.load(vocab_path, merges_path)) {
                std::cout << "✅ Загружена модель: " << vocab_path << std::endl;
                return true;
            }
        }
    }
    
    return false;
}

// ======================================================================
// Тестовый класс
// ======================================================================

class FastTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Базовая конфигурация для тестов
        config_.vocab_size = 1000;
        config_.cache_size = 100;
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
    
    void TearDown() override {
        tokenizer_.reset();
    }
    
    /**
     * @brief Проверить, что токенизатор готов к работе
     */
    bool tokenizer_ready() const {
        return has_model_ && model_loaded_ && tokenizer_ != nullptr;
    }
    
    TokenizerConfig config_;
    std::unique_ptr<FastBPETokenizer> tokenizer_;
    bool has_model_ = false;
    bool model_loaded_ = false;
    
    // Тестовые строки
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
 * @test Простое создание токенизатора без загрузки модели
 */
TEST_F(FastTokenizerTest, Creation) {
    FastBPETokenizer tokenizer(config_);
    EXPECT_EQ(tokenizer.vocab_size(), 0);
    EXPECT_EQ(tokenizer.merges_count(), 0);
    EXPECT_EQ(tokenizer.unknown_id(), 0);
    
    std::cout << "✅ Токенизатор создан с конфигурацией:" << std::endl;
    std::cout << "   vocab_size: " << config_.vocab_size << std::endl;
    std::cout << "   cache_size: " << config_.cache_size << std::endl;
    std::cout << "   byte_level: " << std::boolalpha << config_.byte_level << std::endl;
}

/**
 * @test Создание с разными параметрами
 */
TEST_F(FastTokenizerTest, CreationWithDifferentConfigs) {
    std::vector<TokenizerConfig> configs = {
        {1000, 100, true, true},   // Маленький
        {32000, 10000, true, true}, // Средний
        {50000, 50000, false, false} // Большой без кэша
    };
    
    for (size_t i = 0; i < configs.size(); ++i) {
        FastBPETokenizer tokenizer(configs[i]);
        EXPECT_EQ(tokenizer.vocab_size(), 0);
        std::cout << "✅ Конфигурация " << i+1 << " работает" << std::endl;
    }
}

// ======================================================================
// Тесты загрузки модели
// ======================================================================

/**
 * @test Загрузка словаря из файлов
 */
TEST_F(FastTokenizerTest, LoadVocabulary) {
    if (!has_model_) {
        GTEST_SKIP() << "❌ Файлы модели не найдены. Сначала обучите токенизатор.";
    }
    
    EXPECT_TRUE(model_loaded_);
    EXPECT_GT(tokenizer_->vocab_size(), 0);
    EXPECT_GT(tokenizer_->merges_count(), 0);
    
    std::cout << "📚 Загружено токенов: " << tokenizer_->vocab_size() << std::endl;
    std::cout << "🔗 Правил слияния: " << tokenizer_->merges_count() << std::endl;
}

/**
 * @test Загрузка несуществующего файла
 */
TEST_F(FastTokenizerTest, LoadNonExistentFile) {
    FastBPETokenizer tokenizer(config_);
    
    bool loaded = tokenizer.load("nonexistent.json", "nonexistent.txt");
    EXPECT_FALSE(loaded);
    std::cout << "✅ Корректная обработка отсутствующего файла" << std::endl;
}

// ======================================================================
// Тесты кодирования
// ======================================================================

/**
 * @test Простое кодирование одного символа
 */
TEST_F(FastTokenizerTest, SimpleEncode) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::string text = "int";
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "📝 '" << text << "' -> " << tokens.size() << " токенов" << std::endl;
    
    for (size_t i = 0; i < std::min<size_t>(5, tokens.size()); ++i) {
        std::cout << "   Токен " << i << ": ID " << tokens[i] << std::endl;
    }
}

/**
 * @test Кодирование с пробелом
 */
TEST_F(FastTokenizerTest, EncodeWithSpace) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
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
        std::cout << "📝 '" << text << "' (" << text.size() << " символов) -> " 
                  << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование чисел
 */
TEST_F(FastTokenizerTest, EncodeNumber) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
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
        std::cout << "🔢 Число '" << num << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование ключевых слов C++
 */
TEST_F(FastTokenizerTest, EncodeKeywords) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
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
        std::cout << "🔑 Ключевое слово '" << kw << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование операторов
 */
TEST_F(FastTokenizerTest, EncodeOperators) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
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
        std::cout << "➡️ Оператор '" << op << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование длинного текста
 */
TEST_F(FastTokenizerTest, LongText) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    // Создаем текст разной длины
    std::vector<size_t> sizes = {10, 100, 1000, 10000};
    
    for (size_t size : sizes) {
        std::string text(size, 'a');
        auto tokens = tokenizer_->encode(text);
        
        EXPECT_GT(tokens.size(), 0);
        std::cout << "📏 Текст " << size << " символов -> " << tokens.size() << " токенов" << std::endl;
        
        // Проверяем примерное соотношение
        double ratio = static_cast<double>(tokens.size()) / size;
        std::cout << "   Соотношение токены/символы: " << std::fixed << std::setprecision(3) << ratio << std::endl;
    }
}

/**
 * @test Кодирование UTF-8 символов
 */
TEST_F(FastTokenizerTest, EncodeUTF8) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::vector<std::string> utf8_strings = {
        "Привет",           // Русские буквы
        "Hello 世界",        // Китайские иероглифы
        "café",             // Буква с диакритикой
        "€100",             // Символ евро
        "🔥 C++ 🔥",         // Эмодзи
        "München",          // Немецкий умлаут
        "français"          // Французский
    };
    
    for (const auto& text : utf8_strings) {
        auto tokens = tokenizer_->encode(text);
        EXPECT_GT(tokens.size(), 0);
        std::cout << "🌍 '" << text << "' -> " << tokens.size() << " токенов" << std::endl;
    }
}

// ======================================================================
// Тесты декодирования
// ======================================================================

/**
 * @test Простое декодирование
 */
TEST_F(FastTokenizerTest, SimpleDecode) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::string original = "int main()";
    auto tokens = tokenizer_->encode(original);
    auto decoded = tokenizer_->decode(tokens);
    
    std::cout << "🔄 Roundtrip тест:" << std::endl;
    std::cout << "   Оригинал:  '" << original << "'" << std::endl;
    std::cout << "   Декод.:    '" << decoded << "'" << std::endl;
    std::cout << "   Токенов:   " << tokens.size() << std::endl;
    
    // Проверяем, что все символы сохранились
    for (char c : original) {
        if (decoded.find(c) == std::string::npos) {
            std::cout << "   ⚠️ Отсутствует символ: '" << c << "'" << std::endl;
        }
    }
}

/**
 * @test Roundtrip для всех тестовых строк
 */
TEST_F(FastTokenizerTest, RoundtripAllStrings) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
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
    std::cout << "📊 Roundtrip成功率: " << std::fixed << std::setprecision(1)
              << pass_rate << "% (" << passed << "/" << total << ")" << std::endl;
    
    EXPECT_GE(pass_rate, 90.0);
}

// ======================================================================
// Тесты пакетной обработки
// ======================================================================

/**
 * @test Пакетное кодирование
 */
TEST_F(FastTokenizerTest, BatchEncode) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::vector<std::string> texts = test_strings_;
    std::vector<std::string_view> views;
    for (const auto& t : texts) views.push_back(t);
    
    auto results = tokenizer_->encode_batch(views);
    
    EXPECT_EQ(results.size(), texts.size());
    std::cout << "📦 Пакетная обработка " << results.size() << " текстов:" << std::endl;
    
    for (size_t i = 0; i < std::min<size_t>(5, results.size()); ++i) {
        std::cout << "   Текст " << i << ": " << results[i].size() << " токенов" << std::endl;
    }
}

/**
 * @test Сравнение последовательной и пакетной обработки
 */
TEST_F(FastTokenizerTest, BatchVsSequential) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
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
        }
    }
    
    double match_rate = 100.0 * matches / texts.size();
    std::cout << "📊 Совпадение batch vs sequential: " << match_rate << "%" << std::endl;
    
    EXPECT_GE(match_rate, 90.0);
}

// ======================================================================
// Тесты производительности
// ======================================================================

/**
 * @test Производительность кодирования
 */
TEST_F(FastTokenizerTest, EncodePerformance) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::string text(10000, 'a');
    const int iterations = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto tokens = tokenizer_->encode(text);
        volatile size_t dummy = tokens.size();
        (void)dummy;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ms_per_encode = duration.count() / static_cast<double>(iterations);
    double chars_per_second = (text.size() * iterations) / (duration.count() / 1000.0);
    
    std::cout << "\n⚡ Производительность encode:" << std::endl;
    std::cout << "   Размер текста: " << text.size() << " символов" << std::endl;
    std::cout << "   Итераций: " << iterations << std::endl;
    std::cout << "   Общее время: " << duration.count() << " мс" << std::endl;
    std::cout << "   Среднее время: " << std::fixed << std::setprecision(3) 
              << ms_per_encode << " мс" << std::endl;
    std::cout << "   Скорость: " << std::fixed << std::setprecision(0)
              << chars_per_second << " символов/сек" << std::endl;
}

// ======================================================================
// Тесты кэширования
// ======================================================================

/**
 * @test Эффективность кэширования
 */
TEST_F(FastTokenizerTest, CacheEfficiency) {
    TokenizerConfig cache_config = config_;
    cache_config.cache_size = 10;
    cache_config.enable_cache = true;
    
    FastBPETokenizer tokenizer(cache_config);
    
    if (!load_tokenizer(tokenizer)) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    // Кодируем одни и те же тексты несколько раз
    std::vector<std::string> texts = {"int", "main", "return", "void"};
    
    // Первый проход (заполнение кэша)
    for (int i = 0; i < 3; ++i) {
        for (const auto& text : texts) {
            tokenizer.encode(text);
        }
    }
    
    auto stats = tokenizer.stats();
    std::cout << "\n📊 Статистика кэша:" << std::endl;
    std::cout << "   Попаданий: " << stats.cache_hits << std::endl;
    std::cout << "   Промахов: " << stats.cache_misses << std::endl;
    std::cout << "   Процент попаданий: " << stats.cache_hit_rate() * 100 << "%" << std::endl;
    
    EXPECT_GT(stats.cache_hits, 0);
}

// ======================================================================
// Тесты граничных случаев
// ======================================================================

/**
 * @test Пустой текст
 */
TEST_F(FastTokenizerTest, EmptyText) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::string empty = "";
    auto tokens = tokenizer_->encode(empty);
    EXPECT_EQ(tokens.size(), 0);
    
    auto decoded = tokenizer_->decode(tokens);
    EXPECT_EQ(decoded, "");
    
    std::cout << "✅ Пустой текст обработан корректно" << std::endl;
}

/**
 * @test Очень длинный текст
 */
TEST_F(FastTokenizerTest, VeryLongText) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::string text(100000, 'a');  // 100KB текста
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "📏 Очень длинный текст (100KB) -> " << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Смешанные символы
 */
TEST_F(FastTokenizerTest, MixedCharacters) {
    if (!tokenizer_ready()) {
        GTEST_SKIP() << "❌ Модель не загружена";
    }
    
    std::string text = "int x = 42; // комментарий с цифрами 123 и символами !@#$%";
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "📝 Смешанный текст -> " << tokens.size() << " токенов" << std::endl;
}

// ======================================================================
// Запуск тестов
// ======================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n🔧 Запуск тестов FastBPETokenizer\n" << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    std::cout << "\n✅ Тестирование завершено. Код возврата: " << result << std::endl;
    
    return result;
}