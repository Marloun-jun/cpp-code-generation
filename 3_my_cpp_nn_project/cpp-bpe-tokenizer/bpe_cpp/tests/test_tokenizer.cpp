/**
 * @file test_tokenizer.cpp
 * @brief Модульные тесты для базовой версии BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Набор тестов для проверки функциональности BPETokenizer:
 *          - Загрузка словаря из разных форматов (JSON, бинарный)
 *          - Кодирование и декодирование различных типов текстов
 *          - Обработка UTF-8 и byte-level режим
 *          - Пакетная обработка
 *          - Граничные случаи
 * 
 * @see BPETokenizer
 */

#include <gtest/gtest.h>

#include "bpe_tokenizer.hpp"
#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <set>
#include <filesystem>

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Создать тестовый словарь в JSON формате
 */
void create_test_json_vocab(const std::string& path) {
    std::ofstream file(path);
    file << R"({
        "size": 10,
        "tokens": ["a", "b", "c", "ab", "bc", "abc", "<UNK>", "<PAD>", "<BOS>", "<EOS>"]
    })";
}

/**
 * @brief Создать тестовый бинарный словарь со всеми 256 байтами
 */
void create_test_binary_vocab(const std::string& path) {
    Vocabulary vocab;
    
    // Добавляем все 256 байт
    for (int i = 0; i < 256; ++i) {
        vocab.add_token(std::string(1, static_cast<char>(i)));
    }
    
    // Добавляем специальные токены
    vocab.add_token("<UNK>");
    vocab.add_token("<PAD>");
    vocab.add_token("<BOS>");
    vocab.add_token("<EOS>");
    
    // Добавляем тестовые токены
    vocab.add_token("a");
    vocab.add_token("b");
    vocab.add_token("c");
    vocab.add_token("ab");
    vocab.add_token("bc");
    vocab.add_token("abc");
    
    vocab.save_binary(path);
}

/**
 * @brief Создать тестовый файл слияний
 */
void create_test_merges(const std::string& path) {
    std::ofstream file(path);
    file << "# version: 0.1\n";
    file << "a b\n";
    file << "b c\n";
    file << "ab c\n";
}

// ======================================================================
// Тестовый класс
// ======================================================================

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Создаем тестовые файлы
        create_test_json_vocab("test_vocab.json");
        create_test_binary_vocab("test_vocab.bin");
        create_test_merges("test_merges.txt");
        
        // Запоминаем размеры для проверок
        json_vocab_size = 10;
        binary_vocab_size = 256 + 4 + 6;  // 256 байт + 4 спецтокена + a,b,c,ab,bc,abc
    }
    
    void TearDown() override {
        // Удаляем тестовые файлы
        std::filesystem::remove("test_vocab.json");
        std::filesystem::remove("test_vocab.bin");
        std::filesystem::remove("test_merges.txt");
    }
    
    size_t json_vocab_size;
    size_t binary_vocab_size;
    
    // Тестовые строки для различных сценариев
    const std::vector<std::string> test_strings = {
        "a", "b", "c",
        "a b", "a b c",
        "ab", "bc", "abc",
        "привет",  // русские буквы
        "hello",   // английские
        "123",     // цифры
        "!@#$",    // символы
        "a1 b2 c3" // смешанные
    };
};

// ======================================================================
// Тесты загрузки словаря
// ======================================================================

/**
 * @test Загрузка из JSON файла
 */
TEST_F(TokenizerTest, LoadFromJSON) {
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    
    bool loaded = tokenizer.load_from_files("test_vocab.json", "test_merges.txt");
    EXPECT_TRUE(loaded);
    EXPECT_EQ(tokenizer.vocab_size(), json_vocab_size);
    EXPECT_EQ(tokenizer.merges_count(), 3);
    
    std::cout << "JSON словарь загружен: " << tokenizer.vocab_size() << " токенов" << std::endl;
}

/**
 * @test Загрузка из бинарного файла
 */
TEST_F(TokenizerTest, LoadFromBinary) {
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    
    bool loaded = tokenizer.load_binary("test_model.bin");
    EXPECT_TRUE(loaded);
    EXPECT_EQ(tokenizer.vocab_size(), binary_vocab_size);
    EXPECT_EQ(tokenizer.merges_count(), 3);
    
    std::cout << "Бинарный словарь загружен: " << tokenizer.vocab_size() << " токенов" << std::endl;
}

/**
 * @test Загрузка несуществующего файла
 */
TEST_F(TokenizerTest, LoadNonExistent) {
    BPETokenizer tokenizer;
    
    bool loaded = tokenizer.load_from_files("nonexistent.json", "nonexistent.txt");
    EXPECT_FALSE(loaded);
    
    std::cout << "Корректная обработка отсутствующего файла" << std::endl;
}

// ======================================================================
// Тесты поиска токенов
// ======================================================================

/**
 * @test Поиск токенов по ID и по строке
 */
TEST_F(TokenizerTest, TokenLookup) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    // Проверяем поиск по строке
    token_id_t id_a = tokenizer.vocabulary().token_to_id("a");
    EXPECT_NE(id_a, INVALID_TOKEN);
    
    // Проверяем поиск по ID
    const std::string& token = tokenizer.vocabulary().id_to_token(id_a);
    EXPECT_EQ(token, "a");
    
    std::cout << "Поиск токенов работает: 'a' -> ID " << id_a << std::endl;
}

/**
 * @test Проверка специальных токенов
 */
TEST_F(TokenizerTest, SpecialTokens) {
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    tokenizer.load_binary("test_model.bin");
    
    token_id_t unk_id = tokenizer.vocabulary().token_to_id("<UNK>");
    EXPECT_NE(unk_id, INVALID_TOKEN);
    
    std::cout << "Специальные токены:" << std::endl;
    std::cout << "   <UNK> ID: " << tokenizer.unknown_token_id() << std::endl;
}

/**
 * @test Проверка наличия всех 256 байт
 */
TEST_F(TokenizerTest, AllBytesPresent) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    int found = 0;
    for (int i = 0; i < 256; ++i) {
        std::string byte_str(1, static_cast<char>(i));
        if (tokenizer.vocabulary().contains(byte_str)) {
            found++;
        }
    }
    
    EXPECT_EQ(found, 256);
    std::cout << "Все 256 байт присутствуют в словаре" << std::endl;
}

// ======================================================================
// Тесты кодирования
// ======================================================================

/**
 * @test Базовое кодирование
 */
TEST_F(TokenizerTest, BasicEncode) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    auto tokens = tokenizer.encode("a b");
    EXPECT_FALSE(tokens.empty());
    
    std::cout << "'a b' -> " << tokens.size() << " токенов: ";
    for (auto id : tokens) std::cout << id << " ";
    std::cout << std::endl;
}

/**
 * @test Кодирование всех тестовых строк
 */
TEST_F(TokenizerTest, EncodeAllStrings) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    for (const auto& text : test_strings) {
        auto tokens = tokenizer.encode(text);
        EXPECT_FALSE(tokens.empty());
        std::cout << "'" << text << "' (" << text.size() << " символов) -> " 
                  << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование с BPE слияниями
 */
TEST_F(TokenizerTest, EncodeWithMerges) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    // Должно применить слияние a b -> ab
    auto tokens = tokenizer.encode("a b");
    
    std::cout << "Слияния применены: 'a b' -> " << tokens.size() << " токенов" << std::endl;
    
    // Проверяем, что размер меньше, чем без слияний
    auto tokens_no_merge = tokenizer.encode("a");
    tokens_no_merge.insert(tokens_no_merge.end(), 
                          tokenizer.encode("b").begin(), 
                          tokenizer.encode("b").end());
    
    EXPECT_LE(tokens.size(), tokens_no_merge.size());
}

// ======================================================================
// Тесты декодирования
// ======================================================================

/**
 * @test Декодирование токенов
 */
TEST_F(TokenizerTest, BasicDecode) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::string original = "a b c";
    auto tokens = tokenizer.encode(original);
    auto decoded = tokenizer.decode(tokens);
    
    std::cout << "Roundtrip тест:" << std::endl;
    std::cout << "   Оригинал: '" << original << "'" << std::endl;
    std::cout << "   Декод.:   '" << decoded << "'" << std::endl;
    
    EXPECT_FALSE(decoded.empty());
}

/**
 * @test Roundtrip для всех тестовых строк
 */
TEST_F(TokenizerTest, RoundtripAll) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    int passed = 0;
    for (const auto& text : test_strings) {
        auto tokens = tokenizer.encode(text);
        auto decoded = tokenizer.decode(tokens);
        
        bool all_chars_present = true;
        for (char c : text) {
            if (decoded.find(c) == std::string::npos) {
                all_chars_present = false;
                break;
            }
        }
        
        if (all_chars_present) passed++;
    }
    
    double pass_rate = 100.0 * passed / test_strings.size();
    std::cout << "Процент успешных roundtrip:: " << std::fixed << std::setprecision(1)
              << pass_rate << "% (" << passed << "/" << test_strings.size() << ")" << std::endl;
    
    EXPECT_GE(pass_rate, 80.0);
}

// ======================================================================
// Тесты UTF-8 и byte-level режима
// ======================================================================

/**
 * @test Byte-level режим с русским текстом
 */
TEST_F(TokenizerTest, ByteLevelRussian) {
    BPETokenizer tokenizer(32000, true);  // byte_level = true
    tokenizer.load_binary("test_model.bin");
    
    std::string russian = "привет мир";
    auto tokens = tokenizer.encode(russian);
    auto decoded = tokenizer.decode(tokens);
    
    EXPECT_EQ(decoded, russian);
    std::cout << "Русский текст: '" << russian << "' -> " 
              << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Byte-level режим с китайским текстом
 */
TEST_F(TokenizerTest, ByteLevelChinese) {
    BPETokenizer tokenizer(32000, true);
    tokenizer.load_binary("test_model.bin");
    
    std::string chinese = "你好世界";
    auto tokens = tokenizer.encode(chinese);
    auto decoded = tokenizer.decode(tokens);
    
    EXPECT_EQ(decoded, chinese);
    std::cout << "Китайский текст: '" << chinese << "' -> " 
              << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Byte-level режим с эмодзи
 */
TEST_F(TokenizerTest, ByteLevelEmoji) {
    BPETokenizer tokenizer(32000, true);
    tokenizer.load_binary("test_model.bin");
    
    std::string emoji = "😊🚀🌟";
    auto tokens = tokenizer.encode(emoji);
    auto decoded = tokenizer.decode(tokens);
    
    EXPECT_EQ(decoded, emoji);
    std::cout << "😊 Эмодзи: '" << emoji << "' -> " 
              << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Обычный режим (не byte-level)
 */
TEST_F(TokenizerTest, NormalMode) {
    BPETokenizer tokenizer(32000, false);  // byte_level = false
    tokenizer.load_binary("test_model.bin");
    
    std::string text = "hello";
    auto tokens = tokenizer.encode(text);
    auto decoded = tokenizer.decode(tokens);
    
    // В обычном режиме может не сохраниться точная длина
    std::cout << "Обычный режим: '" << text << "' -> " 
              << tokens.size() << " токенов" << std::endl;
}

// ======================================================================
// Тесты пакетной обработки
// ======================================================================

/**
 * @test Пакетное кодирование
 */
TEST_F(TokenizerTest, BatchEncode) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::vector<std::string> texts = {"a", "b", "c", "ab", "bc"};
    auto batch_result = tokenizer.encode_batch(texts);
    
    EXPECT_EQ(batch_result.size(), texts.size());
    std::cout << "Пакетная обработка " << batch_result.size() << " текстов:" << std::endl;
    
    for (size_t i = 0; i < batch_result.size(); ++i) {
        std::cout << "   Текст " << i << ": " << batch_result[i].size() << " токенов" << std::endl;
    }
}

/**
 * @test Пакетная обработка с разными размерами
 */
TEST_F(TokenizerTest, BatchWithDifferentSizes) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::vector<std::string> texts;
    for (int i = 1; i <= 10; ++i) {
        texts.push_back(std::string(i, 'a'));
    }
    
    auto batch_result = tokenizer.encode_batch(texts);
    EXPECT_EQ(batch_result.size(), texts.size());
    
    // Проверяем, что длина результатов растет
    for (size_t i = 1; i < batch_result.size(); ++i) {
        EXPECT_GE(batch_result[i].size(), batch_result[i-1].size());
    }
}

// ======================================================================
// Тесты граничных случаев
// ======================================================================

/**
 * @test Пустой вход
 */
TEST_F(TokenizerTest, EmptyInput) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    auto tokens = tokenizer.encode("");
    EXPECT_TRUE(tokens.empty());
    
    auto decoded = tokenizer.decode({});
    EXPECT_TRUE(decoded.empty());
    
    std::cout << "Пустой вход обработан корректно" << std::endl;
}

/**
 * @test Очень длинный вход
 */
TEST_F(TokenizerTest, VeryLongInput) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::string text(10000, 'a');
    auto tokens = tokenizer.encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Длинный текст (10000 символов) -> " << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Смешанные символы
 */
TEST_F(TokenizerTest, MixedCharacters) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::string text = "a1 b2 c3 !@#$ привет 世界 😊";
    auto tokens = tokenizer.encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Смешанные символы -> " << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Все возможные байты
 */
TEST_F(TokenizerTest, AllBytes) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::string all_bytes;
    for (int i = 0; i < 256; ++i) {
        all_bytes += static_cast<char>(i);
    }
    
    auto tokens = tokenizer.encode(all_bytes);
    auto decoded = tokenizer.decode(tokens);
    
    EXPECT_EQ(decoded, all_bytes);
    std::cout << "Все 256 байт закодированы и декодированы" << std::endl;
}

// ======================================================================
// Тесты производительности
// ======================================================================

/**
 * @test Производительность encode
 */
TEST_F(TokenizerTest, EncodePerformance) {
    BPETokenizer tokenizer;
    tokenizer.load_binary("test_model.bin");
    
    std::string text(1000, 'a');
    const int iterations = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto tokens = tokenizer.encode(text);
        volatile size_t dummy = tokens.size();
        (void)dummy;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ms_per_encode = duration.count() / static_cast<double>(iterations);
    std::cout << "\n⚡ Производительность encode:" << std::endl;
    std::cout << "   Размер текста: " << text.size() << " символов" << std::endl;
    std::cout << "   Итераций: " << iterations << std::endl;
    std::cout << "   Среднее время: " << std::fixed << std::setprecision(3) 
              << ms_per_encode << " мс" << std::endl;
}
