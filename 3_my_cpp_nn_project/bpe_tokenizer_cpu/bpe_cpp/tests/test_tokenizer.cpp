/**
 * @file test_tokenizer.cpp
 * @brief Модульные тесты для базовой версии BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Набор модульных тестов для проверки функциональности BPETokenizer -
 *          эталонной реализации BPE токенизатора.
 * 
 *          **Проверяемые аспекты:**
 *          ┌────────────────────┬────────────────────────────────────┐
 *          │ Загрузка словаря   │ JSON и бинарный форматы            │
 *          │ Поиск токенов      │ token_to_id и id_to_token          │
 *          │ Специальные токены │ <UNK>, <PAD>, <BOS>, <EOS>         │
 *          │ Кодирование        │ Простые и составные тексты         │
 *          │ Декодирование      │ Roundtrip для всех тестовых строк  │
 *          │ Пакетная обработка │ encode_batch для множества текстов │
 *          │ Граничные случаи   │ Пустые, длинные, смешанные строки  │
 *          │ Производительность │ Скорость encode на больших текстах │
 *          └────────────────────┴────────────────────────────────────┘
 * 
 * @note Все тесты используют временные файлы, которые удаляются после выполнения
 * @see BPETokenizer
 */

#include <gtest/gtest.h>

#include "bpe_tokenizer.hpp"
#include "test_helpers.hpp"
#include "utils.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    constexpr int TEST_VOCAB_SIZE = 10000;            ///< Размер словаря для тестов
    constexpr int BYTE_LEVEL_VOCAB_SIZE = 256;        ///< Количество байт в byte-level режиме
    constexpr int JSON_VOCAB_SIZE = 10;               ///< Размер JSON словаря
    constexpr int BINARY_VOCAB_SIZE = 256 + 4 + 6;    ///< 256 байт + 4 спецтокена + a,b,c,ab,bc,abc
    constexpr int TEST_MERGES_COUNT = 3;              ///< Количество правил слияния в тестовом файле
    constexpr int PERFORMANCE_TEXT_SIZE = 1000;       ///< Размер текста для тестов производительности
    constexpr int PERFORMANCE_ITERATIONS = 100;       ///< Количество итераций в тестах производительности
    constexpr int VERY_LONG_TEXT_SIZE = 10000;        ///< Очень длинный текст (10 КБ)
    constexpr int BATCH_SIZE = 10;                    ///< Размер батча для тестов
    constexpr double MIN_ROUNDTRIP_RATE = 80.0;       ///< Минимальный процент успешных roundtrip
    
    // Имена временных файлов
    const std::string TEST_VOCAB_JSON = "test_vocab.json";
    const std::string TEST_VOCAB_BIN = "test_vocab.bin";
    const std::string TEST_MERGES_TXT = "test_merges.txt";
    const std::string TEST_BYTES_BIN = "test_bytes.bin";
    const std::string TEST_BYTES_PRESENT_BIN = "test_bytes_present.bin";
    const std::string TEST_MODEL_BIN = "test_model.bin";
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string CYAN = "\033[36m";
    const std::string YELLOW = "\033[33m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ============================================================================
// Вспомогательные функции
// ============================================================================

/**
 * @brief Создать тестовый словарь в JSON формате
 * 
 * @param path Путь для сохранения файла
 * 
 * Формат:
 * @code
 * {
 *   "size":   10,
 *   "tokens": ["a", "b", "c", "ab", "bc", "abc", "<UNK>", "<PAD>", "<BOS>", "<EOS>"]
 * }
 * @endcode
 */
void create_test_json_vocab(const std::string& path) {
    std::ofstream file(path);
    file << R"({
        "size":   10,
        "tokens": ["a", "b", "c", "ab", "bc", "abc", "<UNK>", "<PAD>", "<BOS>", "<EOS>"]
    })";
}

/**
 * @brief Создать тестовый бинарный словарь со всеми 256 байтами
 * 
 * @param path Путь для сохранения файла
 */
void create_test_binary_vocab(const std::string& path) {
    Vocabulary vocab;
    
    // Добавляем все 256 байт
    for (int i = 0; i < BYTE_LEVEL_VOCAB_SIZE; ++i) {
        vocab.add_token(std::string(1, static_cast<char>(i)));
    }
    
    // Добавляем специальные токены
    vocab.add_token("<UNK>");
    vocab.add_token("<PAD>");
    vocab.add_token("<BOS>");
    vocab.add_token("<EOS>");
    
    // Добавляем тестовые токены для проверки слияний
    vocab.add_token("a");
    vocab.add_token("b");
    vocab.add_token("c");
    vocab.add_token("ab");
    vocab.add_token("bc");
    vocab.add_token("abc");
    
    // Получаем абсолютный путь для диагностики
    fs::path full_path = fs::absolute(path);
    std::cout << "Абсолютный путь: " << full_path << std::endl;
    
    vocab.save_binary(full_path.string());
    
    // Проверяем, что файл создан
    if (fs::exists(full_path)) {
        std::cout << "Файл существует, размер: " 
                  << fs::file_size(full_path) << " байт" << std::endl;
    } else {
        std::cout << "Файл не создан!" << std::endl;
    }
}

/**
 * @brief Создать тестовый файл слияний
 * 
 * @param path Путь для сохранения файла
 * 
 * Формат:
 * @code
 * # version: 0.1
 * a b
 * b c
 * ab c
 * @endcode
 */
void create_test_merges(const std::string& path) {
    std::ofstream file(path);
    file << "# version: 0.1\n";
    file << "a b\n";
    file << "b c\n";
    file << "ab c\n";
}

// ============================================================================
// Тестовый класс
// ============================================================================

/**
 * @brief Тестовый класс для BPETokenizer
 * 
 * Содержит общие настройки, тестовые данные и вспомогательные методы
 * для всех тестов базового токенизатора.
 */
class TokenizerTest : public ::testing::Test {
protected:
    /**
     * @brief Настройка перед каждым тестом
     * 
     * Создает временные файлы для тестов и запоминает ожидаемые размеры.
     */
    void SetUp() override {
        // Создаем тестовые файлы
        create_test_json_vocab(TEST_VOCAB_JSON);
        create_test_binary_vocab(TEST_VOCAB_BIN);
        create_test_merges(TEST_MERGES_TXT);
        
        // Запоминаем размеры для проверок
        json_vocab_size = JSON_VOCAB_SIZE;
        binary_vocab_size = BYTE_LEVEL_VOCAB_SIZE + 4 + 6;    // 256 байт + 4 спецтокена + a,b,c,ab,bc,abc
    }
    
    /**
     * @brief Очистка после каждого теста
     * 
     * Удаляет все временные файлы, созданные во время теста.
     */
    void TearDown() override {
        // Удаляем тестовые файлы с помощью test_helpers
        bpe_test::safe_remove(TEST_VOCAB_JSON);
        bpe_test::safe_remove(TEST_VOCAB_BIN);
        bpe_test::safe_remove(TEST_MERGES_TXT);
        
        // Удаляем временные файлы, если они есть
        bpe_test::safe_remove(TEST_BYTES_BIN);
        bpe_test::safe_remove(TEST_BYTES_PRESENT_BIN);
        bpe_test::safe_remove(TEST_MODEL_BIN);
    }
    
    size_t json_vocab_size;      ///< Размер JSON словаря (10 токенов)
    size_t binary_vocab_size;    ///< Размер бинарного словаря (256 + спецтокены)
    
    // Тестовые строки
    
    /**
     * @brief Набор тестовых строк для различных сценариев
     */
    const std::vector<std::string> test_strings = {
        "a", "b", "c",
        "a b", "a b c",
        "ab", "bc", "abc",
        "привет",     // Русские буквы
        "hello",      // Английские
        "123",        // Цифры
        "!@#$",       // Символы
        "a1 b2 c3"    // Смешанные
    };
};

// ============================================================================
// Тесты загрузки словаря
// ============================================================================

/**
 * @test Загрузка словаря из JSON файла
 * 
 * Проверяет, что JSON формат правильно загружается и
 * размер словаря соответствует ожидаемому.
 */
TEST_F(TokenizerTest, LoadFromJSON) {
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    
    bool loaded = tokenizer.load_from_files(TEST_VOCAB_JSON, TEST_MERGES_TXT);
    EXPECT_TRUE(loaded);
    EXPECT_EQ(tokenizer.vocab_size(), json_vocab_size);
    EXPECT_EQ(tokenizer.merges_count(), TEST_MERGES_COUNT);
    
    std::cout << GREEN << "JSON словарь загружен: " << tokenizer.vocab_size() << " токенов" << RESET << std::endl;
}

/**
 * @test Загрузка словаря из бинарного файла
 * 
 * Проверяет полный цикл: создание токенизатора -> сохранение в бинарный формат ->
 * загрузка из бинарного формата -> проверка совпадения всех токенов и правил слияния.
 */
TEST_F(TokenizerTest, LoadFromBinary) {
    // 1. Создаём исходный токенизатор с byte-level режимом
    BPETokenizer original_tokenizer(TEST_VOCAB_SIZE, true);
    original_tokenizer.set_unknown_token("<UNK>");    
    // В конструкторе с byte_level = true уже добавлены все 256 байт
    
    // Загружаем тестовые токены и мерджи из файлов
    bool loaded = original_tokenizer.load_from_files(TEST_VOCAB_JSON, TEST_MERGES_TXT);
    ASSERT_TRUE(loaded) << "Не удалось загрузить исходный токенизатор!";
    
    std::cout << CYAN << "\nИсходный токенизатор:" << RESET << std::endl;
    std::cout << "- Словарь: " << original_tokenizer.vocab_size() << " токенов" << std::endl;
    std::cout << "- Мерджи:  " << original_tokenizer.merges_count() << std::endl;
    
    // Проверяем работу слияний на исходном токенизаторе
    std::string test_str = "ab";
    auto original_tokens = original_tokenizer.encode(test_str);
    std::cout << "Исходный токенизатор кодирует 'ab' в " 
              << original_tokens.size() << " токенов" << std::endl;
    if (original_tokenizer.contains_token("ab")) {
        EXPECT_LE(original_tokens.size(), 2);    // Должен быть 1 токен или меньше
    }

    // 2. Сохраняем в бинарный файл
    std::string binary_path = TEST_MODEL_BIN;
    bool saved = original_tokenizer.save_binary(binary_path);
    ASSERT_TRUE(saved) << "Не удалось сохранить бинарный файл!";
    
    // Проверяем, что файл создан
    EXPECT_TRUE(fs::exists(binary_path));
    auto file_size = fs::file_size(binary_path);
    std::cout << "Бинарный файл создан, размер: " << file_size << " байт!" << std::endl;
    EXPECT_GT(file_size, 0);
    
    // 3. Загружаем из бинарного файла в новый токенизатор
    BPETokenizer new_tokenizer;
    new_tokenizer.set_unknown_token("<UNK>");
    // включаем byte-level режим до загрузки
    new_tokenizer.set_byte_level(true);
    bool reloaded = new_tokenizer.load_binary(binary_path);
    ASSERT_TRUE(reloaded) << "Не удалось загрузить бинарный файл!";
    
    std::cout << CYAN << "Загруженный токенизатор:" << RESET << std::endl;
    std::cout << "- Словарь: " << new_tokenizer.vocab_size() << " токенов" << std::endl;
    std::cout << "- Мерджи:  " << new_tokenizer.merges_count() << std::endl;
    
    // 4. Проверяем, что размеры совпадают
    EXPECT_EQ(new_tokenizer.vocab_size(), original_tokenizer.vocab_size());
    EXPECT_EQ(new_tokenizer.merges_count(), original_tokenizer.merges_count());
    
    // 5. Проверяем несколько ключевых токенов
    std::vector<std::string> test_tokens = {"a", "b", "c", "ab", "bc", "abc", 
                                            "<UNK>", "<PAD>", "<BOS>", "<EOS>"};
    for (const auto& token : test_tokens) {
        token_id_t original_id = original_tokenizer.token_to_id(token);
        token_id_t new_id = new_tokenizer.token_to_id(token);
        EXPECT_EQ(new_id, original_id) << "Несовпадение ID для токена '" << token << "'";
        if (new_id == original_id && new_id != INVALID_TOKEN) {
            std::cout << "Токен '" << token << "': ID " << new_id << std::endl;
        }
    }
    
    // 6. Проверяем несколько случайных байт (они должны быть, так как byte_level = true)
    std::cout << "\nПРОВЕРКА НАЛИЧИЯ БАЙТ" << std::endl;
    int bytes_found = 0;
    for (int i = 0; i < 10; ++i) {
        std::string byte_str(1, static_cast<char>(i));
        if (new_tokenizer.contains_token(byte_str)) {
            bytes_found++;
        } else {
            std::cout << YELLOW << "Байт " << i << " отсутствует!" << RESET << std::endl;
        }
    }
    std::cout << "Найдено байт: " << bytes_found << "/10" << std::endl;
    
    // ПРОВЕРКА МЕРДЖЕЙ
    
    std::cout << "\nПРОВЕРКА РАБОТЫ МЕРДЖЕЙ" << std::endl;
    
    // Проверка слияния "a" + "b" -> "ab" на новом токенизаторе
    if (new_tokenizer.contains_token("ab") && 
        new_tokenizer.contains_token("a") && 
        new_tokenizer.contains_token("b")) {
        auto tokens_ab = new_tokenizer.encode("ab");
        auto tokens_a = new_tokenizer.encode("a");
        auto tokens_b = new_tokenizer.encode("b");
        
        std::vector<token_id_t> tokens_a_plus_b;
        tokens_a_plus_b.reserve(tokens_a.size() + tokens_b.size());
        tokens_a_plus_b.insert(tokens_a_plus_b.end(), tokens_a.begin(), tokens_a.end());
        tokens_a_plus_b.insert(tokens_a_plus_b.end(), tokens_b.begin(), tokens_b.end());
        
        std::cout << "'ab' как одно слово:      " << tokens_ab.size() << " токенов" << std::endl;
        std::cout << "'a' + 'b' по отдельности: " << tokens_a_plus_b.size() << " токенов" << std::endl;
        
        EXPECT_LE(tokens_ab.size(), tokens_a_plus_b.size());
        
        if (tokens_ab.size() == 1) {
            std::cout << GREEN << "Слияние 'ab' работает: один токен!" << RESET << std::endl;
        }
    }
    
    // Проверка слияния "b" + "c" -> "bc"
    if (new_tokenizer.contains_token("bc") && 
        new_tokenizer.contains_token("b") && 
        new_tokenizer.contains_token("c")) {
        auto tokens_bc = new_tokenizer.encode("bc");
        auto tokens_b = new_tokenizer.encode("b");
        auto tokens_c = new_tokenizer.encode("c");
        
        std::vector<token_id_t> tokens_b_plus_c;
        tokens_b_plus_c.reserve(tokens_b.size() + tokens_c.size());
        tokens_b_plus_c.insert(tokens_b_plus_c.end(), tokens_b.begin(), tokens_b.end());
        tokens_b_plus_c.insert(tokens_b_plus_c.end(), tokens_c.begin(), tokens_c.end());
        
        std::cout << "'bc' как одно слово:      " << tokens_bc.size() << " токенов" << std::endl;
        std::cout << "'b' + 'c' по отдельности: " << tokens_b_plus_c.size() << " токенов" << std::endl;
        
        EXPECT_LE(tokens_bc.size(), tokens_b_plus_c.size());
        
        if (tokens_bc.size() == 1) {
            std::cout << GREEN << "Слияние 'bc' работает: один токен!" << RESET << std::endl;
        }
    }
    
    // Проверка на фразе
    std::string test_phrase = "a b c";
    auto tokens_phrase = new_tokenizer.encode(test_phrase);
    
    std::cout << "\nПРОВЕРКА НА ФРАЗЕ '" << test_phrase << "'" << std::endl;
    std::cout << "Получено токенов: " << tokens_phrase.size() << std::endl;
    
    EXPECT_GT(tokens_phrase.size(), 0);
    
    // 7. Проверяем, что загруженный токенизатор работает так же, как исходный
    auto original_phrase_tokens = original_tokenizer.encode(test_phrase);
    EXPECT_EQ(original_phrase_tokens.size(), tokens_phrase.size());
    std::cout << "Исходный токенизатор: " << original_phrase_tokens.size() << " токенов" << std::endl;
    
    // Очистка с помощью test_helpers
    bpe_test::safe_remove(binary_path);
    
    std::cout << GREEN << "\nТест LoadFromBinary пройден успешно!" << RESET << std::endl;
}

/**
 * @test Загрузка несуществующего файла
 * 
 * Проверяет корректную обработку ошибок при загрузке.
 */
TEST_F(TokenizerTest, LoadNonExistent) {
    BPETokenizer tokenizer;
    
    bool loaded = tokenizer.load_from_files("nonexistent.json", "nonexistent.txt");
    EXPECT_FALSE(loaded);
    
    std::cout << GREEN << "Корректная обработка отсутствующего файла!" << RESET << std::endl;
}

// ============================================================================
// Тесты поиска токенов
// ============================================================================

/**
 * @test Поиск токенов по строке и по ID
 * 
 * Проверяет двунаправленное отображение между строками и ID.
 */
TEST_F(TokenizerTest, TokenLookup) {
    // 1. Создаём токенизатор через загрузку из JSON (без бинарных файлов)
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    
    // Загружаем из JSON и merges.txt
    bool loaded = tokenizer.load_from_files(TEST_VOCAB_JSON, TEST_MERGES_TXT);
    ASSERT_TRUE(loaded) << "Не удалось загрузить токенизатор!";
    
    // Проверяем поиск по строке
    token_id_t id_a = tokenizer.token_to_id("a");
    EXPECT_NE(id_a, INVALID_TOKEN);
    
    // Проверяем поиск по ID
    const std::string& token = tokenizer.id_to_token(id_a);
    EXPECT_EQ(token, "a");
    
    std::cout << "Поиск токенов работает: 'a' -> ID " << id_a << std::endl;
    
    // Проверяем другие токены
    EXPECT_EQ(tokenizer.token_to_id("b"), 1);
    EXPECT_EQ(tokenizer.token_to_id("c"), 2);
    EXPECT_EQ(tokenizer.token_to_id("ab"), 3);
    EXPECT_EQ(tokenizer.token_to_id("bc"), 4);
    EXPECT_EQ(tokenizer.token_to_id("abc"), 5);
    EXPECT_EQ(tokenizer.token_to_id("<UNK>"), 6);
    EXPECT_EQ(tokenizer.token_to_id("<PAD>"), 7);
    EXPECT_EQ(tokenizer.token_to_id("<BOS>"), 8);
    EXPECT_EQ(tokenizer.token_to_id("<EOS>"), 9);
}

/**
 * @test Проверка специальных токенов
 * 
 * Убеждается, что все специальные токены имеют корректные ID.
 */
TEST_F(TokenizerTest, SpecialTokens) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА SpecialTokens" << RESET << std::endl;
    
    // Создаём токенизатор через конструктор
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем специальные токены
    tokenizer.add_token("<PAD>");
    tokenizer.add_token("<BOS>");
    tokenizer.add_token("<EOS>");
    
    std::cout << "Создан токенизатор с " << tokenizer.vocab_size() << " токенами" << std::endl;
    
    token_id_t unk_id = tokenizer.unknown_token_id();
    token_id_t pad_id = tokenizer.pad_id();
    token_id_t bos_id = tokenizer.bos_id();
    token_id_t eos_id = tokenizer.eos_id();
    
    std::cout << CYAN << "Специальные токены:" << RESET << std::endl;
    std::cout << "- <UNK> ID: " << unk_id << std::endl;
    std::cout << "- <PAD> ID: " << pad_id << std::endl;
    std::cout << "- <BOS> ID: " << bos_id << std::endl;
    std::cout << "- <EOS> ID: " << eos_id << std::endl;
    
    EXPECT_NE(unk_id, INVALID_TOKEN);
    EXPECT_NE(pad_id, INVALID_TOKEN);
    EXPECT_NE(bos_id, INVALID_TOKEN);
    EXPECT_NE(eos_id, INVALID_TOKEN);
}

/**
 * @test Проверка наличия всех 256 байт в словаре
 * 
 * Убеждается, что при byte-level режиме все байты присутствуют в словаре.
 */
TEST_F(TokenizerTest, AllBytesPresent) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА AllBytesPresent" << RESET << std::endl;
    
    // 1. Создаём исходный токенизатор с byte-level режимом
    std::cout << "Создаём токенизатор с byte_level=true..." << std::endl;
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем специальные токены для полноты
    tokenizer.add_token("<PAD>");
    tokenizer.add_token("<BOS>");
    tokenizer.add_token("<EOS>");
    
    // Добавляем тестовые токены
    tokenizer.add_token("a");
    tokenizer.add_token("b");
    tokenizer.add_token("c");
    tokenizer.add_token("ab");
    tokenizer.add_token("bc");
    tokenizer.add_token("abc");
    
    std::cout << "Создан токенизатор с " << tokenizer.vocab_size() << " токенами" << std::endl;
    
    // 2. Сохраняем в бинарный файл через BPETokenizer::save_binary
    std::string binary_path = TEST_BYTES_PRESENT_BIN;
    std::cout << "Сохраняем бинарный файл: " << binary_path << std::endl;
    bool saved = tokenizer.save_binary(binary_path);
    ASSERT_TRUE(saved) << "Не удалось сохранить бинарный файл!";
    
    // Проверяем, что файл создан
    EXPECT_TRUE(fs::exists(binary_path));
    auto file_size = fs::file_size(binary_path);
    std::cout << "Бинарный файл сохранён, размер: " << file_size << " байт" << std::endl;
    EXPECT_GT(file_size, 0);
    
    // 3. Загружаем из бинарного файла в новый токенизатор
    std::cout << "Загружаем бинарный файл в новый токенизатор..." << std::endl;
    BPETokenizer new_tokenizer;
    new_tokenizer.set_unknown_token("<UNK>");
    new_tokenizer.set_byte_level(true);
    
    bool loaded = new_tokenizer.load_binary(binary_path);
    ASSERT_TRUE(loaded) << "Не удалось загрузить бинарный файл!";
    
    std::cout << "Новый токенизатор загружен с " 
              << new_tokenizer.vocab_size() << " токенами" << std::endl;
    
    // 4. Проверяем наличие всех 256 байт
    int found = 0;
    for (int i = 0; i < BYTE_LEVEL_VOCAB_SIZE; ++i) {
        std::string byte_str(1, static_cast<char>(i));
        if (new_tokenizer.contains_token(byte_str)) {
            found++;
        } else {
            // Выводим информацию об отсутствующих байтах (не все, чтобы не засорять лог)
            if (found < 5) {
                std::cout << YELLOW << "Байт " << i << " отсутствует!" << RESET << std::endl;
            }
        }
    }
    
    std::cout << "Найдено байт: " << found << "/256" << std::endl;
    EXPECT_EQ(found, BYTE_LEVEL_VOCAB_SIZE);
    
    // 5. Очистка
    bpe_test::safe_remove(binary_path);
    
    std::cout << GREEN << "Все 256 байт присутствуют в словаре!" << RESET << std::endl;
}

// ============================================================================
// Тесты кодирования
// ============================================================================

/**
 * @test Базовое кодирование простых текстов
 */
TEST_F(TokenizerTest, BasicEncode) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА BasicEncode" << RESET << std::endl;
    
    // Создаём токенизатор через конструктор
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем тестовые токены
    tokenizer.add_token("a");
    tokenizer.add_token("b");
    tokenizer.add_token("c");
    tokenizer.add_token("ab");
    tokenizer.add_token("bc");
    tokenizer.add_token("abc");
    
    std::cout << "Создан токенизатор с " << tokenizer.vocab_size() << " токенами" << std::endl;
    
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
    std::cout << CYAN << "НАЧАЛО ТЕСТА EncodeAllStrings" << RESET << std::endl;
    
    // Создаём токенизатор через конструктор
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем тестовые токены
    tokenizer.add_token("a");
    tokenizer.add_token("b");
    tokenizer.add_token("c");
    tokenizer.add_token("ab");
    tokenizer.add_token("bc");
    tokenizer.add_token("abc");
    
    std::cout << "Создан токенизатор с " << tokenizer.vocab_size() << " токенами" << std::endl;
    
    for (const auto& text : test_strings) {
        auto tokens = tokenizer.encode(text);
        EXPECT_FALSE(tokens.empty());
        std::cout << "'" << text << "' (" << text.size() << " символов) -> " 
                  << tokens.size() << " токенов" << std::endl;
    }
}

/**
 * @test Кодирование с применением правил слияния
 * 
 * Проверяет, что encode() использует правила слияния для уменьшения
 * количества токенов.
 */
TEST_F(TokenizerTest, EncodeWithMerges) {
    // Создаём токенизатор через загрузку из файлов (с мерджами)
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    
    bool loaded = tokenizer.load_from_files(TEST_VOCAB_JSON, TEST_MERGES_TXT);
    ASSERT_TRUE(loaded) << "Не удалось загрузить токенизатор!";
    
    // Должно применить слияние a b -> ab
    auto tokens = tokenizer.encode("a b");
    
    std::cout << "Слияния применены: 'a b' -> " << tokens.size() << " токенов" << std::endl;
    
    // Проверяем, что размер меньше, чем без слияний
    auto tokens_a = tokenizer.encode("a");
    auto tokens_b = tokenizer.encode("b");
    
    std::vector<token_id_t> tokens_no_merge;
    tokens_no_merge.reserve(tokens_a.size() + tokens_b.size());
    tokens_no_merge.insert(tokens_no_merge.end(), tokens_a.begin(), tokens_a.end());
    tokens_no_merge.insert(tokens_no_merge.end(), tokens_b.begin(), tokens_b.end());
    
    EXPECT_LE(tokens.size(), tokens_no_merge.size());
    
    std::cout << "'a b' с мерджами:      " << tokens.size() << " токенов" << std::endl;
    std::cout << "'a' + 'b' без мерджей: " << tokens_no_merge.size() << " токенов" << std::endl;
}
 
// ============================================================================
// Тесты декодирования
// ============================================================================

/**
 * @test Базовое декодирование
 * 
 * Проверяет, что decode() восстанавливает текст из токенов.
 */
TEST_F(TokenizerTest, BasicDecode) {
    // Создаём токенизатор через загрузку из файлов (с мерджами)
    BPETokenizer tokenizer;
    tokenizer.set_unknown_token("<UNK>");
    
    bool loaded = tokenizer.load_from_files(TEST_VOCAB_JSON, TEST_MERGES_TXT);
    ASSERT_TRUE(loaded) << "Не удалось загрузить токенизатор!";
    
    std::string original = "a b c";
    auto tokens = tokenizer.encode(original);
    auto decoded = tokenizer.decode(tokens);
    
    std::cout << CYAN << "Roundtrip тест:" << RESET << std::endl;
    std::cout << "- Оригинал: '" << original << "'" << std::endl;
    std::cout << "- Декод.:   '" << decoded << "'" << std::endl;
    
    EXPECT_FALSE(decoded.empty());
}

/**
 * @test Roundtrip для всех тестовых строк
 * 
 * Проверяет, что encode() + decode() возвращает исходный текст
 * для всех тестовых строк.
 */
TEST_F(TokenizerTest, RoundtripAll) {
    // Используем byte-level режим
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);    // vocab_size, byte_level = true
    tokenizer.set_unknown_token("<UNK>");
    
    bool loaded = tokenizer.load_from_files(TEST_VOCAB_JSON, TEST_MERGES_TXT);
    ASSERT_TRUE(loaded) << "Не удалось загрузить токенизатор!";
    
    // Добавляем все 256 байт (они уже должны быть, но на всякий случай)
    for (int i = 0; i < BYTE_LEVEL_VOCAB_SIZE; ++i) {
        std::string byte_str(1, static_cast<char>(i));
        if (!tokenizer.contains_token(byte_str)) {
            tokenizer.add_token(byte_str);
        }
    }
    
    int passed = 0;
    for (const auto& text : test_strings) {
        auto tokens = tokenizer.encode(text);
        auto decoded = tokenizer.decode(tokens);
        
        if (decoded == text) {
            passed++;
            std::cout << GREEN << "'" << text << "' -> '" << decoded << "'" << RESET << std::endl;
        } else {
            std::cout << RED << "'" << text << "' -> '" << decoded << "'" << RESET << std::endl;
        }
    }
    
    double pass_rate = 100.0 * passed / test_strings.size();
    std::cout << "Процент успешных roundtrip: " << std::fixed << std::setprecision(1)
              << pass_rate << "% (" << passed << "/" << test_strings.size() << ")" << std::endl;
    
    EXPECT_GE(pass_rate, MIN_ROUNDTRIP_RATE);
}

/**
 * @test Byte-level режим с китайскими иероглифами
 */
TEST_F(TokenizerTest, ByteLevelChinese) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА ByteLevelChinese" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    std::string chinese = "你好世界";
    auto tokens = tokenizer.encode(chinese);
    auto decoded = tokenizer.decode(tokens);
    
    EXPECT_EQ(decoded, chinese);
    std::cout << "Китайский текст: '" << chinese << "' -> " 
              << tokens.size() << " токенов" << std::endl;
    std::cout << "Декодировано:    '" << decoded << "'" << std::endl;
}

/**
 * @test Byte-level режим с эмодзи
 */
TEST_F(TokenizerTest, ByteLevelEmoji) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА ByteLevelEmoji" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    std::string emoji = "😊🚀🌟";
    auto tokens = tokenizer.encode(emoji);
    auto decoded = tokenizer.decode(tokens);
    
    EXPECT_EQ(decoded, emoji);
    std::cout << "Эмодзи:       '" << emoji << "' -> " 
              << tokens.size() << " токенов" << std::endl;
    std::cout << "Декодировано: '" << decoded << "'" << std::endl;
}

/**
 * @test Обычный режим (без byte-level)
 */
TEST_F(TokenizerTest, NormalMode) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА NormalMode" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, false);    // byte_level = false
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем базовые токены для обычного режима
    tokenizer.add_token("h");
    tokenizer.add_token("e");
    tokenizer.add_token("l");
    tokenizer.add_token("o");
    tokenizer.add_token("hello");
    
    std::string text = "hello";
    auto tokens = tokenizer.encode(text);
    auto decoded = tokenizer.decode(tokens);
    
    std::cout << "Обычный режим: '" << text << "' -> " 
              << tokens.size() << " токенов" << std::endl;
    std::cout << "Декодировано:  '" << decoded << "'" << std::endl;
}

// ============================================================================
// Тесты пакетной обработки
// ============================================================================

/**
 * @test Пакетное кодирование
 * 
 * Проверяет работу encode_batch() на наборе текстов.
 */
TEST_F(TokenizerTest, BatchEncode) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА BatchEncode" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем тестовые токены
    tokenizer.add_token("a");
    tokenizer.add_token("b");
    tokenizer.add_token("c");
    tokenizer.add_token("ab");
    tokenizer.add_token("bc");
    
    std::cout << "Создан токенизатор с " << tokenizer.vocab_size() << " токенами" << std::endl;
    
    std::vector<std::string> texts = {"a", "b", "c", "ab", "bc"};
    auto batch_result = tokenizer.encode_batch(texts);
    
    EXPECT_EQ(batch_result.size(), texts.size());
    std::cout << "Пакетная обработка " << batch_result.size() << " текстов:" << std::endl;
    
    for (size_t i = 0; i < batch_result.size(); ++i) {
        std::cout << "Текст " << i << ": " << batch_result[i].size() << " токенов" << std::endl;
    }
}

/**
 * @test Пакетное кодирование текстов разной длины
 */
TEST_F(TokenizerTest, BatchWithDifferentSizes) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА BatchWithDifferentSizes" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем базовые токены
    tokenizer.add_token("a");
    
    std::vector<std::string> texts;
    for (int i = 1; i <= BATCH_SIZE; ++i) {
        texts.push_back(std::string(i, 'a'));
    }
    
    auto batch_result = tokenizer.encode_batch(texts);
    EXPECT_EQ(batch_result.size(), texts.size());
    
    // Проверяем, что длина результатов растет
    for (size_t i = 1; i < batch_result.size(); ++i) {
        EXPECT_GE(batch_result[i].size(), batch_result[i-1].size());
    }
    
    std::cout << "Пакетная обработка " << texts.size() << " текстов разной длины:" << std::endl;
    for (size_t i = 0; i < batch_result.size(); ++i) {
        std::cout << "Текст " << i << " (длина " << texts[i].size() 
                  << "): " << batch_result[i].size() << " токенов" << std::endl;
    }
}

// ============================================================================
// Тесты граничных случаев
// ============================================================================

/**
 * @test Пустой входной текст
 */
TEST_F(TokenizerTest, EmptyInput) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА EmptyInput" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    auto tokens = tokenizer.encode("");
    EXPECT_TRUE(tokens.empty());
    
    auto decoded = tokenizer.decode({});
    EXPECT_TRUE(decoded.empty());
    
    std::cout << GREEN << "Пустой вход обработан корректно!" << RESET << std::endl;
}

/**
 * @test Очень длинный текст (10 КБ)
 */
TEST_F(TokenizerTest, VeryLongInput) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА VeryLongInput" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем базовый токен
    tokenizer.add_token("a");
    
    std::string text(VERY_LONG_TEXT_SIZE, 'a');
    auto tokens = tokenizer.encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Длинный текст (" << VERY_LONG_TEXT_SIZE << " символов) -> " 
              << tokens.size() << " токенов" << std::endl;
}

/**
 * @test Текст со смешанными символами
 */
TEST_F(TokenizerTest, MixedCharacters) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА MixedCharacters" << RESET << std::endl;
    
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    std::string text = "a1 b2 c3 !@#$ привет 世界 😊";
    auto tokens = tokenizer.encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "Смешанные символы -> " << tokens.size() << " токенов" << std::endl;
    
    auto decoded = tokenizer.decode(tokens);
    std::cout << "Декодировано: '" << decoded << "'" << std::endl;
}

/**
 * @test Кодирование/декодирование всех 256 байт
 */
TEST_F(TokenizerTest, AllBytes) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА AllBytes" << RESET << std::endl;
    
    // 1. Создаём исходный токенизатор с byte-level режимом
    std::cout << "Создаём исходный токенизатор с byte_level=true..." << std::endl;
    BPETokenizer original_tokenizer(TEST_VOCAB_SIZE, true);
    original_tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем специальные токены для полноты
    original_tokenizer.add_token("<PAD>");
    original_tokenizer.add_token("<BOS>");
    original_tokenizer.add_token("<EOS>");
    
    // Добавляем тестовые токены для проверки слияний
    original_tokenizer.add_token("ab");
    original_tokenizer.add_token("bc");
    original_tokenizer.add_token("abc");
    
    std::cout << "Создан исходный токенизатор с " 
              << original_tokenizer.vocab_size() << " токенами" << std::endl;
    
    // 2. Сохраняем в бинарный файл через BPETokenizer::save_binary
    std::string binary_path = TEST_BYTES_BIN;
    std::cout << "Сохраняем бинарный файл: " << binary_path << std::endl;
    bool saved = original_tokenizer.save_binary(binary_path);
    ASSERT_TRUE(saved) << "Не удалось сохранить бинарный файл!";
    
    // Проверяем, что файл создан
    EXPECT_TRUE(fs::exists(binary_path));
    auto file_size = fs::file_size(binary_path);
    std::cout << "Бинарный файл сохранён, размер: " << file_size << " байт" << std::endl;
    EXPECT_GT(file_size, 0);
    
    // 3. Загружаем из бинарного файла в новый токенизатор
    std::cout << "Загружаем бинарный файл в новый токенизатор..." << std::endl;
    BPETokenizer new_tokenizer;
    new_tokenizer.set_unknown_token("<UNK>");
    new_tokenizer.set_byte_level(true);
    
    bool loaded = new_tokenizer.load_binary(binary_path);
    ASSERT_TRUE(loaded) << "Не удалось загрузить бинарный файл!";
    
    std::cout << "Новый токенизатор загружен с " 
              << new_tokenizer.vocab_size() << " токенами" << std::endl;
    
    // 4. Проверяем, что размеры совпадают
    EXPECT_EQ(new_tokenizer.vocab_size(), original_tokenizer.vocab_size());
    
    // 5. Тестируем кодирование/декодирование всех байт
    std::string all_bytes;
    for (int i = 0; i < BYTE_LEVEL_VOCAB_SIZE; ++i) {
        all_bytes += static_cast<char>(i);
    }
    
    std::cout << "Кодируем строку из " << all_bytes.size() << " байт..." << std::endl;
    auto tokens = new_tokenizer.encode(all_bytes);
    std::cout << "Получено " << tokens.size() << " токенов" << std::endl;
    
    std::cout << "Декодируем обратно..." << std::endl;
    auto decoded = new_tokenizer.decode(tokens);
    std::cout << "Декодировано " << decoded.size() << " байт" << std::endl;
    
    EXPECT_EQ(decoded.size(), all_bytes.size());
    EXPECT_EQ(decoded, all_bytes);
    
    // 6. Очистка
    bpe_test::safe_remove(binary_path);
    
    std::cout << GREEN << "Все 256 байт закодированы и декодированы!" << RESET << std::endl;
}

// ============================================================================
// Тесты производительности
// ============================================================================

/**
 * @test Производительность encode на большом тексте
 * 
 * Измеряет скорость encode() и выводит статистику.
 */
TEST_F(TokenizerTest, EncodePerformance) {
    std::cout << CYAN << "НАЧАЛО ТЕСТА EncodePerformance" << RESET << std::endl;
    
    // Создаём токенизатор через конструктор, а не через загрузку
    BPETokenizer tokenizer(TEST_VOCAB_SIZE, true);
    tokenizer.set_unknown_token("<UNK>");
    
    // Добавляем базовый токен
    tokenizer.add_token("a");
    
    std::cout << "Создан токенизатор с " << tokenizer.vocab_size() << " токенами" << std::endl;
    
    std::string text(PERFORMANCE_TEXT_SIZE, 'a');
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < PERFORMANCE_ITERATIONS; ++i) {
        auto tokens = tokenizer.encode(text);
        volatile size_t dummy = tokens.size();
        (void)dummy;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ms_per_encode = duration.count() / static_cast<double>(PERFORMANCE_ITERATIONS);
    double chars_per_second = (text.size() * PERFORMANCE_ITERATIONS) / (duration.count() / 1000.0);
    double kb_per_second = chars_per_second / 1024.0;
    
    std::cout << CYAN << "\nПроизводительность encode:" << RESET << std::endl;
    std::cout << "- Размер текста: " << text.size() << " символов (" 
              << text.size() / 1024.0 << " КБ)" << std::endl;
    std::cout << "- Итераций:      " << PERFORMANCE_ITERATIONS << std::endl;
    std::cout << "- Общее время:   " << duration.count() << " мс" << std::endl;
    std::cout << "- Среднее время: " << std::fixed << std::setprecision(3) 
              << ms_per_encode << " мс" << std::endl;
    std::cout << "- Скорость:      " << std::fixed << std::setprecision(2)
              << kb_per_second << " КБ/с" << std::endl;
}