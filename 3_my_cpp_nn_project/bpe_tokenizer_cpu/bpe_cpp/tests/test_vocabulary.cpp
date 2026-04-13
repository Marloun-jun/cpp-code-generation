/**
 * @file test_vocabulary.cpp
 * @brief Модульные тесты для класса Vocabulary
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Набор тестов для проверки функциональности класса Vocabulary,
 *          который обеспечивает двустороннее отображение между токенами и их ID.
 * 
 *          **Проверяемые компоненты:**
 *          ┌─────────────────────┬─────────────────────────────────────┐
 *          │ Базовые операции    │ add_token, token_to_id, id_to_token │
 *          │ Дубликаты           │ Повторное добавление токена         │
 *          │ Специальные токены  │ <PAD>, <UNK>, <BOS>, <EOS>, <MASK>  │
 *          │ Граничные случаи    │ Несуществующие токены/ID, пустой    │
 *          │ Сериализация        │ JSON (массив и ID форматы)          │
 *          │ Файловый ввод/вывод │ Сохранение и загрузка               │
 *          │ Бинарный формат     │ save_binary / load_binary           │
 *          │ Производительность  │ Добавление и поиск многих токенов   │
 *          │ Константность       │ const-корректность методов          │
 *          │ Очистка             │ clear(), empty()                    │
 *          └─────────────────────┴─────────────────────────────────────┘
 * 
 * @note Все тесты используют временные файлы, которые удаляются после выполнения
 * @see Vocabulary
 */

#include <gtest/gtest.h>

#include "test_helpers.hpp"
#include "vocabulary.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    constexpr int NUM_TOKENS_PERFORMANCE = 10000;    ///< Количество токенов для тестов производительности
    constexpr int BYTE_LEVEL_SIZE = 256;             ///< Количество байт в byte-level режиме
    constexpr size_t MIN_FREQ_TEST = 2;              ///< Минимальная частота для тестового конструктора
    
    // Имена временных файлов
    const std::string TEST_VOCAB_JSON = "test_vocab.json";
    const std::string TEST_VOCAB_BIN = "test_vocab.bin";
    const std::string CORRUPTED_BIN = "corrupted.bin";
    
#ifdef _WIN32
    const std::string PROTECTED_PATH = "C:\\Windows\\System32\\test.json";
#else
    const std::string PROTECTED_PATH = "/root/test.json";
#endif
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string CYAN = "\033[36m";
    const std::string YELLOW = "\033[33m";
}

// ============================================================================
// Тесты базовых операций
// ============================================================================

/**
 * @test Добавление и поиск токенов
 * 
 * Проверяет:
 * - add_token()                - Возвращает последовательные ID
 * - token_to_id()              - Находит токены
 * - id_to_token()              - Возвращает правильные строки
 * - contains() и contains_id() - Работают корректно
 */
TEST(VocabularyTest, AddAndRetrieve) {
    Vocabulary vocab;
    
    auto id1 = vocab.add_token("hello");
    auto id2 = vocab.add_token("world");
    auto id3 = vocab.add_token("!");
    
    EXPECT_EQ(id1, 0);
    EXPECT_EQ(id2, 1);
    EXPECT_EQ(id3, 2);
    EXPECT_EQ(vocab.size(), 3);
    
    EXPECT_EQ(vocab.token_to_id("hello"), 0);
    EXPECT_EQ(vocab.token_to_id("world"), 1);
    EXPECT_EQ(vocab.token_to_id("!"), 2);
    
    EXPECT_EQ(vocab.id_to_token(0), "hello");
    EXPECT_EQ(vocab.id_to_token(1), "world");
    EXPECT_EQ(vocab.id_to_token(2), "!");
    
    EXPECT_TRUE(vocab.contains("hello"));
    EXPECT_TRUE(vocab.contains_id(1));
    EXPECT_FALSE(vocab.contains("nonexistent"));
    EXPECT_FALSE(vocab.contains_id(999));
}

/**
 * @test Добавление дубликатов
 * 
 * Проверяет, что повторное добавление того же токена
 * возвращает существующий ID, а не создает новый.
 */
TEST(VocabularyTest, DuplicateToken) {
    Vocabulary vocab;
    
    auto id1 = vocab.add_token("test");
    auto id2 = vocab.add_token("test");
    auto id3 = vocab.add_token("TEST");
    
    EXPECT_EQ(id1, id2);
    EXPECT_NE(id1, id3);
    EXPECT_EQ(vocab.size(), 2);
    
    EXPECT_EQ(vocab.token_to_id("test"), id1);
    EXPECT_EQ(vocab.id_to_token(id1), "test");
}

/**
 * @test Добавление специальных токенов
 * 
 * Проверяет, что add_special_tokens() корректно добавляет
 * стандартные специальные токены и не дублирует их.
 */
TEST(VocabularyTest, SpecialTokens) {
    Vocabulary vocab;
    
    std::vector<std::string> specials = {
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"
    };
    
    vocab.add_special_tokens(specials);
    
    EXPECT_EQ(vocab.size(), specials.size());
    
    for (const auto& token : specials) {
        EXPECT_TRUE(vocab.contains(token));
        EXPECT_NE(vocab.token_to_id(token), INVALID_TOKEN);
    }
    
    auto first_id = vocab.token_to_id("<PAD>");
    vocab.add_special_tokens({"<PAD>"});
    EXPECT_EQ(vocab.token_to_id("<PAD>"), first_id);
    EXPECT_EQ(vocab.size(), specials.size());
}

// ============================================================================
// Тесты граничных случаев
// ============================================================================

/**
 * @test Поиск несуществующего токена
 * 
 * Проверяет, что token_to_id() возвращает INVALID_TOKEN
 * для токена, отсутствующего в словаре.
 */
TEST(VocabularyTest, InvalidToken) {
    Vocabulary vocab;
    vocab.add_token("existing");
    
    EXPECT_EQ(vocab.token_to_id("nonexistent"), INVALID_TOKEN);
    EXPECT_FALSE(vocab.contains("nonexistent"));
}

/**
 * @test Поиск по несуществующему ID
 * 
 * Проверяет, что id_to_token() выбрасывает исключение
 * для ID вне диапазона.
 */
TEST(VocabularyTest, InvalidId) {
    Vocabulary vocab;
    vocab.add_token("token");
    
    EXPECT_THROW(vocab.id_to_token(999), std::out_of_range);
    EXPECT_FALSE(vocab.contains_id(999));
}

/**
 * @test Пустой словарь
 * 
 * Проверяет поведение всех методов для пустого словаря.
 */
TEST(VocabularyTest, EmptyVocabulary) {
    Vocabulary vocab;
    
    EXPECT_EQ(vocab.size(), 0);
    EXPECT_TRUE(vocab.empty());
    EXPECT_EQ(vocab.token_to_id("anything"), INVALID_TOKEN);
    EXPECT_THROW(vocab.id_to_token(0), std::out_of_range);
}

// ============================================================================
// Тесты сериализации
// ============================================================================

/**
 * @test JSON сериализация (формат C++ реализации)
 * 
 * Проверяет, что to_json() создает корректный JSON в формате:
 * {"size": N, "tokens": ["token1", "token2", ...]}
 */
TEST(VocabularyTest, JsonSerialization) {
    Vocabulary vocab;
    vocab.add_token("first");
    vocab.add_token("second");
    vocab.add_token("third");
    vocab.add_special_tokens({"<SPECIAL>"});
    
    auto json = vocab.to_json();
    
    // Проверяем структуру JSON в формате C++ реализации
    EXPECT_TRUE(json.contains("tokens"));
    EXPECT_TRUE(json["tokens"].is_array());
    EXPECT_TRUE(json.contains("size"));
    EXPECT_EQ(json["size"], 4);
    
    const auto& tokens = json["tokens"];
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0], "first");
    EXPECT_EQ(tokens[1], "second");
    EXPECT_EQ(tokens[2], "third");
    EXPECT_EQ(tokens[3], "<SPECIAL>");
    
    Vocabulary vocab2;
    vocab2.from_json(json);
    
    EXPECT_EQ(vocab.size(), vocab2.size());
    EXPECT_EQ(vocab.token_to_id("first"), vocab2.token_to_id("first"));
    EXPECT_EQ(vocab.token_to_id("second"), vocab2.token_to_id("second"));
    EXPECT_EQ(vocab.token_to_id("third"), vocab2.token_to_id("third"));
    EXPECT_EQ(vocab.token_to_id("<SPECIAL>"), vocab2.token_to_id("<SPECIAL>"));
    EXPECT_EQ(vocab.id_to_token(0), vocab2.id_to_token(0));
    EXPECT_EQ(vocab.id_to_token(1), vocab2.id_to_token(1));
    EXPECT_EQ(vocab.id_to_token(2), vocab2.id_to_token(2));
    EXPECT_EQ(vocab.id_to_token(3), vocab2.id_to_token(3));
}

/**
 * @test JSON сериализация пустого словаря
 */
TEST(VocabularyTest, EmptyJsonSerialization) {
    Vocabulary vocab;
    
    auto json = vocab.to_json();
    
    // Пустой словарь может либо не иметь поля tokens, либо иметь пустой массив
    if (json.contains("tokens")) {
        EXPECT_TRUE(json["tokens"].empty());
        EXPECT_EQ(json["size"], 0);
    } else {
        EXPECT_TRUE(json.empty());
    }
    
    Vocabulary vocab2;
    EXPECT_NO_THROW(vocab2.from_json(json));
    EXPECT_TRUE(vocab2.empty());
}

/**
 * @test Загрузка из JSON в формате массива
 * 
 * Проверяет поддержку формата {"size": N, "tokens": [...]}
 */
TEST(VocabularyTest, JsonFromArrayFormat) {
    // Тест для формата с массивом (ваш формат)
    nlohmann::json array_json;
    array_json["size"] = 3;
    array_json["tokens"] = {"alpha", "beta", "gamma"};
    
    Vocabulary vocab_array;
    vocab_array.from_json(array_json);
    
    EXPECT_EQ(vocab_array.size(), 3);
    EXPECT_EQ(vocab_array.token_to_id("alpha"), 0);
    EXPECT_EQ(vocab_array.token_to_id("beta"), 1);
    EXPECT_EQ(vocab_array.token_to_id("gamma"), 2);
}

/**
 * @test Загрузка из JSON в формате ID->токен
 * 
 * Проверяет обратную совместимость с форматом {"0": "token", "1": "token", ...}
 */
TEST(VocabularyTest, JsonFromIdFormat) {
    // Тест для обратной совместимости с форматом ID->токен
    nlohmann::json id_json;
    id_json["0"] = "alpha";
    id_json["1"] = "beta";
    id_json["2"] = "gamma";
    
    Vocabulary vocab_id;
    vocab_id.from_json(id_json);
    
    EXPECT_EQ(vocab_id.size(), 3);
    EXPECT_EQ(vocab_id.token_to_id("alpha"), 0);
    EXPECT_EQ(vocab_id.token_to_id("beta"), 1);
    EXPECT_EQ(vocab_id.token_to_id("gamma"), 2);
}

/**
 * @test Загрузка и сохранение в формате C++
 * 
 * Проверяет полный цикл сериализации в формате C++ реализации.
 */
TEST(VocabularyTest, LoadCppFormat) {
    // Создаем JSON в формате C++ реализации
    nlohmann::json cpp_format;
    cpp_format["size"] = 5;
    cpp_format["tokens"] = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<CPP>"};
    
    Vocabulary vocab;
    vocab.from_json(cpp_format);
    
    EXPECT_EQ(vocab.size(), 5);
    EXPECT_EQ(vocab.token_to_id("<PAD>"), 0);
    EXPECT_EQ(vocab.token_to_id("<UNK>"), 1);
    EXPECT_EQ(vocab.token_to_id("<BOS>"), 2);
    EXPECT_EQ(vocab.token_to_id("<EOS>"), 3);
    EXPECT_EQ(vocab.token_to_id("<CPP>"), 4);
    
    // Проверяем обратную конвертацию
    auto back_to_json = vocab.to_json();
    EXPECT_EQ(back_to_json["size"], 5);
    EXPECT_EQ(back_to_json["tokens"].size(), 5);
    EXPECT_EQ(back_to_json["tokens"][0], "<PAD>");
}

// ============================================================================
// Тесты файлового ввода/вывода
// ============================================================================

/**
 * @test Сохранение и загрузка из JSON файла
 */
TEST(VocabularyTest, FileIO) {
    Vocabulary vocab;
    vocab.add_token("test1");
    vocab.add_token("test2");
    vocab.add_token("test3");
    
    EXPECT_TRUE(vocab.save(TEST_VOCAB_JSON));
    EXPECT_TRUE(fs::exists(TEST_VOCAB_JSON));
    
    auto file_size = fs::file_size(TEST_VOCAB_JSON);
    EXPECT_GT(file_size, 0);
    
    Vocabulary vocab2;
    EXPECT_TRUE(vocab2.load(TEST_VOCAB_JSON));
    
    EXPECT_EQ(vocab.size(), vocab2.size());
    EXPECT_EQ(vocab.token_to_id("test1"), vocab2.token_to_id("test1"));
    EXPECT_EQ(vocab.token_to_id("test2"), vocab2.token_to_id("test2"));
    EXPECT_EQ(vocab.token_to_id("test3"), vocab2.token_to_id("test3"));
    
    auto all_tokens = vocab2.get_all_tokens();
    EXPECT_EQ(all_tokens.size(), 3);
    EXPECT_EQ(all_tokens[0], "test1");
    EXPECT_EQ(all_tokens[1], "test2");
    EXPECT_EQ(all_tokens[2], "test3");
    
    bpe_test::safe_remove(TEST_VOCAB_JSON);
}

/**
 * @test Загрузка несуществующего файла
 */
TEST(VocabularyTest, LoadNonExistentFile) {
    Vocabulary vocab;
    EXPECT_FALSE(vocab.load("nonexistent.json"));
}

/**
 * @test Сохранение в защищенную директорию
 */
TEST(VocabularyTest, SaveToProtectedDir) {
#ifdef _WIN32
    GTEST_SKIP() << "Тест пропущен на Windows";
#else
    Vocabulary vocab;
    vocab.add_token("test");
    
    EXPECT_FALSE(vocab.save(PROTECTED_PATH));
#endif
}

// ============================================================================
// Тесты бинарного формата
// ============================================================================

/**
 * @test Сохранение и загрузка в бинарном формате
 * 
 * Проверяет полный цикл бинарной сериализации:
 * - Сохранение всех 256 байт и специальных токенов
 * - Загрузка и проверка целостности
 */
TEST(VocabularyTest, BinaryIO) {
    Vocabulary vocab;
    
    // Добавляем все 256 байт
    for (int i = 0; i < BYTE_LEVEL_SIZE; ++i) {
        vocab.add_token(std::string(1, static_cast<char>(i)));
    }
    
    vocab.add_special_tokens({"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"});
    
    std::cout << CYAN << "\nСоздан словарь с " << vocab.size() << " токенами" << RESET << std::endl;
    
    std::cout << "Сохраняем в " << TEST_VOCAB_BIN << "..." << std::endl;
    bool save_result = vocab.save_binary(TEST_VOCAB_BIN);
    std::cout << "save_binary вернул: " << (save_result ? GREEN + "true" : "false") << RESET << std::endl;
    EXPECT_TRUE(save_result);
    
    bool file_exists = fs::exists(TEST_VOCAB_BIN);
    std::cout << "Файл существует: " << (file_exists ? GREEN + "да" : "нет") << RESET << std::endl;
    EXPECT_TRUE(file_exists);
    
    if (file_exists) {
        auto file_size = fs::file_size(TEST_VOCAB_BIN);
        std::cout << "Размер файла: " << file_size << " байт" << std::endl;
        EXPECT_GT(file_size, 0);
    }
    
    std::cout << "Загружаем из " << TEST_VOCAB_BIN << "..." << std::endl;
    Vocabulary vocab2;
    bool load_result = vocab2.load_binary(TEST_VOCAB_BIN);
    std::cout << "load_binary вернул: " << (load_result ? GREEN + "true" : "false") << RESET << std::endl;
    EXPECT_TRUE(load_result);
    
    if (load_result) {
        std::cout << "Загружен словарь с " << vocab2.size() << " токенами" << std::endl;
        
        EXPECT_EQ(vocab.size(), vocab2.size());
        
        std::cout << CYAN << "Проверка специальных токенов:" << RESET << std::endl;
        std::cout << "- <PAD> ID: " << vocab2.token_to_id("<PAD>") << std::endl;
        std::cout << "- <UNK> ID: " << vocab2.token_to_id("<UNK>") << std::endl;
        std::cout << "- <BOS> ID: " << vocab2.token_to_id("<BOS>") << std::endl;
        std::cout << "- <EOS> ID: " << vocab2.token_to_id("<EOS>") << std::endl;
        std::cout << "- <MASK> ID: " << vocab2.token_to_id("<MASK>") << std::endl;
        
        int missing = 0;
        for (int i = 0; i < BYTE_LEVEL_SIZE; ++i) {
            std::string byte_str(1, static_cast<char>(i));
            if (!vocab2.contains(byte_str)) {
                missing++;
                if (missing <= 10) {
                    std::cout << YELLOW << "Отсутствует байт " << i << " (0x" << std::hex << i << std::dec << ")" << RESET << std::endl;
                }
            }
        }
        std::cout << "Всего отсутствует байт: " << missing << std::endl;
        EXPECT_EQ(missing, 0);
    }
    
    bpe_test::safe_remove(TEST_VOCAB_BIN);
}

/**
 * @test Загрузка поврежденного бинарного файла
 */
TEST(VocabularyTest, LoadCorruptedBinary) {
    std::ofstream file(CORRUPTED_BIN, std::ios::binary);
    
    // Записываем некорректные данные
    uint32_t bad_size = 1000000;    // Слишком большой размер
    file.write(reinterpret_cast<const char*>(&bad_size), sizeof(bad_size));
    file.close();
    
    Vocabulary vocab;
    EXPECT_FALSE(vocab.load_binary(CORRUPTED_BIN));
    
    bpe_test::safe_remove(CORRUPTED_BIN);
}

/**
 * @test Загрузка несуществующего бинарного файла
 */
TEST(VocabularyTest, LoadNonExistentBinary) {
    Vocabulary vocab;
    EXPECT_FALSE(vocab.load_binary("nonexistent.bin"));
}

// ============================================================================
// Тесты производительности
// ============================================================================

/**
 * @test Добавление множества токенов
 */
TEST(VocabularyTest, AddManyTokens) {
    Vocabulary vocab;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_TOKENS_PERFORMANCE; ++i) {
        vocab.add_token("token_" + std::to_string(i));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_EQ(vocab.size(), NUM_TOKENS_PERFORMANCE);
    std::cout << CYAN << "\nДобавление " << NUM_TOKENS_PERFORMANCE << " токенов: " 
              << duration.count() << " мс" << RESET << std::endl;
}

/**
 * @test Скорость поиска токенов
 */
TEST(VocabularyTest, TokenLookupSpeed) {
    Vocabulary vocab;
    
    for (int i = 0; i < NUM_TOKENS_PERFORMANCE; ++i) {
        vocab.add_token("token_" + std::to_string(i));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < NUM_TOKENS_PERFORMANCE; ++i) {
        auto id = vocab.token_to_id("token_" + std::to_string(i));
        EXPECT_NE(id, INVALID_TOKEN);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << CYAN << "Поиск " << NUM_TOKENS_PERFORMANCE << " токенов: " 
              << duration.count() << " мс" << RESET << std::endl;
}

// ============================================================================
// Тесты для вспомогательных методов
// ============================================================================

/**
 * @test Получение всех токенов
 */
TEST(VocabularyTest, GetAllTokens) {
    Vocabulary vocab;
    
    std::vector<std::string> expected = {"first", "second", "third"};
    
    for (const auto& token : expected) {
        vocab.add_token(token);
    }
    
    auto tokens = vocab.get_all_tokens();
    
    EXPECT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]);
    }
}

/**
 * @test next_id() и max_id()
 */
TEST(VocabularyTest, NextAndMaxId) {
    Vocabulary vocab;
    
    EXPECT_EQ(vocab.next_id(), 0);
    EXPECT_EQ(vocab.max_id(), INVALID_TOKEN);
    
    vocab.add_token("first");
    EXPECT_EQ(vocab.next_id(), 1);
    EXPECT_EQ(vocab.max_id(), 0);
    
    vocab.add_token("second");
    EXPECT_EQ(vocab.next_id(), 2);
    EXPECT_EQ(vocab.max_id(), 1);
}

// ============================================================================
// Альтернативные тесты для конструкторов
// ============================================================================

/**
 * @test Проверка создания словаря с начальными токенами
 * (имитация конструктора с вектором)
 */
TEST(VocabularyTest, InitializeWithTokens) {
    Vocabulary vocab;
    std::vector<std::string> tokens = {"a", "b", "c"};
    
    for (const auto& token : tokens) {
        vocab.add_token(token);
    }
    
    EXPECT_EQ(vocab.size(), 3);
    EXPECT_EQ(vocab.token_to_id("a"), 0);
    EXPECT_EQ(vocab.token_to_id("b"), 1);
    EXPECT_EQ(vocab.token_to_id("c"), 2);
}

/**
 * @test Проверка создания словаря с частотной картой
 * (имитация конструктора с freq map)
 */
TEST(VocabularyTest, InitializeWithFreqMap) {
    std::unordered_map<std::string, size_t> freq = {
        {"a", 10}, {"b", 5}, {"c", 1}, {"d", 2}
    };
    
    Vocabulary vocab;
    
    for (const auto& [ch, count] : freq) {
        if (count >= MIN_FREQ_TEST) {
            vocab.add_token(ch);
        }
    }
    
    EXPECT_EQ(vocab.size(), 3);    // a, b, d (c исключен, freq=1 < 2)
    EXPECT_TRUE(vocab.contains("a"));
    EXPECT_TRUE(vocab.contains("b"));
    EXPECT_FALSE(vocab.contains("c"));
    EXPECT_TRUE(vocab.contains("d"));
}

/**
 * @test Добавление токена с перемещением
 */
TEST(VocabularyTest, MoveAddToken) {
    Vocabulary vocab;
    
    std::string token = "hello";
    auto id1 = vocab.add_token(std::move(token));
    
    EXPECT_EQ(id1, 0);
    EXPECT_TRUE(vocab.contains("hello"));
    
    std::string empty_token;
    auto id2 = vocab.add_token(std::move(empty_token));
    EXPECT_NE(id2, id1);
}

// ============================================================================
// Тесты константности
// ============================================================================

/**
 * @test Проверка константной корректности
 */
TEST(VocabularyTest, ConstCorrectness) {
    const Vocabulary vocab_const;    // Пустой константный словарь
    
    EXPECT_EQ(vocab_const.size(), 0);
    EXPECT_TRUE(vocab_const.empty());
    EXPECT_EQ(vocab_const.token_to_id("anything"), INVALID_TOKEN);
    EXPECT_THROW(vocab_const.id_to_token(0), std::out_of_range);
    
    // Проверка методов, которые должны быть const
    auto all_tokens = vocab_const.get_all_tokens();
    EXPECT_TRUE(all_tokens.empty());
    
    EXPECT_EQ(vocab_const.next_id(), 0);
    EXPECT_EQ(vocab_const.max_id(), INVALID_TOKEN);
}

// ============================================================================
// Тесты уникальности и очистки
// ============================================================================

/**
 * @test Проверка уникальности ID
 */
TEST(VocabularyTest, UniqueIds) {
    Vocabulary vocab;
    std::set<token_id_t> ids;
    
    for (int i = 0; i < 100; ++i) {
        auto id = vocab.add_token("token" + std::to_string(i));
        EXPECT_TRUE(ids.find(id) == ids.end());
        ids.insert(id);
    }
    
    EXPECT_EQ(ids.size(), 100);
}

/**
 * @test Очистка словаря
 */
TEST(VocabularyTest, Clear) {
    Vocabulary vocab;
    
    vocab.add_token("one");
    vocab.add_token("two");
    vocab.add_token("three");
    
    EXPECT_EQ(vocab.size(), 3);
    
    vocab.clear();
    
    EXPECT_EQ(vocab.size(), 0);
    EXPECT_TRUE(vocab.empty());
    EXPECT_EQ(vocab.token_to_id("one"), INVALID_TOKEN);
    EXPECT_THROW(vocab.id_to_token(0), std::out_of_range);
}

/**
 * @test Многократное добавление одного токена
 */
TEST(VocabularyTest, RepeatedAdd) {
    Vocabulary vocab;
    
    auto first_id = vocab.add_token("test");
    
    for (int i = 0; i < 100; ++i) {
        auto id = vocab.add_token("test");
        EXPECT_EQ(id, first_id);
    }
    
    EXPECT_EQ(vocab.size(), 1);
}