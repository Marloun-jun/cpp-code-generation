/**
 * @file test_vocabulary.cpp
 * @brief Модульные тесты для класса Vocabulary
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Набор тестов для проверки функциональности Vocabulary:
 *          - Добавление и поиск токенов
 *          - Обработка дубликатов
 *          - Специальные токены
 *          - Граничные случаи
 *          - Сериализация в JSON
 *          - Сохранение/загрузка в файлы
 *          - Бинарный формат
 * 
 * @see Vocabulary
 */

#include <gtest/gtest.h>
#include "vocabulary.hpp"

#include <fstream>
#include <filesystem>

using namespace bpe;

// ======================================================================
// Тесты базовых операций
// ======================================================================

/**
 * @test Добавление и получение токенов
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
    
    // Поиск по токену
    EXPECT_EQ(vocab.token_to_id("hello"), 0);
    EXPECT_EQ(vocab.token_to_id("world"), 1);
    EXPECT_EQ(vocab.token_to_id("!"), 2);
    
    // Поиск по ID
    EXPECT_EQ(vocab.id_to_token(0), "hello");
    EXPECT_EQ(vocab.id_to_token(1), "world");
    EXPECT_EQ(vocab.id_to_token(2), "!");
    
    // Проверка наличия
    EXPECT_TRUE(vocab.contains("hello"));
    EXPECT_TRUE(vocab.contains_id(1));
    EXPECT_FALSE(vocab.contains("nonexistent"));
    EXPECT_FALSE(vocab.contains_id(999));
}

/**
 * @test Добавление дубликатов
 */
TEST(VocabularyTest, DuplicateToken) {
    Vocabulary vocab;
    
    auto id1 = vocab.add_token("test");
    auto id2 = vocab.add_token("test");
    auto id3 = vocab.add_token("TEST");  // Регистрозависимо
    
    EXPECT_EQ(id1, id2);
    EXPECT_NE(id1, id3);
    EXPECT_EQ(vocab.size(), 2);
    
    // Проверяем, что дубликат не изменил словарь
    EXPECT_EQ(vocab.token_to_id("test"), id1);
    EXPECT_EQ(vocab.id_to_token(id1), "test");
}

/**
 * @test Добавление специальных токенов
 */
TEST(VocabularyTest, SpecialTokens) {
    Vocabulary vocab;
    
    std::vector<std::string> specials = {
        "<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"
    };
    
    vocab.add_special_tokens(specials);
    
    EXPECT_EQ(vocab.size(), specials.size());
    
    // Проверяем, что все специальные токены добавлены
    for (const auto& token : specials) {
        EXPECT_TRUE(vocab.contains(token));
        EXPECT_NE(vocab.token_to_id(token), INVALID_TOKEN);
    }
    
    // Повторное добавление не должно изменять ID
    auto first_id = vocab.token_to_id("<PAD>");
    vocab.add_special_tokens({"<PAD>"});
    EXPECT_EQ(vocab.token_to_id("<PAD>"), first_id);
    EXPECT_EQ(vocab.size(), specials.size());
}

// ======================================================================
// Тесты граничных случаев
// ======================================================================

/**
 * @test Поиск несуществующего токена
 */
TEST(VocabularyTest, InvalidToken) {
    Vocabulary vocab;
    vocab.add_token("existing");
    
    EXPECT_EQ(vocab.token_to_id("nonexistent"), INVALID_TOKEN);
    EXPECT_FALSE(vocab.contains("nonexistent"));
}

/**
 * @test Поиск по несуществующему ID
 */
TEST(VocabularyTest, InvalidId) {
    Vocabulary vocab;
    vocab.add_token("token");
    
    EXPECT_EQ(vocab.id_to_token(999), "");
    EXPECT_FALSE(vocab.contains_id(999));
    
    // Граничные значения
    EXPECT_EQ(vocab.id_to_token(static_cast<token_id_t>(-1)), "");
}

/**
 * @test Пустой словарь
 */
TEST(VocabularyTest, EmptyVocabulary) {
    Vocabulary vocab;
    
    EXPECT_EQ(vocab.size(), 0);
    EXPECT_TRUE(vocab.empty());
    EXPECT_EQ(vocab.token_to_id("anything"), INVALID_TOKEN);
    EXPECT_EQ(vocab.id_to_token(0), "");
}

// ======================================================================
// Тесты сериализации
// ======================================================================

/**
 * @test Сериализация в JSON и обратно
 */
TEST(VocabularyTest, JsonSerialization) {
    Vocabulary vocab;
    vocab.add_token("first");
    vocab.add_token("second");
    vocab.add_token("third");
    vocab.add_special_tokens({"<SPECIAL>"});
    
    auto json = vocab.to_json();
    
    // Проверяем структуру JSON
    EXPECT_TRUE(json.contains("size"));
    EXPECT_TRUE(json.contains("tokens"));
    EXPECT_EQ(json["size"], vocab.size());
    EXPECT_EQ(json["tokens"].size(), vocab.size());
    
    // Восстанавливаем
    Vocabulary vocab2;
    vocab2.from_json(json);
    
    // Сравниваем
    EXPECT_EQ(vocab.size(), vocab2.size());
    EXPECT_EQ(vocab.token_to_id("first"), vocab2.token_to_id("first"));
    EXPECT_EQ(vocab.token_to_id("second"), vocab2.token_to_id("second"));
    EXPECT_EQ(vocab.token_to_id("third"), vocab2.token_to_id("third"));
    EXPECT_EQ(vocab.token_to_id("<SPECIAL>"), vocab2.token_to_id("<SPECIAL>"));
    EXPECT_EQ(vocab.id_to_token(1), vocab2.id_to_token(1));
}

/**
 * @test Сериализация пустого словаря
 */
TEST(VocabularyTest, EmptyJsonSerialization) {
    Vocabulary vocab;
    
    auto json = vocab.to_json();
    
    EXPECT_EQ(json["size"], 0);
    EXPECT_TRUE(json["tokens"].empty());
    
    Vocabulary vocab2;
    vocab2.from_json(json);
    
    EXPECT_TRUE(vocab2.empty());
}

/**
 * @test Загрузка из JSON с разными форматами
 */
TEST(VocabularyTest, JsonFromDifferentFormats) {
    // Формат массива
    nlohmann::json array_json = nlohmann::json::array();
    array_json.push_back("a");
    array_json.push_back("b");
    array_json.push_back("c");
    
    Vocabulary vocab_array;
    vocab_array.from_json(array_json);
    
    EXPECT_EQ(vocab_array.size(), 3);
    EXPECT_EQ(vocab_array.token_to_id("a"), 0);
    EXPECT_EQ(vocab_array.token_to_id("c"), 2);
    
    // Формат объекта с tokens
    nlohmann::json object_json;
    object_json["size"] = 3;
    object_json["tokens"] = {"x", "y", "z"};
    
    Vocabulary vocab_object;
    vocab_object.from_json(object_json);
    
    EXPECT_EQ(vocab_object.size(), 3);
    EXPECT_EQ(vocab_object.token_to_id("x"), 0);
    EXPECT_EQ(vocab_object.token_to_id("z"), 2);
}

// ======================================================================
// Тесты файлового ввода/вывода
// ======================================================================

/**
 * @test Сохранение и загрузка в JSON файл
 */
TEST(VocabularyTest, FileIO) {
    Vocabulary vocab;
    vocab.add_token("test1");
    vocab.add_token("test2");
    vocab.add_token("test3");
    
    std::string test_path = "test_vocab.json";
    
    // Сохраняем
    EXPECT_TRUE(vocab.save(test_path));
    EXPECT_TRUE(std::filesystem::exists(test_path));
    
    // Проверяем размер файла
    auto file_size = std::filesystem::file_size(test_path);
    EXPECT_GT(file_size, 0);
    
    // Загружаем
    Vocabulary vocab2;
    EXPECT_TRUE(vocab2.load(test_path));
    
    // Сравниваем
    EXPECT_EQ(vocab.size(), vocab2.size());
    EXPECT_EQ(vocab.token_to_id("test1"), vocab2.token_to_id("test1"));
    EXPECT_EQ(vocab.token_to_id("test2"), vocab2.token_to_id("test2"));
    EXPECT_EQ(vocab.token_to_id("test3"), vocab2.token_to_id("test3"));
    
    // Проверяем все токены
    auto all_tokens = vocab2.get_all_tokens();
    EXPECT_EQ(all_tokens.size(), 3);
    EXPECT_EQ(all_tokens[0], "test1");
    EXPECT_EQ(all_tokens[1], "test2");
    EXPECT_EQ(all_tokens[2], "test3");
    
    // Очистка
    std::filesystem::remove(test_path);
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
    Vocabulary vocab;
    vocab.add_token("test");
    
    EXPECT_FALSE(vocab.save("/root/test.json"));
}

// ======================================================================
// Тесты бинарного формата
// ======================================================================

/**
 * @test Сохранение и загрузка в бинарный файл
 */
TEST(VocabularyTest, BinaryIO) {
    Vocabulary vocab;
    
    // Добавляем все 256 байт
    for (int i = 0; i < 256; ++i) {
        vocab.add_token(std::string(1, static_cast<char>(i)));
    }
    
    // Добавляем специальные токены
    vocab.add_special_tokens({"<PAD>", "<UNK>", "<BOS>", "<EOS>"});
    
    std::string test_path = "test_vocab.bin";
    
    // Сохраняем
    EXPECT_TRUE(vocab.save_binary(test_path));
    EXPECT_TRUE(std::filesystem::exists(test_path));
    
    // Загружаем
    Vocabulary vocab2;
    EXPECT_TRUE(vocab2.load_binary(test_path));
    
    // Сравниваем размер
    EXPECT_EQ(vocab.size(), vocab2.size());
    
    // Проверяем несколько случайных токенов
    EXPECT_EQ(vocab.token_to_id("a"), vocab2.token_to_id("a"));
    EXPECT_EQ(vocab.token_to_id("z"), vocab2.token_to_id("z"));
    EXPECT_EQ(vocab.token_to_id("<UNK>"), vocab2.token_to_id("<UNK>"));
    
    // Проверяем, что все 256 байт сохранились
    for (int i = 0; i < 256; ++i) {
        std::string byte_str(1, static_cast<char>(i));
        EXPECT_TRUE(vocab2.contains(byte_str));
    }
    
    // Очистка
    std::filesystem::remove(test_path);
}

/**
 * @test Загрузка поврежденного бинарного файла
 */
TEST(VocabularyTest, LoadCorruptedBinary) {
    // Создаем поврежденный файл
    std::string test_path = "corrupted.bin";
    std::ofstream file(test_path, std::ios::binary);
    
    // Записываем неверный размер
    uint32_t bad_size = 1000000;
    file.write(reinterpret_cast<const char*>(&bad_size), sizeof(bad_size));
    file.close();
    
    Vocabulary vocab;
    EXPECT_FALSE(vocab.load_binary(test_path));
    
    std::filesystem::remove(test_path);
}

// ======================================================================
// Тесты производительности
// ======================================================================

/**
 * @test Скорость добавления большого количества токенов
 */
TEST(VocabularyTest, AddManyTokens) {
    Vocabulary vocab;
    
    const int num_tokens = 10000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_tokens; ++i) {
        vocab.add_token("token_" + std::to_string(i));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    EXPECT_EQ(vocab.size(), num_tokens);
    std::cout << "⏱️  Добавление " << num_tokens << " токенов: " 
              << duration.count() << " мс" << std::endl;
}

/**
 * @test Скорость поиска токенов
 */
TEST(VocabularyTest, TokenLookupSpeed) {
    Vocabulary vocab;
    
    // Заполняем словарь
    const int num_tokens = 10000;
    for (int i = 0; i < num_tokens; ++i) {
        vocab.add_token("token_" + std::to_string(i));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Ищем каждый токен
    for (int i = 0; i < num_tokens; ++i) {
        auto id = vocab.token_to_id("token_" + std::to_string(i));
        EXPECT_NE(id, INVALID_TOKEN);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "⏱️  Поиск " << num_tokens << " токенов: " 
              << duration.count() << " мс" << std::endl;
}

// ======================================================================
// Тесты для get_all_tokens
// ======================================================================

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
