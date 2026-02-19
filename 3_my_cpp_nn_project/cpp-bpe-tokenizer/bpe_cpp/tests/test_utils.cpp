/**
 * @file test_utils.cpp
 * @brief Модульные тесты для вспомогательных утилит
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Набор тестов для проверки функциональности утилит:
 *          - Чтение/запись файлов
 *          - Форматирование размеров
 *          - Валидация UTF-8
 *          - Экранирование строк
 *          - Таймер для измерения времени
 * 
 * @see utils.hpp
 */

#include <gtest/gtest.h>
#include "utils.hpp"

#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>

using namespace bpe::utils;

// ======================================================================
// Тесты для работы с файлами
// ======================================================================

/**
 * @test Чтение и запись файла
 */
TEST(UtilsTest, ReadWriteFile) {
    std::string test_content = "Hello, World!";
    std::string test_path = "test_file.txt";
    
    // Запись
    EXPECT_TRUE(write_file(test_path, test_content));
    EXPECT_TRUE(std::filesystem::exists(test_path));
    
    // Чтение
    auto read_content = read_file(test_path);
    EXPECT_EQ(read_content, test_content);
    
    // Проверка размера
    auto file_size = std::filesystem::file_size(test_path);
    EXPECT_EQ(file_size, test_content.size());
    
    // Очистка
    std::filesystem::remove(test_path);
}

/**
 * @test Чтение несуществующего файла
 */
TEST(UtilsTest, ReadNonExistentFile) {
    EXPECT_THROW(read_file("nonexistent_file.txt"), std::runtime_error);
}

/**
 * @test Запись в бинарный файл
 */
TEST(UtilsTest, WriteBinaryFile) {
    std::string test_content;
    for (int i = 0; i < 256; ++i) {
        test_content += static_cast<char>(i);
    }
    std::string test_path = "test_binary.bin";
    
    EXPECT_TRUE(write_file(test_path, test_content));
    EXPECT_TRUE(std::filesystem::exists(test_path));
    
    auto read_content = read_file(test_path);
    EXPECT_EQ(read_content.size(), 256);
    EXPECT_EQ(read_content, test_content);
    
    std::filesystem::remove(test_path);
}

/**
 * @test Запись в защищенную директорию
 */
TEST(UtilsTest, WriteToProtectedDir) {
    EXPECT_FALSE(write_file("/root/test.txt", "test"));
}

// ======================================================================
// Тесты для форматирования размеров
// ======================================================================

/**
 * @test Форматирование размера в байтах
 */
TEST(UtilsTest, FormatSize) {
    // Байты
    EXPECT_EQ(format_size(0), "0.00 B");
    EXPECT_EQ(format_size(500), "500.00 B");
    EXPECT_EQ(format_size(1023), "1023.00 B");
    
    // Килобайты
    EXPECT_EQ(format_size(1024), "1.00 KB");
    EXPECT_EQ(format_size(1500), "1.46 KB");
    EXPECT_EQ(format_size(1024 * 1024 - 1), "1024.00 KB");
    
    // Мегабайты
    EXPECT_EQ(format_size(1024 * 1024), "1.00 MB");
    EXPECT_EQ(format_size(1500 * 1024), "1.46 MB");
    EXPECT_EQ(format_size(1500000), "1.43 MB");
    
    // Гигабайты
    EXPECT_EQ(format_size(1024LL * 1024 * 1024), "1.00 GB");
    EXPECT_EQ(format_size(1500LL * 1024 * 1024), "1.46 GB");
    EXPECT_EQ(format_size(1500000000), "1.40 GB");
    
    // Терабайты
    EXPECT_EQ(format_size(1024LL * 1024 * 1024 * 1024), "1.00 TB");
    EXPECT_EQ(format_size(1500LL * 1024 * 1024 * 1024), "1.46 TB");
}

/**
 * @test Граничные значения
 */
TEST(UtilsTest, FormatSizeEdgeCases) {
    EXPECT_EQ(format_size(1), "1.00 B");
    EXPECT_EQ(format_size(1024), "1.00 KB");
    EXPECT_EQ(format_size(1024 * 1024), "1.00 MB");
    EXPECT_EQ(format_size(1024LL * 1024 * 1024), "1.00 GB");
    EXPECT_EQ(format_size(1024LL * 1024 * 1024 * 1024), "1.00 TB");
}

// ======================================================================
// Тесты для валидации UTF-8
// ======================================================================

/**
 * @test Валидные UTF-8 строки
 */
TEST(UtilsTest, ValidUTF8) {
    // ASCII
    EXPECT_TRUE(is_valid_utf8(""));
    EXPECT_TRUE(is_valid_utf8("Hello"));
    EXPECT_TRUE(is_valid_utf8("Hello World! 123"));
    
    // Русские буквы
    EXPECT_TRUE(is_valid_utf8("Привет"));
    EXPECT_TRUE(is_valid_utf8("Привет мир!"));
    
    // Китайские иероглифы
    EXPECT_TRUE(is_valid_utf8("你好"));
    EXPECT_TRUE(is_valid_utf8("世界"));
    
    // Японские
    EXPECT_TRUE(is_valid_utf8("こんにちは"));
    
    // Эмодзи
    EXPECT_TRUE(is_valid_utf8("😊"));
    EXPECT_TRUE(is_valid_utf8("🔥 C++ 🔥"));
    
    // Смешанные
    EXPECT_TRUE(is_valid_utf8("Hello Привет 世界 😊"));
}

/**
 * @test Невалидные UTF-8 строки
 */
TEST(UtilsTest, InvalidUTF8) {
    // Неверный стартовый байт
    EXPECT_FALSE(is_valid_utf8("\xFF"));
    EXPECT_FALSE(is_valid_utf8("\xC0"));
    
    // Незавершенные последовательности
    EXPECT_FALSE(is_valid_utf8("\xC2"));  // Должен быть еще один байт
    EXPECT_FALSE(is_valid_utf8("\xE0\xA0"));  // Должно быть 3 байта
    EXPECT_FALSE(is_valid_utf8("\xF0\x90\x80"));  // Должно быть 4 байта
    
    // Неверные continuation байты
    EXPECT_FALSE(is_valid_utf8("\xC2\xC0"));  // Второй байт не 10xxxxxx
    EXPECT_FALSE(is_valid_utf8("\xE0\xA0\xC0"));
    
    // Overlong encoding
    EXPECT_FALSE(is_valid_utf8("\xC0\x80"));  // ASCII символ в 2 байтах
    EXPECT_FALSE(is_valid_utf8("\xE0\x80\x80"));  // ASCII в 3 байтах
    EXPECT_FALSE(is_valid_utf8("\xF0\x80\x80\x80"));  // ASCII в 4 байтах
    
    // Surrogate символы
    EXPECT_FALSE(is_valid_utf8("\xED\xA0\x80"));  // Surrogate
    
    // Выход за пределы Unicode
    EXPECT_FALSE(is_valid_utf8("\xF4\x90\x80\x80"));  // > U+10FFFF
}

/**
 * @test Смесь валидного и невалидного
 */
TEST(UtilsTest, MixedUTF8) {
    std::string mixed = "Hello" + std::string("\xFF") + "World";
    EXPECT_FALSE(is_valid_utf8(mixed));
}

// ======================================================================
// Тесты для экранирования строк
// ======================================================================

/**
 * @test Экранирование спецсимволов
 */
TEST(UtilsTest, EscapeString) {
    // Управляющие символы
    EXPECT_EQ(escape_string("\n"), "\\n");
    EXPECT_EQ(escape_string("\r"), "\\r");
    EXPECT_EQ(escape_string("\t"), "\\t");
    EXPECT_EQ(escape_string("\b"), "\\b");
    EXPECT_EQ(escape_string("\f"), "\\f");
    
    // Комбинации
    EXPECT_EQ(escape_string("Hello\nWorld"), "Hello\\nWorld");
    EXPECT_EQ(escape_string("Tab\tHere"), "Tab\\tHere");
    EXPECT_EQ(escape_string("Carriage\rReturn"), "Carriage\\rReturn");
    
    // Кавычки и слеши
    EXPECT_EQ(escape_string("Quote\"Test"), "Quote\\\"Test");
    EXPECT_EQ(escape_string("Back\\Slash"), "Back\\\\Slash");
    EXPECT_EQ(escape_string("Single'Quote"), "Single'Quote");  // Не экранируется по умолчанию
    
    // Пустая строка
    EXPECT_EQ(escape_string(""), "");
}

/**
 * @test Экранирование контрольных символов
 */
TEST(UtilsTest, EscapeControlChars) {
    // Символы от 0x00 до 0x1F
    for (int i = 0; i <= 0x1F; ++i) {
        if (i == '\n' || i == '\r' || i == '\t' || i == '\b' || i == '\f') {
            continue;  // Уже проверено в другом тесте
        }
        std::string input(1, static_cast<char>(i));
        auto escaped = escape_string(input);
        EXPECT_NE(escaped.find("\\x"), std::string::npos);
    }
    
    // DEL символ (0x7F)
    std::string del(1, static_cast<char>(0x7F));
    EXPECT_NE(escape_string(del).find("\\x7f"), std::string::npos);
}

/**
 * @test Экранирование UTF-8 символов
 */
TEST(UtilsTest, EscapeUTF8) {
    // Русские буквы должны остаться без изменений
    EXPECT_EQ(escape_string("Привет"), "Привет");
    
    // Китайские иероглифы
    EXPECT_EQ(escape_string("你好"), "你好");
    
    // Эмодзи
    EXPECT_EQ(escape_string("😊"), "😊");
    
    // Смешанные
    EXPECT_EQ(escape_string("Hello Привет 世界 😊"), "Hello Привет 世界 😊");
}

/**
 * @test Экранирование длинных строк
 */
TEST(UtilsTest, EscapeLongString) {
    std::string long_str(10000, 'a');
    long_str += "\n\t\"\\";
    long_str += std::string(10000, 'b');
    
    auto escaped = escape_string(long_str);
    
    EXPECT_GT(escaped.size(), long_str.size());
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
    EXPECT_NE(escaped.find("\\t"), std::string::npos);
    EXPECT_NE(escaped.find("\\\""), std::string::npos);
    EXPECT_NE(escaped.find("\\\\"), std::string::npos);
}

// ======================================================================
// Тесты для таймера
// ======================================================================

/**
 * @test Базовые функции таймера
 */
TEST(UtilsTest, TimerBasic) {
    Timer timer;
    
    // Сразу после создания время должно быть близко к 0
    EXPECT_LT(timer.elapsed(), 0.001);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto elapsed = timer.elapsed();
    
    EXPECT_GE(elapsed, 0.009);
    EXPECT_LT(elapsed, 0.05);
}

/**
 * @test Сброс таймера
 */
TEST(UtilsTest, TimerReset) {
    Timer timer;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto elapsed1 = timer.elapsed();
    EXPECT_GT(elapsed1, 0.0);
    
    timer.reset();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    auto elapsed2 = timer.elapsed();
    
    EXPECT_GT(elapsed2, 0.004);
    EXPECT_LT(elapsed2, 0.01);
    
    // После сброса время должно быть меньше
    EXPECT_LT(elapsed2, elapsed1);
}

/**
 * @test Точность таймера
 */
TEST(UtilsTest, TimerPrecision) {
    Timer timer;
    
    // Очень короткая задержка
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    auto elapsed = timer.elapsed();
    
    // Должно быть больше 0, но меньше 1 мс
    EXPECT_GT(elapsed, 0.0);
    EXPECT_LT(elapsed, 0.001);
}

/**
 * @test Многократные измерения
 */
TEST(UtilsTest, TimerMultipleMeasurements) {
    Timer timer;
    
    std::vector<double> measurements;
    
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        measurements.push_back(timer.elapsed());
    }
    
    // Проверяем, что время монотонно растет
    for (size_t i = 1; i < measurements.size(); ++i) {
        EXPECT_GT(measurements[i], measurements[i-1]);
    }
}

// ======================================================================
// Тесты для комбинированных операций
// ======================================================================

/**
 * @test Чтение и валидация UTF-8 файла
 */
TEST(UtilsTest, ReadAndValidateUTF8) {
    std::string test_path = "test_utf8.txt";
    std::string test_content = "Hello Привет 世界 😊";
    
    EXPECT_TRUE(write_file(test_path, test_content));
    
    auto read_content = read_file(test_path);
    EXPECT_TRUE(is_valid_utf8(read_content));
    EXPECT_EQ(read_content, test_content);
    
    std::filesystem::remove(test_path);
}

/**
 * @test Чтение, экранирование и запись
 */
TEST(UtilsTest, ReadEscapeWrite) {
    std::string test_path = "test_escape.txt";
    std::string test_content = "Line1\nLine2\tLine3\"Quote\"";
    
    EXPECT_TRUE(write_file(test_path, test_content));
    
    auto read_content = read_file(test_path);
    auto escaped = escape_string(read_content);
    
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
    EXPECT_NE(escaped.find("\\t"), std::string::npos);
    EXPECT_NE(escaped.find("\\\""), std::string::npos);
    
    std::filesystem::remove(test_path);
}
