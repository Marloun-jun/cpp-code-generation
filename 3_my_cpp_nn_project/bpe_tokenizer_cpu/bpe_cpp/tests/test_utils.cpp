/**
 * @file test_utils.cpp
 * @brief Модульные тесты для вспомогательных утилит
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Набор тестов для проверки функциональности всех утилит,
 *          используемых в проекте BPE токенизатора.
 */

#include <gtest/gtest.h>

#include "utils.hpp"
#include "test_helpers.hpp"  // Добавлен include для test_helpers

#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <cstdio>

namespace fs = std::filesystem;
using namespace bpe::utils;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr int SLEEP_MS_SHORT = 5;
    constexpr int SLEEP_MS_MEDIUM = 10;
    constexpr int SLEEP_US_SHORT = 100;
    constexpr int BINARY_SIZE = 256;
    constexpr int LONG_STRING_SIZE = 10000;
    constexpr double EPSILON = 0.005;
    
    const std::string TEST_FILE_TXT = "test_file.txt";
    const std::string TEST_BINARY_BIN = "test_binary.bin";
    const std::string TEST_UTF8_TXT = "test_utf8.txt";
    const std::string TEST_ESCAPE_TXT = "test_escape.txt";
}

// ======================================================================
// Тесты для работы с файлами
// ======================================================================

TEST(UtilsTest, ReadWriteFile) {
    std::string test_content = "Hello, World!";
    
    EXPECT_TRUE(write_file(TEST_FILE_TXT, test_content));
    EXPECT_TRUE(fs::exists(TEST_FILE_TXT));
    
    auto read_content = read_file(TEST_FILE_TXT);
    EXPECT_EQ(read_content, test_content);
    
    auto file_size = fs::file_size(TEST_FILE_TXT);
    EXPECT_EQ(file_size, test_content.size());
    
    bpe_test::safe_remove(TEST_FILE_TXT);
}

TEST(UtilsTest, ReadNonExistentFile) {
    EXPECT_THROW(read_file("nonexistent_file.txt"), std::runtime_error);
}

TEST(UtilsTest, WriteBinaryFile) {
    std::string test_content;
    for (int i = 0; i < BINARY_SIZE; ++i) {
        test_content += static_cast<char>(i);
    }
    
    EXPECT_TRUE(write_file(TEST_BINARY_BIN, test_content));
    EXPECT_TRUE(fs::exists(TEST_BINARY_BIN));
    
    auto read_content = read_file(TEST_BINARY_BIN);
    EXPECT_EQ(read_content.size(), BINARY_SIZE);
    EXPECT_EQ(read_content, test_content);
    
    bpe_test::safe_remove(TEST_BINARY_BIN);
}

TEST(UtilsTest, WriteToProtectedDir) {
#ifdef _WIN32
    GTEST_SKIP() << "Тест пропущен на Windows";
#else
    EXPECT_FALSE(write_file("/root/test.txt", "test"));
#endif
}

// ======================================================================
// Тесты для форматирования размеров
// ======================================================================

TEST(UtilsTest, FormatSize) {
    EXPECT_EQ(format_size(0), "0.00 байт");
    EXPECT_EQ(format_size(500), "500.00 байт");
    EXPECT_EQ(format_size(1023), "1023.00 байт");
    
    EXPECT_EQ(format_size(1024), "1.00 КБ");
    EXPECT_EQ(format_size(1500), "1.46 КБ");
    EXPECT_EQ(format_size(1024 * 1024 - 1), "1024.00 КБ");
    
    EXPECT_EQ(format_size(1024 * 1024), "1.00 МБ");
    EXPECT_EQ(format_size(1500 * 1024), "1.46 МБ");
    EXPECT_EQ(format_size(1500000), "1.43 МБ");
    
    EXPECT_EQ(format_size(1024LL * 1024 * 1024), "1.00 ГБ");
    EXPECT_EQ(format_size(1500LL * 1024 * 1024), "1.46 ГБ");
    EXPECT_EQ(format_size(1500000000), "1.40 ГБ");
    
    EXPECT_EQ(format_size(1024LL * 1024 * 1024 * 1024), "1.00 ТБ");
    EXPECT_EQ(format_size(1500LL * 1024 * 1024 * 1024), "1.46 ТБ");
}

TEST(UtilsTest, FormatSizeEdgeCases) {
    EXPECT_EQ(format_size(1), "1.00 байт");
    EXPECT_EQ(format_size(1024), "1.00 КБ");
    EXPECT_EQ(format_size(1024 * 1024), "1.00 МБ");
    EXPECT_EQ(format_size(1024LL * 1024 * 1024), "1.00 ГБ");
    EXPECT_EQ(format_size(1024LL * 1024 * 1024 * 1024), "1.00 ТБ");
}

// ======================================================================
// Тесты для валидации UTF-8
// ======================================================================

TEST(UtilsTest, ValidUTF8) {
    EXPECT_TRUE(is_valid_utf8(""));
    EXPECT_TRUE(is_valid_utf8("Hello"));
    EXPECT_TRUE(is_valid_utf8("Hello World! 123"));
    
    EXPECT_TRUE(is_valid_utf8("Привет"));
    EXPECT_TRUE(is_valid_utf8("Привет мир!"));
    
    EXPECT_TRUE(is_valid_utf8("你好"));
    EXPECT_TRUE(is_valid_utf8("世界"));
    
    EXPECT_TRUE(is_valid_utf8("こんにちは"));
    
    EXPECT_TRUE(is_valid_utf8("😊"));
    EXPECT_TRUE(is_valid_utf8("🔥 C++ 🔥"));
    
    EXPECT_TRUE(is_valid_utf8("Hello Привет 世界 😊"));
}

TEST(UtilsTest, InvalidUTF8) {
    EXPECT_FALSE(is_valid_utf8("\xFF"));
    EXPECT_FALSE(is_valid_utf8("\xC0"));
    EXPECT_FALSE(is_valid_utf8("\xC1"));
    
    EXPECT_FALSE(is_valid_utf8("\xC2"));
    EXPECT_FALSE(is_valid_utf8("\xE0\xA0"));
    EXPECT_FALSE(is_valid_utf8("\xF0\x90\x80"));
    
    EXPECT_FALSE(is_valid_utf8("\xC2\xC0"));
    EXPECT_FALSE(is_valid_utf8("\xE0\xA0\xC0"));
    EXPECT_FALSE(is_valid_utf8("\xF0\x90\x80\xC0"));
    
    EXPECT_FALSE(is_valid_utf8("\xC0\x80"));
    EXPECT_FALSE(is_valid_utf8("\xE0\x80\x80"));
    EXPECT_FALSE(is_valid_utf8("\xF0\x80\x80\x80"));
    
    EXPECT_FALSE(is_valid_utf8("\xC1\xBF"));
    EXPECT_FALSE(is_valid_utf8("\xE0\x9F\x80"));
    EXPECT_FALSE(is_valid_utf8("\xF0\x8F\x80\x80"));
    
    EXPECT_FALSE(is_valid_utf8("\xED\xA0\x80"));
    EXPECT_FALSE(is_valid_utf8("\xED\xBF\xBF"));
    
    EXPECT_FALSE(is_valid_utf8("\xF4\x90\x80\x80"));
}

TEST(UtilsTest, MixedUTF8) {
    std::string mixed = "Hello" + std::string("\xFF") + "World";
    EXPECT_FALSE(is_valid_utf8(mixed));
}

// ======================================================================
// Тесты для экранирования строк
// ======================================================================

TEST(UtilsTest, EscapeString) {
    EXPECT_EQ(escape_string("\n"), "\\n");
    EXPECT_EQ(escape_string("\r"), "\\r");
    EXPECT_EQ(escape_string("\t"), "\\t");
    EXPECT_EQ(escape_string("\b"), "\\b");
    EXPECT_EQ(escape_string("\f"), "\\f");
    EXPECT_EQ(escape_string("\v"), "\\v");
    EXPECT_EQ(escape_string("\a"), "\\a");
    
    EXPECT_EQ(escape_string("Hello\nWorld"), "Hello\\nWorld");
    EXPECT_EQ(escape_string("Tab\tHere"), "Tab\\tHere");
    EXPECT_EQ(escape_string("Carriage\rReturn"), "Carriage\\rReturn");
    
    EXPECT_EQ(escape_string("Quote\"Test"), "Quote\\\"Test");
    EXPECT_EQ(escape_string("Back\\Slash"), "Back\\\\Slash");
    EXPECT_EQ(escape_string("Single'Quote"), "Single\\'Quote");
    
    EXPECT_EQ(escape_string(""), "");
}

TEST(UtilsTest, EscapeControlChars) {
    for (int i = 0; i <= 0x1F; ++i) {
        if (i == '\n' || i == '\r' || i == '\t' || i == '\b' || i == '\f' || 
            i == '\v' || i == '\a') {
            continue;
        }
        std::string input(1, static_cast<char>(i));
        auto escaped = escape_string(input);
        EXPECT_NE(escaped.find("\\x"), std::string::npos);
    }
    
    std::string del(1, static_cast<char>(0x7F));
    auto escaped = escape_string(del);
    EXPECT_NE(escaped.find("\\x7f"), std::string::npos);
}

TEST(UtilsTest, EscapeUTF8) {
    EXPECT_EQ(escape_string("Привет"), "Привет");
    EXPECT_EQ(escape_string("你好"), "你好");
    EXPECT_EQ(escape_string("😊"), "😊");
    EXPECT_EQ(escape_string("Hello Привет 世界 😊"), "Hello Привет 世界 😊");
}

TEST(UtilsTest, EscapeLongString) {
    std::string long_str(LONG_STRING_SIZE, 'a');
    long_str += "\n\t\"\\";
    long_str += std::string(LONG_STRING_SIZE, 'b');
    
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

TEST(UtilsTest, TimerBasic) {
    Timer timer;
    
    EXPECT_LT(timer.elapsed(), 0.001);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS_MEDIUM));
    auto elapsed = timer.elapsed();
    
    EXPECT_GE(elapsed, SLEEP_MS_MEDIUM / 1000.0 - EPSILON);
    EXPECT_LT(elapsed, SLEEP_MS_MEDIUM / 1000.0 + EPSILON * 2);
}

TEST(UtilsTest, TimerReset) {
    Timer timer;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS_MEDIUM));
    auto elapsed1 = timer.elapsed();
    EXPECT_GT(elapsed1, 0.0);
    
    timer.reset();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS_SHORT));
    auto elapsed2 = timer.elapsed();
    
    EXPECT_GE(elapsed2, SLEEP_MS_SHORT / 1000.0 - EPSILON);
    EXPECT_LT(elapsed2, SLEEP_MS_SHORT / 1000.0 + EPSILON * 2);
    
    EXPECT_LT(elapsed2, elapsed1);
}

TEST(UtilsTest, TimerPrecision) {
    Timer timer;
    
    std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_US_SHORT));
    auto elapsed = timer.elapsed();
    
    EXPECT_GT(elapsed, 0.0);
    EXPECT_LT(elapsed, 0.001);
}

TEST(UtilsTest, TimerMultipleMeasurements) {
    Timer timer;
    
    std::vector<double> measurements;
    
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS_SHORT));
        measurements.push_back(timer.elapsed());
    }
    
    for (size_t i = 1; i < measurements.size(); ++i) {
        EXPECT_GT(measurements[i], measurements[i-1]);
    }
}

TEST(UtilsTest, TimerMilliseconds) {
    Timer timer;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_MS_MEDIUM));
    
    auto elapsed_ms = timer.elapsed_ms();
    EXPECT_GE(elapsed_ms, SLEEP_MS_MEDIUM - 1);
    EXPECT_LT(elapsed_ms, SLEEP_MS_MEDIUM + 5);
}

TEST(UtilsTest, TimerMicroseconds) {
    Timer timer;
    
    std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_US_SHORT));
    
    auto elapsed_us = timer.elapsed_us();
    EXPECT_GE(elapsed_us, SLEEP_US_SHORT - 20);
    EXPECT_LT(elapsed_us, SLEEP_US_SHORT + 200);
}

// ======================================================================
// Тесты для обработки строк (только те функции, которые есть в utils.hpp)
// ======================================================================

TEST(UtilsTest, StringCase) {
    EXPECT_EQ(to_lower("Hello World"), "hello world");
    EXPECT_EQ(to_lower("123 ABC"), "123 abc");
    EXPECT_EQ(to_upper("Hello World"), "HELLO WORLD");
    EXPECT_EQ(to_upper("123 abc"), "123 ABC");
    EXPECT_EQ(to_lower(""), "");
    EXPECT_EQ(to_upper(""), "");
}

TEST(UtilsTest, StringTrim) {
    EXPECT_EQ(trim("  hello  "), "hello");
    EXPECT_EQ(trim("\t\n  hello  \n\t"), "hello");
    EXPECT_EQ(trim("hello"), "hello");
    EXPECT_EQ(trim("  "), "");
    EXPECT_EQ(trim(""), "");
}

// ======================================================================
// Тесты для комбинированных операций
// ======================================================================

TEST(UtilsTest, ReadAndValidateUTF8) {
    std::string test_content = "Hello Привет 世界 😊";
    
    EXPECT_TRUE(write_file(TEST_UTF8_TXT, test_content));
    
    auto read_content = read_file(TEST_UTF8_TXT);
    EXPECT_TRUE(is_valid_utf8(read_content));
    EXPECT_EQ(read_content, test_content);
    
    bpe_test::safe_remove(TEST_UTF8_TXT);
}

TEST(UtilsTest, ReadEscapeWrite) {
    std::string test_content = "Line1\nLine2\tLine3\"Quote\"";
    
    EXPECT_TRUE(write_file(TEST_ESCAPE_TXT, test_content));
    
    auto read_content = read_file(TEST_ESCAPE_TXT);
    auto escaped = escape_string(read_content);
    
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
    EXPECT_NE(escaped.find("\\t"), std::string::npos);
    EXPECT_NE(escaped.find("\\\""), std::string::npos);
    
    bpe_test::safe_remove(TEST_ESCAPE_TXT);
}

TEST(UtilsTest, FormatAndEscape) {
    std::string text = "File size: " + format_size(1234567);
    auto escaped = escape_string(text);
    
    EXPECT_EQ(escaped, "File size: 1.18 МБ");
    EXPECT_TRUE(is_valid_utf8(escaped));
}

TEST(UtilsTest, TrimAndEscape) {
    std::string text = "  \t\n  Hello World  \n\t  ";
    auto trimmed = trim(text);
    auto escaped = escape_string(trimmed);
    
    EXPECT_EQ(escaped, "Hello World");
}

TEST(UtilsTest, CaseAndEscape) {
    std::string text = "Hello\nWorld";
    auto lower = to_lower(text);
    auto escaped = escape_string(lower);
    
    EXPECT_EQ(escaped, "hello\\nworld");
}

TEST(UtilsTest, FileOperationsAndValidation) {
    std::string test_content = "Line1\nLine2\nLine3";
    
    EXPECT_TRUE(write_file(TEST_FILE_TXT, test_content));
    EXPECT_TRUE(fs::exists(TEST_FILE_TXT));
    
    auto read_content = read_file(TEST_FILE_TXT);
    EXPECT_TRUE(is_valid_utf8(read_content));
    EXPECT_EQ(read_content, test_content);
    
    auto size = format_size(read_content.size());
    EXPECT_NE(size.find("байт"), std::string::npos);
    
    bpe_test::safe_remove(TEST_FILE_TXT);
    EXPECT_FALSE(fs::exists(TEST_FILE_TXT));
}

TEST(UtilsTest, TimerAndString) {
    Timer timer;
    
    std::string text = "test string";
    auto lower = to_lower(text);
    auto upper = to_upper(text);
    
    EXPECT_LT(timer.elapsed_ms(), 1.0);
    EXPECT_EQ(lower, "test string");
    EXPECT_EQ(upper, "TEST STRING");
}

TEST(UtilsTest, UTF8AndEscape) {
    std::string text = "Привет\nМир";
    
    EXPECT_TRUE(is_valid_utf8(text));
    
    auto escaped = escape_string(text);
    EXPECT_NE(escaped.find("\\n"), std::string::npos);
    EXPECT_TRUE(is_valid_utf8(escaped));
}

TEST(UtilsTest, FormatAndTrim) {
    std::string text = "  " + format_size(1024) + "  ";
    auto trimmed = trim(text);
    
    EXPECT_EQ(trimmed, "1.00 КБ");
}

TEST(UtilsTest, MultipleOperations) {
    std::string text = "  \n  Hello World!  \n  ";
    
    auto trimmed = trim(text);
    auto lower = to_lower(trimmed);
    auto escaped = escape_string(lower);
    
    EXPECT_EQ(escaped, "hello world!");
    EXPECT_TRUE(is_valid_utf8(escaped));
}

TEST(UtilsTest, TimerMicrosecondsComparison) {
    Timer timer;
    
    auto elapsed_us1 = timer.elapsed_us();
    EXPECT_GE(elapsed_us1, 0.0);
    
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    
    auto elapsed_us2 = timer.elapsed_us();
    EXPECT_GT(elapsed_us2, elapsed_us1);
}

TEST(UtilsTest, TimerMillisecondsComparison) {
    Timer timer;
    
    auto elapsed_ms1 = timer.elapsed_ms();
    EXPECT_GE(elapsed_ms1, 0.0);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    
    auto elapsed_ms2 = timer.elapsed_ms();
    EXPECT_GT(elapsed_ms2, elapsed_ms1);
}