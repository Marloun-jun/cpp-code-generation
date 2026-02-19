/**
 * @file test_compatibility.cpp
 * @brief Тесты совместимости между C++ и Python реализациями BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Набор тестов для проверки, что C++ токенизатор ведет себя так же,
 *          как и эталонная Python реализация. Включает:
 *          - Загрузку той же модели
 *          - Кодирование различных типов текстов
 *          - Декодирование и roundtrip проверки
 *          - Пакетную обработку
 *          - Сравнение с сохраненными выходами Python
 * 
 * @note Для полных тестов требуются файлы модели от Python реализации
 * @see FastBPETokenizer
 * @see BPETokenizer (Python)
 */

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include "fast_tokenizer.hpp"
#include "bpe_tokenizer.hpp"
#include "utils.hpp"

#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загрузить ожидаемые результаты из Python
 */
std::vector<std::vector<uint32_t>> load_python_outputs(const std::string& filename) {
    std::vector<std::vector<uint32_t>> outputs;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        return outputs;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::vector<uint32_t> tokens;
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(std::stoul(token));
        }
        outputs.push_back(tokens);
    }
    
    return outputs;
}

/**
 * @brief Сохранить результаты C++ для сравнения с Python
 */
void save_cpp_outputs(const std::string& filename, 
                      const std::vector<std::vector<uint32_t>>& outputs) {
    std::ofstream file(filename);
    for (const auto& tokens : outputs) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) file << ",";
            file << tokens[i];
        }
        file << "\n";
    }
}

/**
 * @brief Проверить UTF-8 корректность строки
 */
bool is_valid_utf8(const std::string& str) {
    try {
        // Простая проверка: пробуем декодировать как UTF-8
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        converter.from_bytes(str);
        return true;
    } catch (...) {
        return false;
    }
}

// ======================================================================
// Тестовый класс
// ======================================================================

class CompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Настройка для тестов совместимости
        config_.vocab_size = 32000;
        config_.cache_size = 10000;
        config_.byte_level = true;
        config_.enable_cache = true;
        config_.enable_profiling = true;  // Включаем для сбора статистики
        
        // Определяем возможные пути к файлам
        python_paths_ = {
            "../../bpe/vocab.json",
            "../bpe/vocab.json",
            "bpe/vocab.json",
            "vocab.json"
        };
        
        merges_paths_ = {
            "../../bpe/merges.txt",
            "../bpe/merges.txt",
            "bpe/merges.txt",
            "merges.txt"
        };
        
        // Пути для сохранения результатов
        results_dir_ = "compatibility_results";
        
        // Создаем директорию для результатов
        std::filesystem::create_directories(results_dir_);
    }
    
    void TearDown() override {
        // Выводим статистику после тестов
        if (tokenizer_loaded_) {
            auto stats = tokenizer_.stats();
            std::cout << "\n📊 Статистика токенизатора:" << std::endl;
            std::cout << "   Кэш: " << stats.cache_hit_rate() * 100 << "% попаданий" << std::endl;
            std::cout << "   Среднее время encode: " 
                      << std::fixed << std::setprecision(3)
                      << stats.avg_encode_time_ms() << " мс" << std::endl;
        }
    }
    
    /**
     * @brief Загрузить токенизатор (с поиском по разным путям)
     */
    bool loadTokenizer() {
        if (tokenizer_loaded_) return true;
        
        for (size_t i = 0; i < python_paths_.size(); ++i) {
            if (tokenizer_.load(python_paths_[i], merges_paths_[i])) {
                std::cout << "✅ Загружен словарь: " << python_paths_[i] << std::endl;
                tokenizer_loaded_ = true;
                loaded_path_ = python_paths_[i];
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @brief Проверить наличие Python файлов
     */
    bool hasPythonFiles() const {
        for (const auto& path : python_paths_) {
            if (std::ifstream(path).good()) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Получить путь к файлу с Python выходами
     */
    std::string getPythonOutputPath(const std::string& test_name) const {
        return results_dir_ + "/python_" + test_name + ".txt";
    }
    
    /**
     * @brief Получить путь к файлу с C++ выходами
     */
    std::string getCppOutputPath(const std::string& test_name) const {
        return results_dir_ + "/cpp_" + test_name + ".txt";
    }
    
    TokenizerConfig config_;
    FastBPETokenizer tokenizer_{config_};
    bool tokenizer_loaded_ = false;
    std::string loaded_path_;
    
    std::vector<std::string> python_paths_;
    std::vector<std::string> merges_paths_;
    std::string results_dir_;
    
    // Тестовые строки для различных сценариев
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
        "for (int i = 0; i < 10; ++i) {}",
        "while (running) { process(); }",
        "switch (value) { case 1: break; }",
        "try { throw std::runtime_error(\"error\"); } catch (...) {}",
        
        // Шаблоны и классы
        "template<typename T>",
        "class MyClass { public: void method(); };",
        "struct Point { int x, y; };",
        "enum Color { RED, GREEN, BLUE };",
        "namespace my_namespace { class MyClass {}; }",
        
        // Функции
        "void func(int a, float b) { return a + b; }",
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
        
        // Русские комментарии
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        "// комментарий с числами 123 и символами !@#$%",
        "// 🔥 emoji комментарий"
    };
};

// ======================================================================
// Тесты
// ======================================================================

/**
 * @test Загрузка того же словаря, что и Python версия
 */
TEST_F(CompatibilityTest, LoadSameVocabulary) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "❌ Python vocabulary files not found. Run python tokenizer first.";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    EXPECT_GT(tokenizer_.vocab_size(), 0);
    EXPECT_GT(tokenizer_.merges_count(), 0);
    
    std::cout << "📚 C++ токенизатор загрузил " << tokenizer_.vocab_size() 
              << " токенов из " << loaded_path_ << std::endl;
    std::cout << "🔗 Правил слияния: " << tokenizer_.merges_count() << std::endl;
}

/**
 * @test Кодирование простого текста
 */
TEST_F(CompatibilityTest, EncodeSimpleText) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::string text = "int main()";
    auto tokens = tokenizer_.encode(text);
    
    EXPECT_GT(tokens.size(), 0);
    std::cout << "📝 Текст '" << text << "' закодирован в " 
              << tokens.size() << " токенов" << std::endl;
    
    // Проверяем первые несколько токенов
    for (size_t i = 0; i < std::min<size_t>(5, tokens.size()); ++i) {
        std::cout << "   Токен " << i << ": ID " << tokens[i] << std::endl;
    }
}

/**
 * @test Кодирование всех тестовых строк
 */
TEST_F(CompatibilityTest, EncodeAllTestStrings) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::vector<std::vector<uint32_t>> all_outputs;
    
    for (const auto& text : test_strings_) {
        auto tokens = tokenizer_.encode(text);
        all_outputs.push_back(tokens);
        
        std::cout << "📌 '" << text.substr(0, 30) 
                  << (text.length() > 30 ? "..." : "") 
                  << "' -> " << tokens.size() << " токенов" << std::endl;
        
        EXPECT_GT(tokens.size(), 0);
    }
    
    // Сохраняем для сравнения с Python
    save_cpp_outputs(getCppOutputPath("encode_all"), all_outputs);
    std::cout << "💾 Результаты сохранены в " << getCppOutputPath("encode_all") << std::endl;
}

/**
 * @test Кодирование с русскими комментариями
 */
TEST_F(CompatibilityTest, EncodeRussianComments) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::vector<std::string> russian_texts = {
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        "// комментарий с числами 123 и символами !@#$%",
        "// 🔥 emoji комментарий"
    };
    
    for (const auto& text : russian_texts) {
        auto tokens = tokenizer_.encode(text);
        EXPECT_GT(tokens.size(), 0);
        
        // Проверяем, что декодирование работает
        auto decoded = tokenizer_.decode(tokens);
        
        std::cout << "\n📌 Исходный:  '" << text << "'" << std::endl;
        std::cout << "   Декод.:    '" << decoded << "'" << std::endl;
        std::cout << "   Токенов:   " << tokens.size() << std::endl;
        
        // Проверяем, что все символы присутствуют
        bool all_chars_present = true;
        for (char c : text) {
            if (decoded.find(c) == std::string::npos && c > 0) {
                all_chars_present = false;
                std::cout << "   ⚠️ Отсутствует символ: '" << c << "' (код " << int(c) << ")" << std::endl;
            }
        }
        
        EXPECT_TRUE(all_chars_present) << "Не все символы сохранились для: " << text;
    }
}

/**
 * @test Roundtrip тест (кодирование + декодирование)
 */
TEST_F(CompatibilityTest, EncodeDecodeRoundtrip) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    int total_tests = 0;
    int passed_tests = 0;
    std::vector<std::string> failures;
    
    for (const auto& text : test_strings_) {
        total_tests++;
        
        auto tokens = tokenizer_.encode(text);
        auto decoded = tokenizer_.decode(tokens);
        
        // В byte-level режиме проверяем посимвольно
        bool all_chars_present = true;
        std::string missing_chars;
        
        for (char c : text) {
            if (decoded.find(c) == std::string::npos) {
                all_chars_present = false;
                missing_chars += c;
            }
        }
        
        if (all_chars_present) {
            passed_tests++;
        } else {
            failures.push_back(text);
            std::cout << "\n❌ Провал для: '" << text << "'" << std::endl;
            std::cout << "   Отсутствуют символы: '" << missing_chars << "'" << std::endl;
            std::cout << "   Декодировано: '" << decoded << "'" << std::endl;
        }
    }
    
    double success_rate = 100.0 * passed_tests / total_tests;
    std::cout << "\n📊 Roundtrip成功率: " << std::fixed << std::setprecision(1)
              << success_rate << "% (" << passed_tests << "/" << total_tests << ")" << std::endl;
    
    EXPECT_GE(success_rate, 95.0) << "Слишком много неудачных roundtrip тестов";
}

/**
 * @test Пакетная обработка
 */
TEST_F(CompatibilityTest, BatchEncode) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    // Берем первые 20 тестовых строк
    std::vector<std::string> texts(test_strings_.begin(), 
                                    test_strings_.begin() + std::min(20, (int)test_strings_.size()));
    
    std::vector<std::string_view> views;
    for (const auto& t : texts) views.push_back(t);
    
    auto batch_result = tokenizer_.encode_batch(views);
    
    EXPECT_EQ(batch_result.size(), texts.size());
    std::cout << "📦 Пакетная обработка " << batch_result.size() << " текстов:" << std::endl;
    
    size_t total_tokens = 0;
    for (size_t i = 0; i < batch_result.size(); ++i) {
        std::cout << "   Текст " << i << ": " << batch_result[i].size() << " токенов" << std::endl;
        total_tokens += batch_result[i].size();
    }
    std::cout << "   Всего токенов: " << total_tokens << std::endl;
    
    // Проверяем последовательно
    for (size_t i = 0; i < texts.size(); ++i) {
        auto single = tokenizer_.encode(texts[i]);
        EXPECT_EQ(single.size(), batch_result[i].size()) 
            << "Размеры не совпадают для текста " << i;
    }
}

/**
 * @test Производительность encode
 */
TEST_F(CompatibilityTest, EncodePerformance) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::string large_text;
    for (int i = 0; i < 100; ++i) {
        large_text += test_strings_[i % test_strings_.size()] + "\n";
    }
    
    const int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto tokens = tokenizer_.encode(large_text);
        benchmark::DoNotOptimize(tokens.data());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double ms_per_encode = duration.count() / static_cast<double>(iterations);
    double tokens_per_second = (large_text.size() * iterations) / (duration.count() / 1000.0);
    
    std::cout << "\n⚡ Производительность encode:" << std::endl;
    std::cout << "   Размер текста: " << large_text.size() << " байт" << std::endl;
    std::cout << "   Среднее время: " << std::fixed << std::setprecision(2) 
              << ms_per_encode << " мс" << std::endl;
    std::cout << "   Скорость: " << std::fixed << std::setprecision(0)
              << tokens_per_second << " байт/сек" << std::endl;
    
    auto stats = tokenizer_.stats();
    std::cout << "   Статистика кэша: " << stats.cache_hit_rate() * 100 << "% попаданий" << std::endl;
}

/**
 * @test Сравнение с Python (если есть сохраненные выходы)
 */
TEST_F(CompatibilityTest, CompareWithPython) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    // Пытаемся загрузить выходы Python
    auto python_outputs = load_python_outputs(getPythonOutputPath("encode_all"));
    
    if (python_outputs.empty()) {
        std::cout << "\n⚠️ Нет сохраненных выходов Python для сравнения." << std::endl;
        std::cout << "   Чтобы создать их, запустите:" << std::endl;
        std::cout << "   python3 -c \"" << std::endl;
        std::cout << "   import json" << std::endl;
        std::cout << "   from bpe import BPETokenizer" << std::endl;
        std::cout << "   tokenizer = BPETokenizer()" << std::endl;
        std::cout << "   tokenizer.load('vocab.json', 'merges.txt')" << std::endl;
        std::cout << "   texts = [...]  # те же тексты" << std::endl;
        std::cout << "   with open('compatibility_results/python_encode_all.txt', 'w') as f:" << std::endl;
        std::cout << "       for text in texts:" << std::endl;
        std::cout << "           tokens = tokenizer.encode(text)" << std::endl;
        std::cout << "           f.write(','.join(map(str, tokens)) + '\\n')" << std::endl;
        std::cout << "   \"" << std::endl;
        GTEST_SKIP() << "Python outputs not found";
    }
    
    // Получаем C++ выходы
    std::vector<std::vector<uint32_t>> cpp_outputs;
    for (const auto& text : test_strings_) {
        cpp_outputs.push_back(tokenizer_.encode(text));
    }
    
    // Сравниваем
    EXPECT_EQ(python_outputs.size(), cpp_outputs.size());
    
    int matches = 0;
    for (size_t i = 0; i < std::min(python_outputs.size(), cpp_outputs.size()); ++i) {
        if (python_outputs[i] == cpp_outputs[i]) {
            matches++;
        } else {
            std::cout << "\n❌ Несовпадение для текста " << i << ":" << std::endl;
            std::cout << "   Python: " << python_outputs[i].size() << " токенов" << std::endl;
            std::cout << "   C++:    " << cpp_outputs[i].size() << " токенов" << std::endl;
        }
    }
    
    double match_rate = 100.0 * matches / cpp_outputs.size();
    std::cout << "\n📊 Совпадение с Python: " << std::fixed << std::setprecision(1)
              << match_rate << "% (" << matches << "/" << cpp_outputs.size() << ")" << std::endl;
    
    EXPECT_GE(match_rate, 90.0) << "Слишком большое расхождение с Python";
}

/**
 * @test Проверка специальных токенов
 */
TEST_F(CompatibilityTest, SpecialTokens) {
    if (!hasPythonFiles()) {
        GTEST_SKIP() << "Python vocabulary files not found";
    }
    
    ASSERT_TRUE(loadTokenizer());
    
    std::cout << "\n🔖 Специальные токены:" << std::endl;
    std::cout << "   <UNK> ID: " << tokenizer_.unknown_id() << std::endl;
    std::cout << "   <PAD> ID: " << tokenizer_.pad_id() << std::endl;
    std::cout << "   <BOS> ID: " << tokenizer_.bos_id() << std::endl;
    std::cout << "   <EOS> ID: " << tokenizer_.eos_id() << std::endl;
    
    // Проверяем, что ID различаются
    std::set<uint32_t> ids = {
        tokenizer_.unknown_id(),
        tokenizer_.pad_id(),
        tokenizer_.bos_id(),
        tokenizer_.eos_id()
    };
    
    EXPECT_EQ(ids.size(), 4) << "Специальные токены должны иметь разные ID";
}
