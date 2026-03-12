/**
 * @file simple_example.cpp
 * @brief Простой пример использования BPE токенизатора для начинающих
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Эта программа демонстрирует базовые возможности BPE токенизатора
 *          в максимально простой и понятной форме. Идеально подходит для
 *          первого знакомства с библиотекой.
 * 
 *          Основные демонстрируемые концепции:
 *          - Загрузка обученной модели из файлов
 *          - Кодирование (encode)      - преобразование текста в токены
 *          - Декодирование (decode)    - восстановление текста из токенов
 *          - Проверка Roundtrip        - код -> токены -> код
 *          - Статистика токенизации
 * 
 *          Поддерживаемые токенизаторы:
 *          - BPETokenizer (базовая реализация)
 *          - FastTokenizer (оптимизированная версия с кэшированием)
 * 
 * @usage ./simple_example [options]
 * 
 * @options
 *   --verbose, -v    Подробный вывод (показывает все примеры полностью)
 *   --all, -a        Тестировать все примеры (по умолчанию только один)
 *   --fast, -f       Использовать FastTokenizer вместо базового
 *   --help, -h       Показать эту справку
 * 
 * @example
 *   # Запуск с базовым токенизатором (рекомендуется для начала)
 *   ./simple_example
 *   
 *   # Запуск с FastTokenizer и подробным выводом
 *   ./simple_example --fast --verbose
 *   
 *   # Тестирование всех примеров
 *   ./simple_example --all
 * 
 * @note Для работы требуются файлы модели:
 *       - models/bpe_8000/cpp_vocab.json
 *       - models/bpe_8000/cpp_merges.txt
 *       (можно получить через python tools/convert_vocab.py)
 * 
 * @see BPETokenizer
 * @see FastTokenizer
 * @see utils::Timer
 */

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <memory>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;
using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr int WIDTH = 60;
    constexpr size_t DEFAULT_VOCAB_SIZE = 8000;
    constexpr size_t DEFAULT_CACHE_SIZE = 10000;
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ======================================================================
// Вспомогательные функции для вывода
// ======================================================================

/**
 * @brief Выводит заголовок раздела
 */
void print_header(const std::string& title) {
    // Верхняя граница
    std::cout << "\n" << BOLD << "┌";
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << "┐\n";
    
    // Центрируем заголовок
    int total_padding = WIDTH - static_cast<int>(title.size());
    int left_padding = total_padding / 2;
    int right_padding = total_padding - left_padding;
    
    std::cout << "│";
    for (int i = 0; i < left_padding; ++i) std::cout << ' ';
    std::cout << title;
    for (int i = 0; i < right_padding; ++i) std::cout << ' ';
    std::cout << "│\n";
    
    // Нижняя граница
    std::cout << "└";
    for (int i = 0; i < WIDTH; ++i) std::cout << '-';
    std::cout << "┘" << RESET << std::endl;
}

/**
 * @brief Экранирует специальные символы в строке для безопасного вывода
 * 
 * @param str Исходная строка
 * @return std::string Строка с экранированными символами
 */
std::string escape_string(const std::string& str) {
    std::string result;
    result.reserve(str.size() * 2);
    
    for (char c : str) {
        switch (c) {
            case '\n': result += "\\n"; break;
            case '\t': result += "\\t"; break;
            case '\r': result += "\\r"; break;
            case '\"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            default:
                if (std::isprint(static_cast<unsigned char>(c))) {
                    result += c;
                } else {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\x%02X", static_cast<unsigned char>(c));
                    result += buf;
                }
                break;
        }
    }
    
    return result;
}

// ======================================================================
// Класс для поиска файлов модели
// ======================================================================

/**
 * @brief Класс для поиска файлов модели по различным путям
 */
class ModelFinder {
private:
    std::vector<std::pair<std::string, std::string>> candidates_;
    
public:
    ModelFinder() {
        // ===== C++ модели (приоритет 1) =====
        candidates_.emplace_back("../models/bpe_8000/cpp_vocab.json", 
                                 "../models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("../models/bpe_10000/cpp_vocab.json", 
                                 "../models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("../models/bpe_12000/cpp_vocab.json", 
                                 "../models/bpe_12000/cpp_merges.txt");
        
        // ===== Если запуск из той же директории =====
        candidates_.emplace_back("models/bpe_8000/cpp_vocab.json", 
                                 "models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("models/bpe_10000/cpp_vocab.json", 
                                 "models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("models/bpe_12000/cpp_vocab.json", 
                                 "models/bpe_12000/cpp_merges.txt");
        
        // ===== Python модели (приоритет 2) =====
        candidates_.emplace_back("../../bpe_python/models/bpe_8000/vocab.json", 
                                 "../../bpe_python/models/bpe_8000/merges.txt");
        candidates_.emplace_back("../../bpe_python/models/bpe_10000/vocab.json", 
                                 "../../bpe_python/models/bpe_10000/merges.txt");
        candidates_.emplace_back("../../bpe_python/models/bpe_12000/vocab.json", 
                                 "../../bpe_python/models/bpe_12000/merges.txt");
    }
    
    /**
     * @brief Найти файлы модели для BPETokenizer
     */
    bool find_for_bpe(BPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
        std::cout << "Поиск файлов модели для BPETokenizer..." << std::endl;
        
        for (const auto& [vpath, mpath] : candidates_) {
            std::cout << "  Проверка: " << vpath << std::endl;
            
            if (fs::exists(vpath) && fs::exists(mpath)) {
                vocab_path = vpath;
                merges_path = mpath;
                
                std::cout << GREEN << "  ✓ Найдены: " << vpath << RESET << std::endl;
                
                tokenizer.set_byte_level(true);
                tokenizer.set_unknown_token("<UNK>");
                
                if (tokenizer.load_from_files(vpath, mpath)) {
                    return true;
                }
            }
        }
        
        std::cout << RED << "  ✗ Файлы не найдены!" << RESET << std::endl;
        return false;
    }
    
    /**
     * @brief Найти файлы модели для FastTokenizer
     */
    bool find_for_fast(FastBPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
        std::cout << "Поиск файлов модели для FastTokenizer..." << std::endl;
        
        for (const auto& [vpath, mpath] : candidates_) {
            std::cout << "  Проверка: " << vpath << std::endl;
            
            if (fs::exists(vpath) && fs::exists(mpath)) {
                vocab_path = vpath;
                merges_path = mpath;
                
                std::cout << GREEN << "  ✓ Найдены: " << vpath << RESET << std::endl;
                
                if (tokenizer.load(vpath, mpath)) {
                    return true;
                }
            }
        }
        
        std::cout << RED << "  ✗ Файлы не найдены!" << RESET << std::endl;
        return false;
    }
};

// ======================================================================
// Тестовые данные
// ======================================================================

/**
 * @brief Возвращает набор тестовых примеров C++ кода разной сложности
 * 
 * @return std::vector<std::pair<std::string, std::string>> 
 *         Вектор пар (описание, код) для тестирования
 */
std::vector<std::pair<std::string, std::string>> get_test_examples() {
    return {
        {"Простое выражение", "int x = 42;"},
        {"Русский комментарий", "// это комментарий на русском языке"},
        {"Смешанный текст", "std::cout << \"Привет, мир!\" << std::endl;"},
        {"Шаблон функции", "template<typename T> T max(T a, T b) { return a > b ? a : b; }"},
        {"Класс", "class MyClass { public: MyClass() = default; };"},
        {"Include директива", "#include <iostream>"},
        {"Эмодзи", "// 🔥 C++ code with emoji 😊"},
        
        // Функция факториала - без отступов в начале строк
        {"Функция факториала",
         "// Функция для вычисления факториала\n"
         "int factorial(int n) {\n"
         "    if (n <= 1) return 1;\n"
         "    return n * factorial(n - 1);\n"
         "}"},
        
        // Сложный шаблон - без отступов в начале строк
        {"Сложный шаблон",
         "template<typename T, typename U>\n"
         "struct Pair {\n"
         "    T first;\n"
         "    U second;\n"
         "    \n"
         "    Pair(const T& f, const U& s) : first(f), second(s) {}\n"
         "    \n"
         "    void print() const {\n"
         "        std::cout << \"(\" << first << \", \" << second << \")\" << std::endl;\n"
         "    }\n"
         "};"},
        
        // Лямбда и алгоритмы - без отступов в начале строк
        {"Лямбда и алгоритмы",
         "std::vector<int> vec = {1, 2, 3, 4, 5};\n"
         "std::transform(vec.begin(), vec.end(), vec.begin(),\n"
         "              [](int x) { return x * x; });\n"
         "auto it = std::find_if(vec.begin(), vec.end(),\n"
         "                       [](int x) { return x > 10; });"}
    };
}

/**
 * @brief Вывести статистику по токенам для BPETokenizer
 * 
 * @param tokens Вектор токенов
 * @param vocab Словарь для преобразования ID в строки
 */
void print_bpe_token_stats(const std::vector<token_id_t>& tokens, const Vocabulary& vocab) {
    std::map<size_t, int> length_distribution;
    std::set<token_id_t> unique_tokens(tokens.begin(), tokens.end());
    
    // Собираем статистику по длинам токенов
    for (auto id : unique_tokens) {
        std::string token = vocab.id_to_token(id);
        length_distribution[token.length()]++;
    }
    
    std::cout << "\n" << CYAN << "Детальная статистика токенов:" << RESET << "\n";
    std::cout << "  • всего токенов: " << tokens.size() << "\n";
    std::cout << "  • уникальных токенов: " << unique_tokens.size() << "\n";
    std::cout << "  • повторяемость: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(tokens.size()) / unique_tokens.size()) << " раз/токен\n";
    
    // Показываем распределение длин
    std::cout << "  • распределение по длинам:\n";
    for (const auto& [len, count] : length_distribution) {
        std::cout << "      " << std::setw(2) << len << " симв.: " 
                  << std::setw(3) << count << " токенов\n";
    }
}

/**
 * @brief Вывести статистику по токенам для FastTokenizer
 * 
 * @param tokens Вектор токенов
 */
void print_fast_token_stats(const std::vector<uint32_t>& tokens) {
    std::set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
    
    std::cout << "\n" << CYAN << "Детальная статистика токенов:" << RESET << "\n";
    std::cout << "  • всего токенов: " << tokens.size() << "\n";
    std::cout << "  • уникальных токенов: " << unique_tokens.size() << "\n";
    std::cout << "  • повторяемость: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(tokens.size()) / unique_tokens.size()) << " раз/токен\n";
}

// ======================================================================
// Классы-обертки для унифицированного интерфейса
// ======================================================================

/**
 * @brief Обертка для BPETokenizer с единообразным интерфейсом
 */
class BPETokenizerWrapper {
public:
    BPETokenizer tokenizer;
    
    BPETokenizerWrapper() {
        tokenizer.set_byte_level(true);
        tokenizer.set_unknown_token("<UNK>");
    }
    
    /**
     * @brief Загрузить модель из файлов
     */
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load_from_files(vocab_path, merges_path);
    }
    
    /**
     * @brief Получить размер словаря
     */
    size_t vocab_size() const {
        return tokenizer.vocab_size();
    }
    
    /**
     * @brief Получить имя токенизатора
     */
    std::string name() const {
        return "BPETokenizer (базовый)";
    }
    
    /**
     * @brief Закодировать текст в токены
     */
    std::vector<uint32_t> encode(const std::string& text) {
        auto tokens = tokenizer.encode(text);
        return std::vector<uint32_t>(tokens.begin(), tokens.end());
    }
    
    /**
     * @brief Декодировать токены обратно в текст
     */
    std::string decode(const std::vector<uint32_t>& tokens) {
        std::vector<token_id_t> old_tokens(tokens.begin(), tokens.end());
        return tokenizer.decode(old_tokens);
    }
    
    /**
     * @brief Получить строковое представление токена по ID
     */
    std::string token_to_string(uint32_t id) {
        return tokenizer.vocabulary().id_to_token(id);
    }
    
    /**
     * @brief Получить ссылку на внутренний токенизатор
     */
    BPETokenizer& get_tokenizer() { return tokenizer; }
    
    /**
     * @brief Вывести статистику (для совместимости)
     */
    void print_stats() const {}
};

/**
 * @brief Обертка для FastTokenizer с дополнительной статистикой
 */
class FastTokenizerWrapper {
public:
    FastBPETokenizer tokenizer;
    TokenizerStats stats;
    
    FastTokenizerWrapper() 
        : tokenizer(TokenizerConfig{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, true}) {}
    
    /**
     * @brief Загрузить модель из файлов
     */
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load(vocab_path, merges_path);
    }
    
    /**
     * @brief Получить размер словаря
     */
    size_t vocab_size() const {
        return tokenizer.vocab_size();
    }
    
    /**
     * @brief Получить имя токенизатора
     */
    std::string name() const {
        return "FastTokenizer (оптимизированный)";
    }
    
    /**
     * @brief Закодировать текст в токены
     */
    std::vector<uint32_t> encode(const std::string& text) {
        auto result = tokenizer.encode(text);
        stats = tokenizer.stats();
        return result;
    }
    
    /**
     * @brief Декодировать токены обратно в текст
     */
    std::string decode(const std::vector<uint32_t>& tokens) {
        auto result = tokenizer.decode(tokens);
        stats = tokenizer.stats();
        return result;
    }
    
    /**
     * @brief Получить строковое представление токена по ID
     * 
     * @note FastTokenizer не хранит строки токенов, только ID
     */
    std::string token_to_string(uint32_t id) {
        return "ID:" + std::to_string(id);
    }
    
    /**
     * @brief Вывести статистику кэширования
     */
    void print_stats() const {
        std::cout << "  • cache hits: " << stats.cache_hits << "\n";
        std::cout << "  • cache misses: " << stats.cache_misses << "\n";
        std::cout << "  • hit rate: " << std::fixed << std::setprecision(1)
                  << (stats.cache_hit_rate() * 100) << "%\n";
    }
    
    /**
     * @brief Получить ссылку на внутренний токенизатор
     */
    FastBPETokenizer& get_tokenizer() { return tokenizer; }
};

// ======================================================================
// Функция для тестирования одного токенизатора
// ======================================================================

/**
 * @brief Тестирование токенизатора на наборе примеров
 * 
 * @tparam TokenizerWrapper Тип обертки токенизатора
 * @param wrapper Указатель на обертку токенизатора
 * @param test_examples Вектор тестовых примеров
 * @param verbose Подробный вывод
 */
template<typename Wrapper>
void run_tests(Wrapper* wrapper, 
               const std::vector<std::pair<std::string, std::string>>& test_examples,
               bool verbose) {
    
    int success_count = 0;
    
    for (size_t idx = 0; idx < test_examples.size(); ++idx) {
        const auto& [desc, code] = test_examples[idx];
        
        // Верхняя линия
        std::cout << "\n";
        for (int i = 0; i < WIDTH; ++i) std::cout << '-';
        std::cout << "\n";

        std::cout << BOLD << "Пример " << (idx + 1) << "/" << test_examples.size() 
                << ": " << desc << RESET << "\n";

        // Нижняя линия
        for (int i = 0; i < WIDTH; ++i) std::cout << '-';
        std::cout << "\n";

        // Показываем текст (полностью или сокращенно)
        std::cout << "Исходный текст (" << code.size() << " байт):\n";
        if (verbose) {
            std::cout << code << "\n";
        } else {
            std::string preview = code.substr(0, 200);
            if (code.size() > 200) preview += "...";
            std::cout << preview << "\n";
        }
        
        // Кодирование
        utils::Timer timer;
        auto tokens = wrapper->encode(code);
        double encode_time = timer.elapsed();
        
        // Декодирование
        timer.reset();
        auto decoded = wrapper->decode(tokens);
        double decode_time = timer.elapsed();
        
        // Проверка Roundtrip
        bool success = (decoded == code);
        if (success) success_count++;
        
        // Вывод результатов
        std::cout << "\nРезультаты:\n";
        std::cout << "  • токенов: " << tokens.size() << "\n";
        std::cout << "  • время encode: " << std::fixed << std::setprecision(3) 
                  << encode_time * 1000 << " мс\n";
        std::cout << "  • время decode: " << decode_time * 1000 << " мс\n";
        std::cout << "  • скорость encode: " 
                  << std::fixed << std::setprecision(2)
                  << (code.size() / 1024.0 / encode_time) << " КБ/с\n";
        std::cout << "  • Roundtrip: " << (success ? GREEN + "✓ УСПЕХ" : RED + "✗ НЕУДАЧА") 
                  << RESET << "\n";
        
        // Детальная статистика если нужно
        if (verbose) {
            if constexpr (std::is_same_v<Wrapper, BPETokenizerWrapper>) {
                print_bpe_token_stats(std::vector<token_id_t>(tokens.begin(), tokens.end()), 
                                     wrapper->get_tokenizer().vocabulary());
            } else {
                print_fast_token_stats(tokens);
            }
        }
        
        // Показываем первые 10 токенов
        std::cout << "\nПервые 10 токенов:\n";
        for (size_t i = 0; i < std::min(size_t(10), tokens.size()); ++i) {
            std::string token_str = wrapper->token_to_string(tokens[i]);
            std::string escaped = escape_string(token_str);
            if (escaped.size() > 30) escaped = escaped.substr(0, 27) + "...";
            std::cout << "  " << std::setw(6) << tokens[i] << ": '" 
                      << escaped << "'\n";
        }
    }
    
    // Итоговая статистика
    std::cout << "\n" << BOLD;
    for (int i = 0; i < WIDTH; ++i) std::cout << '=';
    std::cout << RESET << "\n";

    std::cout << BOLD << "ИТОГОВАЯ СТАТИСТИКА:" << RESET << "\n";

    for (int i = 0; i < WIDTH; ++i) std::cout << '=';
    std::cout << "\n";
    
    std::cout << "  • токенизатор: " << wrapper->name() << "\n";
    std::cout << "  • размер словаря: " << wrapper->vocab_size() << " токенов\n";
    std::cout << "  • протестировано примеров: " << test_examples.size() << "\n";
    std::cout << "  • успешных Roundtrip: " << success_count << "/" 
              << test_examples.size() << " ("
              << std::fixed << std::setprecision(1)
              << (100.0 * success_count / test_examples.size()) << "%)\n";
    
    wrapper->print_stats();
}

// ======================================================================
// Основная функция
// ======================================================================

/**
 * @brief Точка входа в программу
 * 
 * @param argc Количество аргументов
 * @param argv Массив аргументов
 * @return int 0 при успехе, 1 при ошибке
 */
int main(int argc, char* argv[]) {
    // ======================================================================
    // Приветствие
    // ======================================================================
    
    print_header("BPE TOKENIZER - ПРОСТОЙ ПРИМЕР");
    std::cout << "Демонстрация базовых операций кодирования/декодирования\n\n";
    
    // ======================================================================
    // Парсинг аргументов командной строки
    // ======================================================================
    
    bool verbose = false;
    bool test_all = false;
    bool use_fast = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--all" || arg == "-a") {
            test_all = true;
        } else if (arg == "--fast" || arg == "-f") {
            use_fast = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --verbose, -v    Подробный вывод\n";
            std::cout << "  --all, -a        Тестировать все примеры\n";
            std::cout << "  --fast, -f       Использовать FastTokenizer\n";
            std::cout << "  --help, -h       Показать справку\n";
            return 0;
        }
    }
    
    try {
        // ======================================================================
        // 1. ВЫБОР ТЕКСТА ДЛЯ ТЕСТИРОВАНИЯ
        // ======================================================================
        
        std::vector<std::pair<std::string, std::string>> test_examples;
        
        if (test_all) {
            test_examples = get_test_examples();
            std::cout << "Режим: тестирование всех " << test_examples.size() 
                      << " примеров\n";
        } else {
            test_examples = {
                {"Функция факториала (C++)", R"(
// Функция для вычисления факториала
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
                )"}
            };
            std::cout << "Режим: тестирование одного примера "
                      << "(используйте --all для всех)\n";
        }
        
        // ======================================================================
        // 2. СОЗДАНИЕ И ЗАГРУЗКА ТОКЕНИЗАТОРА
        // ======================================================================
        
        ModelFinder finder;
        std::string vocab_path, merges_path;
        bool loaded = false;
        
        if (use_fast) {
            // FastTokenizer
            std::cout << "\n" << CYAN << "Создание FastTokenizer..." << RESET << std::endl;
            auto fast = std::make_unique<FastTokenizerWrapper>();
            
            print_header("ЗАГРУЗКА МОДЕЛИ");
            loaded = finder.find_for_fast(fast->get_tokenizer(), vocab_path, merges_path);
            
            if (loaded) {
                std::cout << "\n" << GREEN << "✓ Модель загружена успешно!" << RESET << "\n";
                std::cout << "  • словарь: " << vocab_path << "\n";
                std::cout << "  • слияния: " << merges_path << "\n";
                std::cout << "  • размер словаря: " << fast->vocab_size() << " токенов\n";
                
                // Запуск тестов
                run_tests(fast.get(), test_examples, verbose);
            }
            
        } else {
            // BPETokenizer
            std::cout << "\n" << CYAN << "Создание BPETokenizer..." << RESET << std::endl;
            auto bpe = std::make_unique<BPETokenizerWrapper>();
            
            print_header("ЗАГРУЗКА МОДЕЛИ");
            loaded = finder.find_for_bpe(bpe->get_tokenizer(), vocab_path, merges_path);
            
            if (loaded) {
                std::cout << "\n" << GREEN << "✓ Модель загружена успешно!" << RESET << "\n";
                std::cout << "  • словарь: " << vocab_path << "\n";
                std::cout << "  • слияния: " << merges_path << "\n";
                std::cout << "  • размер словаря: " << bpe->vocab_size() << " токенов\n";
                std::cout << "  • правил слияния: " << bpe->get_tokenizer().merges_count() << "\n";
                
                // Запуск тестов
                run_tests(bpe.get(), test_examples, verbose);
            }
        }
        
        // ======================================================================
        // 3. ОБРАБОТКА ОШИБКИ ЗАГРУЗКИ
        // ======================================================================
        
        if (!loaded) {
            std::cerr << RED << "\n❌ НЕ УДАЛОСЬ ЗАГРУЗИТЬ МОДЕЛЬ!" << RESET << "\n";
            std::cerr << "\nУбедитесь, что файлы существуют в одном из путей:\n";
            std::cerr << "  • ../models/bpe_8000/cpp_vocab.json\n";
            std::cerr << "  • models/bpe_8000/cpp_vocab.json\n";
            std::cerr << "  • ../../bpe_python/models/bpe_8000/vocab.json\n";
            std::cerr << "\nЧтобы конвертировать Python модели в C++ формат:\n";
            std::cerr << YELLOW << "  cd ../tools/ && python convert_vocab.py --model-size 8000" 
                      << RESET << std::endl;
            return 1;
        }
        
        // ======================================================================
        // 4. УСПЕШНОЕ ЗАВЕРШЕНИЕ
        // ======================================================================
        
        std::cout << "\n" << GREEN << BOLD << "✓ Пример успешно завершен!" << RESET << "\n";
        std::cout << "Используйте --help для просмотра дополнительных опций.\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "\n❌ Ошибка выполнения: " << e.what() << RESET << std::endl;
        return 1;
    } catch (...) {
        std::cerr << RED << "\n❌ Неизвестная ошибка" << RESET << std::endl;
        return 1;
    }
    
    return 0;
}