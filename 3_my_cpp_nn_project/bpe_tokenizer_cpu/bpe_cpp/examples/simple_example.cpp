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
 *          **Демонстрируемые концепции:**
 *          ┌────────────────────┬────────────────────────────────────────┐
 *          │ Загрузка модели    │ Из файлов vocab.json + merges.txt      │
 *          │ Кодирование        │ Текст -> последовательность токенов    │
 *          │ Декодирование      │ Токены -> восстановленный текст        │
 *          │ Roundtrip проверка │ Код -> токены -> код (должен совпасть) │
 *          │ Статистика         │ Размер словаря, количество токенов     │
 *          └────────────────────┴────────────────────────────────────────┘
 * 
 *          **Режимы работы:**
 *          - Базовый - Тестирует один пример (функция факториала)
 *          --all     - Тестирует все 8 примеров разной сложности
 *          --fast    - Использует оптимизированную версию FastTokenizer
 *          --verbose - Показывает детальную статистику по токенам
 * 
 * @usage ./simple_example [options]
 * 
 * @options
 *   --verbose, -v - Подробный вывод (показывает все примеры полностью)
 *   --all, -a     - Тестировать все примеры (по умолчанию только один)
 *   --fast, -f    - Использовать FastTokenizer вместо базового
 *   --help, -h    - Показать справку
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
 *       - models/bpe_10000/cpp_vocab.json
 *       - models/bpe_10000/cpp_merges.txt
 *       (можно получить через python tools/convert_vocab.py)
 * 
 * @see BPETokenizer, FastTokenizer, utils::Timer
 */

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    constexpr int WIDTH = 60;                       ///< Ширина таблиц для вывода
    constexpr size_t DEFAULT_VOCAB_SIZE = 10000;    ///< Размер словаря по умолчанию
    constexpr size_t DEFAULT_CACHE_SIZE = 10000;    ///< Размер кэша для FastTokenizer
    
    // ANSI цвета для красивого вывода
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ============================================================================
// Вспомогательные функции для вывода
// ============================================================================

/**
 * @brief Выводит заголовок раздела с красивым оформлением
 * 
 * @param title Заголовок для вывода
 * 
 * @code
 * print_header("ЗАГРУЗКА МОДЕЛИ");
 * // Вывод:
 * // ┌────────────────────────────────────────────────────────────┐
 * // │                      ЗАГРУЗКА МОДЕЛИ                       │
 * // └────────────────────────────────────────────────────────────┘
 * @endcode
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
 * 
 * @code
 * std::string s = "Hello\nWorld\t!";
 * std::cout << escape_string(s);    // Вывод: Hello\nWorld\t!
 * @endcode
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

// ============================================================================
// Класс для поиска файлов модели
// ============================================================================

/**
 * @brief Класс для поиска файлов модели по различным путям
 * 
 * Пытается найти файлы vocab.json и merges.txt в различных расположениях:
 * - ../models/bpe_10000/               - Из папки bpe_cpp
 * - models/bpe_10000/                  - Текущая директория
 * - ../../bpe_python/models/bpe_10000/ - Из корня проекта
 */
class ModelFinder {
private:
    std::vector<std::pair<std::string, std::string>> candidates_;
    
public:
    ModelFinder() {
        // C++ модели из папки проекта
        candidates_.emplace_back("../models/bpe_8000/cpp_vocab.json", 
                                "../models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("../models/bpe_10000/cpp_vocab.json", 
                                "../models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("../models/bpe_12000/cpp_vocab.json", 
                                "../models/bpe_12000/cpp_merges.txt");
        
        // C++ модели в текущей директории
        candidates_.emplace_back("models/bpe_8000/cpp_vocab.json", 
                                "models/bpe_8000/cpp_merges.txt");
        candidates_.emplace_back("models/bpe_10000/cpp_vocab.json", 
                                "models/bpe_10000/cpp_merges.txt");
        candidates_.emplace_back("models/bpe_12000/cpp_vocab.json", 
                                "models/bpe_12000/cpp_merges.txt");
        
        // Python модели из корня проекта
        candidates_.emplace_back("../../bpe_python/models/bpe_8000/vocab.json", 
                                "../../bpe_python/models/bpe_8000/merges.txt");
        candidates_.emplace_back("../../bpe_python/models/bpe_10000/vocab.json", 
                                "../../bpe_python/models/bpe_10000/merges.txt");
        candidates_.emplace_back("../../bpe_python/models/bpe_12000/vocab.json", 
                                "../../bpe_python/models/bpe_12000/merges.txt");
    }
    
    /**
     * @brief Найти файлы модели для BPETokenizer
     * 
     * @param tokenizer Ссылка на токенизатор для загрузки
     * @param vocab_path [out] Путь к найденному vocab.json
     * @param merges_path [out] Путь к найденному merges.txt
     * @return true если модель найдена и загружена
     */
    bool find_for_bpe(BPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
        std::cout << "Поиск файлов модели для BPETokenizer..." << std::endl;
        
        for (const auto& [vpath, mpath] : candidates_) {
            std::cout << "Проверка: " << vpath << std::endl;
            
            if (fs::exists(vpath) && fs::exists(mpath)) {
                vocab_path = vpath;
                merges_path = mpath;
                
                std::cout << GREEN << "Найдены: " << vpath << RESET << std::endl;
                
                tokenizer.set_byte_level(true);
                tokenizer.set_unknown_token("<UNK>");
                
                if (tokenizer.load_from_files(vpath, mpath)) {
                    return true;
                }
            }
        }
        
        std::cout << RED << "Файлы не найдены!" << RESET << std::endl;
        return false;
    }
    
    /**
     * @brief Найти файлы модели для FastTokenizer
     * 
     * @param tokenizer Ссылка на токенизатор для загрузки
     * @param vocab_path [out] Путь к найденному vocab.json
     * @param merges_path [out] Путь к найденному merges.txt
     * @return true если модель найдена и загружена
     */
    bool find_for_fast(FastBPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
        std::cout << "Поиск файлов модели для FastTokenizer..." << std::endl;
        
        for (const auto& [vpath, mpath] : candidates_) {
            std::cout << "Проверка: " << vpath << std::endl;
            
            if (fs::exists(vpath) && fs::exists(mpath)) {
                vocab_path = vpath;
                merges_path = mpath;
                
                std::cout << GREEN << "Найдены: " << vpath << RESET << std::endl;
                
                if (tokenizer.load(vpath, mpath)) {
                    return true;
                }
            }
        }
        
        std::cout << RED << "Файлы не найдены!" << RESET << std::endl;
        return false;
    }
};

// ============================================================================
// Тестовые данные
// ============================================================================

/**
 * @brief Возвращает набор тестовых примеров C++ кода разной сложности
 * 
 * @return std::vector<std::pair<std::string, std::string>> 
 *         Вектор пар (описание, код) для тестирования
 * 
 * Содержит 8 примеров разной сложности:
 * 1. Простое выражение
 * 2. Шаблон функции
 * 3. Класс
 * 4. Include директива
 * 5. Функция факториала
 * 6. Сложный шаблон Pair
 * 7. Лямбда и алгоритмы
 * 8. Сложный класс с методами
 */
std::vector<std::pair<std::string, std::string>> get_test_examples() {
    return {
        {"1. Простое выражение", "int x = 42;"},
        
        {"2. Шаблон функции", 
         "template<typename T> T max(T a, T b) { return a > b ? a : b; }"},
        
        {"3. Класс", 
         "class MyClass { public: MyClass() = default; };"},
        
        {"4. Include директива", 
         "#include <iostream>"},
        
        {"5. Функция факториала",
         "// Function to calculate factorial\n"
         "int factorial(int n) {\n"
         "    if (n <= 1) return 1;\n"
         "    return n * factorial(n - 1);\n"
         "}"},
        
        {"6. Сложный шаблон Pair",
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
        
        {"7. Лямбда и алгоритмы",
         "std::vector<int> vec = {1, 2, 3, 4, 5};\n"
         "std::transform(vec.begin(), vec.end(), vec.begin(),\n"
         "              [](int x) { return x * x; });\n"
         "auto it = std::find_if(vec.begin(), vec.end(),\n"
         "                       [](int x) { return x > 10; });"},
        
        {"8. Сложный класс с методами",
         "class Calculator {\n"
         "private:\n"
         "    int value;\n"
         "public:\n"
         "    Calculator() : value(0) {}\n"
         "    explicit Calculator(int v) : value(v) {}\n"
         "    \n"
         "    int get() const { return value; }\n"
         "    void set(int v) { value = v; }\n"
         "    \n"
         "    Calculator operator+(const Calculator& other) const {\n"
         "        return Calculator(value + other.value);\n"
         "    }\n"
         "    \n"
         "    friend std::ostream& operator<<(std::ostream& os, const Calculator& c) {\n"
         "        return os << \"Calc(\" << c.value << \")\";\n"
         "    }\n"
         "};"}
    };
}

// ============================================================================
// Функции для вывода статистики по токенам
// ============================================================================

/**
 * @brief Вывести детальную статистику по токенам для BPETokenizer
 * 
 * @param tokens Вектор токенов
 * @param vocab Словарь для преобразования ID в строки
 * 
 * Выводит:
 * - Общее количество токенов
 * - Количество уникальных токенов
 * - Среднюю повторяемость
 * - Распределение токенов по длинам
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
    std::cout << "- Всего токенов:      " << tokens.size() << "\n";
    std::cout << "- Уникальных токенов: " << unique_tokens.size() << "\n";
    std::cout << "- Повторяемость:      " << std::fixed << std::setprecision(2)
              << (static_cast<double>(tokens.size()) / unique_tokens.size()) << " раз/токен\n";
    
    // Показываем распределение длин
    std::cout << "- Распределение по длинам:\n";
    for (const auto& [len, count] : length_distribution) {
        std::cout << "    " << std::setw(2) << len << " симв.: " 
                  << std::setw(3) << count << " токенов\n";
    }
}

/**
 * @brief Вывести детальную статистику по токенам для FastTokenizer
 * 
 * @param tokens Вектор токенов
 * 
 * FastTokenizer не хранит строки токенов, поэтому показывает только
 * базовую статистику.
 */
void print_fast_token_stats(const std::vector<uint32_t>& tokens) {
    std::set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
    
    std::cout << "\n" << CYAN << "Детальная статистика токенов:" << RESET << "\n";
    std::cout << "- Всего токенов:      " << tokens.size() << "\n";
    std::cout << "- Уникальных токенов: " << unique_tokens.size() << "\n";
    std::cout << "- Повторяемость:      " << std::fixed << std::setprecision(2)
              << (static_cast<double>(tokens.size()) / unique_tokens.size()) << " раз/токен\n";
}

// ============================================================================
// Классы-обертки для унифицированного интерфейса
// ============================================================================

/**
 * @brief Обертка для BPETokenizer с единообразным интерфейсом
 * 
 * Позволяет использовать BPETokenizer и FastTokenizer в одном шаблоне run_tests
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
     * @brief Вывести статистику (заглушка для совместимости)
     */
    void print_stats() const {}
};

/**
 * @brief Обертка для FastTokenizer с дополнительной статистикой
 * 
 * Добавляет сбор статистики кэширования к базовому функционалу
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
        std::cout << "- cache hits:   " << stats.cache_hits << "\n";
        std::cout << "- cache misses: " << stats.cache_misses << "\n";
        std::cout << "- hit rate:     " << std::fixed << std::setprecision(1)
                  << (stats.cache_hit_rate() * 100) << "%\n";
    }
    
    /**
     * @brief Получить ссылку на внутренний токенизатор
     */
    FastBPETokenizer& get_tokenizer() { return tokenizer; }
};

// ============================================================================
// Функция для тестирования одного токенизатора
// ============================================================================

/**
 * @brief Тестирование токенизатора на наборе примеров
 * 
 * @tparam Wrapper Тип обертки токенизатора
 * @param wrapper Указатель на обертку токенизатора
 * @param test_examples Вектор тестовых примеров
 * @param verbose Подробный вывод (показывать полный код)
 * 
 * Для каждого примера:
 * 1. Кодирует текст в токены (с измерением времени)
 * 2. Декодирует обратно (с измерением времени)
 * 3. Проверяет roundtrip (декодированный == исходный)
 * 4. Выводит статистику
 */
template<typename Wrapper>
void run_tests(Wrapper* wrapper, 
               const std::vector<std::pair<std::string, std::string>>& test_examples,
               bool verbose) {
    
    int success_count = 0;
    
    for (size_t idx = 0; idx < test_examples.size(); ++idx) {
        const auto& [desc, code] = test_examples[idx];
        
        // Разделитель
        std::cout << "\n";
        for (int i = 0; i < WIDTH; ++i) std::cout << '-';
        std::cout << "\n";

        std::cout << BOLD << "Пример " << (idx + 1) << "/" << test_examples.size() 
                << ": " << desc << RESET << "\n";

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
        std::cout << "- Токенов:         " << tokens.size() << "\n";
        std::cout << "- Время encode:    " << std::fixed << std::setprecision(3) 
                  << encode_time * 1000 << " мс\n";
        std::cout << "- Время decode:    " << decode_time * 1000 << " мс\n";
        std::cout << "- Скорость encode: " 
                  << std::fixed << std::setprecision(2)
                  << (code.size() / 1024.0 / encode_time) << " КБ/с\n";
        std::cout << "- Roundtrip:       " << (success ? GREEN + "УСПЕХ" : RED + "НЕУДАЧА") 
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
    
    std::cout << "- Токенизатор:             " << wrapper->name() << "\n";
    std::cout << "- Размер словаря:          " << wrapper->vocab_size() << " токенов\n";
    std::cout << "- Протестировано примеров: " << test_examples.size() << "\n";
    std::cout << "- Успешных Roundtrip:      " << success_count << "/" 
              << test_examples.size() << " ("
              << std::fixed << std::setprecision(1)
              << (100.0 * success_count / test_examples.size()) << "%)\n";
    
    wrapper->print_stats();
}

// ============================================================================
// Основная функция
// ============================================================================

/**
 * @brief Точка входа в программу
 * 
 * @param argc Количество аргументов
 * @param argv Массив аргументов
 * @return int 0 при успехе, 1 при ошибке
 */
int main(int argc, char* argv[]) {
    // ============================================================================
    // Приветствие
    // ============================================================================
    
    print_header("BPE TOKENIZER - ПРОСТОЙ ПРИМЕР");
    std::cout << "Демонстрация базовых операций кодирования/декодирования\n\n";
    
    // ============================================================================
    // Парсинг аргументов командной строки
    // ============================================================================
    
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
            std::cout << "--verbose, -v - Подробный вывод\n";
            std::cout << "--all, -a     - Тестировать все примеры\n";
            std::cout << "--fast, -f    - Использовать FastTokenizer\n";
            std::cout << "--help, -h    - Показать справку\n";
            return 0;
        }
    }
    
    try {
        // ============================================================================
        // 1. ВЫБОР ТЕКСТА ДЛЯ ТЕСТИРОВАНИЯ
        // ============================================================================
        
        std::vector<std::pair<std::string, std::string>> test_examples;
        
        if (test_all) {
            test_examples = get_test_examples();
            std::cout << "Режим: тестирование всех " << test_examples.size() 
                      << " примеров\n";
        } else {
            test_examples = {
                {"Функция факториала (C++)", R"(
// Function to calculate factorial
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
                )"}
            };
            std::cout << "Режим: тестирование одного примера "
                      << "(используйте --all для всех)\n";
        }
        
        // ============================================================================
        // 2. СОЗДАНИЕ И ЗАГРУЗКА ТОКЕНИЗАТОРА
        // ============================================================================
        
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
                std::cout << "\n" << GREEN << "Модель загружена успешно!" << RESET << "\n";
                std::cout << "- Словарь:        " << vocab_path << "\n";
                std::cout << "- Слияния:        " << merges_path << "\n";
                std::cout << "- Размер словаря: " << fast->vocab_size() << " токенов\n";
                
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
                std::cout << "\n" << GREEN << "Модель загружена успешно!" << RESET << "\n";
                std::cout << "- Словарь:        " << vocab_path << "\n";
                std::cout << "- Слияния:        " << merges_path << "\n";
                std::cout << "- Размер словаря: " << bpe->vocab_size() << " токенов\n";
                std::cout << "- Правил слияния: " << bpe->get_tokenizer().merges_count() << "\n";
                
                // Запуск тестов
                run_tests(bpe.get(), test_examples, verbose);
            }
        }
        
        // ============================================================================
        // 3. ОБРАБОТКА ОШИБКИ ЗАГРУЗКИ
        // ============================================================================
        
        if (!loaded) {
            std::cerr << RED << "\nНЕ УДАЛОСЬ ЗАГРУЗИТЬ МОДЕЛЬ!" << RESET << "\n";
            std::cerr << "\nУбедитесь, что файлы существуют в одном из путей:\n";
            std::cerr << "../models/bpe_10000/cpp_vocab.json\n";
            std::cerr << "models/bpe_10000/cpp_vocab.json\n";
            std::cerr << "../../bpe_python/models/bpe_10000/vocab.json\n";
            std::cerr << "\nЧтобы конвертировать Python модели в C++ формат:\n";
            std::cerr << YELLOW << "cd ../tools/ && python convert_vocab.py --model-size 10000" 
                      << RESET << std::endl;
            return 1;
        }
        
        // ============================================================================
        // 4. УСПЕШНОЕ ЗАВЕРШЕНИЕ
        // ============================================================================
        
        std::cout << "\n" << GREEN << BOLD << "Пример успешно завершен!" << RESET << "\n";
        std::cout << "Используйте --help для просмотра дополнительных опций.\n";
        
    } catch (const std::exception& e) {
        std::cerr << RED << "\nОшибка выполнения: " << e.what() << RESET << std::endl;
        return 1;
    } catch (...) {
        std::cerr << RED << "\nНеизвестная ошибка!" << RESET << std::endl;
        return 1;
    }
    
    return 0;
}