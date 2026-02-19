/**
 * @file simple_example.cpp
 * @brief Простой пример использования BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Демонстрация базовых возможностей BPE токенизатора:
 *          - Загрузка модели из файлов
 *          - Кодирование текста в токены
 *          - Декодирование токенов обратно в текст
 *          - Проверка roundtrip (кодирование + декодирование)
 *          - Вывод статистики по токенам
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
 *   # Запуск с базовым токенизатором
 *   ./simple_example
 *   
 *   # Запуск с FastTokenizer и подробным выводом
 *   ./simple_example --fast --verbose
 *   
 *   # Тестирование всех примеров
 *   ./simple_example --all
 * 
 * @note Для работы требуются файлы модели:
 *       - models/cpp_vocab.json
 *       - models/cpp_merges.txt
 *       (можно получить через python tools/convert_vocab.py)
 * 
 * @see BPETokenizer
 * @see FastTokenizer
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

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Поиск файлов модели по разным путям (для BPETokenizer)
 */
bool find_bpe_model_files(BPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
        {"../../models/cpp_vocab.json", "../../models/cpp_merges.txt"},
        {"vocab.json", "merges.txt"},
        {"../../bpe/vocab.json", "../../bpe/merges.txt"}
    };
    
    for (const auto& [vpath, mpath] : candidates) {
        std::ifstream vfile(vpath);
        std::ifstream mfile(mpath);
        
        if (vfile.good() && mfile.good()) {
            vocab_path = vpath;
            merges_path = mpath;
            
            if (tokenizer.load_from_files(vpath, mpath)) {
                return true;
            }
        }
    }
    
    return false;
}

/**
 * @brief Поиск файлов модели по разным путям (для FastTokenizer)
 */
bool find_fast_model_files(FastBPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
        {"../../models/cpp_vocab.json", "../../models/cpp_merges.txt"},
        {"vocab.json", "merges.txt"},
        {"../../bpe/vocab.json", "../../bpe/merges.txt"}
    };
    
    for (const auto& [vpath, mpath] : candidates) {
        std::ifstream vfile(vpath);
        std::ifstream mfile(mpath);
        
        if (vfile.good() && mfile.good()) {
            vocab_path = vpath;
            merges_path = mpath;
            
            if (tokenizer.load(vpath, mpath)) {
                return true;
            }
        }
    }
    
    return false;
}

/**
 * @brief Создать тестовые примеры разной сложности
 */
std::vector<std::pair<std::string, std::string>> get_test_examples() {
    return {
        {"Простое выражение", "int x = 42;"},
        {"Русский комментарий", "// это комментарий на русском языке"},
        {"Смешанный текст", "std::cout << \"Привет, мир!\" << std::endl;"},
        {"Шаблон", "template<typename T> T max(T a, T b) { return a > b ? a : b; }"},
        {"Класс", "class MyClass { public: MyClass() = default; };"},
        {"Include", "#include <iostream>"},
        {"Эмодзи", "// 🔥 C++ code with emoji 😊"}
    };
}

/**
 * @brief Вывести статистику по токенам (для BPETokenizer)
 */
void print_bpe_token_stats(const std::vector<token_id_t>& tokens, const Vocabulary& vocab) {
    std::map<size_t, int> length_distribution;
    std::set<token_id_t> unique_tokens(tokens.begin(), tokens.end());
    
    for (auto id : unique_tokens) {
        std::string token = vocab.id_to_token(id);
        length_distribution[token.length()]++;
    }
    
    std::cout << "\n📊 Статистика токенов:\n";
    std::cout << "  • Всего токенов: " << tokens.size() << "\n";
    std::cout << "  • Уникальных токенов: " << unique_tokens.size() << "\n";
    std::cout << "  • Соотношение: " << std::fixed << std::setprecision(2)
              << (100.0 * unique_tokens.size() / tokens.size()) << "%\n";
}

/**
 * @brief Вывести статистику по токенам (для FastTokenizer)
 */
void print_fast_token_stats(const std::vector<uint32_t>& tokens) {
    std::set<uint32_t> unique_tokens(tokens.begin(), tokens.end());
    
    std::cout << "\n📊 Статистика токенов:\n";
    std::cout << "  • Всего токенов: " << tokens.size() << "\n";
    std::cout << "  • Уникальных токенов: " << unique_tokens.size() << "\n";
    std::cout << "  • Соотношение: " << std::fixed << std::setprecision(2)
              << (100.0 * unique_tokens.size() / tokens.size()) << "%\n";
}

// ======================================================================
// Классы-обертки с доступом к внутренним токенизаторам
// ======================================================================

class BPETokenizerWrapper {
public:
    BPETokenizer tokenizer;
    
    BPETokenizerWrapper() {
        tokenizer.set_byte_level(true);
        tokenizer.set_unknown_token("<UNK>");
    }
    
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load_from_files(vocab_path, merges_path);
    }
    
    size_t vocab_size() const {
        return tokenizer.vocab_size();
    }
    
    std::vector<uint32_t> encode(const std::string& text) {
        auto tokens = tokenizer.encode(text);
        return std::vector<uint32_t>(tokens.begin(), tokens.end());
    }
    
    std::string decode(const std::vector<uint32_t>& tokens) {
        std::vector<token_id_t> old_tokens(tokens.begin(), tokens.end());
        return tokenizer.decode(old_tokens);
    }
    
    std::string token_to_string(uint32_t id) {
        return tokenizer.vocabulary().id_to_token(id);
    }
    
    BPETokenizer& get_tokenizer() { return tokenizer; }
};

class FastTokenizerWrapper {
public:
    FastBPETokenizer tokenizer;
    TokenizerStats stats;
    
    FastTokenizerWrapper() : tokenizer(TokenizerConfig{32000, 10000, true, true}) {}
    
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load(vocab_path, merges_path);
    }
    
    size_t vocab_size() const {
        return tokenizer.vocab_size();
    }
    
    std::vector<uint32_t> encode(const std::string& text) {
        auto result = tokenizer.encode(text);
        stats = tokenizer.stats();
        return result;
    }
    
    std::string decode(const std::vector<uint32_t>& tokens) {
        auto result = tokenizer.decode(tokens);
        stats = tokenizer.stats();
        return result;
    }
    
    std::string token_to_string(uint32_t id) {
        return "ID:" + std::to_string(id);
    }
    
    void print_stats() const {
        std::cout << "    • Cache hits: " << stats.cache_hits << "\n";
        std::cout << "    • Cache misses: " << stats.cache_misses << "\n";
        std::cout << "    • Hit rate: " << (stats.cache_hit_rate() * 100) << "%\n";
    }
    
    FastBPETokenizer& get_tokenizer() { return tokenizer; }
};

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "🔧 BPE TOKENIZER - ПРОСТОЙ ПРИМЕР\n";
    std::cout << "========================================\n\n";
    
    // Парсинг аргументов
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
        // Создание и загрузка токенизатора
        // ======================================================================
        
        std::string vocab_path, merges_path;
        bool loaded = false;
        
        if (use_fast) {
            std::cout << "📦 Создание FastTokenizer...\n";
            auto fast = std::make_unique<FastTokenizerWrapper>();
            loaded = find_fast_model_files(fast->get_tokenizer(), vocab_path, merges_path);
            
            if (loaded) {
                std::cout << "  ✅ Загружен: " << vocab_path << "\n";
                std::cout << "  ✅ Загружен: " << merges_path << "\n";
                std::cout << "  📚 Размер словаря: " << fast->vocab_size() << " токенов\n";
                std::cout << std::endl;
                
                // ======================================================================
                // Выбор текста для тестирования
                // ======================================================================
                
                std::vector<std::pair<std::string, std::string>> test_examples;
                
                if (test_all) {
                    test_examples = get_test_examples();
                } else {
                    test_examples = {
                        {"C++ код с комментариями", R"(
                // Функция для вычисления факториала
                int factorial(int n) {
                    if (n <= 1) return 1;
                    return n * factorial(n - 1);
                }
            )"}
                    };
                }
                
                // ======================================================================
                // Тестирование каждого примера
                // ======================================================================
                
                int success_count = 0;
                
                for (const auto& [desc, code] : test_examples) {
                    std::cout << "\n" << std::string(50, '-') << "\n";
                    std::cout << "📝 " << desc << "\n";
                    std::cout << std::string(50, '-') << "\n";
                    
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
                    auto tokens = fast->encode(code);
                    double encode_time = timer.elapsed();
                    
                    // Декодирование
                    timer.reset();
                    auto decoded = fast->decode(tokens);
                    double decode_time = timer.elapsed();
                    
                    // Проверка roundtrip
                    bool success = (decoded == code);
                    if (success) success_count++;
                    
                    std::cout << "\n📊 Результаты:\n";
                    std::cout << "  • Токенов: " << tokens.size() << "\n";
                    std::cout << "  • Время encode: " << std::fixed << std::setprecision(3) 
                              << encode_time * 1000 << " ms\n";
                    std::cout << "  • Время decode: " << decode_time * 1000 << " ms\n";
                    std::cout << "  • Скорость: " << (code.size() / 1024.0 / encode_time) 
                              << " KB/s\n";
                    std::cout << "  • Roundtrip: " << (success ? "✅ УСПЕХ" : "❌ НЕУДАЧА") << "\n";
                    
                    if (verbose) {
                        print_fast_token_stats(tokens);
                    }
                    
                    // Показываем первые токены
                    std::cout << "\n🔤 Первые 10 токенов:\n";
                    for (size_t i = 0; i < std::min(size_t(10), tokens.size()); ++i) {
                        std::string token_str = fast->token_to_string(tokens[i]);
                        std::string escaped = utils::escape_string(token_str);
                        if (escaped.size() > 30) escaped = escaped.substr(0, 27) + "...";
                        std::cout << "  " << std::setw(4) << tokens[i] << ": '" 
                                  << escaped << "'\n";
                    }
                }
                
                // ======================================================================
                // Итоговая статистика
                // ======================================================================
                
                std::cout << "\n" << std::string(50, '=') << "\n";
                std::cout << "📊 ИТОГОВАЯ СТАТИСТИКА\n";
                std::cout << std::string(50, '=') << "\n";
                
                std::cout << "  • Токенизатор: FastTokenizer\n";
                std::cout << "  • Протестировано примеров: " << test_examples.size() << "\n";
                std::cout << "  • Успешных roundtrip: " << success_count << "/" 
                          << test_examples.size() << " ("
                          << std::fixed << std::setprecision(1)
                          << (100.0 * success_count / test_examples.size()) << "%)\n";
                
                fast->print_stats();
            }
            
        } else {
            std::cout << "📦 Создание BPETokenizer...\n";
            auto bpe = std::make_unique<BPETokenizerWrapper>();
            loaded = find_bpe_model_files(bpe->get_tokenizer(), vocab_path, merges_path);
            
            if (loaded) {
                std::cout << "  ✅ Загружен: " << vocab_path << "\n";
                std::cout << "  ✅ Загружен: " << merges_path << "\n";
                std::cout << "  📚 Размер словаря: " << bpe->vocab_size() << " токенов\n";
                std::cout << std::endl;
                
                // ======================================================================
                // Выбор текста для тестирования
                // ======================================================================
                
                std::vector<std::pair<std::string, std::string>> test_examples;
                
                if (test_all) {
                    test_examples = get_test_examples();
                } else {
                    test_examples = {
                        {"C++ код с комментариями", R"(
                // Функция для вычисления факториала
                int factorial(int n) {
                    if (n <= 1) return 1;
                    return n * factorial(n - 1);
                }
            )"}
                    };
                }
                
                // ======================================================================
                // Тестирование каждого примера
                // ======================================================================
                
                int success_count = 0;
                
                for (const auto& [desc, code] : test_examples) {
                    std::cout << "\n" << std::string(50, '-') << "\n";
                    std::cout << "📝 " << desc << "\n";
                    std::cout << std::string(50, '-') << "\n";
                    
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
                    auto tokens = bpe->encode(code);
                    double encode_time = timer.elapsed();
                    
                    // Декодирование
                    timer.reset();
                    auto decoded = bpe->decode(tokens);
                    double decode_time = timer.elapsed();
                    
                    // Проверка roundtrip
                    bool success = (decoded == code);
                    if (success) success_count++;
                    
                    std::cout << "\n📊 Результаты:\n";
                    std::cout << "  • Токенов: " << tokens.size() << "\n";
                    std::cout << "  • Время encode: " << std::fixed << std::setprecision(3) 
                              << encode_time * 1000 << " ms\n";
                    std::cout << "  • Время decode: " << decode_time * 1000 << " ms\n";
                    std::cout << "  • Скорость: " << (code.size() / 1024.0 / encode_time) 
                              << " KB/s\n";
                    std::cout << "  • Roundtrip: " << (success ? "✅ УСПЕХ" : "❌ НЕУДАЧА") << "\n";
                    
                    // Показываем первые токены
                    std::cout << "\n🔤 Первые 10 токенов:\n";
                    for (size_t i = 0; i < std::min(size_t(10), tokens.size()); ++i) {
                        std::string token_str = bpe->token_to_string(tokens[i]);
                        std::string escaped = utils::escape_string(token_str);
                        if (escaped.size() > 30) escaped = escaped.substr(0, 27) + "...";
                        std::cout << "  " << std::setw(4) << tokens[i] << ": '" 
                                  << escaped << "'\n";
                    }
                }
                
                // ======================================================================
                // Итоговая статистика
                // ======================================================================
                
                std::cout << "\n" << std::string(50, '=') << "\n";
                std::cout << "📊 ИТОГОВАЯ СТАТИСТИКА\n";
                std::cout << std::string(50, '=') << "\n";
                
                std::cout << "  • Токенизатор: BPETokenizer\n";
                std::cout << "  • Протестировано примеров: " << test_examples.size() << "\n";
                std::cout << "  • Успешных roundtrip: " << success_count << "/" 
                          << test_examples.size() << " ("
                          << std::fixed << std::setprecision(1)
                          << (100.0 * success_count / test_examples.size()) << "%)\n";
            }
        }
        
        if (!loaded) {
            std::cerr << "❌ Не удалось загрузить модель!\n";
            std::cerr << "   Убедитесь, что файлы существуют:\n";
            std::cerr << "   - models/cpp_vocab.json\n";
            std::cerr << "   - models/cpp_merges.txt\n";
            std::cerr << "\n   Запустите для конвертации:\n";
            std::cerr << "   python tools/convert_vocab.py\n";
            return 1;
        }
        
        std::cout << "\n✅ Пример успешно завершен!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}