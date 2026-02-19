/**
 * @file compare_with_python.cpp
 * @brief Инструмент для сравнения C++ и Python реализаций токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 2.0.0
 * 
 * @details Утилита для валидации C++ токенизатора против Python эталона.
 *          Поддерживает различные режимы работы:
 *          - Кодирование/декодирование текста
 *          - Пакетная обработка нескольких примеров
 *          - Сравнение с эталонными результатами из файла
 *          - Сохранение результатов в JSON
 *          - Подробная статистика производительности
 * 
 *          Поддерживаемые токенизаторы:
 *          - BPETokenizer (базовая реализация)
 *          - FastTokenizer (оптимизированная версия)
 * 
 * @usage ./compare_with_python <input_file> <vocab_file> <merges_file> [options]
 * 
 * @param input_file  Входной файл с текстом или JSON массивом токенов
 * @param vocab_file  Файл словаря в формате JSON
 * @param merges_file Файл правил слияния
 * 
 * @options
 *   --decode           Режим декодирования (иначе кодирование)
 *   --batch            Пакетный режим (файл содержит список текстов)
 *   --compare <file>   Сравнить с результатами из файла
 *   --output <file>    Сохранить результат в файл
 *   --verbose          Подробный вывод
 *   --fast             Использовать FastTokenizer вместо базового
 *   --stats            Показать статистику
 *   --help             Показать эту справку
 * 
 * @example
 *   # Кодирование текста
 *   ./compare_with_python input.txt vocab.json merges.txt
 *   
 *   # Кодирование с сохранением и сравнением
 *   ./compare_with_python input.txt vocab.json merges.txt --output result.json --compare expected.json
 *   
 *   # Пакетное кодирование с FastTokenizer
 *   ./compare_with_python texts.txt vocab.json merges.txt --batch --fast --stats
 *   
 *   # Декодирование токенов
 *   ./compare_with_python tokens.json vocab.json merges.txt --decode
 * 
 * @note Для корректной работы требуются файлы модели, обученные на Python
 * @see BPETokenizer
 * @see FastTokenizer
 */

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <memory>

#include <nlohmann/json.hpp>

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

void print_help(const char* prog_name) {
    std::cout << "Использование: " << prog_name 
              << " <input_file> <vocab_file> <merges_file> [options]\n"
              << "\n"
              << "Опции:\n"
              << "  --decode           Режим декодирования (иначе кодирование)\n"
              << "  --batch            Пакетный режим (файл содержит список текстов)\n"
              << "  --compare <file>   Сравнить с результатами из файла\n"
              << "  --output <file>    Сохранить результат в файл\n"
              << "  --verbose          Подробный вывод\n"
              << "  --fast             Использовать FastTokenizer вместо базового\n"
              << "  --stats            Показать статистику\n"
              << "  --help             Показать эту справку\n"
              << std::endl;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

nlohmann::json read_json(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    nlohmann::json j;
    file >> j;
    return j;
}

void save_json(const nlohmann::json& j, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot save to file: " + path);
    }
    file << j.dump(2);
}

std::vector<std::string> read_texts(const std::string& path) {
    std::vector<std::string> texts;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            texts.push_back(line);
        }
    }
    
    return texts;
}

bool compare_tokens(const std::vector<token_id_t>& a, 
                    const std::vector<token_id_t>& b,
                    bool verbose = false) {
    if (a.size() != b.size()) {
        if (verbose) {
            std::cerr << "  ❌ Размер разный: " << a.size() << " vs " << b.size() << std::endl;
        }
        return false;
    }
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            if (verbose) {
                std::cerr << "  ❌ Несовпадение на позиции " << i << ": "
                          << a[i] << " vs " << b[i] << std::endl;
            }
            return false;
        }
    }
    
    return true;
}

// ======================================================================
// Интерфейс для работы с разными токенизаторами
// ======================================================================

class ITokenizerWrapper {
public:
    virtual ~ITokenizerWrapper() = default;
    virtual std::vector<token_id_t> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<token_id_t>& tokens) = 0;
    virtual size_t vocab_size() const = 0;
    virtual std::string name() const = 0;
    virtual void print_stats() const {}
};

class BPETokenizerWrapper : public ITokenizerWrapper {
private:
    BPETokenizer tokenizer;
    
public:
    BPETokenizerWrapper(const std::string& vocab_path, const std::string& merges_path) {
        tokenizer.set_byte_level(true);
        tokenizer.set_unknown_token("<UNK>");
        if (!tokenizer.load_from_files(vocab_path, merges_path)) {
            throw std::runtime_error("Failed to load BPETokenizer");
        }
    }
    
    std::vector<token_id_t> encode(const std::string& text) override {
        return tokenizer.encode(text);
    }
    
    std::string decode(const std::vector<token_id_t>& tokens) override {
        return tokenizer.decode(tokens);
    }
    
    size_t vocab_size() const override {
        return tokenizer.vocab_size();
    }
    
    std::string name() const override {
        return "BPETokenizer";
    }
};

class FastTokenizerWrapper : public ITokenizerWrapper {
private:
    FastBPETokenizer tokenizer;
    TokenizerStats stats;
    
public:
    FastTokenizerWrapper(const std::string& vocab_path, const std::string& merges_path) 
        : tokenizer(TokenizerConfig{32000, 10000, true, true}) {
        if (!tokenizer.load(vocab_path, merges_path)) {
            throw std::runtime_error("Failed to load FastTokenizer");
        }
    }
    
    std::vector<token_id_t> encode(const std::string& text) override {
        auto result = tokenizer.encode(text);
        stats = tokenizer.stats();  // Обновляем статистику
        return result;
    }
    
    std::string decode(const std::vector<token_id_t>& tokens) override {
        auto result = tokenizer.decode(tokens);
        stats = tokenizer.stats();  // Обновляем статистику
        return result;
    }
    
    size_t vocab_size() const override {
        return tokenizer.vocab_size();
    }
    
    std::string name() const override {
        return "FastTokenizer";
    }
    
    void print_stats() const override {
        std::cout << "  • Попаданий в кэш: " << stats.cache_hits << "\n";
        std::cout << "  • Промахов кэша: " << stats.cache_misses << "\n";
        std::cout << "  • Эффективность кэша: " 
                  << std::fixed << std::setprecision(1) << (stats.cache_hit_rate() * 100) << "%\n";
    }
};

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        print_help(argv[0]);
        return 0;
    }
    
    if (argc < 4) {
        std::cerr << "❌ Недостаточно аргументов!" << std::endl;
        print_help(argv[0]);
        return 1;
    }
    
    // ======================================================================
    // Парсинг аргументов
    // ======================================================================
    
    std::string input_path = argv[1];
    std::string vocab_path = argv[2];
    std::string merges_path = argv[3];
    
    bool decode_mode = false;
    bool batch_mode = false;
    bool verbose = false;
    bool use_fast = false;
    bool show_stats = false;
    std::string compare_path;
    std::string output_path;
    
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--decode") {
            decode_mode = true;
        } else if (arg == "--batch") {
            batch_mode = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--fast") {
            use_fast = true;
        } else if (arg == "--stats") {
            show_stats = true;
        } else if (arg == "--compare" && i + 1 < argc) {
            compare_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "❌ Неизвестная опция: " << arg << std::endl;
            print_help(argv[0]);
            return 1;
        }
    }
    
    // ======================================================================
    // Заголовок
    // ======================================================================
    
    std::cout << "========================================\n";
    std::cout << "🔍 СРАВНЕНИЕ C++ И PYTHON ТОКЕНИЗАТОРОВ\n";
    std::cout << "========================================\n\n";
    
    std::cout << "📂 Входной файл:  " << input_path << std::endl;
    std::cout << "📂 Словарь:       " << vocab_path << std::endl;
    std::cout << "📂 Слияния:       " << merges_path << std::endl;
    std::cout << "🔄 Режим:         " << (decode_mode ? "декодирование" : "кодирование") << std::endl;
    std::cout << "📦 Режим пакетный: " << (batch_mode ? "да" : "нет") << std::endl;
    std::cout << "⚡ Токенизатор:   " << (use_fast ? "FastTokenizer" : "BPETokenizer") << std::endl;
    std::cout << std::endl;
    
    try {
        // ======================================================================
        // Инициализация токенизатора
        // ======================================================================
        
        bpe::utils::Timer timer;
        
        // Создаем соответствующий токенизатор
        std::unique_ptr<ITokenizerWrapper> tokenizer;
        
        if (use_fast) {
            tokenizer = std::make_unique<FastTokenizerWrapper>(vocab_path, merges_path);
        } else {
            tokenizer = std::make_unique<BPETokenizerWrapper>(vocab_path, merges_path);
        }
        
        double load_time = timer.elapsed() * 1000;
        std::cout << "✅ " << tokenizer->name() << " загружен за " 
                  << std::fixed << std::setprecision(2) << load_time << " мс" << std::endl;
        std::cout << "📚 Размер словаря: " << tokenizer->vocab_size() << std::endl;
        std::cout << std::endl;
        
        // ======================================================================
        // Обработка в зависимости от режима
        // ======================================================================
        
        nlohmann::json result;
        double process_time = 0;
        
        timer.reset();
        
        if (decode_mode) {
            // ==================================================================
            // Режим декодирования
            // ==================================================================
            
            if (batch_mode) {
                // Пакетное декодирование
                auto json_data = read_json(input_path);
                
                if (!json_data.is_array()) {
                    std::cerr << "❌ Ожидался JSON массив для пакетного режима" << std::endl;
                    return 1;
                }
                
                std::vector<std::vector<token_id_t>> batch;
                for (const auto& item : json_data) {
                    batch.push_back(item.get<std::vector<token_id_t>>());
                }
                
                std::vector<std::string> decoded;
                for (const auto& tokens : batch) {
                    decoded.push_back(tokenizer->decode(tokens));
                }
                
                result = decoded;
            } else {
                // Одиночное декодирование
                auto json_data = read_json(input_path);
                std::vector<token_id_t> tokens = json_data.get<std::vector<token_id_t>>();
                
                std::string decoded = tokenizer->decode(tokens);
                result = decoded;
            }
        } else {
            // ==================================================================
            // Режим кодирования
            // ==================================================================
            
            if (batch_mode) {
                // Пакетное кодирование
                auto texts = read_texts(input_path);
                
                std::vector<std::vector<token_id_t>> encoded;
                for (const auto& text : texts) {
                    encoded.push_back(tokenizer->encode(text));
                }
                
                result = encoded;
            } else {
                // Одиночное кодирование
                std::string text = read_file(input_path);
                auto tokens = tokenizer->encode(text);
                result = tokens;
            }
        }
        
        process_time = timer.elapsed() * 1000;
        
        std::cout << "⏱️  Время обработки: " << std::fixed << std::setprecision(2) 
                  << process_time << " мс" << std::endl;
        
        // ======================================================================
        // Сравнение с эталоном
        // ======================================================================
        
        if (!compare_path.empty()) {
            std::cout << "\n🔍 Сравнение с эталоном: " << compare_path << std::endl;
            
            auto expected = read_json(compare_path);
            
            bool match = false;
            
            if (result.is_array() && expected.is_array()) {
                if (result.size() == expected.size()) {
                    match = true;
                    
                    if (!result.empty() && result[0].is_array()) {
                        // Сравнение пакетных результатов
                        for (size_t i = 0; i < result.size() && match; ++i) {
                            std::vector<token_id_t> a = result[i].get<std::vector<token_id_t>>();
                            std::vector<token_id_t> b = expected[i].get<std::vector<token_id_t>>();
                            
                            if (!compare_tokens(a, b, verbose)) {
                                match = false;
                                std::cout << "  ❌ Несовпадение в элементе " << i << std::endl;
                            }
                        }
                    } else {
                        // Сравнение одиночных результатов
                        if (result[0].is_number()) {
                            std::vector<token_id_t> a = result.get<std::vector<token_id_t>>();
                            std::vector<token_id_t> b = expected.get<std::vector<token_id_t>>();
                            match = compare_tokens(a, b, verbose);
                        } else if (result[0].is_string()) {
                            std::vector<std::string> a = result.get<std::vector<std::string>>();
                            std::vector<std::string> b = expected.get<std::vector<std::string>>();
                            match = (a == b);
                        }
                    }
                }
            } else if (result.is_array() && expected.is_array()) {
                std::vector<token_id_t> a = result.get<std::vector<token_id_t>>();
                std::vector<token_id_t> b = expected.get<std::vector<token_id_t>>();
                match = compare_tokens(a, b, verbose);
            } else if (result.is_string() && expected.is_string()) {
                match = (result.get<std::string>() == expected.get<std::string>());
            }
            
            std::cout << "  Результат: " << (match ? "✅ СОВПАДАЕТ" : "❌ НЕ СОВПАДАЕТ") << std::endl;
        }
        
        // ======================================================================
        // Сохранение результата
        // ======================================================================
        
        if (!output_path.empty()) {
            save_json(result, output_path);
            std::cout << "\n💾 Результат сохранен в: " << output_path << std::endl;
        } else if (verbose) {
            std::cout << "\n📄 Результат:\n" << result.dump(2) << std::endl;
        }
        
        // ======================================================================
        // Статистика
        // ======================================================================
        
        if (show_stats) {
            std::cout << "\n📊 Статистика:\n";
            
            if (!decode_mode && !batch_mode) {
                std::string text = read_file(input_path);
                auto tokens = result.get<std::vector<token_id_t>>();
                
                std::cout << "  • Размер текста: " << text.size() << " символов\n";
                std::cout << "  • Количество токенов: " << tokens.size() << "\n";
                std::cout << "  • Сжатие: " << std::fixed << std::setprecision(1)
                          << (100.0 * (1.0 - static_cast<double>(tokens.size()) / text.size())) << "%\n";
                std::cout << "  • Скорость: " << std::fixed << std::setprecision(0)
                          << (text.size() / (process_time / 1000.0)) << " символов/сек\n";
            }
            
            tokenizer->print_stats();
        }
        
        std::cout << "\n✅ Готово!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Ошибка: " << e.what() << std::endl;
        return 1;
    }
}