/**
 * @file train_example.cpp
 * @brief Пример параллельного обучения BPE токенизатора на корпусе C++ кода
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 2.0.0
 * 
 * @details Демонстрация обучения токенизатора с использованием:
 *          - Параллельной обработки (OpenMP)
 *          - Прогресс-баров
 *          - Сохранения модели в разных форматах
 *          - Валидации после обучения
 * 
 * @compile g++ -std=c++17 -O3 -fopenmp -Iinclude train_example.cpp -o train_example -lfast_tokenizer
 * @run ./train_example [--quick] [--save-binary] [--validate]
 */

#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>
#include <set>

using namespace bpe;

// ======================================================================
// Класс для измерения времени с автоматическим логированием
// ======================================================================

class ScopedTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;
    
public:
    ScopedTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
        std::cout << "  ⏱️  " << std::left << std::setw(30) << name_ << ": " 
                  << std::right << std::setw(8) << duration.count() / 1000.0 
                  << " сек" << std::endl;
    }
};

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загрузка корпуса из файла
 */
std::vector<std::string> load_corpus(const std::string& path, bool verbose = true) {
    std::cout << "📖 Загрузка корпуса из: " << path << std::endl;
    
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    
    std::vector<std::string> corpus;
    std::string line;
    size_t total_size = 0;
    size_t line_count = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        if (!line.empty()) {
            corpus.push_back(line);
            total_size += line.size();
            
            if (verbose && line_count % 100000 == 0) {
                std::cout << "  Загружено " << line_count << " строк..." << std::endl;
            }
        }
    }
    
    if (verbose) {
        std::cout << "  ✅ Загружено примеров: " << corpus.size() << std::endl;
        std::cout << "  📊 Общий размер: " << total_size << " байт ("
                  << std::fixed << std::setprecision(2) 
                  << total_size / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "  📏 Средний размер: " << total_size / corpus.size() << " байт" << std::endl;
        
        // Статистика по уникальным символам
        std::set<char> unique_chars;
        for (const auto& text : corpus) {
            unique_chars.insert(text.begin(), text.end());
        }
        std::cout << "  🔤 Уникальных символов: " << unique_chars.size() << std::endl;
    }
    
    return corpus;
}

/**
 * @brief Создать тестовый корпус для демонстрации (если нет реального)
 */
std::vector<std::string> create_demo_corpus(size_t num_examples = 1000) {
    std::cout << "📝 Создание демо-корпуса из " << num_examples << " примеров..." << std::endl;
    
    std::vector<std::string> examples = {
        "int x = 42;",
        "std::cout << \"Hello\" << std::endl;",
        "for (int i = 0; i < 10; ++i) { sum += i; }",
        "if (condition) { do_something(); }",
        "class MyClass { public: void method(); };",
        "#include <iostream>",
        "template<typename T> T max(T a, T b) { return a > b ? a : b; }",
        "auto lambda = [](int x) { return x * x; };",
        "// это комментарий на русском языке",
        "/* многострочный\n комментарий */"
    };
    
    std::vector<std::string> corpus;
    corpus.reserve(num_examples);
    
    for (size_t i = 0; i < num_examples; ++i) {
        corpus.push_back(examples[i % examples.size()] + " // " + std::to_string(i));
    }
    
    std::cout << "  ✅ Создано " << corpus.size() << " примеров" << std::endl;
    return corpus;
}

/**
 * @brief Валидация обученной модели
 */
void validate_model(FastBPETokenizer& tokenizer, const std::vector<std::string>& corpus, size_t num_samples = 100) {
    std::cout << "\n🔍 Валидация модели на " << num_samples << " примерах..." << std::endl;
    
    size_t correct = 0;
    size_t total_chars = 0;
    size_t total_tokens = 0;
    
    // Берем случайные примеры из корпуса
    std::vector<size_t> indices;
    for (size_t i = 0; i < num_samples && i < corpus.size(); ++i) {
        indices.push_back(i);
    }
    
    for (size_t idx : indices) {
        const auto& text = corpus[idx];
        
        auto tokens = tokenizer.encode(text);
        auto decoded = tokenizer.decode(tokens);
        
        total_chars += text.size();
        total_tokens += tokens.size();
        
        if (decoded == text) {
            correct++;
        }
        
        if (idx < 5) {  // Показываем первые 5 примеров
            std::cout << "\n  Пример " << idx << ":\n";
            std::cout << "    Исходный: " << text.substr(0, 50) 
                      << (text.size() > 50 ? "..." : "") << "\n";
            std::cout << "    Декод.:   " << decoded.substr(0, 50) 
                      << (decoded.size() > 50 ? "..." : "") << "\n";
            std::cout << "    Токенов:  " << tokens.size() << "\n";
        }
    }
    
    double accuracy = 100.0 * correct / num_samples;
    double compression = 100.0 * (1.0 - static_cast<double>(total_tokens) / total_chars);
    
    std::cout << "\n📊 Результаты валидации:" << std::endl;
    std::cout << "  • Точность roundtrip: " << std::fixed << std::setprecision(1)
              << accuracy << "% (" << correct << "/" << num_samples << ")" << std::endl;
    std::cout << "  • Сжатие: " << std::fixed << std::setprecision(1)
              << compression << "%" << std::endl;
    std::cout << "  • Средняя длина токена: " << std::fixed << std::setprecision(2)
              << static_cast<double>(total_chars) / total_tokens << " символов" << std::endl;
}

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "🚀 ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРА\n";
    std::cout << "========================================\n\n";
    
    // Парсинг аргументов
    bool quick_mode = false;
    bool save_binary = false;
    bool validate = false;
    std::string corpus_path = "../../data/corpus/train_code.txt";
    size_t vocab_size = 8000;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick" || arg == "-q") {
            quick_mode = true;
            vocab_size = 1000;
        } else if (arg == "--save-binary" || arg == "-b") {
            save_binary = true;
        } else if (arg == "--validate" || arg == "-v") {
            validate = true;
        } else if (arg == "--corpus" && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoul(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --quick, -q           Быстрый режим (меньше токенов)\n";
            std::cout << "  --save-binary, -b     Сохранить в бинарном формате\n";
            std::cout << "  --validate, -v        Валидация после обучения\n";
            std::cout << "  --corpus PATH         Путь к корпусу\n";
            std::cout << "  --vocab-size N        Размер словаря\n";
            std::cout << "  --help, -h            Показать справку\n";
            return 0;
        }
    }
    
    try {
        // ======================================================================
        // Загрузка корпуса
        // ======================================================================
        
        std::vector<std::string> corpus;
        
        // Проверяем существование файла
        if (std::filesystem::exists(corpus_path)) {
            corpus = load_corpus(corpus_path);
        } else {
            std::cout << "⚠️  Файл не найден: " << corpus_path << std::endl;
            std::cout << "   Создаю демо-корпус для тестирования..." << std::endl;
            corpus = create_demo_corpus(quick_mode ? 100 : 1000);
        }
        
        if (corpus.empty()) {
            std::cerr << "❌ Корпус пуст!" << std::endl;
            return 1;
        }
        
        // ======================================================================
        // Конфигурация токенизатора
        // ======================================================================
        
        TokenizerConfig config;
        config.vocab_size = vocab_size;
        config.cache_size = 10000;
        config.byte_level = true;
        config.enable_cache = true;
        config.enable_profiling = true;
        
        std::cout << "\n⚙️  Конфигурация обучения:\n";
        std::cout << "  • Размер словаря: " << config.vocab_size << std::endl;
        std::cout << "  • Byte-level: " << (config.byte_level ? "да" : "нет") << std::endl;
        std::cout << "  • Кэширование: " << (config.enable_cache ? "да" : "нет") << std::endl;
        std::cout << "  • Профилирование: " << (config.enable_profiling ? "да" : "нет") << std::endl;
        std::cout << "  • Потоков: " << std::thread::hardware_concurrency() << std::endl;
        
        // ======================================================================
        // Создание токенизатора
        // ======================================================================
        
        FastBPETokenizer tokenizer(config);
        
        // ======================================================================
        // Обучение
        // ======================================================================
        
        std::cout << "\n🔄 Начало обучения..." << std::endl;
        
        auto stats = tokenizer.stats();
        
        {
            ScopedTimer timer("Общее время обучения");
            tokenizer.parallel_train(corpus, vocab_size);
        }
        
        // ======================================================================
        // Статистика после обучения
        // ======================================================================
        
        auto final_stats = tokenizer.stats();
        
        std::cout << "\n📊 Статистика обучения:\n";
        std::cout << "  • Итоговый размер словаря: " << tokenizer.vocab_size() << std::endl;
        std::cout << "  • Правил слияния: " << tokenizer.merges_count() << std::endl;
        
        // Показываем первые несколько токенов для примера
        std::cout << "\n🔤 Первые 20 токенов:\n";
        for (uint32_t i = 0; i < std::min<uint32_t>(20, tokenizer.vocab_size()); ++i) {
            // Для FastTokenizer нет прямого доступа к токенам по ID
            // В реальном коде можно было бы вывести что-то типа tokenizer.id_to_token(i)
            std::cout << "  " << std::setw(4) << i << ": [токен " << i << "]" << std::endl;
        }
        
        // ======================================================================
        // Валидация
        // ======================================================================
        
        if (validate) {
            validate_model(tokenizer, corpus, quick_mode ? 10 : 100);
        }
        
        // ======================================================================
        // Сохранение модели
        // ======================================================================
        
        std::cout << "\n💾 Сохранение модели..." << std::endl;
        
        // Создаем директорию для моделей
        std::filesystem::create_directories("../../bpe");
        
        if (save_binary) {
            std::string model_path = "../../bpe/model_trained.bin";
            {
                ScopedTimer timer("Сохранение в бинарный формат");
                tokenizer.save_binary(model_path);
            }
            std::cout << "  ✅ Бинарная модель: " << model_path << std::endl;
        } else {
            std::string vocab_path = "../../bpe/vocab_trained.json";
            std::string merges_path = "../../bpe/merges_trained.txt";
            
            {
                ScopedTimer timer("Сохранение в текстовый формат");
                tokenizer.save(vocab_path, merges_path);
            }
            
            std::cout << "  ✅ Словарь: " << vocab_path << std::endl;
            std::cout << "  ✅ Слияния: " << merges_path << std::endl;
            
            // Показываем размеры файлов
            auto vocab_size_bytes = std::filesystem::file_size(vocab_path);
            auto merges_size_bytes = std::filesystem::file_size(merges_path);
            
            std::cout << "  📦 Размер словаря: " << std::fixed << std::setprecision(2)
                      << vocab_size_bytes / 1024.0 / 1024.0 << " MB" << std::endl;
            std::cout << "  📦 Размер слияний: " << merges_size_bytes / 1024.0 << " KB" << std::endl;
        }
        
        // ======================================================================
        // Тест загрузки сохраненной модели
        // ======================================================================
        
        std::cout << "\n🔄 Тест загрузки сохраненной модели..." << std::endl;
        
        FastBPETokenizer loaded_tokenizer(config);
        
        if (save_binary) {
            if (loaded_tokenizer.load_binary("../../bpe/model_trained.bin")) {
                std::cout << "  ✅ Модель успешно загружена из бинарного файла" << std::endl;
                std::cout << "  📚 Размер словаря: " << loaded_tokenizer.vocab_size() << std::endl;
            }
        } else {
            if (loaded_tokenizer.load("../../bpe/vocab_trained.json", "../../bpe/merges_trained.txt")) {
                std::cout << "  ✅ Модель успешно загружена из текстовых файлов" << std::endl;
                std::cout << "  📚 Размер словаря: " << loaded_tokenizer.vocab_size() << std::endl;
            }
        }
        
        // ======================================================================
        // Итог
        // ======================================================================
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::cout << "\n📊 Сводка:" << std::endl;
        std::cout << "  • Размер корпуса: " << corpus.size() << " примеров" << std::endl;
        std::cout << "  • Размер словаря: " << tokenizer.vocab_size() << std::endl;
        std::cout << "  • Правил слияния: " << tokenizer.merges_count() << std::endl;
        
        if (!save_binary) {
            std::cout << "\n💡 Для использования модели:" << std::endl;
            std::cout << "   FastBPETokenizer tokenizer(config);" << std::endl;
            std::cout << "   tokenizer.load(\"../../bpe/vocab_trained.json\", "
                      << "\"../../bpe/merges_trained.txt\");" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}