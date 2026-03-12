/**
 * @file train_example.cpp
 * @brief Пример параллельного обучения BPE токенизатора на корпусе C++ кода
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Эта программа демонстрирует полный цикл обучения токенизатора:
 *          **Загрузка корпуса**         - чтение текстов из файла или создание демо-данных
 *          **Параллельное обучение**    - использование OpenMP для ускорения
 *          **Прогресс-бары**            - визуализация процесса обучения
 *          **Валидация**                - проверка качества обученной модели
 *          **Сохранение**               - экспорт в текстовом (JSON/TXT) или бинарном формате
 *          **Тест загрузки**            - проверка, что сохраненная модель корректно загружается
 * 
 *          Программа поддерживает аргументы командной строки:
 *          - `--quick, -q`          - быстрый режим (меньше токенов, для тестов)
 *          - `--save-binary, -b`    - сохранить в бинарном формате
 *          - `--validate, -v`       - выполнить валидацию после обучения
 *          - `--corpus PATH`        - путь к файлу корпуса
 *          - `--vocab-size N`       - размер словаря (по умолчанию 8000)
 *          - `--help, -h`           - показать справку
 * 
 * @compile g++ -std=c++17 -O3 -fopenmp -Iinclude train_example.cpp -o train_example -lfast_tokenizer -lpthread
 * @run   ./train_example --corpus ../../../data/corpus/train_code.txt --vocab-size 8000 --validate
 * 
 * @see FastBPETokenizer
 * @see TokenizerConfig
 * @see ScopedTimer
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
#include <random>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;
using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr int WIDTH = 60;
    constexpr size_t DEFAULT_VOCAB_SIZE = 8000;
    constexpr size_t DEFAULT_CACHE_SIZE = 10000;
    constexpr size_t QUICK_VOCAB_SIZE = 1000;
    constexpr size_t QUICK_CORPUS_SIZE = 100;
    constexpr size_t NORMAL_CORPUS_SIZE = 1000;
    constexpr size_t VALIDATION_SAMPLES = 100;
    constexpr size_t QUICK_VALIDATION_SAMPLES = 10;
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ======================================================================
// Класс для измерения времени с автоматическим логированием
// ======================================================================

/**
 * @brief RAII таймер с автоматическим выводом при разрушении
 * 
 * Удобен для измерения времени выполнения блоков кода.
 * Автоматически выводит имя операции и затраченное время.
 */
class ScopedTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;
    bool print_on_destroy_;
    
public:
    /**
     * @brief Конструктор - запоминает время старта и имя операции
     * @param name Имя операции для вывода
     * @param print Выводить ли результат при разрушении
     */
    ScopedTimer(const std::string& name, bool print = true) 
        : name_(name), print_on_destroy_(print) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Деструктор - вычисляет и выводит прошедшее время
     */
    ~ScopedTimer() {
        if (print_on_destroy_) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
            std::cout << "  " << BOLD << "⏱" << RESET << "  " 
                      << std::left << std::setw(30) << name_ << ": " 
                      << std::right << std::setw(8) << duration.count() / 1000.0 
                      << " с" << std::endl;
        }
    }
    
    /**
     * @brief Получить прошедшее время в миллисекундах
     */
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    /**
     * @brief Сбросить таймер
     */
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
};

// ======================================================================
// Вспомогательные функции
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
 * @brief Выводит прогресс-бар
 * 
 * @param current Текущее значение
 * @param total Общее значение
 * @param width Ширина прогресс-бара
 */
void print_progress_bar(size_t current, size_t total, int width = 50) {
    float ratio = static_cast<float>(current) / total;
    int bar_width = static_cast<int>(ratio * width);
    
    std::cout << "\r  [";
    for (int i = 0; i < width; ++i) {
        if (i < bar_width) {
            std::cout << (i < bar_width - 1 ? "=" : ">");
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(ratio * 100.0) << "%";
    std::cout.flush();
}

/**
 * @brief Загружает корпус текстов из файла
 * 
 * Читает файл построчно, игнорируя пустые строки.
 * Выводит подробную статистику о загруженных данных.
 * 
 * @param path Путь к файлу корпуса
 * @param verbose Выводить ли подробную статистику
 * @return std::vector<std::string> Вектор строк корпуса
 * @throws std::runtime_error если файл не удалось открыть
 */
std::vector<std::string> load_corpus(const std::string& path, bool verbose = true) {
    if (verbose) {
        std::cout << CYAN << "📂 Загрузка корпуса из: " << path << RESET << std::endl;
    }
    
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + path);
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
        std::cout << GREEN << "  ✓ Загружено примеров: " << corpus.size() << RESET << std::endl;
        std::cout << "  • общий размер: " << total_size << " байт ("
                  << std::fixed << std::setprecision(2) 
                  << total_size / 1024.0 / 1024.0 << " МБ)" << std::endl;
        std::cout << "  • средний размер: " << total_size / corpus.size() << " байт" << std::endl;
        
        // Статистика по уникальным символам
        std::set<char> unique_chars;
        for (const auto& text : corpus) {
            unique_chars.insert(text.begin(), text.end());
        }
        std::cout << "  • уникальных символов: " << unique_chars.size() << std::endl;
    }
    
    return corpus;
}

/**
 * @brief Создает демонстрационный корпус для тестирования, если реальный не найден
 * 
 * Генерирует указанное количество примеров на основе небольшого набора
 * шаблонов C++ кода. Каждый пример получает уникальный ID.
 * 
 * @param num_examples Количество примеров для генерации
 * @return std::vector<std::string> Вектор сгенерированных примеров
 */
std::vector<std::string> create_demo_corpus(size_t num_examples = 1000) {
    std::cout << YELLOW << "⚠ Создание демо-корпуса из " << num_examples << " примеров..." << RESET << std::endl;
    
    // Базовые шаблоны C++ кода
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
        
        if ((i + 1) % 100 == 0) {
            print_progress_bar(i + 1, num_examples);
        }
    }
    
    std::cout << "\n" << GREEN << "  ✓ Создано " << corpus.size() << " примеров" << RESET << std::endl;
    return corpus;
}

/**
 * @brief Выполняет валидацию обученной модели
 * 
 * Проверяет, что encode + decode возвращают исходный текст (Roundtrip).
 * Вычисляет:
 * - Точность (процент успешных преобразований)
 * - Степень сжатия (насколько токены компактнее исходного текста)
 * - Среднюю длину токена
 * 
 * @param tokenizer Ссылка на обученный токенизатор
 * @param corpus Корпус текстов для валидации
 * @param num_samples Количество случайных примеров для проверки
 */
void validate_model(FastBPETokenizer& tokenizer, 
                   const std::vector<std::string>& corpus, 
                   size_t num_samples = VALIDATION_SAMPLES) {
    
    print_header("ВАЛИДАЦИЯ МОДЕЛИ");
    std::cout << "Проверка на " << num_samples << " случайных примерах..." << std::endl;
    
    size_t correct = 0;
    size_t total_chars = 0;
    size_t total_tokens = 0;
    
    // Генерируем случайные индексы для выборки
    std::vector<size_t> indices;
    if (corpus.size() <= num_samples) {
        for (size_t i = 0; i < corpus.size(); ++i) indices.push_back(i);
    } else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, static_cast<int>(corpus.size() - 1));
        
        while (indices.size() < num_samples) {
            size_t idx = dis(gen);
            if (std::find(indices.begin(), indices.end(), idx) == indices.end()) {
                indices.push_back(idx);
            }
        }
    }
    
    std::sort(indices.begin(), indices.end());
    
    std::cout << "\n------------------------------------------------------------\n";
    
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        const auto& text = corpus[idx];
        
        auto tokens = tokenizer.encode(text);
        auto decoded = tokenizer.decode(tokens);
        
        total_chars += text.size();
        total_tokens += tokens.size();
        
        if (decoded == text) {
            correct++;
        }
        
        // Показываем первые 5 примеров для наглядности
        if (i < 5) {
            std::cout << "\n  Пример " << idx << ":\n";
            
            std::string preview_text = text.substr(0, 50);
            if (text.size() > 50) preview_text += "...";
            
            std::string preview_decoded = decoded.substr(0, 50);
            if (decoded.size() > 50) preview_decoded += "...";
            
            std::cout << "    Исходный: " << preview_text << "\n";
            std::cout << "    Декод.:   " << preview_decoded << "\n";
            std::cout << "    Токенов:  " << tokens.size() << "\n";
            std::cout << "    Результат: " << (decoded == text ? GREEN + "✓" : RED + "✗") 
                      << RESET << "\n";
        }
        
        print_progress_bar(i + 1, indices.size());
    }
    
    std::cout << "\n\n" << CYAN << "Результаты валидации:" << RESET << std::endl;
    
    double accuracy = 100.0 * correct / indices.size();
    double compression = 100.0 * (1.0 - static_cast<double>(total_tokens) / total_chars);
    double avg_token_len = static_cast<double>(total_chars) / total_tokens;
    
    std::cout << "  • точность Roundtrip: " << std::fixed << std::setprecision(1)
              << accuracy << "% (" << correct << "/" << indices.size() << ")" << std::endl;
    std::cout << "  • сжатие: " << std::fixed << std::setprecision(1)
              << compression << "%" << std::endl;
    std::cout << "  • средняя длина токена: " << std::fixed << std::setprecision(2)
              << avg_token_len << " символов" << std::endl;
    
    // Показываем статистику токенизатора
    auto stats = tokenizer.stats();
    std::cout << "  • cache hit rate: " << std::fixed << std::setprecision(1)
              << (stats.cache_hit_rate() * 100) << "%" << std::endl;
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
    print_header("ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРА");
    
    // ======================================================================
    // Парсинг аргументов командной строки
    // ======================================================================
    
    bool quick_mode = false;
    bool save_binary = false;
    bool validate = false;
    std::string corpus_path = "data/corpus/train_code.txt";
    size_t vocab_size = DEFAULT_VOCAB_SIZE;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick" || arg == "-q") {
            quick_mode = true;
            vocab_size = QUICK_VOCAB_SIZE;
            std::cout << YELLOW << "Быстрый режим: размер словаря уменьшен до " 
                      << vocab_size << RESET << std::endl;
        } else if (arg == "--save-binary" || arg == "-b") {
            save_binary = true;
        } else if (arg == "--validate" || arg == "-v") {
            validate = true;
        } else if (arg == "--corpus" && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoul(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "\nИспользование: " << argv[0] << " [options]\n";
            std::cout << "  --quick, -q          Быстрый режим (меньше токенов)\n";
            std::cout << "  --save-binary, -b    Сохранить в бинарном формате\n";
            std::cout << "  --validate, -v       Валидация после обучения\n";
            std::cout << "  --corpus PATH        Путь к корпусу\n";
            std::cout << "  --vocab-size N       Размер словаря\n";
            std::cout << "  --help, -h           Показать справку\n";
            return 0;
        }
    }
    
    try {
        // ======================================================================
        // 1. ЗАГРУЗКА КОРПУСА
        // ======================================================================
        
        std::vector<std::string> corpus;
        
        // Пробуем разные пути к корпусу
        std::vector<std::string> corpus_paths = {
            corpus_path,                                           // пользовательский путь
            "../../../data/corpus/train_code.txt",                 // NS/data/corpus/train_code.txt
            "../../data/corpus/train_code.txt",                    // bpe_tokenizer_cpu/data/corpus/train_code.txt
            "../../bpe_python/data/corpus/train_code.txt",         // bpe_python/data/corpus/train_code.txt
            "../data/corpus/train_code.txt"                        // examples/data/corpus/train_code.txt
        };
        
        bool corpus_loaded = false;
        for (const auto& path : corpus_paths) {
            if (fs::exists(path)) {
                corpus = load_corpus(path);
                corpus_loaded = true;
                break;
            }
        }
        
        if (!corpus_loaded) {
            std::cout << YELLOW << "Файл не найден. Искали в:" << RESET << std::endl;
            for (const auto& path : corpus_paths) {
                std::cout << "  - " << path << std::endl;
            }
            corpus = create_demo_corpus(quick_mode ? QUICK_CORPUS_SIZE : NORMAL_CORPUS_SIZE);
        }
        
        if (corpus.empty()) {
            std::cerr << RED << "❌ Корпус пуст!" << RESET << std::endl;
            return 1;
        }
        
        // ======================================================================
        // 2. КОНФИГУРАЦИЯ ТОКЕНИЗАТОРА
        // ======================================================================
        
        TokenizerConfig config;
        config.vocab_size = vocab_size;
        config.cache_size = DEFAULT_CACHE_SIZE;
        config.byte_level = true;
        config.enable_cache = true;
        config.enable_profiling = true;
        
        std::cout << "\n" << CYAN << "🔧 Конфигурация обучения:" << RESET << std::endl;
        std::cout << "------------------------------------------------------------\n";
        std::cout << "  • размер словаря:    " << config.vocab_size << std::endl;
        std::cout << "  • byte-level:        " << (config.byte_level ? "да" : "нет") << std::endl;
        std::cout << "  • кэширование:       " << (config.enable_cache ? "да" : "нет") << std::endl;
        std::cout << "  • профилирование:    " << (config.enable_profiling ? "да" : "нет") << std::endl;
        std::cout << "  • потоков CPU:       " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "------------------------------------------------------------\n";
        
        // ======================================================================
        // 3. СОЗДАНИЕ ТОКЕНИЗАТОРА
        // ======================================================================
        
        FastBPETokenizer tokenizer(config);
        
        // ======================================================================
        // 4. ПАРАЛЛЕЛЬНОЕ ОБУЧЕНИЕ
        // ======================================================================
        
        std::cout << "\n" << CYAN << "🚀 Начало обучения..." << RESET << std::endl;
        
        {
            ScopedTimer timer("Общее время обучения");
            tokenizer.parallel_train(corpus, vocab_size);
        }
        
        // ======================================================================
        // 5. СТАТИСТИКА ПОСЛЕ ОБУЧЕНИЯ
        // ======================================================================
        
        std::cout << "\n" << CYAN << "📊 Статистика обучения:" << RESET << std::endl;
        std::cout << "  • итоговый размер словаря: " << tokenizer.vocab_size() << std::endl;
        std::cout << "  • правил слияния: " << tokenizer.merges_count() << std::endl;
        
        auto train_stats = tokenizer.stats();
        std::cout << "  • всего encode вызовов: " << train_stats.encode_calls << std::endl;
        std::cout << "  • среднее время encode: " 
                  << std::fixed << std::setprecision(3) 
                  << train_stats.avg_encode_time_ms() << " мс" << std::endl;
        
        // ======================================================================
        // 6. ВАЛИДАЦИЯ (ОПЦИОНАЛЬНО)
        // ======================================================================
        
        if (validate) {
            validate_model(tokenizer, corpus, 
                          quick_mode ? QUICK_VALIDATION_SAMPLES : VALIDATION_SAMPLES);
        }
        
        // ======================================================================
        // 7. СОХРАНЕНИЕ МОДЕЛИ
        // ======================================================================
        
        std::cout << "\n" << CYAN << "💾 Сохранение модели..." << RESET << std::endl;
        
        // Создаем директорию для моделей
        std::string model_dir = "../models/bpe_" + std::to_string(vocab_size);
        fs::create_directories(model_dir);
        
        if (save_binary) {
            std::string model_path = model_dir + "/model_trained.bin";
            {
                ScopedTimer timer("Сохранение в бинарный формат");
                tokenizer.save_binary(model_path);
            }
            
            if (fs::exists(model_path)) {
                auto file_size = fs::file_size(model_path);
                std::cout << GREEN << "  ✓ Бинарная модель: " << model_path << RESET << std::endl;
                std::cout << "    размер: " << std::fixed << std::setprecision(2)
                          << file_size / 1024.0 / 1024.0 << " МБ" << std::endl;
            }
        } else {
            std::string vocab_path = model_dir + "/cpp_vocab.json";
            std::string merges_path = model_dir + "/cpp_merges.txt";
            
            {
                ScopedTimer timer("Сохранение в текстовый формат");
                tokenizer.save(vocab_path, merges_path);
            }
            
            std::cout << GREEN << "  ✓ Словарь: " << vocab_path << RESET << std::endl;
            std::cout << GREEN << "  ✓ Слияния: " << merges_path << RESET << std::endl;
            
            if (fs::exists(vocab_path)) {
                auto vocab_size_bytes = fs::file_size(vocab_path);
                std::cout << "    размер словаря: " << std::fixed << std::setprecision(2)
                          << vocab_size_bytes / 1024.0 / 1024.0 << " МБ" << std::endl;
            }
            
            if (fs::exists(merges_path)) {
                auto merges_size_bytes = fs::file_size(merges_path);
                std::cout << "    размер слияний: " << merges_size_bytes / 1024.0 << " КБ" << std::endl;
            }
        }
        
        // ======================================================================
        // 8. ТЕСТ ЗАГРУЗКИ СОХРАНЕННОЙ МОДЕЛИ
        // ======================================================================
        
        std::cout << "\n" << CYAN << "🔄 Тест загрузки сохраненной модели..." << RESET << std::endl;
        
        FastBPETokenizer loaded_tokenizer(config);
        bool load_success = false;
        std::string model_dir_test = "../models/bpe_" + std::to_string(vocab_size);
        
        if (save_binary) {
            std::string model_path = model_dir_test + "/model_trained.bin";
            if (fs::exists(model_path)) {
                {
                    ScopedTimer timer("Загрузка бинарной модели");
                    load_success = loaded_tokenizer.load_binary(model_path);
                }
                
                if (load_success) {
                    std::cout << GREEN << "  ✓ Модель успешно загружена из бинарного файла!" << RESET << std::endl;
                    std::cout << "    размер словаря: " << loaded_tokenizer.vocab_size() << std::endl;
                }
            }
        } else {
            std::string vocab_path = model_dir_test + "/cpp_vocab.json";
            std::string merges_path = model_dir_test + "/cpp_merges.txt";
            
            if (fs::exists(vocab_path) && fs::exists(merges_path)) {
                {
                    ScopedTimer timer("Загрузка текстовой модели");
                    load_success = loaded_tokenizer.load(vocab_path, merges_path);
                }
                
                if (load_success) {
                    std::cout << GREEN << "  ✓ Модель успешно загружена из текстовых файлов!" << RESET << std::endl;
                    std::cout << "    размер словаря: " << loaded_tokenizer.vocab_size() << std::endl;
                    
                    if (loaded_tokenizer.vocab_size() == tokenizer.vocab_size()) {
                        std::cout << GREEN << "  ✓ Размер словаря совпадает с оригиналом" << RESET << std::endl;
                    } else {
                        std::cout << YELLOW << "  ⚠ Размер словаря отличается" << RESET << std::endl;
                    }
                }
            }
        }
        
        if (!load_success) {
            std::cout << YELLOW << "  ⚠ Не удалось загрузить сохраненную модель" << RESET << std::endl;
        }
        
        // ======================================================================
        // 9. ИТОГ
        // ======================================================================
        
        print_header("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО");
        
        std::cout << "\n" << CYAN << "📋 Сводка:" << RESET << std::endl;
        std::cout << "  • размер корпуса: " << corpus.size() << " примеров" << std::endl;
        std::cout << "  • размер словаря: " << tokenizer.vocab_size() << std::endl;
        std::cout << "  • правил слияния: " << tokenizer.merges_count() << std::endl;
        
        std::cout << "\n" << CYAN << "📝 Для использования обученной модели в коде:" << RESET << std::endl;
        if (!save_binary) {
            std::cout << "  FastBPETokenizer tokenizer(config);" << std::endl;
            std::cout << "  tokenizer.load(\"../models/bpe_" << vocab_size << "/cpp_vocab.json\",\n"
                      << "                \"../models/bpe_" << vocab_size << "/cpp_merges.txt\");" << std::endl;
        } else {
            std::cout << "  FastBPETokenizer tokenizer(config);" << std::endl;
            std::cout << "  tokenizer.load_binary(\"../models/bpe_" << vocab_size << "/model_trained.bin\");" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << RED << "\n❌ Ошибка: " << e.what() << RESET << std::endl;
        return 1;
    } catch (...) {
        std::cerr << RED << "\n❌ Неизвестная ошибка" << RESET << std::endl;
        return 1;
    }
    
    std::cout << "\n" << GREEN << BOLD << "✓ Программа успешно завершена!" << RESET << std::endl;
    return 0;
}