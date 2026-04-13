/**
 * @file bench_comparison.cpp
 * @brief Прямое сравнение производительности базовой и оптимизированной версий BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Этот бенчмарк использует библиотеку Google Benchmark для всестороннего сравнения
 *          двух реализаций токенизатора:
 *          - BPETokenizer (базовая версия)       - простая, понятная реализация
 *          - FastBPETokenizer (оптимизированная) - использует SIMD, пулы памяти, кэширование
 * 
 *          Измеряемые метрики:
 *          - Скорость encode/decode для текстов разного размера (1 КБ - 256 КБ)
 *          - Производительность пакетной обработки (1-64 текста)
 *          - Влияние размера словаря (8000, 10000, 12000) на скорость
 *          - Использование оперативной памяти (RSS)
 *          - Время загрузки моделей с диска
 *          - Эффективность кэширования (FastTokenizer)
 *          - Итоговое ускорение (speedup) оптимизированной версии
 * 
 * @note Для запуска требуется:
 *       1. Собранный проект с бенчмарками (см. CMakeLists.txt в директории benchmarks/)
 *       2. Наличие файлов моделей в директории ../models/ или ../../bpe_python/models/
 *       3. Библиотека Google Benchmark (скачивается автоматически через CMake)
 * 
 * @see BPETokenizer
 * @see FastBPETokenizer
 * @see TokenizerConfig
 */

#include <benchmark/benchmark.h>

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <thread>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include <array>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using namespace bpe;

// ======================================================================
// Константы и настройки
// ======================================================================

namespace {
    /** Размеры словарей из проекта */
    constexpr std::array<size_t, 3> VOCAB_SIZES = {8000, 10000, 12000};
    constexpr size_t DEFAULT_VOCAB_SIZE = 10000;
    constexpr size_t MAX_VOCAB_SIZE = 12000;
    
    /** Размеры текста для бенчмарков (от 1 КБ до 256 КБ) */
    constexpr size_t MIN_TEXT_SIZE = 1 << 10;    // 1 КБ
    constexpr size_t MAX_TEXT_SIZE = 1 << 18;    // 256 КБ
    
    /** Размеры батчей для пакетной обработки */
    constexpr size_t MIN_BATCH_SIZE = 1;
    constexpr size_t MAX_BATCH_SIZE = 64;
    
    /** Конфигурация токенизатора по умолчанию */
    constexpr std::string_view UNKNOWN_TOKEN = "<UNK>";
    constexpr bool DEFAULT_BYTE_LEVEL = true;
    constexpr uint32_t DEFAULT_CACHE_SIZE = 10000;
    
    /** Задержка для стабилизации памяти (мс) */
    constexpr auto MEMORY_STABILIZATION_DELAY = std::chrono::milliseconds(100);
    
    /** Количество итераций для проверки воспроизводимости */
    constexpr int REPEATABILITY_ITERATIONS = 10;
    
    /** Количество итераций для измерения памяти */
    constexpr int MEMORY_ITERATIONS = 5;
    
    /** Пути для поиска тестового корпуса */
    constexpr std::array<const char*, 4> CORPUS_SEARCH_PATHS = {
        "../benchmarks/bench_data/sample_code.txt",
        "benchmarks/bench_data/sample_code.txt",
        "./bench_data/sample_code.txt",
        "../../benchmarks/bench_data/sample_code.txt"
    };
}

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загружает тестовый корпус кода C++ из файла или использует встроенный пример
 * 
 * Функция кэширует результат после первого вызова.
 * При ошибках не выводит сообщения, просто возвращает встроенный пример.
 * 
 * @return std::string Строка с тестовым кодом для бенчмарков
 */
std::string load_test_corpus() {
    static std::string cached_corpus;
    static bool loaded = false;
    
    if (loaded) {
        return cached_corpus;
    }
    
    // Пробуем разные пути к файлу
    for (const auto& path : CORPUS_SEARCH_PATHS) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
            if (!content.empty()) {
                cached_corpus = std::move(content);
                loaded = true;
                return cached_corpus;
            }
        }
    }
    
    // Встроенный тестовый код (репрезентативный пример C++ кода)
    cached_corpus = R"(
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

template<typename T>
class Vector {
private:
    T* data;
    size_t size;
    size_t capacity;
    
public:
    Vector() : data(nullptr), size(0), capacity(0) {}
    
    ~Vector() { delete[] data; }
    
    void push_back(const T& value) {
        if (size >= capacity) {
            reserve(capacity == 0 ? 1 : capacity * 2);
        }
        data[size++] = value;
    }
    
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity) {
            T* new_data = new T[new_capacity];
            for (size_t i = 0; i < size; ++i) {
                new_data[i] = data[i];
            }
            delete[] data;
            data = new_data;
            capacity = new_capacity;
        }
    }
    
    size_t get_size() const { return size; }
    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }
};

int main() {
    Vector<int> v;
    for (int i = 0; i < 1000; ++i) {
        v.push_back(i);
    }
    
    int sum = 0;
    for (size_t i = 0; i < v.get_size(); ++i) {
        sum += v[i];
    }
    
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
)";
    
    loaded = true;
    return cached_corpus;
}

/**
 * @brief Создает текст заданного размера из тестового корпуса
 * 
 * Функция многократно повторяет базовый текст до достижения целевого размера.
 * 
 * @param target_size Желаемый размер текста в байтах
 * @return std::string Текст указанного размера
 */
std::string create_text_of_size(size_t target_size) {
    static std::string base_text = load_test_corpus();
    
    if (base_text.empty()) {
        return std::string(target_size, ' ');
    }
    
    std::string result;
    result.reserve(target_size);
    
    while (result.size() < target_size) {
        size_t chunk_size = std::min(base_text.size(), target_size - result.size());
        result.append(base_text, 0, chunk_size);
    }
    
    return result;
}

// ======================================================================
// Класс ModelLoader
// ======================================================================

/**
 * @brief Управляет загрузкой моделей токенизаторов разных размеров
 * 
 * Автоматически ищет модели в нескольких стандартных расположениях.
 * При ошибках возвращает false, не выводя сообщения.
 */
class ModelLoader {
private:
    struct ModelPaths {
        std::string vocab;
        std::string merges;
        size_t size;
        
        /**
         * @brief Проверяет существование обоих файлов модели
         * @return true если оба файла существуют
         */
        bool exists() const {
            return !vocab.empty() && !merges.empty() &&
                   std::filesystem::exists(vocab) && 
                   std::filesystem::exists(merges);
        }
    };
    
    std::unordered_map<size_t, ModelPaths> size_index_;
    
    /** Базовые пути для поиска моделей */
    static constexpr std::array<const char*, 2> MODEL_BASES = {
        "../models/bpe_",
        "../../bpe_python/models/bpe_"
    };
    
    /** Имена файлов для разных версий */
    static constexpr std::array<const char*, 2> VOCAB_FILENAMES = {
        "/cpp_vocab.json",
        "/vocab.json"
    };
    
public:
    ModelLoader() {
        initialize_paths();
    }
    
    /**
     * @brief Загружает модель в базовый токенизатор
     * @param tokenizer Целевой токенизатор
     * @param size Размер словаря
     * @return true если загрузка успешна
     */
    bool load_basic_model(BPETokenizer& tokenizer, 
                          size_t size = DEFAULT_VOCAB_SIZE) const {
        auto it = size_index_.find(size);
        if (it == size_index_.end() || !it->second.exists()) {
            return false;
        }
        
        const auto& paths = it->second;
        
        tokenizer.set_byte_level(DEFAULT_BYTE_LEVEL);
        tokenizer.set_unknown_token(std::string(UNKNOWN_TOKEN));
        
        return tokenizer.load_from_files(paths.vocab, paths.merges);
    }
    
    /**
     * @brief Загружает модель в быстрый токенизатор
     * @param tokenizer Целевой токенизатор
     * @param size Размер словаря
     * @return true если загрузка успешна
     */
    bool load_fast_model(FastBPETokenizer& tokenizer, 
                         size_t size = DEFAULT_VOCAB_SIZE) const {
        auto it = size_index_.find(size);
        if (it == size_index_.end() || !it->second.exists()) {
            return false;
        }
        
        const auto& paths = it->second;
        
        TokenizerConfig config{
            static_cast<uint32_t>(size), 
            DEFAULT_CACHE_SIZE, 
            DEFAULT_BYTE_LEVEL
        };
        
        return tokenizer.load(paths.vocab, paths.merges);
    }
    
    /**
     * @brief Проверяет доступность модели заданного размера
     * @param size Размер словаря
     * @return true если модель существует
     */
    bool is_model_available(size_t size) const {
        auto it = size_index_.find(size);
        return it != size_index_.end() && it->second.exists();
    }
    
    /**
     * @brief Возвращает список доступных размеров моделей
     * @return vector размеров
     */
    std::vector<size_t> get_available_sizes() const {
        std::vector<size_t> available;
        for (size_t size : VOCAB_SIZES) {
            if (is_model_available(size)) {
                available.push_back(size);
            }
        }
        return available;
    }
    
private:
    /**
     * @brief Инициализирует пути к моделям в порядке приоритета
     */
    void initialize_paths() {
        // Проверяем все комбинации путей и имен файлов
        for (size_t size : VOCAB_SIZES) {
            for (const auto& base : MODEL_BASES) {
                std::string full_base = std::string(base) + std::to_string(size);
                
                for (const auto& vocab_name : VOCAB_FILENAMES) {
                    ModelPaths paths;
                    paths.vocab = full_base + vocab_name;
                    paths.size = size;
                    
                    // Формируем имя для merges.txt
                    size_t pos = paths.vocab.find("vocab.json");
                    if (pos != std::string::npos) {
                        paths.merges = paths.vocab;
                        paths.merges.replace(pos, 10, "merges.txt");
                    } else {
                        pos = paths.vocab.find("cpp_vocab.json");
                        if (pos != std::string::npos) {
                            paths.merges = paths.vocab;
                            paths.merges.replace(pos, 14, "cpp_merges.txt");
                        }
                    }
                    
                    if (paths.exists()) {
                        size_index_[size] = paths;
                        break;
                    }
                }
                
                if (size_index_.find(size) != size_index_.end()) {
                    break;
                }
            }
        }
    }
};

// ======================================================================
// Класс MemoryMeasurer
// ======================================================================

/**
 * @brief Измеряет использование памяти при выполнении операций
 * 
 * Использует платформозависимые вызовы для получения RSS (Resident Set Size).
 * Поддерживает Windows, macOS и Linux.
 */
class MemoryMeasurer {
public:
    struct MemoryStats {
        size_t before_rss;          ///< Память до операции (байт)
        size_t after_rss;           ///< Память после операции (байт)
        size_t diff_rss;            ///< Разница (байт)
        size_t peak_rss;            ///< Пиковое значение (байт)
        size_t operations_count;    ///< Количество операций
        
        /**
         * @brief Возвращает среднее потребление на операцию
         * @return байт на операцию
         */
        double get_average_per_op() const {
            return operations_count > 0 ? 
                   static_cast<double>(diff_rss) / operations_count : 0.0;
        }
    };
    
    /**
     * @brief Измеряет память при многократном выполнении операции
     * @param op Измеряемая операция
     * @param iterations Количество повторений
     * @return Статистика использования памяти
     */
    static MemoryStats measure_operation(const std::function<void()>& op, 
                                         size_t iterations = 1) {
        // Прогрев и стабилизация
        op();
        std::this_thread::sleep_for(MEMORY_STABILIZATION_DELAY);
        
        size_t before = getCurrentRSS();
        size_t peak = before;
        
        for (size_t i = 0; i < iterations; ++i) {
            op();
            
            size_t current = getCurrentRSS();
            peak = std::max(peak, current);
            
            if (i < iterations - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        size_t after = getCurrentRSS();
        
        return {before, after, after - before, peak, iterations};
    }
    
private:
    /**
     * @brief Получает текущий размер RSS (Resident Set Size)
     * @return Размер в байтах или 0 при ошибке
     */
    static size_t getCurrentRSS() {
    #ifdef _WIN32
        PROCESS_MEMORY_COUNTERS info;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
            return info.WorkingSetSize;
        }
        return 0;
    #elif defined(__APPLE__)
        struct mach_task_basic_info info;
        mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                      (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
            return info.resident_size;
        }
        return 0;
    #elif defined(__linux__)
        long rss_pages = 0;
        std::ifstream statm("/proc/self/statm");
        if (statm.is_open()) {
            long total_pages;
            if (statm >> total_pages >> rss_pages) {
                return static_cast<size_t>(rss_pages) * sysconf(_SC_PAGESIZE);
            }
        }
        return 0;
    #else
        return 0;
    #endif
    }
};

// ======================================================================
// Кастомный репортер
// ======================================================================

/**
 * @brief Расширенный репортер для Google Benchmark
 * 
 * Добавляет сводную таблицу с ускорением оптимизированной версии
 * по сравнению с базовой.
 */
class ComparisonReporter : public ::benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        std::cout << "\n============================================================\n";
        std::cout << "СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ БАЗОВОГО И БЫСТРОГО ТОКЕНИЗАТОРОВ";
        std::cout << "\n============================================================\n\n";
        return ConsoleReporter::ReportContext(context);
    }
    
    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);
        
        std::unordered_map<std::string, std::vector<double>> basic_times;
        std::unordered_map<std::string, std::vector<double>> fast_times;
        
        for (const auto& run : reports) {
            std::string name = run.run_name.str();
            double time = run.real_accumulated_time;
            
            if (name.find("BPETokenizer") != std::string::npos && 
                name.find("Fast") == std::string::npos) {
                basic_times[name].push_back(time);
            } else if (name.find("FastBPETokenizer") != std::string::npos) {
                fast_times[name].push_back(time);
            }
        }
        
        std::cout << "\nИТОГОВОЕ УСКОРЕНИЕ ПО КАТЕГОРИЯМ:\n";
        
        double total_basic = 0.0, total_fast = 0.0;
        int count_basic = 0, count_fast = 0;
        
        for (const auto& [name, times] : basic_times) {
            double basic_avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            
            std::string fast_name = name;
            size_t pos = fast_name.find("BPETokenizer");
            if (pos != std::string::npos) {
                fast_name.replace(pos, 12, "FastBPETokenizer");
            }
            
            auto fit = fast_times.find(fast_name);
            if (fit != fast_times.end()) {
                double fast_avg = std::accumulate(fit->second.begin(), 
                                                   fit->second.end(), 0.0) / fit->second.size();
                double speedup = basic_avg / fast_avg;
                
                std::cout << std::left << std::setw(40) << name.substr(0, 39) 
                          << ": " << std::fixed << std::setprecision(2) 
                          << speedup << "x\n";
                
                total_basic += basic_avg;
                total_fast += fast_avg;
                ++count_basic;
                ++count_fast;
            }
        }
        
        if (count_basic > 0 && count_fast > 0) {
            double overall_speedup = total_basic / total_fast;
            std::cout << std::left << std::setw(40) << "ОБЩЕЕ УСКОРЕНИЕ" 
                      << ": " << std::fixed << std::setprecision(2) 
                      << overall_speedup << "x\n";
        }
    }
};

// Глобальный экземпляр загрузчика моделей
static ModelLoader g_loader;

// ======================================================================
// Бенчмарки: Базовый токенизатор
// ======================================================================

/**
 * @brief Кодирование короткого текста базовым токенизатором
 */
static void BM_Basic_EncodeShort(benchmark::State& state) {
    BPETokenizer tokenizer;
    
    if (!g_loader.load_basic_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const std::string text = "int main() { return 0; }";
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel("BPETokenizer (short)");
}
BENCHMARK(BM_Basic_EncodeShort)->Unit(benchmark::kMicrosecond);

/**
 * @brief Кодирование текста разного размера базовым токенизатором
 */
static void BM_Basic_EncodeLong(benchmark::State& state) {
    BPETokenizer tokenizer;
    
    if (!g_loader.load_basic_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const size_t target_size = static_cast<size_t>(state.range(0));
    const std::string text = create_text_of_size(target_size);
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel("BPETokenizer (long)");
}
BENCHMARK(BM_Basic_EncodeLong)
    ->RangeMultiplier(2)
    ->Range(MIN_TEXT_SIZE, MAX_TEXT_SIZE)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Декодирование базовым токенизатором
 */
static void BM_Basic_Decode(benchmark::State& state) {
    BPETokenizer tokenizer;
    
    if (!g_loader.load_basic_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    const auto tokens = tokenizer.encode(text);
    
    for (auto _ : state) {
        auto decoded = tokenizer.decode(tokens);
        benchmark::DoNotOptimize(decoded);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel("BPETokenizer decode");
}
BENCHMARK(BM_Basic_Decode)->Unit(benchmark::kMicrosecond);

/**
 * @brief Пакетное кодирование базовым токенизатором
 */
static void BM_Basic_BatchEncode(benchmark::State& state) {
    BPETokenizer tokenizer;
    
    if (!g_loader.load_basic_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<std::string> texts;
    
    const std::string base = load_test_corpus();
    texts.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        texts.push_back(base + " // " + std::to_string(i));
    }
    
    for (auto _ : state) {
        auto results = tokenizer.encode_batch(texts);
        benchmark::DoNotOptimize(results);
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("BPETokenizer batch");
}
BENCHMARK(BM_Basic_BatchEncode)
    ->RangeMultiplier(2)
    ->Range(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Бенчмарки: Быстрый токенизатор
// ======================================================================

/**
 * @brief Кодирование короткого текста быстрым токенизатором
 */
static void BM_Fast_EncodeShort(benchmark::State& state) {
    const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!g_loader.load_fast_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const std::string text = "int main() { return 0; }";
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel("FastBPETokenizer (short)");
}
BENCHMARK(BM_Fast_EncodeShort)->Unit(benchmark::kMicrosecond);

/**
 * @brief Кодирование текста разного размера быстрым токенизатором
 */
static void BM_Fast_EncodeLong(benchmark::State& state) {
    const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!g_loader.load_fast_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const size_t target_size = static_cast<size_t>(state.range(0));
    const std::string text = create_text_of_size(target_size);
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel("FastBPETokenizer (long)");
}
BENCHMARK(BM_Fast_EncodeLong)
    ->RangeMultiplier(2)
    ->Range(MIN_TEXT_SIZE, MAX_TEXT_SIZE)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Декодирование быстрым токенизатором
 */
static void BM_Fast_Decode(benchmark::State& state) {
    const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!g_loader.load_fast_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    const auto tokens = tokenizer.encode(text);
    
    for (auto _ : state) {
        auto decoded = tokenizer.decode(tokens);
        benchmark::DoNotOptimize(decoded);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetLabel("FastBPETokenizer decode");
}
BENCHMARK(BM_Fast_Decode)->Unit(benchmark::kMicrosecond);

/**
 * @brief Пакетное кодирование быстрым токенизатором
 */
static void BM_Fast_BatchEncode(benchmark::State& state) {
    const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!g_loader.load_fast_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<std::string> texts;
    std::vector<std::string_view> views;
    
    const std::string base = load_test_corpus();
    texts.reserve(batch_size);
    views.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        texts.push_back(base + " // " + std::to_string(i));
        views.push_back(texts.back());
    }
    
    for (auto _ : state) {
        auto results = tokenizer.encode_batch(views);
        benchmark::DoNotOptimize(results);
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("FastBPETokenizer batch");
}
BENCHMARK(BM_Fast_BatchEncode)
    ->RangeMultiplier(2)
    ->Range(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Сравнительные бенчмарки с разными размерами словаря
// ======================================================================

/**
 * @brief Сравнение производительности при разных размерах словаря
 * 
 * Измеряет и базовую, и быструю версии для каждого размера словаря.
 */
static void BM_Compare_VocabSizes(benchmark::State& state) {
    const size_t vocab_size = static_cast<size_t>(state.range(0));
    
    if (!g_loader.is_model_available(vocab_size)) {
        state.SkipWithError(("Модель размером " + std::to_string(vocab_size) + 
                             " недоступна!").c_str());
        return;
    }
    
    BPETokenizer basic_tokenizer;
    if (!g_loader.load_basic_model(basic_tokenizer, vocab_size)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const TokenizerConfig config{static_cast<uint32_t>(vocab_size), 
                                  DEFAULT_CACHE_SIZE, 
                                  DEFAULT_BYTE_LEVEL};
    FastBPETokenizer fast_tokenizer(config);
    if (!g_loader.load_fast_model(fast_tokenizer, vocab_size)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    std::vector<token_id_t> basic_tokens;
    
    // Измеряем базовую версию
    for (auto _ : state) {
        basic_tokens = basic_tokenizer.encode(text);
        benchmark::DoNotOptimize(basic_tokens);
    }
    
    // Измеряем быструю версию
    for (auto _ : state) {
        auto fast_tokens = fast_tokenizer.encode(text);
        benchmark::DoNotOptimize(fast_tokens);
    }
    
    state.counters["VocabSize"] = vocab_size;
    state.counters["BasicTokens"] = basic_tokens.size();
    state.SetLabel("Сравнение vocab=" + std::to_string(vocab_size));
}
BENCHMARK(BM_Compare_VocabSizes)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(12000)
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Бенчмарки использования памяти
// ======================================================================

/**
 * @brief Измерение использования памяти базовым токенизатором
 */
static void BM_Basic_MemoryUsage(benchmark::State& state) {
    BPETokenizer tokenizer;
    
    if (!g_loader.load_basic_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    tokenizer.encode(text);    // Прогрев
    
    for (auto _ : state) {
        state.PauseTiming();
        
        auto stats = MemoryMeasurer::measure_operation([&]() {
            auto tokens = tokenizer.encode(text);
            benchmark::DoNotOptimize(tokens);
        });
        
        state.counters["MemoryDelta_KB"] = stats.diff_rss / 1024.0;
        state.counters["PeakMemory_KB"] = stats.peak_rss / 1024.0;
        
        state.ResumeTiming();
    }
    
    state.SetLabel("BPETokenizer memory");
}
BENCHMARK(BM_Basic_MemoryUsage)->Iterations(MEMORY_ITERATIONS)->Unit(benchmark::kMillisecond);

/**
 * @brief Измерение использования памяти быстрым токенизатором
 */
static void BM_Fast_MemoryUsage(benchmark::State& state) {
    const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!g_loader.load_fast_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    tokenizer.encode(text);    // Прогрев
    
    for (auto _ : state) {
        state.PauseTiming();
        
        auto stats = MemoryMeasurer::measure_operation([&]() {
            auto tokens = tokenizer.encode(text);
            benchmark::DoNotOptimize(tokens);
        });
        
        state.counters["MemoryDelta_KB"] = stats.diff_rss / 1024.0;
        state.counters["PeakMemory_KB"] = stats.peak_rss / 1024.0;
        
        state.ResumeTiming();
    }
    
    state.SetLabel("FastBPETokenizer memory");
}
BENCHMARK(BM_Fast_MemoryUsage)->Iterations(MEMORY_ITERATIONS)->Unit(benchmark::kMillisecond);

// ======================================================================
// Бенчмарки времени загрузки моделей
// ======================================================================

/**
 * @brief Время загрузки модели базовым токенизатором
 */
static void BM_Basic_LoadTime(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        BPETokenizer tokenizer;
        state.ResumeTiming();
        
        if (!g_loader.load_basic_model(tokenizer)) {
            state.SkipWithError("Не удалось загрузить базовую модель!");
            return;
        }
    }
    
    state.SetLabel("BPETokenizer load time");
}
BENCHMARK(BM_Basic_LoadTime)->Unit(benchmark::kMillisecond);

/**
 * @brief Время загрузки модели быстрым токенизатором
 */
static void BM_Fast_LoadTime(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
        FastBPETokenizer tokenizer(config);
        state.ResumeTiming();
        
        if (!g_loader.load_fast_model(tokenizer)) {
            state.SkipWithError("Не удалось загрузить fast-модель!");
            return;
        }
    }
    
    state.SetLabel("FastBPETokenizer load time");
}
BENCHMARK(BM_Fast_LoadTime)->Unit(benchmark::kMillisecond);

// ======================================================================
// Бенчмарки воспроизводимости (стабильности) результатов
// ======================================================================

/**
 * @brief Проверка воспроизводимости результатов базового токенизатора
 * 
 * Вычисляет среднее, стандартное отклонение и коэффициент вариации.
 */
static void BM_Basic_Repeatability(benchmark::State& state) {
    BPETokenizer tokenizer;
    
    if (!g_loader.load_basic_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить базовую модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    std::vector<double> measurements;
    measurements.reserve(REPEATABILITY_ITERATIONS);
    
    for (int i = 0; i < REPEATABILITY_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::micro>(end - start).count();
        measurements.push_back(duration);
        benchmark::DoNotOptimize(tokens);
    }
    
    const double mean = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    double sq_sum = 0.0;
    for (double val : measurements) {
        sq_sum += (val - mean) * (val - mean);
    }
    const double stdev = std::sqrt(sq_sum / measurements.size());
    
    state.counters["Mean_us"] = mean;
    state.counters["Stdev_us"] = stdev;
    state.counters["CV_percent"] = (stdev / mean) * 100.0;
    
    state.SetLabel("BPETokenizer repeatability");
}
BENCHMARK(BM_Basic_Repeatability)->Iterations(1)->Unit(benchmark::kMicrosecond);

/**
 * @brief Проверка воспроизводимости результатов быстрого токенизатора
 * 
 * Вычисляет среднее, стандартное отклонение и коэффициент вариации.
 */
static void BM_Fast_Repeatability(benchmark::State& state) {
    const TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!g_loader.load_fast_model(tokenizer)) {
        state.SkipWithError("Не удалось загрузить fast-модель!");
        return;
    }
    
    const std::string text = load_test_corpus();
    std::vector<double> measurements;
    measurements.reserve(REPEATABILITY_ITERATIONS);
    
    for (int i = 0; i < REPEATABILITY_ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::micro>(end - start).count();
        measurements.push_back(duration);
        benchmark::DoNotOptimize(tokens);
    }
    
    const double mean = std::accumulate(measurements.begin(), measurements.end(), 0.0) / measurements.size();
    double sq_sum = 0.0;
    for (double val : measurements) {
        sq_sum += (val - mean) * (val - mean);
    }
    const double stdev = std::sqrt(sq_sum / measurements.size());
    
    state.counters["Mean_us"] = mean;
    state.counters["Stdev_us"] = stdev;
    state.counters["CV_percent"] = (stdev / mean) * 100.0;
    
    state.SetLabel("FastBPETokenizer repeatability");
}
BENCHMARK(BM_Fast_Repeatability)->Iterations(1)->Unit(benchmark::kMicrosecond);

// ======================================================================
// Точка входа
// ======================================================================

/**
 * @brief Точка входа в программу бенчмарков
 * 
 * Инициализирует Google Benchmark, запускает все зарегистрированные
 * бенчмарки с кастомным репортером.
 * 
 * @param argc Количество аргументов командной строки
 * @param argv Массив аргументов командной строки
 * @return int Код возврата (0 при успехе)
 */
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    
    // Используем кастомный репортер для сводного сравнения
    ComparisonReporter reporter;
    ::benchmark::RunSpecifiedBenchmarks(&reporter);
    
    return 0;
}