/**
 * @file bench_fast_tokenizer.cpp
 * @brief Комплексное тестирование производительности оптимизированной версии BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Набор тестов производительности с использованием Google Benchmark,
 *          фокусирующийся на возможностях FastBPETokenizer:
 * 
 *          Основные тесты:
 *          - Базовые операции encode/decode с разными размерами входных данных
 *          - Пакетная обработка (batch encoding) для оптимизации throughput
 *          - Эффективность кэширования при повторяющихся паттернах
 *          - Масштабирование в многопоточных сценариях
 *          - Влияние размера словаря на производительность (8000, 10000, 12000)
 *          - Прямое сравнение с оригинальным BPETokenizer
 * 
 *          Измеряемые метрики:
 *          - Время выполнения (микросекунды/миллисекунды)
 *          - Пропускная способность (байты/токены в секунду)
 *          - Hit rate кэша при разных размерах кэша
 *          - Ускорение относительно базовой версии
 * 
 * @note Для запуска требуется собранный проект с бенчмарками
 * @see FastBPETokenizer
 * @see BPETokenizer
 * @see TokenizerConfig
 */

#include <benchmark/benchmark.h>

#include "fast_tokenizer.hpp"
#include "bpe_tokenizer.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <array>
#include <algorithm>
#include <numeric>

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
    /** Размеры для бенчмарков */
    constexpr size_t MIN_TEXT_SIZE = 1 << 10;    // 1 КБ
    constexpr size_t MAX_TEXT_SIZE = 1 << 18;    // 256 КБ
    constexpr size_t MIN_TOKENS = 1 << 8;        // 256 токенов
    constexpr size_t MAX_TOKENS = 1 << 16;       // 65536 токенов
    constexpr size_t MIN_BATCH_SIZE = 1;
    constexpr size_t MAX_BATCH_SIZE = 64;
    constexpr size_t MIN_CACHE_SIZE = 100;
    constexpr size_t MAX_CACHE_SIZE = 10000;
    constexpr size_t MAX_THREADS = 8;
    
    /** Размеры словарей из проекта */
    constexpr std::array<size_t, 3> VOCAB_SIZES = {8000, 10000, 12000};
    constexpr size_t DEFAULT_VOCAB_SIZE = 10000;
    
    /** Параметры кэша */
    constexpr size_t DEFAULT_CACHE_SIZE = 10000;
    constexpr bool DEFAULT_BYTE_LEVEL = true;
    constexpr bool DEFAULT_ENABLE_CACHE = true;
    
    /** Размер для сравнения (100 КБ) */
    constexpr size_t COMPARISON_TEXT_SIZE = 100000;
    
    /** Количество текстов для тестов кэша */
    constexpr size_t CACHE_TEST_TEXTS = 1000;
    
    /** Количество итераций для теста памяти */
    constexpr int MEMORY_ITERATIONS = 100;
    
    /** Количество итераций для decode бенчмарков */
    constexpr int DECODE_ITERATIONS = 1000;
    
    /** Задержка для стабилизации (мс) */
    constexpr auto STABILIZATION_DELAY = std::chrono::milliseconds(10);
    
    /** Пути для поиска тестового корпуса */
    static constexpr std::array<const char*, 4> CORPUS_SEARCH_PATHS = {
        "../benchmarks/bench_data/sample_code.txt",
        "benchmarks/bench_data/sample_code.txt",
        "./bench_data/sample_code.txt",
        "../../benchmarks/bench_data/sample_code.txt"
    };
}

// ======================================================================
// Вспомогательные классы и функции
// ======================================================================

/**
 * @brief Класс для кэширования тестового корпуса
 * 
 * Обеспечивает единый доступ к тестовым данным с кэшированием
 * и автоматическим поиском файлов.
 */
class TestCorpus {
private:
    std::string cached_corpus_;
    bool loaded_ = false;
    
public:
    /**
     * @brief Загружает тестовый корпус C++ кода из файла или возвращает встроенный пример
     * 
     * Функция кэширует результат после первого вызова.
     * При ошибках не выводит сообщения, просто возвращает встроенный пример.
     * 
     * @return const std::string& Ссылка на строку с тестовым кодом
     */
    const std::string& get() {
        if (loaded_) {
            return cached_corpus_;
        }
        
        // Пробуем разные пути к файлу
        for (const auto& path : CORPUS_SEARCH_PATHS) {
            std::ifstream file(path);
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                cached_corpus_ = buffer.str();
                loaded_ = true;
                return cached_corpus_;
            }
        }
        
        // Если файл не найден, используем встроенный пример
        cached_corpus_ = R"(
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

namespace example {

template<typename T>
class Vector {
private:
    std::vector<T> data_;
    
public:
    Vector() = default;
    void push_back(const T& value) { data_.push_back(value); }
    size_t size() const { return data_.size(); }
    T& operator[](size_t index) { return data_[index]; }
    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
};

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::transform(numbers.begin(), numbers.end(), numbers.begin(),
                   [](int x) { return x * x; });
    
    for (const auto& num : numbers) {
        std::cout << num << std::endl;
    }
    
    Vector<std::string> strings;
    strings.push_back("Hello");
    strings.push_back("World");
    
    return 0;
}

}    // namespace example
)";
        
        loaded_ = true;
        return cached_corpus_;
    }
    
    /**
     * @brief Очищает кэш (для тестирования)
     */
    void clear() {
        cached_corpus_.clear();
        loaded_ = false;
    }
};

// ======================================================================
// Глобальные объекты
// ======================================================================

/** Глобальный экземпляр тестового корпуса для использования в бенчмарках */
TestCorpus g_testCorpus;

/**
 * @brief Класс для генерации тестовых корпусов с контролем разнообразия
 * 
 * Позволяет создавать наборы текстов с заданным уровнем разнообразия
 * для тестирования кэширования.
 */
class CorpusGenerator {
private:
    std::vector<std::string> templates_;
    std::mt19937 rng_;
    
public:
    CorpusGenerator() : rng_(std::random_device{}()) {
        templates_ = {
            g_testCorpus.get(),
            R"(#include <iostream>
int main() { 
    std::cout << "test" << std::endl; 
    return 0; 
})",
            R"(class MyClass {
public:
    MyClass() = default;
    void doSomething() const {}
private:
    int value_ = 42;
};)"
        };
    }
    
    /**
     * @brief Генерирует корпус текстов с заданным разнообразием
     * 
     * @param count Количество текстов для генерации
     * @param diversity Коэффициент разнообразия (0.0 - все одинаковые, 1.0 - все разные)
     * @return std::vector<std::string> Вектор сгенерированных текстов
     */
    std::vector<std::string> generate(size_t count, double diversity = 0.3) {
        std::vector<std::string> result;
        result.reserve(count);
        
        std::uniform_real_distribution<> dist(0.0, 1.0);
        std::uniform_int_distribution<> temp_idx(0, static_cast<int>(templates_.size() - 1));
        
        for (size_t i = 0; i < count; ++i) {
            std::string text;
            
            if (dist(rng_) < diversity) {
                text = templates_[temp_idx(rng_)];
            } else {
                text = templates_[0];
            }
            
            text += "\n// ID: " + std::to_string(i);
            result.push_back(std::move(text));
        }
        
        return result;
    }
};

/**
 * @brief Класс для поиска и кэширования путей к файлам моделей
 * 
 * Автоматически ищет модели в стандартных расположениях и кэширует результаты.
 * При ошибках возвращает пустой результат, не выводя сообщения.
 */
class ModelPathFinder {
public:
    struct PathCache {
        std::string vocab;
        std::string merges;
        
        /**
         * @brief Проверяет существование обоих файлов
         * @return true если оба файла существуют
         */
        bool exists() const {
            return !vocab.empty() && !merges.empty() &&
                   std::filesystem::exists(vocab) && 
                   std::filesystem::exists(merges);
        }
    };
    
private:
    std::unordered_map<size_t, PathCache> cache_;
    
    /** Возможные пути для поиска моделей */
    static constexpr std::array<const char*, 2> MODEL_LOCATIONS = {
        "../models/bpe_",
        "../../bpe_python/models/bpe_"
    };
    
    /** Имена файлов для разных версий */
    static constexpr std::array<const char*, 2> VOCAB_FILENAMES = {
        "/cpp_vocab.json",
        "/vocab.json"
    };
    
public:
    /**
     * @brief Находит пути к файлам модели для указанного размера
     * 
     * @param size Размер словаря (8000, 10000, 12000)
     * @return PathCache Структура с путями (может быть пустой)
     */
    PathCache find_for_size(size_t size) {
        auto it = cache_.find(size);
        if (it != cache_.end()) {
            return it->second;
        }
        
        PathCache result;
        
        // Проверяем все комбинации путей и имен файлов
        for (const auto& location : MODEL_LOCATIONS) {
            std::string base = std::string(location) + std::to_string(size);
            
            for (const auto& vocab_name : VOCAB_FILENAMES) {
                result.vocab = base + vocab_name;
                
                // Формируем имя для merges.txt
                size_t pos = result.vocab.find("vocab.json");
                if (pos != std::string::npos) {
                    result.merges = result.vocab;
                    result.merges.replace(pos, 10, "merges.txt");
                } else {
                    pos = result.vocab.find("cpp_vocab.json");
                    if (pos != std::string::npos) {
                        result.merges = result.vocab;
                        result.merges.replace(pos, 14, "cpp_merges.txt");
                    }
                }
                
                if (result.exists()) {
                    cache_[size] = result;
                    return result;
                }
            }
        }
        
        return PathCache{};
    }
};

/** Глобальный экземпляр для использования в бенчмарках */
ModelPathFinder g_pathFinder;

/**
 * @brief Измеряет текущее использование памяти (RSS)
 * 
 * Реализация для разных платформ:
 * - Windows: GetProcessMemoryInfo
 * - macOS:   task_info
 * - Linux:   чтение /proc/self/statm
 * 
 * @return size_t Размер памяти в байтах, 0 при ошибке
 */
size_t getCurrentRSS() {
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
    std::ifstream statm_file("/proc/self/statm");
    
    if (statm_file.is_open()) {
        long total_pages;    // Первое значение (пропускаем)
        if (statm_file >> total_pages >> rss_pages) {
            // Определяем размер страницы в зависимости от платформы
            long page_size = 0;
            
            #ifdef _SC_PAGESIZE
                page_size = sysconf(_SC_PAGESIZE);
            #elif defined(_SC_PAGE_SIZE)
                page_size = sysconf(_SC_PAGE_SIZE);
            #else
                // Значение по умолчанию для большинства систем - 4096 байт
                page_size = 4096;
            #endif
            
            return static_cast<size_t>(rss_pages) * static_cast<size_t>(page_size);
        }
    }
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief Загружает модель указанного размера
 * 
 * @tparam Tokenizer Тип токенизатора
 * @param tokenizer Ссылка на токенизатор для загрузки
 * @param size Размер словаря
 * @param state Состояние бенчмарка
 * @return true если модель успешно загружена
 */
template<typename Tokenizer>
bool load_model_by_size(Tokenizer& tokenizer, size_t size, benchmark::State& state) {
    auto paths = g_pathFinder.find_for_size(size);
    
    if (!paths.exists()) {
        state.SkipWithError(("Не удалось найти модель " + std::to_string(size)).c_str());
        return false;
    }
    
    if (!tokenizer.load(paths.vocab, paths.merges)) {
        state.SkipWithError(("Ошибка загрузки модели " + std::to_string(size)).c_str());
        return false;
    }
    
    return true;
}

/**
 * @brief Загружает модель размером 10000 (по умолчанию)
 * 
 * @tparam Tokenizer Тип токенизатора
 * @param tokenizer Ссылка на токенизатор
 * @param state Состояние бенчмарка
 * @return true если модель успешно загружена
 */
template<typename Tokenizer>
bool load_default_model(Tokenizer& tokenizer, benchmark::State& state) {
    return load_model_by_size(tokenizer, DEFAULT_VOCAB_SIZE, state);
}

/**
 * @brief Создает текст заданного размера из тестового корпуса
 * 
 * @param target_size Желаемый размер в байтах
 * @return std::string Текст указанного размера
 */
std::string create_text_of_size(size_t target_size) {
    const std::string& base = g_testCorpus.get();
    
    if (base.empty()) {
        return std::string(target_size, ' ');
    }
    
    std::string result;
    result.reserve(target_size);
    
    while (result.size() < target_size) {
        size_t chunk_size = std::min(base.size(), target_size - result.size());
        result.append(base, 0, chunk_size);
    }
    
    return result;
}

// ======================================================================
// Бенчмарки для FastTokenizer
// ======================================================================

/**
 * @brief Тест производительности encode с разными размерами входного текста
 * 
 * Измеряет скорость кодирования текстов от 1 КБ до 256 КБ.
 * Собирает статистику:
 * - Среднее время encode
 * - Пропускная способность (токены/сек)
 * - Cache hit rate
 */
static void BM_FastTokenizer_Encode(benchmark::State& state) {
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    config.enable_profiling = true;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = vocab_size;
    
    const size_t target_size = static_cast<size_t>(state.range(0));
    const std::string text = create_text_of_size(target_size);
    
    if (text.empty()) {
        state.SkipWithError("Тестовый текст пуст!");
        return;
    }
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * target_size);
    state.SetLabel("FastTokenizer encode");
    
    const auto stats = tokenizer.stats();
    state.counters["AvgEncodeTime_us"] = stats.avg_encode_time_ms() * 1000.0;
    state.counters["TokensPerSec"] = static_cast<double>(target_size) / (stats.avg_encode_time_ms() / 1000.0);
    state.counters["CacheHitRate_%"] = stats.cache_hit_rate();
}

BENCHMARK(BM_FastTokenizer_Encode)
    ->RangeMultiplier(2)
    ->Range(MIN_TEXT_SIZE, MAX_TEXT_SIZE)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест производительности decode с разным количеством токенов
 * 
 * Измеряет скорость декодирования для разного количества токенов.
 */
static void BM_FastTokenizer_Decode(benchmark::State& state) {
    static std::unique_ptr<FastBPETokenizer> tokenizer;
    static std::vector<uint32_t> tokens;
    
    if (!tokenizer) {
        tokenizer = std::make_unique<FastBPETokenizer>(
            TokenizerConfig{8000, 10000, true}
        );
        
        // Загружаем модель
        auto paths = g_pathFinder.find_for_size(10000);
        
        if (!paths.exists()) {
            state.SkipWithError("Не удалось найти модель для decode бенчмарка!");
            return;
        }
        
        if (!tokenizer->load(paths.vocab, paths.merges)) {
            state.SkipWithError("Не удалось загрузить модель для decode бенчмарка!");
            return;
        }
        
        // Загружаем тестовый текст и кодируем его
        tokens = tokenizer->encode(g_testCorpus.get());
    }
    
    size_t total_chars = 0;
    
    for (auto _ : state) {
        std::string decoded = tokenizer->decode(tokens);
        total_chars += decoded.size();
        benchmark::DoNotOptimize(decoded);
    }
    
    state.SetBytesProcessed(total_chars);
    state.SetItemsProcessed(state.iterations());
    
    state.counters["TokensPerText"] = static_cast<double>(tokens.size());
    state.counters["VocabSize"] = static_cast<double>(tokenizer->vocab_size());
    state.counters["DecodeSpeed_MB_s"] = benchmark::Counter(
        static_cast<double>(total_chars) / (1024.0 * 1024.0),
        benchmark::Counter::kIsRate
    );
}

BENCHMARK(BM_FastTokenizer_Decode)
    ->Name("BM_FastTokenizer_Decode")
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(DECODE_ITERATIONS);

/**
 * @brief Тест производительности decode с разными размерами токенов
 * 
 * Измеряет скорость декодирования для различных количеств токенов.
 */
static void BM_FastTokenizer_Decode_Sizes(benchmark::State& state) {
    const size_t num_tokens = static_cast<size_t>(state.range(0));
    
    // Создаем тестовые токены нужного размера
    std::vector<uint32_t> test_tokens;
    test_tokens.reserve(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        test_tokens.push_back(static_cast<uint32_t>(i % DEFAULT_VOCAB_SIZE));
    }
    
    TokenizerConfig config{DEFAULT_VOCAB_SIZE, DEFAULT_CACHE_SIZE, DEFAULT_BYTE_LEVEL};
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    size_t total_chars = 0;
    
    for (auto _ : state) {
        auto decoded = tokenizer.decode(test_tokens);
        total_chars += decoded.size();
        benchmark::DoNotOptimize(decoded);
    }
    
    state.SetBytesProcessed(total_chars);
    state.SetItemsProcessed(state.iterations());
    state.counters["TokensPerText"] = static_cast<double>(num_tokens);
    state.counters["TokensPerSec"] = benchmark::Counter(
        static_cast<double>(state.iterations() * num_tokens),
        benchmark::Counter::kIsRate
    );
}

BENCHMARK(BM_FastTokenizer_Decode_Sizes)
    ->Name("BM_FastTokenizer_Decode_Sizes")
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(16, 4096)
    ->Iterations(500);

/**
 * @brief Тест производительности пакетной обработки
 * 
 * Измеряет эффективность encode_batch для разных размеров батча.
 */
static void BM_FastTokenizer_EncodeBatch(benchmark::State& state) {
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = vocab_size;
    
    const size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<std::string_view> text_views;
    std::vector<std::string> text_storage;
    
    text_storage.reserve(batch_size);
    text_views.reserve(batch_size);
    
    const std::string& base = g_testCorpus.get();
    for (size_t i = 0; i < batch_size; ++i) {
        text_storage.push_back(base + " // batch " + std::to_string(i));
        text_views.push_back(text_storage.back());
    }
    
    for (auto _ : state) {
        auto batch_result = tokenizer.encode_batch(text_views);
        benchmark::DoNotOptimize(batch_result.data());
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
    state.SetLabel("FastTokenizer batch encode");
}

BENCHMARK(BM_FastTokenizer_EncodeBatch)
    ->RangeMultiplier(2)
    ->Range(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест эффективности кэширования
 * 
 * Измеряет cache hit rate при разных размерах кэша.
 */
static void BM_FastTokenizer_CacheEfficiency(benchmark::State& state) {
    const size_t cache_size = static_cast<size_t>(state.range(0));
    
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = true;
    config.cache_size = cache_size;
    config.enable_profiling = true;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = vocab_size;
    
    CorpusGenerator generator;
    auto texts = generator.generate(CACHE_TEST_TEXTS, 0.3);
    
    for (auto _ : state) {
        for (const auto& text : texts) {
            auto tokens = tokenizer.encode(text);
            benchmark::DoNotOptimize(tokens.data());
        }
    }
    
    const auto stats = tokenizer.stats();
    state.counters["CacheHitRate_%"] = stats.cache_hit_rate();
    state.counters["CacheHits"] = static_cast<double>(stats.cache_hits);
    state.counters["CacheMisses"] = static_cast<double>(stats.cache_misses);
    state.SetLabel("Cache efficiency");
}

BENCHMARK(BM_FastTokenizer_CacheEfficiency)
    ->RangeMultiplier(10)
    ->Range(MIN_CACHE_SIZE, MAX_CACHE_SIZE)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Базовый многопоточный тест
 * 
 * Измеряет производительность при параллельном encode из нескольких потоков.
 */
static void BM_FastTokenizer_Multithreaded(benchmark::State& state) {
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = vocab_size;
    
    const std::string& text = g_testCorpus.get();
    const size_t num_threads = static_cast<size_t>(state.range(0));
    
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        state.ResumeTiming();
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&tokenizer, &text]() {
                auto tokens = tokenizer.encode(text);
                benchmark::DoNotOptimize(tokens.data());
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_threads);
    state.SetLabel("Multithreaded encode");
}

BENCHMARK(BM_FastTokenizer_Multithreaded)
    ->RangeMultiplier(2)
    ->Range(1, MAX_THREADS)
    ->Unit(benchmark::kMillisecond);

/**
 * @brief Расширенный многопоточный тест
 * 
 * Более сложный сценарий: каждый поток обрабатывает несколько текстов.
 */
static void BM_FastTokenizer_Multithreaded_Advanced(benchmark::State& state) {
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const size_t num_threads = static_cast<size_t>(state.range(0));
    const size_t texts_per_thread = static_cast<size_t>(state.range(1));
    
    CorpusGenerator generator;
    auto texts = generator.generate(num_threads * texts_per_thread, 0.5);
    
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        state.ResumeTiming();
        
        for (size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&tokenizer, &texts, t, texts_per_thread]() {
                const size_t start_idx = t * texts_per_thread;
                for (size_t i = 0; i < texts_per_thread; ++i) {
                    auto tokens = tokenizer.encode(texts[start_idx + i]);
                    benchmark::DoNotOptimize(tokens.data());
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
    state.SetItemsProcessed(state.iterations() * num_threads * texts_per_thread);
    state.SetLabel(std::to_string(num_threads) + " threads x " + 
                   std::to_string(texts_per_thread) + " texts");
}

BENCHMARK(BM_FastTokenizer_Multithreaded_Advanced)
    ->Args({1, 10})
    ->Args({2, 5})
    ->Args({4, 3})
    ->Args({8, 2})
    ->Unit(benchmark::kMillisecond);

/**
 * @brief Сравнение производительности на моделях разных размеров
 * 
 * Измеряет время encode для моделей с разным размером словаря.
 */
static void BM_FastTokenizer_CompareSizes(benchmark::State& state) {
    const size_t model_size = static_cast<size_t>(state.range(0));
    
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    config.enable_profiling = true;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_model_by_size(tokenizer, model_size, state)) {
        return;
    }
    
    const auto actual_size = tokenizer.vocab_size();
    state.counters["ModelSize"] = static_cast<double>(model_size);
    state.counters["ActualVocabSize"] = static_cast<double>(actual_size);
    
    const std::string& text = g_testCorpus.get();
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
    
    state.SetLabel("Модель " + std::to_string(model_size));
    
    const auto stats = tokenizer.stats();
    state.counters["AvgEncodeTime_us"] = stats.avg_encode_time_ms() * 1000.0;
    state.counters["CacheHitRate_%"] = stats.cache_hit_rate();
}

BENCHMARK(BM_FastTokenizer_CompareSizes)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(12000)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест на утечки памяти
 * 
 * Многократно создает и уничтожает токенизаторы, отслеживая изменение RSS.
 */
static void BM_FastTokenizer_MemoryLeak(benchmark::State& state) {
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    
    const std::string& text = g_testCorpus.get();
    const size_t memory_before = getCurrentRSS();
    size_t operations = 0;
    
    for (auto _ : state) {
        auto tokenizer = std::make_unique<FastBPETokenizer>(config);
        
        if (!load_default_model(*tokenizer, state)) {
            return;
        }
        
        auto tokens = tokenizer->encode(text);
        benchmark::DoNotOptimize(tokens.data());
        
        tokenizer.reset();
        operations++;
    }
    
    std::this_thread::sleep_for(STABILIZATION_DELAY);
    const size_t memory_after = getCurrentRSS();
    
    state.counters["MemoryDelta_KB"] = static_cast<double>(memory_after - memory_before) / 1024.0;
    state.counters["Operations"] = static_cast<double>(operations);
    state.SetLabel("Memory leak test");
}

BENCHMARK(BM_FastTokenizer_MemoryLeak)->Iterations(MEMORY_ITERATIONS)->Unit(benchmark::kMillisecond);

// ======================================================================
// Бенчмарки для оригинального токенизатора
// ======================================================================

/**
 * @brief Бенчмарк оригинального токенизатора
 * 
 * Используется как baseline для сравнения с оптимизированной версией.
 */
static void BM_Original_Encode(benchmark::State& state) {
    BPETokenizer tokenizer;
    tokenizer.set_byte_level(DEFAULT_BYTE_LEVEL);
    tokenizer.set_unknown_token("<UNK>");
    
    auto paths = g_pathFinder.find_for_size(DEFAULT_VOCAB_SIZE);
    
    if (!paths.exists()) {
        state.SkipWithError("Не удалось найти модель!");
        return;
    }
    
    if (!tokenizer.load_from_files(paths.vocab, paths.merges)) {
        state.SkipWithError("Не удалось загрузить токенизатор!");
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = vocab_size;
    
    std::string text = create_text_of_size(COMPARISON_TEXT_SIZE);
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * COMPARISON_TEXT_SIZE);
    state.SetLabel("Original BPETokenizer");
}

BENCHMARK(BM_Original_Encode)->Unit(benchmark::kMillisecond);

/**
 * @brief Бенчмарк быстрого токенизатора для сравнения
 * 
 * Используется вместе с BM_Original_Encode для расчета ускорения.
 */
static void BM_Fast_Encode(benchmark::State& state) {
    TokenizerConfig config;
    config.byte_level = DEFAULT_BYTE_LEVEL;
    config.enable_cache = DEFAULT_ENABLE_CACHE;
    config.cache_size = DEFAULT_CACHE_SIZE;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = vocab_size;
    
    std::string text = create_text_of_size(COMPARISON_TEXT_SIZE);
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * COMPARISON_TEXT_SIZE);
    state.SetLabel("FastBPETokenizer");
}

BENCHMARK(BM_Fast_Encode)->Unit(benchmark::kMillisecond);

// ======================================================================
// Кастомный репортер
// ======================================================================

/**
 * @brief Расширенный репортер для Google Benchmark
 * 
 * Добавляет сводную статистику:
 * - Средний cache hit rate
 * - Ускорение относительно оригинальной версии
 */
class FastTokenizerReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        std::cout << "\n============================================================\n";
        std::cout << "ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ FAST BPE TOKENIZER";
        std::cout << "\n============================================================\n\n";
        return ConsoleReporter::ReportContext(context);
    }
    
    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);
        
        std::cout << "\nДЕТАЛЬНАЯ СТАТИСТИКА:\n";
        
        double total_cache_hits = 0;
        int cache_count = 0;
        
        for (const auto& run : reports) {
            auto it = run.counters.find("CacheHitRate_%");
            if (it != run.counters.end()) {
                total_cache_hits += it->second;
                cache_count++;
            }
        }
        
        if (cache_count > 0) {
            std::cout << "Средний Cache Hit Rate: " 
                      << std::fixed << std::setprecision(2)
                      << (total_cache_hits / cache_count) << "%\n";
        }
        
        // Вычисляем ускорение
        double original_time = 0;
        double fast_time = 0;
        
        for (const auto& run : reports) {
            if (run.run_name.str().find("Original") != std::string::npos) {
                original_time = run.real_accumulated_time;
            }
            if (run.run_name.str().find("Fast_Encode") != std::string::npos) {
                fast_time = run.real_accumulated_time;
            }
        }
        
        if (original_time > 0 && fast_time > 0) {
            double speedup = original_time / fast_time;
            std::cout << "УСКОРЕНИЕ ОТНОСИТЕЛЬНО ОРИГИНАЛА: " 
                      << std::fixed << std::setprecision(2) << speedup << "x\n";
        }
    }
};

// ======================================================================
// Точка входа
// ======================================================================

/**
 * @brief Главная функция бенчмарка
 * 
 * Инициализирует Google Benchmark, прогревает кэш и запускает тесты.
 * 
 * @param argc Количество аргументов командной строки
 * @param argv Массив аргументов командной строки
 * @return int Код возврата (0 при успехе)
 */
int main(int argc, char** argv) {
    // Прогреваем кэш перед запуском бенчмарков
    g_testCorpus.get();
    
    ::benchmark::Initialize(&argc, argv);
    
    // Используем кастомный репортер для расширенной статистики
    FastTokenizerReporter reporter;
    ::benchmark::RunSpecifiedBenchmarks(&reporter);
    
    return 0;
}