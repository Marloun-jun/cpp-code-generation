/**
 * @file bench_fast_tokenizer.cpp
 * @brief Бенчмарки для сравнения производительности BPE токенизаторов
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Набор тестов производительности с использованием Google Benchmark:
 *          - Сравнение fast_tokenizer и оригинального bpe_tokenizer
 *          - Тесты encode/decode с разными размерами входных данных
 *          - Пакетная обработка
 *          - Эффективность кэширования
 *          - Многопоточное использование
 * 
 * @note Для запуска: ./bench_fast_tokenizer
 * @see fast_tokenizer.hpp
 * @see bpe_tokenizer.hpp
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

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загрузить тестовый корпус из файла или создать пример
 * @return Строка с тестовым C++ кодом
 */
std::string load_test_corpus() {
    // Пробуем загрузить из файла
    std::ifstream file("../benchmarks/bench_data/sample_code.txt");
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    // Если файл не найден, используем встроенный пример
    std::cout << "sample_code.txt не найден, использую встроенный пример" << std::endl;
    
    return R"(
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

namespace example {

template<typename T>
class Vector {
public:
    Vector() = default;
    void push_back(const T& value) { data_.push_back(value); }
    size_t size() const { return data_.size(); }
    
private:
    std::vector<T> data_;
};

int main() {
    // Создаем вектор чисел
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    // Применяем алгоритмы
    std::transform(numbers.begin(), numbers.end(), numbers.begin(),
                   [](int x) { return x * x; });
    
    // Выводим результаты
    for (const auto& num : numbers) {
        std::cout << "Number: " << num << std::endl;
    }
    
    // Работа с пользовательским классом
    Vector<std::string> strings;
    strings.push_back("Hello");
    strings.push_back("World");
    
    std::cout << "Vector size: " << strings.size() << std::endl;
    
    return 0;
}

} // namespace example
)";
}

/**
 * @brief Создать большой корпус для тестирования
 * @param size Количество копий базового кода
 * @return Вектор строк с кодом
 */
std::vector<std::string> create_large_corpus(size_t size) {
    std::vector<std::string> corpus;
    corpus.reserve(size);
    
    std::string base_code = load_test_corpus();
    
    for (size_t i = 0; i < size; ++i) {
        corpus.push_back(base_code + " // ID: " + std::to_string(i));
    }
    
    return corpus;
}

/**
 * @brief Создать тексты с повторяющимися паттернами для теста кэша
 * @param count Количество текстов
 * @return Вектор строк
 */
std::vector<std::string> create_cache_test_texts(size_t count) {
    std::vector<std::string> texts;
    texts.reserve(count);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9);
    
    for (size_t i = 0; i < count; ++i) {
        texts.push_back("int var" + std::to_string(dis(gen)) + " = " + 
                       std::to_string(i) + "; // comment " + std::to_string(i % 5));
    }
    
    return texts;
}

/**
 * @brief Загрузить модель с проверкой
 * @param tokenizer Ссылка на токенизатор
 * @param state Состояние бенчмарка
 * @return true если успешно
 */
template<typename Tokenizer>
bool load_model(Tokenizer& tokenizer, benchmark::State& state) {
    // Пробуем разные пути к файлам модели
    const std::vector<std::pair<std::string, std::string>> paths = {
        {"../../bpe/vocab.json", "../../bpe/merges.txt"},
        {"../models/vocab.json", "../models/merges.txt"},
        {"models/vocab.json", "models/merges.txt"},
        {"vocab.json", "merges.txt"}
    };
    
    for (const auto& [vocab_path, merges_path] : paths) {
        if (tokenizer.load(vocab_path, merges_path)) {
            std::cout << "Загружена модель: " << vocab_path << std::endl;
            return true;
        }
    }
    
    state.SkipWithError("Не удалось загрузить файлы модели");
    return false;
}

// ======================================================================
// Бенчмарки для FastTokenizer
// ======================================================================

/**
 * @brief Тест производительности encode с разными размерами текста
 */
static void BM_FastTokenizer_Encode(benchmark::State& state) {
    TokenizerConfig config;
    config.vocab_size = 32000;
    config.cache_size = 10000;
    config.byte_level = true;
    config.enable_profiling = true;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_model(tokenizer, state)) {
        return;
    }
    
    // Подготавливаем текст нужного размера
    std::string text = load_test_corpus();
    size_t target_size = static_cast<size_t>(state.range(0));
    
    while (text.size() < target_size) {
        text += text;
    }
    text.resize(target_size);
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * target_size);
    state.SetLabel("FastTokenizer encode");
    
    // Добавляем метрики
    auto stats = tokenizer.stats();
    state.counters["AvgEncodeTime_us"] = stats.avg_encode_time_ms() * 1000;
    state.counters["TokensPerSec"] = target_size / (stats.avg_encode_time_ms() / 1000);
}

BENCHMARK(BM_FastTokenizer_Encode)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<18)  // от 1KB до 256KB
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест производительности decode с разным количеством токенов
 */
static void BM_FastTokenizer_Decode(benchmark::State& state) {
    TokenizerConfig config;
    config.vocab_size = 32000;
    config.cache_size = 10000;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_model(tokenizer, state)) {
        return;
    }
    
    // Подготавливаем токены
    std::string text = load_test_corpus();
    auto tokens = tokenizer.encode(text);
    
    size_t target_tokens = static_cast<size_t>(state.range(0));
    while (tokens.size() < target_tokens) {
        tokens.insert(tokens.end(), tokens.begin(), tokens.end());
    }
    tokens.resize(target_tokens);
    
    for (auto _ : state) {
        auto decoded = tokenizer.decode(tokens);
        benchmark::DoNotOptimize(decoded.data());
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(state.iterations() * text.size());
    state.SetItemsProcessed(state.iterations() * target_tokens);
    state.SetLabel("FastTokenizer decode");
}

BENCHMARK(BM_FastTokenizer_Decode)
    ->RangeMultiplier(2)
    ->Range(1<<8, 1<<16)  // от 256 до 65536 токенов
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест пакетной обработки
 */
static void BM_FastTokenizer_EncodeBatch(benchmark::State& state) {
    TokenizerConfig config;
    config.vocab_size = 32000;
    config.cache_size = 10000;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_model(tokenizer, state)) {
        return;
    }
    
    size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<std::string_view> text_views;
    std::vector<std::string> text_storage;
    
    text_storage.reserve(batch_size);
    text_views.reserve(batch_size);
    
    std::string base = load_test_corpus();
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
    ->Range(1, 64)  // от 1 до 64 текстов
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест эффективности кэширования
 */
static void BM_FastTokenizer_CacheEfficiency(benchmark::State& state) {
    size_t cache_size = static_cast<size_t>(state.range(0));
    
    TokenizerConfig config;
    config.vocab_size = 32000;
    config.cache_size = cache_size;
    config.enable_cache = true;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_model(tokenizer, state)) {
        return;
    }
    
    auto texts = create_cache_test_texts(1000);
    
    for (auto _ : state) {
        for (const auto& text : texts) {
            auto tokens = tokenizer.encode(text);
            benchmark::DoNotOptimize(tokens.data());
        }
    }
    
    auto stats = tokenizer.stats();
    state.counters["CacheHitRate"] = stats.cache_hit_rate();
    state.counters["CacheHits"] = stats.cache_hits;
    state.counters["CacheMisses"] = stats.cache_misses;
    state.SetLabel("Cache efficiency");
}

BENCHMARK(BM_FastTokenizer_CacheEfficiency)
    ->RangeMultiplier(10)
    ->Range(100, 10000)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Многопоточный тест
 */
static void BM_FastTokenizer_Multithreaded(benchmark::State& state) {
    TokenizerConfig config;
    config.vocab_size = 32000;
    config.cache_size = 10000;
    config.enable_cache = true;
    
    FastBPETokenizer tokenizer(config);
    
    if (!load_model(tokenizer, state)) {
        return;
    }
    
    std::string text = load_test_corpus();
    size_t num_threads = static_cast<size_t>(state.range(0));
    
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
    ->Range(1, 8)
    ->Unit(benchmark::kMillisecond);

// ======================================================================
// Сравнение с оригинальным токенизатором
// ======================================================================

/**
 * @brief Сравнение производительности с оригинальным BPETokenizer
 */
static void BM_CompareWithOriginal(benchmark::State& state) {
    // Загружаем оригинальный токенизатор
    BPETokenizer original(32000, true);
    bool original_loaded = original.load_from_files("../../bpe/vocab.json", "../../bpe/merges.txt");
    
    if (!original_loaded) {
        state.SkipWithError("Не удалось загрузить оригинальный токенизатор");
        return;
    }
    
    // Загружаем быстрый токенизатор
    TokenizerConfig config;
    config.vocab_size = 32000;
    config.cache_size = 10000;
    
    FastBPETokenizer fast(config);
    bool fast_loaded = fast.load("../../bpe/vocab.json", "../../bpe/merges.txt");
    
    if (!fast_loaded) {
        state.SkipWithError("Не удалось загрузить быстрый токенизатор");
        return;
    }
    
    // Подготавливаем текст
    std::string text = load_test_corpus();
    const size_t target_size = 100000;
    while (text.size() < target_size) {
        text += text;
    }
    text.resize(target_size);
    
    // Тестируем оригинальный
    state.SetLabel("Original BPETokenizer");
    for (auto _ : state) {
        auto tokens = original.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
    
    // Переключаемся на быстрый
    state.SetLabel("FastBPETokenizer");
    for (auto _ : state) {
        auto tokens = fast.encode(text);
        benchmark::DoNotOptimize(tokens.data());
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_CompareWithOriginal)
    ->Unit(benchmark::kMillisecond);

// ======================================================================
// Точка входа
// ======================================================================

BENCHMARK_MAIN();