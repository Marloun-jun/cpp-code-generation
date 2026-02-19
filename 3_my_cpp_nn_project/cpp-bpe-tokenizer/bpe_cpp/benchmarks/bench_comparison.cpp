/**
 * @file bench_comparison.cpp
 * @brief Бенчмарк для сравнения базовой и оптимизированной версий токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.1.0
 * 
 * @details Прямое сравнение производительности между BPETokenizer (базовая версия) 
 *          и FastBPETokenizer (оптимизированная версия с SIMD, пулами памяти и кэшированием).
 * 
 *          Измеряемые метрики:
 *          - Скорость encode для текстов разного размера (1KB - 256KB)
 *          - Скорость decode для разного количества токенов
 *          - Производительность пакетной обработки (1-64 текста)
 *          - Влияние размера словаря (1000, 5000, 10000, 32000)
 *          - Использование оперативной памяти (RSS)
 *          - Время загрузки моделей
 *          - Эффективность кэширования (FastTokenizer)
 *          - Итоговое ускорение (speedup) оптимизированной версии
 * 
 *          Результаты позволяют количественно оценить преимущества оптимизаций:
 *          - Ожидаемое ускорение encode: 5-10x
 *          - Ожидаемое ускорение decode: 3-5x
 *          - Экономия памяти: 20-40%
 *          - Эффективность кэша: до 80-90% на повторяющихся текстах
 * 
 * @note Для запуска требуется собранный проект с бенчмарками и наличие файлов моделей
 * @see BPETokenizer
 * @see FastBPETokenizer
 * @see bench_tokenizer.cpp
 * @see bench_fast_tokenizer.cpp
 */

#include <benchmark/benchmark.h>

#include "bpe_tokenizer.hpp"
#include "fast_tokenizer.hpp"

#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>

#ifdef __linux__
#include <unistd.h>
#endif

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загрузить тестовый корпус
 */
std::string load_test_corpus() {
    std::ifstream file("../benchmarks/bench_data/sample_code.txt");
    if (file.is_open()) {
        return std::string((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    }
    
    // Встроенный тестовый код
    return R"(
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
    
    ~Vector() { delete[] data; }
};

int main() {
    Vector<int> v;
    for (int i = 0; i < 1000; ++i) {
        v.push_back(i);
    }
    return 0;
}
)";
}

/**
 * @brief Загрузить модель для обоих токенизаторов
 */
bool load_models(BPETokenizer& basic, FastBPETokenizer& fast) {
    std::vector<std::pair<std::string, std::string>> paths = {
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
        {"../../models/cpp_vocab.json", "../../models/cpp_merges.txt"},
        {"vocab.json", "merges.txt"}
    };
    
    bool basic_loaded = false;
    bool fast_loaded = false;
    
    for (const auto& [vocab, merges] : paths) {
        if (!basic_loaded) {
            basic.set_byte_level(true);
            basic.set_unknown_token("<UNK>");
            if (basic.load_from_files(vocab, merges)) {
                basic_loaded = true;
                std::cout << "Базовая модель загружена: " << vocab << std::endl;
            }
        }
        
        if (!fast_loaded) {
            TokenizerConfig config{32000, 10000, true, true};
            if (fast.load(vocab, merges)) {
                fast_loaded = true;
                std::cout << "Fast модель загружена: " << vocab << std::endl;
            }
        }
        
        if (basic_loaded && fast_loaded) break;
    }
    
    return basic_loaded && fast_loaded;
}

/**
 * @brief Получить использование памяти (RSS)
 */
size_t getCurrentRSS() {
#ifdef _WIN32
    // Windows
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.WorkingSetSize;
#elif __APPLE__
    // macOS
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return 0;
    return info.resident_size;
#else
    // Linux
    long rss = 0;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (fp) {
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return 0;
        }
        fclose(fp);
        rss *= sysconf(_SC_PAGESIZE);
    }
    return rss;
#endif
}

// ======================================================================
// Бенчмарки сравнения encode
// ======================================================================

/**
 * @brief Сравнение encode на коротком тексте
 */
static void BM_Compare_EncodeShort(benchmark::State& state) {
    BPETokenizer basic;
    FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
    
    if (!load_models(basic, fast)) {
        state.SkipWithError("Failed to load models");
        return;
    }
    
    std::string text = "int main() { return 0; }";
    
    // Тестируем базовую версию
    for (auto _ : state) {
        auto tokens = basic.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    state.SetLabel("BPETokenizer (short)");
    
    // Тестируем быструю версию
    for (auto _ : state) {
        auto tokens = fast.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    state.SetLabel("FastBPETokenizer (short)");
}
BENCHMARK(BM_Compare_EncodeShort)->Unit(benchmark::kMicrosecond);

/**
 * @brief Сравнение encode на длинном тексте
 */
static void BM_Compare_EncodeLong(benchmark::State& state) {
    BPETokenizer basic;
    FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
    
    if (!load_models(basic, fast)) {
        state.SkipWithError("Failed to load models");
        return;
    }
    
    std::string text = load_test_corpus();
    size_t target_size = static_cast<size_t>(state.range(0));
    
    while (text.size() < target_size) {
        text += text;
    }
    text.resize(target_size);
    
    // Базовая версия
    for (auto _ : state) {
        auto tokens = basic.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    state.SetLabel("BPETokenizer");
    
    // Быстрая версия
    for (auto _ : state) {
        auto tokens = fast.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    state.SetLabel("FastBPETokenizer");
    
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_Compare_EncodeLong)
    ->RangeMultiplier(2)
    ->Range(1<<10, 1<<18)  // 1KB - 256KB
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Сравнение decode
// ======================================================================

/**
 * @brief Сравнение decode
 */
static void BM_Compare_Decode(benchmark::State& state) {
    BPETokenizer basic;
    FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
    
    if (!load_models(basic, fast)) {
        state.SkipWithError("Failed to load models");
        return;
    }
    
    std::string text = load_test_corpus();
    
    // Кодируем текст обоими токенизаторами
    auto basic_tokens = basic.encode(text);
    auto fast_tokens = fast.encode(text);
    
    // Базовая версия
    for (auto _ : state) {
        auto decoded = basic.decode(basic_tokens);
        benchmark::DoNotOptimize(decoded);
        benchmark::ClobberMemory();
    }
    state.SetLabel("BPETokenizer decode");
    
    // Быстрая версия
    for (auto _ : state) {
        auto decoded = fast.decode(fast_tokens);
        benchmark::DoNotOptimize(decoded);
        benchmark::ClobberMemory();
    }
    state.SetLabel("FastBPETokenizer decode");
    
    state.SetBytesProcessed(state.iterations() * text.size());
}
BENCHMARK(BM_Compare_Decode)->Unit(benchmark::kMicrosecond);

// ======================================================================
// Сравнение пакетной обработки
// ======================================================================

/**
 * @brief Сравнение пакетной обработки
 */
static void BM_Compare_BatchEncode(benchmark::State& state) {
    BPETokenizer basic;
    FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
    
    if (!load_models(basic, fast)) {
        state.SkipWithError("Failed to load models");
        return;
    }
    
    size_t batch_size = static_cast<size_t>(state.range(0));
    std::vector<std::string> texts;
    std::vector<std::string_view> views;
    
    std::string base = load_test_corpus();
    for (size_t i = 0; i < batch_size; ++i) {
        texts.push_back(base + " // " + std::to_string(i));
        views.push_back(texts.back());
    }
    
    // Базовая версия
    for (auto _ : state) {
        auto results = basic.encode_batch(texts);
        benchmark::DoNotOptimize(results);
        benchmark::ClobberMemory();
    }
    state.SetLabel("BPETokenizer batch");
    
    // Быстрая версия
    for (auto _ : state) {
        auto results = fast.encode_batch(views);
        benchmark::DoNotOptimize(results);
        benchmark::ClobberMemory();
    }
    state.SetLabel("FastBPETokenizer batch");
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_Compare_BatchEncode)
    ->RangeMultiplier(2)
    ->Range(1, 64)
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Сравнение с разными размерами словаря
// ======================================================================

/**
 * @brief Сравнение с разными размерами словаря
 */
static void BM_Compare_DifferentVocabSizes(benchmark::State& state) {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    
    BPETokenizer basic(vocab_size, true);
    FastBPETokenizer fast(TokenizerConfig{vocab_size, 10000, true, true});
    
    std::string vocab_path = "models/vocab_" + std::to_string(vocab_size) + ".json";
    std::string merges_path = "models/merges_" + std::to_string(vocab_size) + ".txt";
    
    bool basic_loaded = basic.load_from_files(vocab_path, merges_path);
    bool fast_loaded = fast.load(vocab_path, merges_path);
    
    if (!basic_loaded || !fast_loaded) {
        std::string error_msg = "Failed to load models for size " + std::to_string(vocab_size);
        state.SkipWithError(error_msg.c_str());
        return;
    }
    
    std::string text = load_test_corpus();
    
    // Базовая версия
    for (auto _ : state) {
        auto tokens = basic.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    state.SetLabel(("BPETokenizer vocab=" + std::to_string(vocab_size)).c_str());
    
    // Быстрая версия
    for (auto _ : state) {
        auto tokens = fast.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    state.SetLabel(("FastBPETokenizer vocab=" + std::to_string(vocab_size)).c_str());
}
BENCHMARK(BM_Compare_DifferentVocabSizes)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(32000)
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Сравнение использования памяти
// ======================================================================

/**
 * @brief Сравнение использования памяти
 */
static void BM_Compare_MemoryUsage(benchmark::State& state) {
    BPETokenizer basic;
    FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
    
    if (!load_models(basic, fast)) {
        state.SkipWithError("Failed to load models");
        return;
    }
    
    std::string text = load_test_corpus();
    
    state.PauseTiming();
    size_t basic_mem_before = getCurrentRSS();
    std::vector<token_id_t> basic_tokens;
    state.ResumeTiming();
    
    // Базовая версия
    for (auto _ : state) {
        basic_tokens = basic.encode(text);
        benchmark::DoNotOptimize(basic_tokens);
    }
    
    state.PauseTiming();
    size_t basic_mem_after = getCurrentRSS();
    state.counters["BasicMemoryKB"] = (basic_mem_after - basic_mem_before) / 1024;
    
    size_t fast_mem_before = getCurrentRSS();
    std::vector<uint32_t> fast_tokens;
    state.ResumeTiming();
    
    // Быстрая версия
    for (auto _ : state) {
        fast_tokens = fast.encode(text);
        benchmark::DoNotOptimize(fast_tokens);
    }
    
    state.PauseTiming();
    size_t fast_mem_after = getCurrentRSS();
    state.counters["FastMemoryKB"] = (fast_mem_after - fast_mem_before) / 1024;
    
    if (basic_mem_after - basic_mem_before > 0) {
        state.counters["MemorySavingsPercent"] = 100.0 * (1.0 - static_cast<double>(fast_mem_after - fast_mem_before) / (basic_mem_after - basic_mem_before));
    }
    state.ResumeTiming();
}
BENCHMARK(BM_Compare_MemoryUsage)->Unit(benchmark::kMillisecond);

// ======================================================================
// Сравнение скорости загрузки моделей
// ======================================================================

/**
 * @brief Сравнение времени загрузки моделей
 */
static void BM_Compare_LoadTime(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        BPETokenizer basic;
        FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
        state.ResumeTiming();
        
        // Загрузка базовой версии
        auto start = std::chrono::high_resolution_clock::now();
        bool basic_loaded = basic.load_from_files("models/cpp_vocab.json", "models/cpp_merges.txt");
        auto end = std::chrono::high_resolution_clock::now();
        
        if (!basic_loaded) {
            state.SkipWithError("Failed to load basic model");
            return;
        }
        
        auto basic_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.counters["BasicLoadTime_us"] = basic_duration.count();
        
        // Загрузка быстрой версии
        start = std::chrono::high_resolution_clock::now();
        bool fast_loaded = fast.load("models/cpp_vocab.json", "models/cpp_merges.txt");
        end = std::chrono::high_resolution_clock::now();
        
        if (!fast_loaded) {
            state.SkipWithError("Failed to load fast model");
            return;
        }
        
        auto fast_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        state.counters["FastLoadTime_us"] = fast_duration.count();
        
        if (fast_duration.count() > 0) {
            state.counters["LoadSpeedup"] = static_cast<double>(basic_duration.count()) / fast_duration.count();
        }
    }
}
BENCHMARK(BM_Compare_LoadTime)->Unit(benchmark::kMillisecond);

// ======================================================================
// Сводный отчет
// ======================================================================

/**
 * @brief Сводный отчет со всеми метриками
 */
static void BM_Compare_Summary(benchmark::State& state) {
    BPETokenizer basic;
    FastBPETokenizer fast(TokenizerConfig{32000, 10000, true, true});
    
    if (!load_models(basic, fast)) {
        state.SkipWithError("Failed to load models");
        return;
    }
    
    std::string text = load_test_corpus();
    
    // Измеряем базовую версию
    auto start = std::chrono::high_resolution_clock::now();
    auto basic_tokens = basic.encode(text);
    auto basic_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    // Измеряем быструю версию
    start = std::chrono::high_resolution_clock::now();
    auto fast_tokens = fast.encode(text);
    auto fast_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    
    state.counters["BasicTokens"] = basic_tokens.size();
    state.counters["FastTokens"] = fast_tokens.size();
    state.counters["BasicTime_us"] = basic_time;
    state.counters["FastTime_us"] = fast_time;
    
    if (fast_time > 0) {
        state.counters["Speedup_x"] = static_cast<double>(basic_time) / fast_time;
    }
    
    state.counters["BasicVocabSize"] = basic.vocab_size();
    state.counters["FastVocabSize"] = fast.vocab_size();
    
    auto fast_stats = fast.stats();
    state.counters["FastCacheHits"] = fast_stats.cache_hits;
    state.counters["FastCacheMisses"] = fast_stats.cache_misses;
    
    if (fast_stats.cache_hits + fast_stats.cache_misses > 0) {
        state.counters["FastCacheHitRate"] = 100.0 * fast_stats.cache_hits / 
            (fast_stats.cache_hits + fast_stats.cache_misses);
    }
    
    state.SetLabel("Сводное сравнение");
}
BENCHMARK(BM_Compare_Summary)->Unit(benchmark::kMillisecond);

// ======================================================================
// Точка входа
// ======================================================================

BENCHMARK_MAIN();