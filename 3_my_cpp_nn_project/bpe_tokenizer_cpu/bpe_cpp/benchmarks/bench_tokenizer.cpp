/**
 * @file bench_tokenizer.cpp
 * @brief Бенчмарки для базовой версии BPE токенизатора (BPETokenizer)
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Набор тестов производительности для BPETokenizer с использованием
 *          библиотеки Google Benchmark. Эти тесты служат базой для сравнения
 *          с оптимизированной версией (FastBPETokenizer).
 * 
 *          **Измеряемые операции:**
 *          - Кодирование (encode) коротких и длинных текстов
 *          - Декодирование (decode) обратно в текст
 *          - Пакетная обработка нескольких текстов одновременно
 *          - Влияние размера входного текста на производительность
 *          - Влияние размера словаря (8000, 10000, 12000) на скорость
 *          - Сравнение byte-level и обычного режимов
 * 
 *          **Метрики:**
 *          - Время выполнения (микросекунды/миллисекунды)
 *          - Количество обработанных байт
 *          - Количество обработанных элементов (для пакетной обработки)
 *          - Стандартное отклонение и коэффициент вариации
 * 
 * @note Для запуска требуются обученные модели:
 *       - Базовая модель:         bpe_8000/cpp_vocab.json, bpe_8000/cpp_merges.txt
 *       - Модели разных размеров: bpe_8000/cpp_vocab.json, bpe_10000/cpp_vocab.json,
 *                                 bpe_12000/cpp_vocab.json и соответствующие файлы слияний
 * 
 * @see BPETokenizer
 * @see bench_fast_tokenizer.cpp
 */

#include <benchmark/benchmark.h>

#include "bpe_tokenizer.hpp"
#include "utils.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <thread>
#include <filesystem>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <map>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

using namespace bpe;

// ======================================================================
// Константы и настройки
// ======================================================================

namespace {
    /** Размеры для бенчмарков */
    constexpr size_t MIN_TEXT_SIZE = 1 << 6;     // 64 байт
    constexpr size_t MAX_TEXT_SIZE = 1 << 16;    // 64 КБ
    constexpr size_t DEFAULT_VOCAB_SIZE = 10000;
    
    /** Размеры словарей из проекта */
    constexpr std::array<size_t, 3> VOCAB_SIZES = {8000, 10000, 12000};
    
    /** Количество текстов для пакетной обработки */
    constexpr int BATCH_SIZE = 10;
    
    /** Диапазоны для генерации текстов */
    constexpr size_t MIN_VAR_LEN = 100;
    constexpr size_t MAX_VAR_LEN = 10000;
    constexpr size_t VAR_TEXTS_COUNT = 100;
    
    /** Параметры по умолчанию */
    constexpr bool DEFAULT_BYTE_LEVEL = true;
    constexpr std::string_view UNKNOWN_TOKEN = "<UNK>";
    
    /** Задержка для стабилизации (мс) */
    constexpr auto STABILIZATION_DELAY = std::chrono::milliseconds(10);
    
    /** Количество итераций для тестов повторяемости */
    constexpr int REPEATABILITY_ITERATIONS = 10;
    
    /** Количество итераций для тестов памяти */
    constexpr int MEMORY_ITERATIONS = 10;
    
    /** Пути для поиска тестового корпуса */
    constexpr std::array<const char*, 4> CORPUS_SEARCH_PATHS = {
        "../benchmarks/bench_data/sample_code.txt",
        "benchmarks/bench_data/sample_code.txt",
        "./bench_data/sample_code.txt",
        "../../benchmarks/bench_data/sample_code.txt"
    };
}

// ======================================================================
// Класс для кэширования тестового корпуса
// ======================================================================

/**
 * @brief Класс для кэширования тестового корпуса C++ кода
 * 
 * Позволяет однократно загрузить тестовый корпус из файла или использовать
 * встроенный пример, и затем использовать его во всех бенчмарках без
 * повторной загрузки с диска.
 */
class TestCorpus {
private:
    std::string cached_corpus_;
    bool loaded_ = false;
    
public:
    /**
     * @brief Возвращает тестовый корпус, загружая его при первом вызове
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
#include <map>
#include <set>

namespace benchmark {
    
template<typename T, typename Alloc = std::allocator<T>>
class CustomVector {
private:
    T* data_;
    size_t size_;
    size_t capacity_;
    Alloc alloc_;
    
public:
    CustomVector() : data_(nullptr), size_(0), capacity_(0) {}
    
    explicit CustomVector(const Alloc& alloc) 
        : data_(nullptr), size_(0), capacity_(0), alloc_(alloc) {}
    
    ~CustomVector() {
        for (size_t i = 0; i < size_; ++i) {
            alloc_.destroy(data_ + i);
        }
        alloc_.deallocate(data_, capacity_);
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        data_[size_++] = value;
    }
    
    void reserve(size_t new_capacity) {
        if (new_capacity > capacity_) {
            T* new_data = alloc_.allocate(new_capacity);
            
            for (size_t i = 0; i < size_; ++i) {
                new_data[i] = std::move(data_[i]);
            }
            
            for (size_t i = 0; i < size_; ++i) {
                alloc_.destroy(data_ + i);
            }
            alloc_.deallocate(data_, capacity_);
            
            data_ = new_data;
            capacity_ = new_capacity;
        }
    }
    
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
};

int calculate_fibonacci(int n) {
    if (n <= 1) return n;
    return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2);
}

class Shape {
public:
    virtual ~Shape() = default;
    virtual double area() const = 0;
    virtual double perimeter() const = 0;
};

class Circle : public Shape {
private:
    double radius_;
public:
    explicit Circle(double r) : radius_(r) {}
    double area() const override { return 3.14159 * radius_ * radius_; }
    double perimeter() const override { return 2 * 3.14159 * radius_; }
};

class Rectangle : public Shape {
private:
    double width_, height_;
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    double area() const override { return width_ * height_; }
    double perimeter() const override { return 2 * (width_ + height_); }
};

}    // namespace benchmark

int main() {
    using namespace benchmark;
    
    CustomVector<int> vec;
    for (int i = 0; i < 1000; ++i) {
        vec.push_back(i * i);
    }
    
    std::vector<int> std_vec(vec.size());
    std::copy(&vec[0], &vec[0] + vec.size(), std_vec.begin());
    std::sort(std_vec.begin(), std_vec.end());
    
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>(5.0));
    shapes.push_back(std::make_unique<Rectangle>(3.0, 4.0));
    
    double total_area = 0;
    for (const auto& shape : shapes) {
        total_area += shape->area();
    }
    
    int fib = calculate_fibonacci(10);
    
    std::cout << "Total area: " << total_area << std::endl;
    std::cout << "Fibonacci: " << fib << std::endl;
    
    return 0;
}
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

/** Глобальный экземпляр тестового корпуса */
TestCorpus g_testCorpus;

// ======================================================================
// Класс для поиска и кэширования путей к моделям
// ======================================================================

/**
 * @brief Класс для поиска и кэширования путей к файлам моделей
 * 
 * Позволяет однократно найти файлы моделей для каждого размера словаря
 * и использовать закэшированные пути во всех бенчмарках.
 */
class ModelPathFinder {
public:
    /**
     * @brief Структура для хранения путей к файлам модели
     */
    struct Paths {
        std::string vocab;     ///< Путь к файлу словаря
        std::string merges;    ///< Путь к файлу слияний
        
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
    std::unordered_map<size_t, Paths> cache_;    ///< Кэш найденных путей
    
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
    /**
     * @brief Находит пути к файлам модели для указанного размера
     * 
     * @param size Размер словаря (8000, 10000, 12000)
     * @return Paths Структура с путями (может быть пустой, если модель не найдена)
     */
    Paths find_for_size(size_t size) {
        auto it = cache_.find(size);
        if (it != cache_.end()) {
            return it->second;
        }
        
        Paths result;
        
        // Проверяем все комбинации путей и имен файлов
        for (const auto& base : MODEL_BASES) {
            std::string full_base = std::string(base) + std::to_string(size);
            
            for (const auto& vocab_name : VOCAB_FILENAMES) {
                result.vocab = full_base + vocab_name;
                
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
        
        return Paths{};
    }
    
    /**
     * @brief Очищает кэш (для тестирования)
     */
    void clear() {
        cache_.clear();
    }
};

/** Глобальный экземпляр для поиска путей */
ModelPathFinder g_pathFinder;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загружает модель указанного размера в токенизатор
 * 
 * @tparam Tokenizer Тип токенизатора (BPETokenizer)
 * @param tokenizer Ссылка на токенизатор для загрузки
 * @param size Размер словаря (8000, 10000 или 12000)
 * @param state Состояние бенчмарка (для сообщения об ошибках)
 * @return true если модель успешно загружена, false в противном случае
 */
template<typename Tokenizer>
bool load_model_by_size(Tokenizer& tokenizer, size_t size, benchmark::State& state) {
    auto paths = g_pathFinder.find_for_size(size);
    
    if (!paths.exists()) {
        state.SkipWithError(("Модель " + std::to_string(size) + " не найдена!").c_str());
        return false;
    }
    
    tokenizer.set_byte_level(DEFAULT_BYTE_LEVEL);
    tokenizer.set_unknown_token(std::string(UNKNOWN_TOKEN));
    
    if (!tokenizer.load_from_files(paths.vocab, paths.merges)) {
        state.SkipWithError(("Ошибка загрузки модели " + std::to_string(size)).c_str());
        return false;
    }
    
    return true;
}

/**
 * @brief Загружает модель размером 10000 (по умолчанию)
 * 
 * @tparam Tokenizer Тип токенизатора
 * @param tokenizer Ссылка на токенизатор для загрузки
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
 * @param target_size Желаемый размер текста в байтах
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

/**
 * @brief Создает набор текстов переменной длины для тестирования
 * 
 * @param count Количество текстов для генерации
 * @param min_len Минимальная длина текста в байтах
 * @param max_len Максимальная длина текста в байтах
 * @return std::vector<std::string> Вектор сгенерированных текстов
 */
std::vector<std::string> create_variable_texts(size_t count, size_t min_len, size_t max_len) {
    std::vector<std::string> texts;
    texts.reserve(count);
    
    const std::string& base = g_testCorpus.get();
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> len_dis(static_cast<int>(min_len), static_cast<int>(max_len));
    std::uniform_int_distribution<> pattern_dis(0, 9);
    
    for (size_t i = 0; i < count; ++i) {
        size_t target_len = static_cast<size_t>(len_dis(gen));
        std::string text = base;
        
        while (text.size() < target_len) {
            text += " // var" + std::to_string(pattern_dis(gen)) + 
                   " = " + std::to_string(i);
        }
        text.resize(target_len);
        texts.push_back(std::move(text));
    }
    
    return texts;
}

/**
 * @brief Измеряет текущее использование памяти (RSS)
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
        long total_pages;    // Первое значение (размер программы в страницах)
        if (statm_file >> total_pages >> rss_pages) {
            long page_size = 0;
            
            #ifdef _SC_PAGESIZE
                page_size = sysconf(_SC_PAGESIZE);
            #elif defined(_SC_PAGE_SIZE)
                page_size = sysconf(_SC_PAGE_SIZE);
            #else
                page_size = 4096;    // Значение по умолчанию
            #endif
            
            return static_cast<size_t>(rss_pages) * static_cast<size_t>(page_size);
        }
    }
    return 0;
#else
    return 0;
#endif
}

// ======================================================================
// Бенчмарки базовых операций
// ======================================================================

/**
 * @brief Тест производительности encode на коротком тексте
 * 
 * Измеряет время кодирования очень короткого текста (1-2 строки кода).
 * Этот тест важен для оценки накладных расходов на вызов функций
 * и инициализацию.
 */
static void BM_EncodeShort(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(vocab_size);
    
    const std::string text = "int main() { return 0; }";
    
    size_t total_chars = 0;
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        total_chars += text.size();
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(total_chars);
    state.SetLabel("Короткий текст");
}
BENCHMARK(BM_EncodeShort)->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест производительности encode на длинном тексте
 * 
 * Измеряет время кодирования полного тестового корпуса.
 * Этот тест показывает производительность при обработке типичного объема кода.
 */
static void BM_EncodeLong(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(vocab_size);
    
    const std::string& text = g_testCorpus.get();
    
    size_t total_chars = 0;
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        total_chars += text.size();
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(total_chars);
    state.SetLabel("Длинный текст");
}
BENCHMARK(BM_EncodeLong)->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест производительности decode
 * 
 * Сначала кодирует тестовый текст для получения токенов, затем измеряет
 * время декодирования. Важно для оценки симметричности операций.
 */
static void BM_Decode(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(vocab_size);
    
    const std::string& text = g_testCorpus.get();
    const auto tokens = tokenizer.encode(text);
    state.counters["Tokens"] = static_cast<double>(tokens.size());
    
    size_t total_chars = 0;
    for (auto _ : state) {
        auto decoded = tokenizer.decode(tokens);
        total_chars += decoded.size();
        benchmark::DoNotOptimize(decoded);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(total_chars);
    state.SetItemsProcessed(state.iterations() * tokens.size());
    state.SetLabel("Декодирование");
}
BENCHMARK(BM_Decode)->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест пакетной обработки нескольких текстов
 * 
 * Измеряет производительность при кодировании нескольких небольших текстов
 * за один вызов. Важно для сценариев обучения, где данные подаются батчами.
 */
static void BM_BatchEncode(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(vocab_size);
    
    std::vector<std::string> texts;
    texts.reserve(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        texts.push_back("int main() { return " + std::to_string(i) + "; }");
    }
    
    size_t total_chars = 0;
    for (auto _ : state) {
        auto batch_result = tokenizer.encode_batch(texts);
        for (const auto& t : texts) {
            total_chars += t.size();
        }
        benchmark::DoNotOptimize(batch_result);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(total_chars);
    state.SetItemsProcessed(state.iterations() * texts.size());
    state.SetLabel("Пакетная обработка");
}
BENCHMARK(BM_BatchEncode)->Unit(benchmark::kMicrosecond);

// ======================================================================
// Параметризованные бенчмарки
// ======================================================================

/**
 * @brief Тест производительности encode на текстах разной длины
 * 
 * @param state Содержит размер текста через range(0) от 64 байт до 64 КБ
 * 
 * Позволяет оценить, как масштабируется производительность с ростом
 * размера входного текста. Ожидается линейная зависимость.
 */
static void BM_EncodeVariableLength(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const auto vocab_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(vocab_size);
    
    const size_t text_size = static_cast<size_t>(state.range(0));
    const std::string text = create_text_of_size(text_size);
    
    size_t total_chars = 0;
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        total_chars += text.size();
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(total_chars);
    state.counters["TextSize"] = static_cast<double>(text_size);
    state.SetLabel("Переменная длина");
}

BENCHMARK(BM_EncodeVariableLength)
    ->RangeMultiplier(2)
    ->Range(MIN_TEXT_SIZE, MAX_TEXT_SIZE)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Сравнение производительности при разных размерах словаря
 * 
 * @param state Содержит размер словаря через Arg(): 8000, 10000, 12000
 * 
 * Анализирует влияние размера словаря на скорость токенизации.
 * Больший словарь обычно дает меньше токенов, но требует больше
 * времени на поиск в словаре.
 */
static void BM_EncodeDifferentVocab(benchmark::State& state) {
    const size_t vocab_size = static_cast<size_t>(state.range(0));
    
    BPETokenizer tokenizer(vocab_size, DEFAULT_BYTE_LEVEL);
    
    if (!load_model_by_size(tokenizer, vocab_size, state)) {
        return;
    }
    
    const auto actual_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(actual_size);
    
    const std::string& text = g_testCorpus.get();
    const auto tokens = tokenizer.encode(text);
    state.counters["Tokens"] = static_cast<double>(tokens.size());
    
    for (auto _ : state) {
        auto result = tokenizer.encode(text);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    
    state.SetLabel("Словарь " + std::to_string(vocab_size));
}

BENCHMARK(BM_EncodeDifferentVocab)
    ->Arg(8000)
    ->Arg(10000)
    ->Arg(12000)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Сравнение byte-level и обычного режимов работы
 * 
 * @param state Содержит флаг byte_level: 0 - обычный режим, 1 - byte-level
 * 
 * Byte-level режим обрабатывает текст на уровне байтов, что позволяет
 * работать с любыми Unicode символами, но может быть медленнее из-за
 * необходимости обрабатывать многобайтовые последовательности.
 */
static void BM_CompareModes(benchmark::State& state) {
    const bool byte_level = state.range(0) == 1;
    
    BPETokenizer tokenizer(DEFAULT_VOCAB_SIZE, byte_level);
    
    if (!load_model_by_size(tokenizer, DEFAULT_VOCAB_SIZE, state)) {
        return;
    }
    
    const auto actual_size = tokenizer.vocab_size();
    state.counters["VocabSize"] = static_cast<double>(actual_size);
    state.counters["ByteLevel"] = byte_level ? 1.0 : 0.0;
    
    const std::string& text = g_testCorpus.get();
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetLabel(byte_level ? "Byte-level режим" : "Обычный режим");
}

BENCHMARK(BM_CompareModes)
    ->Arg(0)    // Обычный режим
    ->Arg(1)    // Byte-level режим
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Дополнительные тесты
// ======================================================================

/**
 * @brief Тест на повторяемость результатов
 * 
 * Запускает кодирование несколько раз и проверяет, что результаты
 * идентичны. Также вычисляет статистику времени выполнения.
 */
static void BM_Repeatability(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const std::string& text = g_testCorpus.get();
    std::vector<token_id_t> first_result;
    std::vector<double> times;
    times.reserve(REPEATABILITY_ITERATIONS);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        auto tokens = tokenizer.encode(text);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (first_result.empty()) {
            first_result = tokens;
        } else {
            if (tokens.size() != first_result.size()) {
                state.SkipWithError("Размер результатов отличается!");
                return;
            }
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (tokens[i] != first_result[i]) {
                    state.SkipWithError("Результаты отличаются!");
                    return;
                }
            }
        }
        
        const auto duration = std::chrono::duration<double, std::micro>(end - start).count();
        times.push_back(duration);
        
        benchmark::DoNotOptimize(tokens);
    }
    
    if (!times.empty()) {
        const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double sq_sum = 0.0;
        for (double val : times) {
            sq_sum += (val - mean) * (val - mean);
        }
        const double stdev = std::sqrt(sq_sum / times.size());
        
        state.counters["Mean_us"] = mean;
        state.counters["Stdev_us"] = stdev;
        state.counters["CV_percent"] = (stdev / mean) * 100.0;
    }
    
    state.SetLabel("Повторяемость");
}
BENCHMARK(BM_Repeatability)->Iterations(REPEATABILITY_ITERATIONS)->Unit(benchmark::kMicrosecond);

/**
 * @brief Тест на использование памяти
 * 
 * Измеряет изменение потребления памяти при выполнении операций.
 */
static void BM_MemoryUsage(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_default_model(tokenizer, state)) {
        return;
    }
    
    const std::string& text = g_testCorpus.get();
    
    // Прогрев
    tokenizer.encode(text);
    
    const size_t memory_before = getCurrentRSS();
    std::this_thread::sleep_for(STABILIZATION_DELAY);
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
    }
    
    std::this_thread::sleep_for(STABILIZATION_DELAY);
    const size_t memory_after = getCurrentRSS();
    
    state.counters["MemoryDelta_KB"] = static_cast<double>(memory_after - memory_before) / 1024.0;
    state.SetLabel("Использование памяти");
}
BENCHMARK(BM_MemoryUsage)->Iterations(MEMORY_ITERATIONS)->Unit(benchmark::kMillisecond);

// ======================================================================
// Кастомный репортер
// ======================================================================

/**
 * @brief Расширенный репортер для Google Benchmark
 * 
 * Добавляет сводную статистику по всем тестам:
 * - Общее время выполнения
 * - Статистика по размерам словарей
 * - Сравнение режимов работы
 * - Анализ стабильности результатов
 */
class BPETokenizerReporter : public benchmark::ConsoleReporter {
public:
    bool ReportContext(const Context& context) override {
        std::cout << "\n============================================================\n";
        std::cout << "ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ БАЗОВОГО BPE TOKENIZER";
        std::cout << "\n============================================================\n\n";
        return ConsoleReporter::ReportContext(context);
    }
    
    void ReportRuns(const std::vector<Run>& reports) override {
        ConsoleReporter::ReportRuns(reports);
        
        std::cout << "\nСВОДНАЯ СТАТИСТИКА:\n";
        
        double total_time = 0;
        int bench_count = 0;
        int memory_tests = 0;
        double total_memory_delta = 0;
        
        for (const auto& run : reports) {
            total_time += run.real_accumulated_time;
            bench_count++;
            
            // Собираем информацию о памяти из тестов MemoryUsage
            if (run.run_name.str().find("MemoryUsage") != std::string::npos) {
                auto it = run.counters.find("MemoryDelta_KB");
                if (it != run.counters.end()) {
                    total_memory_delta += it->second;
                    memory_tests++;
                }
            }
        }
        
        std::cout << "Общее время тестов: " << std::fixed << std::setprecision(2) 
                  << total_time << " ms\n";
        std::cout << "Количество тестов:  " << bench_count << "\n";
        
        if (memory_tests > 0) {
            std::cout << "Среднее использование памяти: " 
                      << std::fixed << std::setprecision(2)
                      << (total_memory_delta / memory_tests) << " КБ\n";
        }
        
        // Статистика по размерам словарей
        std::map<size_t, std::pair<double, int>> vocab_stats;
        std::map<size_t, std::pair<double, int>> token_counts;
        
        for (const auto& run : reports) {
            auto it = run.counters.find("VocabSize");
            if (it != run.counters.end()) {
                const size_t vocab_size = static_cast<size_t>(it->second);
                auto& stats = vocab_stats[vocab_size];
                stats.first += run.real_accumulated_time;
                stats.second++;
            }
            
            auto tokens_it = run.counters.find("Tokens");
            if (tokens_it != run.counters.end()) {
                const size_t vocab_size = static_cast<size_t>(run.counters.at("VocabSize"));
                auto& stats = token_counts[vocab_size];
                stats.first += tokens_it->second;
                stats.second++;
            }
        }
        
        if (!vocab_stats.empty()) {
            std::cout << "\nСреднее время по размерам словаря:\n";
            for (const auto& [size, stats] : vocab_stats) {
                const double avg_time = stats.first / stats.second;
                std::cout << "Словарь " << size << ": " 
                          << std::fixed << std::setprecision(2) << avg_time << " мс";
                if (size == DEFAULT_VOCAB_SIZE) {
                    std::cout << " (базовый)";
                }
                std::cout << "\n";
            }
        }
        
        if (!token_counts.empty()) {
            std::cout << "\nСреднее количество токенов:\n";
            for (const auto& [size, stats] : token_counts) {
                const double avg_tokens = stats.first / stats.second;
                std::cout << "Словарь " << size << ": " 
                          << std::fixed << std::setprecision(0) << avg_tokens << " токенов\n";
            }
        }
        
        // Сравнение режимов
        double normal_mode_time = 0;
        double bytelevel_mode_time = 0;
        int normal_count = 0;
        int bytelevel_count = 0;
        
        for (const auto& run : reports) {
            if (run.run_name.str().find("CompareModes") != std::string::npos) {
                auto it = run.counters.find("ByteLevel");
                if (it != run.counters.end()) {
                    if (it->second == 0) {
                        normal_mode_time += run.real_accumulated_time;
                        normal_count++;
                    } else {
                        bytelevel_mode_time += run.real_accumulated_time;
                        bytelevel_count++;
                    }
                }
            }
        }
        
        if (normal_count > 0 && bytelevel_count > 0) {
            const double normal_avg = normal_mode_time / normal_count;
            const double bytelevel_avg = bytelevel_mode_time / bytelevel_count;
            const double slowdown = bytelevel_avg / normal_avg;
            
            std::cout << "\nСравнение режимов:\n";
            std::cout << "- Обычный режим:    " << std::fixed << std::setprecision(2) 
                      << normal_avg << " мс\n";
            std::cout << "- Byte-level режим: " << bytelevel_avg << " мс\n";
            std::cout << "- Замедление:       " << std::fixed << std::setprecision(2)
                      << slowdown << "x\n";
        }
        
        // Анализ повторяемости
        double total_cv = 0;
        int cv_count = 0;
        
        for (const auto& run : reports) {
            if (run.run_name.str().find("Repeatability") != std::string::npos) {
                auto it = run.counters.find("CV_percent");
                if (it != run.counters.end()) {
                    total_cv += it->second;
                    cv_count++;
                }
            }
        }
        
        if (cv_count > 0) {
            std::cout << "\nСтабильность результатов:\n";
            std::cout << "- Средний коэффициент вариации: " 
                      << std::fixed << std::setprecision(2)
                      << (total_cv / cv_count) << "%\n";
        }
        
        std::cout << "\nРЕКОМЕНДАЦИИ:\n";
        std::cout << "- Для ускорения рассмотрите использование FastBPETokenizer\n";
        std::cout << "- Для больших текстов используйте пакетную обработку\n";
        std::cout << "- Результаты можно сравнить с bench_fast_tokenizer\n";
    }
};

// ======================================================================
// Точка входа
// ======================================================================

/**
 * @brief Главная функция бенчмарка
 * 
 * Запускает все зарегистрированные тесты и выводит результаты.
 * 
 * Поддерживаемые аргументы командной строки:
 *   --benchmark_filter=<regex>                - Запустить только тесты, соответствующие regex
 *   --benchmark_list_tests={true|false}       - Вывести список тестов
 *   --benchmark_min_time=<N>                  - Минимальное время выполнения теста (секунды)
 *   --benchmark_out=<filename>                - Сохранить результаты в файл
 *   --benchmark_out_format={json|console|csv} - Формат вывода
 *   --benchmark_repetitions=<N>               - Количество повторений каждого теста
 */
int main(int argc, char** argv) {
    // Прогреваем кэш тестового корпуса
    g_testCorpus.get();
    
    // Инициализация Google Benchmark
    ::benchmark::Initialize(&argc, argv);
    
    // Используем кастомный репортер
    BPETokenizerReporter reporter;
    ::benchmark::RunSpecifiedBenchmarks(&reporter);
    
    return 0;
}