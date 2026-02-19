/**
 * @file bench_tokenizer.cpp
 * @brief Бенчмарки для базовой версии BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Набор тестов производительности для BPETokenizer:
 *          - Кодирование коротких и длинных текстов
 *          - Декодирование
 *          - Пакетная обработка
 *          - Сравнение разных размеров входных данных
 * 
 * @note Для запуска требуется обученная модель в папке models/
 * @see BPETokenizer
 */

#include <benchmark/benchmark.h>

#include "bpe_tokenizer.hpp"
#include "utils.hpp"

#include <fstream>
#include <random>
#include <thread>

using namespace bpe;

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Загрузить тестовый корпус из файла или использовать встроенный
 * @return Строка с тестовым C++ кодом
 */
static std::string load_test_corpus() {
    // Пробуем загрузить из файла
    std::ifstream file("bench_data/sample_code.txt");
    if (file.is_open()) {
        return std::string((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    }
    
    // Тестовый код по умолчанию (более разнообразный)
    return R"(
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
            
            void push_back(const T& value) {
                if (size_ >= capacity_) {
                    reserve(capacity_ == 0 ? 1 : capacity_ * 2);
                }
                data_[size_++] = value;
            }
            
            void reserve(size_t new_capacity) {
                if (new_capacity > capacity_) {
                    T* new_data = alloc_.allocate(new_capacity);
                    
                    // Перемещаем существующие элементы
                    for (size_t i = 0; i < size_; ++i) {
                        new_data[i] = std::move(data_[i]);
                    }
                    
                    // Освобождаем старую память
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
            
            ~CustomVector() {
                for (size_t i = 0; i < size_; ++i) {
                    alloc_.destroy(data_ + i);
                }
                alloc_.deallocate(data_, capacity_);
            }
        };
        
        // Функция для тестирования
        int calculate_fibonacci(int n) {
            if (n <= 1) return n;
            return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2);
        }
        
        // Класс с виртуальными функциями
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
        
        } // namespace benchmark
        
        int main() {
            using namespace benchmark;
            
            // Тестируем вектор
            CustomVector<int> vec;
            for (int i = 0; i < 1000; ++i) {
                vec.push_back(i * i);
            }
            
            // Используем алгоритмы
            std::vector<int> std_vec(vec.size());
            std::copy(&vec[0], &vec[0] + vec.size(), std_vec.begin());
            
            // Сортируем
            std::sort(std_vec.begin(), std_vec.end());
            
            // Работа с полиморфизмом
            std::vector<std::unique_ptr<Shape>> shapes;
            shapes.push_back(std::make_unique<Circle>(5.0));
            shapes.push_back(std::make_unique<Rectangle>(3.0, 4.0));
            
            double total_area = 0;
            for (const auto& shape : shapes) {
                total_area += shape->area();
            }
            
            // Вычисления
            int fib = calculate_fibonacci(10);
            
            std::cout << "Total area: " << total_area << std::endl;
            std::cout << "Fibonacci: " << fib << std::endl;
            
            return 0;
        }
    )";
}

/**
 * @brief Загрузить модель с проверкой существования файлов
 */
static bool load_tokenizer(BPETokenizer& tokenizer, benchmark::State& state) {
    // Пробуем разные пути к файлам модели
    const std::vector<std::pair<std::string, std::string>> paths = {
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
        {"../../models/cpp_vocab.json", "../../models/cpp_merges.txt"},
        {"vocab.json", "merges.txt"}
    };
    
    for (const auto& [vocab_path, merges_path] : paths) {
        if (tokenizer.load_from_files(vocab_path, merges_path)) {
            return true;
        }
    }
    
    state.SkipWithError("Не удалось загрузить модель токенизатора");
    return false;
}

/**
 * @brief Создать тексты разной длины
 */
static std::vector<std::string> create_variable_texts(size_t count, size_t min_len, size_t max_len) {
    std::vector<std::string> texts;
    texts.reserve(count);
    
    std::string base = load_test_corpus();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> len_dis(min_len, max_len);
    
    for (size_t i = 0; i < count; ++i) {
        size_t target_len = len_dis(gen);
        std::string text = base;
        while (text.size() < target_len) {
            text += " // " + std::to_string(i);
        }
        text.resize(target_len);
        texts.push_back(text);
    }
    
    return texts;
}

// ======================================================================
// Бенчмарки
// ======================================================================

/**
 * @brief Кодирование короткого текста (1-2 строки кода)
 */
static void BM_EncodeShort(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_tokenizer(tokenizer, state)) {
        return;
    }
    
    std::string text = "int main() { return 0; }";
    
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
BENCHMARK(BM_EncodeShort);

/**
 * @brief Кодирование длинного текста (весь тестовый корпус)
 */
static void BM_EncodeLong(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_tokenizer(tokenizer, state)) {
        return;
    }
    
    std::string text = load_test_corpus();
    
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
BENCHMARK(BM_EncodeLong);

/**
 * @brief Декодирование токенов обратно в текст
 */
static void BM_Decode(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_tokenizer(tokenizer, state)) {
        return;
    }
    
    std::string text = load_test_corpus();
    auto tokens = tokenizer.encode(text);
    
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
BENCHMARK(BM_Decode);

/**
 * @brief Пакетное кодирование нескольких текстов
 */
static void BM_BatchEncode(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_tokenizer(tokenizer, state)) {
        return;
    }
    
    std::vector<std::string> texts;
    texts.reserve(10);
    for (int i = 0; i < 10; ++i) {
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
    state.SetLabel("Пакетная обработка (10 текстов)");
}
BENCHMARK(BM_BatchEncode);

/**
 * @brief Кодирование текстов разной длины (параметризованный тест)
 */
static void BM_EncodeVariableLength(benchmark::State& state) {
    BPETokenizer tokenizer;
    if (!load_tokenizer(tokenizer, state)) {
        return;
    }
    
    size_t text_size = static_cast<size_t>(state.range(0));
    std::string text = load_test_corpus();
    
    while (text.size() < text_size) {
        text += text;
    }
    text.resize(text_size);
    
    size_t total_chars = 0;
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        total_chars += text.size();
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetBytesProcessed(total_chars);
    state.counters["TextSize"] = text_size;
    state.SetLabel("Переменная длина");
}

BENCHMARK(BM_EncodeVariableLength)
    ->RangeMultiplier(2)
    ->Range(1<<6, 1<<16)  // от 64 байт до 64KB
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Кодирование с разным размером словаря
 */
static void BM_EncodeDifferentVocab(benchmark::State& state) {
    size_t vocab_size = static_cast<size_t>(state.range(0));
    
    // Создаем токенизатор с нужным размером словаря
    BPETokenizer tokenizer(vocab_size, true);
    
    // Пробуем загрузить соответствующую модель
    std::string vocab_path = "models/vocab_" + std::to_string(vocab_size) + ".json";
    std::string merges_path = "models/merges_" + std::to_string(vocab_size) + ".txt";
    
    if (!tokenizer.load_from_files(vocab_path, merges_path)) {
        // Если нет специальной модели, используем стандартную
        if (!tokenizer.load_from_files("models/cpp_vocab.json", "models/cpp_merges.txt")) {
            state.SkipWithError("Не удалось загрузить модель");
            return;
        }
    }
    
    std::string text = load_test_corpus();
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.counters["VocabSize"] = vocab_size;
    state.SetLabel("Размер словаря: " + std::to_string(vocab_size));
}

BENCHMARK(BM_EncodeDifferentVocab)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(32000)
    ->Unit(benchmark::kMicrosecond);

/**
 * @brief Сравнение byte-level и обычного режима
 */
static void BM_CompareModes(benchmark::State& state) {
    bool byte_level = state.range(0) == 1;
    
    BPETokenizer tokenizer(32000, byte_level);
    if (!load_tokenizer(tokenizer, state)) {
        return;
    }
    
    std::string text = load_test_corpus();
    
    for (auto _ : state) {
        auto tokens = tokenizer.encode(text);
        benchmark::DoNotOptimize(tokens);
        benchmark::ClobberMemory();
    }
    
    state.SetLabel(byte_level ? "Byte-level режим" : "Обычный режим");
}

BENCHMARK(BM_CompareModes)
    ->Arg(0)  // Обычный режим
    ->Arg(1)  // Byte-level режим
    ->Unit(benchmark::kMicrosecond);

// ======================================================================
// Точка входа
// ======================================================================

BENCHMARK_MAIN();