/**
 * @file python_bindings.cpp
 * @brief Python биндинги для BPE токенизатора через pybind11
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Этот файл создает Python модуль `bpe_tokenizer_cpp`, который предоставляет
 *          высокопроизводительный доступ к C++ реализации токенизатора из Python.
 * 
 *          **Предоставляемый API:**
 *          ```python
 *          from bpe_tokenizer_cpp import FastBPETokenizer
 *          
 *          # Создание токенизатора
 *          tokenizer = FastBPETokenizer(vocab_size=8000, byte_level=True)
 *          
 *          # Загрузка модели
 *          tokenizer.load("vocab.json", "merges.txt")
 *          
 *          # Использование
 *          tokens = tokenizer.encode("int main() { return 0; }")
 *          text = tokenizer.decode(tokens)
 *          stats = tokenizer.stats
 *          ```
 * 
 * @see FastBPETokenizer
 * @see https://pybind11.readthedocs.io/
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "fast_tokenizer.hpp"
#include "simd_utils.hpp"
#include "optimized_types.hpp"

namespace py = pybind11;
using namespace bpe;

// ======================================================================
// Обертка для FastBPETokenizer
// ======================================================================

/**
 * @brief Python-совместимая обертка для FastBPETokenizer
 * 
 * Предоставляет удобный API для Python, конвертируя C++ типы
 * в Python-совместимые и обрабатывая исключения.
 */
class PyFastBPETokenizer {
private:
    FastBPETokenizer tokenizer;    ///< Внутренний C++ токенизатор

public:
    /**
     * @brief Конструктор с полным набором параметров (БЕЗ ЗНАЧЕНИЙ ПО УМОЛЧАНИЮ)
     */
    PyFastBPETokenizer(size_t vocab_size, 
                       bool byte_level, 
                       size_t cache_size,
                       bool enable_cache,
                       bool enable_profiling,
                       bool use_memory_pool,
                       int num_threads,
                       const std::string& unknown_token,
                       const std::string& pad_token,
                       const std::string& bos_token,
                       const std::string& eos_token,
                       const std::string& mask_token)
        : tokenizer(create_config(vocab_size, byte_level, cache_size, 
                                   enable_cache, enable_profiling, use_memory_pool, 
                                   num_threads, unknown_token, pad_token, 
                                   bos_token, eos_token, mask_token)) {}
    
    /**
     * @brief Конструктор по умолчанию
     */
    PyFastBPETokenizer() : tokenizer(TokenizerConfig{}) {}
    
    /**
     * @brief Конструктор только с размером словаря
     */
    explicit PyFastBPETokenizer(size_t vocab_size) 
        : tokenizer(create_config(vocab_size, true, 10000, true, false, true, 0,
                                   "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>")) {}
    
    /**
     * @brief Конструктор с размером словаря и режимом
     */
    PyFastBPETokenizer(size_t vocab_size, bool byte_level)
        : tokenizer(create_config(vocab_size, byte_level, 10000, true, false, true, 0,
                                   "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>")) {}
    
    /**
     * @brief Загрузить модель из файлов
     */
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load(vocab_path, merges_path);
    }
    
    /**
     * @brief Кодировать текст в токены
     */
    std::vector<uint32_t> encode(const std::string& text) {
        return tokenizer.encode(text);
    }
    
    /**
     * @brief Декодировать токены обратно в текст
     */
    std::string decode(const std::vector<uint32_t>& tokens) {
        return tokenizer.decode(tokens);
    }
    
    /**
     * @brief Пакетное кодирование нескольких текстов
     */
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string>& texts) {
        std::vector<std::string_view> views;
        views.reserve(texts.size());
        for (const auto& t : texts) {
            views.push_back(t);
        }
        return tokenizer.encode_batch(views);
    }
    
    // ==================== Геттеры ====================
    
    size_t vocab_size() const { return tokenizer.vocab_size(); }
    size_t merges_count() const { return tokenizer.merges_count(); }
    uint32_t unknown_id() const { return tokenizer.unknown_id(); }
    uint32_t pad_id() const { return tokenizer.pad_id(); }
    uint32_t bos_id() const { return tokenizer.bos_id(); }
    uint32_t eos_id() const { return tokenizer.eos_id(); }
    uint32_t mask_id() const { return tokenizer.mask_id(); }
    
    /**
     * @brief Получить статистику работы
     */
    py::dict stats() const {
        auto s = tokenizer.stats();
        py::dict d;
        d["encode_calls"] = s.encode_calls;
        d["decode_calls"] = s.decode_calls;
        d["cache_hits"] = s.cache_hits;
        d["cache_misses"] = s.cache_misses;
        d["total_tokens"] = s.total_tokens_processed;
        d["total_encode_time_ms"] = s.total_encode_time_ms;
        d["total_decode_time_ms"] = s.total_decode_time_ms;
        d["avg_encode_time_ms"] = s.avg_encode_time_ms();
        d["avg_decode_time_ms"] = s.avg_decode_time_ms();
        d["cache_hit_rate"] = s.cache_hit_rate();
        return d;
    }
    
    /**
     * @brief Сбросить статистику работы
     */
    void reset_stats() { tokenizer.reset_stats(); }
    
    /**
     * @brief Получить версию токенизатора
     */
    std::string version() const { return "3.4.0"; }
    
    /**
     * @brief Получить информацию о модели
     */
    std::string get_model_info() const { return tokenizer.get_model_info(); }

private:
    /**
     * @brief Вспомогательная функция для создания конфигурации
     */
    static TokenizerConfig create_config(size_t vocab_size,
                                         bool byte_level,
                                         size_t cache_size,
                                         bool enable_cache,
                                         bool enable_profiling,
                                         bool use_memory_pool,
                                         int num_threads,
                                         const std::string& unknown_token,
                                         const std::string& pad_token,
                                         const std::string& bos_token,
                                         const std::string& eos_token,
                                         const std::string& mask_token) {
        TokenizerConfig config;
        config.vocab_size = vocab_size;
        config.cache_size = cache_size;
        config.byte_level = byte_level;
        config.enable_cache = enable_cache;
        config.enable_profiling = enable_profiling;
        config.use_memory_pool = use_memory_pool;
        config.num_threads = num_threads;
        config.unknown_token = unknown_token;
        config.pad_token = pad_token;
        config.bos_token = bos_token;
        config.eos_token = eos_token;
        config.mask_token = mask_token;
        return config;
    }
};

// ======================================================================
// Определение Python модуля
// ======================================================================

/**
 * @brief Инициализация Python модуля bpe_tokenizer_cpp
 */
PYBIND11_MODULE(bpe_tokenizer_cpp, m) {
    m.doc() = "Fast BPE Tokenizer for C++ code - высокопроизводительная реализация BPE токенизатора";
    m.attr("__version__") = "3.4.0";
    
    py::class_<PyFastBPETokenizer>(m, "FastBPETokenizer")
        // ==================== Конструкторы ====================
        
        // Полный конструктор (12 параметров)
        .def(py::init<size_t, bool, size_t, bool, bool, bool, int,
                      const std::string&, const std::string&,
                      const std::string&, const std::string&,
                      const std::string&>(),
             py::arg("vocab_size"),
             py::arg("byte_level"),
             py::arg("cache_size"),
             py::arg("enable_cache"),
             py::arg("enable_profiling"),
             py::arg("use_memory_pool"),
             py::arg("num_threads"),
             py::arg("unknown_token"),
             py::arg("pad_token"),
             py::arg("bos_token"),
             py::arg("eos_token"),
             py::arg("mask_token"),
             "Создает токенизатор с полной конфигурацией")
        
        // Конструктор по умолчанию
        .def(py::init<>(), "Создает токенизатор с настройками по умолчанию")
        
        // Конструктор только с размером словаря
        .def(py::init<size_t>(), 
             py::arg("vocab_size"), 
             "Создает токенизатор с указанным размером словаря")
        
        // Конструктор с размером словаря и режимом
        .def(py::init<size_t, bool>(), 
             py::arg("vocab_size"), 
             py::arg("byte_level"),
             "Создает токенизатор с размером словаря и режимом byte-level")
        
        // ==================== Методы загрузки/сохранения ====================
        
        .def("load", &PyFastBPETokenizer::load,
             py::arg("vocab_path"), py::arg("merges_path"),
             "Загружает модель из файлов словаря и слияний")
        
        // ==================== Методы токенизации ====================
        
        .def("encode", &PyFastBPETokenizer::encode,
             py::arg("text"),
             "Кодирует текст в список ID токенов")
        
        .def("decode", &PyFastBPETokenizer::decode,
             py::arg("tokens"),
             "Декодирует список ID токенов обратно в текст")
        
        .def("encode_batch", &PyFastBPETokenizer::encode_batch,
             py::arg("texts"),
             "Кодирует несколько текстов за один вызов")
        
        // ==================== Свойства только для чтения ====================
        
        .def_property_readonly("vocab_size", &PyFastBPETokenizer::vocab_size,
             "Текущий размер словаря")
        
        .def_property_readonly("merges_count", &PyFastBPETokenizer::merges_count,
             "Количество правил слияния")
        
        .def_property_readonly("unknown_id", &PyFastBPETokenizer::unknown_id,
             "ID токена <UNK>")
        
        .def_property_readonly("pad_id", &PyFastBPETokenizer::pad_id,
             "ID токена <PAD>")
        
        .def_property_readonly("bos_id", &PyFastBPETokenizer::bos_id,
             "ID токена <BOS>")
        
        .def_property_readonly("eos_id", &PyFastBPETokenizer::eos_id,
             "ID токена <EOS>")
        
        .def_property_readonly("mask_id", &PyFastBPETokenizer::mask_id,
             "ID токена <MASK>")
        
        .def_property_readonly("stats", &PyFastBPETokenizer::stats,
             "Статистика работы токенизатора")
        
        // ==================== Дополнительные методы ====================
        
        .def("reset_stats", &PyFastBPETokenizer::reset_stats,
             "Сбрасывает статистику работы")
        
        .def("version", &PyFastBPETokenizer::version,
             "Возвращает версию токенизатора")
        
        .def("get_model_info", &PyFastBPETokenizer::get_model_info,
             "Возвращает информацию о модели")
        
        // ==================== Строковое представление ====================
        
        .def("__repr__", [](const PyFastBPETokenizer& t) {
            return "<FastBPETokenizer vocab_size=" + std::to_string(t.vocab_size()) + 
                   " merges=" + std::to_string(t.merges_count()) + ">";
        });
    
    // ==================== Функции модуля ====================
    
    m.def("version", []() { return "3.4.0"; },
          "Возвращает версию библиотеки");
    
    m.def("has_avx2", []() { return SIMDUtils::has_avx2(); },
          "Проверяет, доступны ли AVX2 оптимизации");
    
    m.def("has_avx", []() { return SIMDUtils::has_avx(); },
          "Проверяет, доступны ли AVX оптимизации");
    
    m.def("has_sse42", []() { return SIMDUtils::has_sse42(); },
          "Проверяет, доступны ли SSE4.2 оптимизации");
    
    m.def("simd_level", []() { return SIMDUtils::get_simd_level(); },
          "Возвращает текущий уровень SIMD оптимизаций");
    
    // ==================== Перенаправление вывода ====================
    
    py::add_ostream_redirect(m, "ostream_redirect");
}