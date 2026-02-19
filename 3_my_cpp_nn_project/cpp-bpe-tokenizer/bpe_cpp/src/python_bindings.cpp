/**
 * @file python_bindings.cpp
 * @brief Python биндинги для BPE токенизатора через pybind11
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Этот файл создает Python модуль bpe_tokenizer_cpp,
 *          предоставляющий доступ к быстрой C++ реализации токенизатора.
 * 
 *          Основные возможности из Python:
 *          - Создание токенизатора с настройками
 *          - Загрузка обученной модели
 *          - Токенизация текста (одиночная и пакетная)
 *          - Декодирование токенов
 *          - Получение статистики и метаданных
 * 
 * @note Требует pybind11 для сборки
 * @see FastBPETokenizer
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // Для автоматической конвертации STL контейнеров
#include <pybind11/iostream.h> // Для перенаправления C++ вывода в Python
#include "simd_utils.hpp"
#include "fast_tokenizer.hpp"

namespace py = pybind11;
using namespace bpe;

/**
 * @brief Обертка для FastBPETokenizer с Python-дружественным интерфейсом
 * 
 * Предоставляет удобный API для Python, конвертируя C++ типы
 * в Python-совместимые и обрабатывая исключения.
 */
class PyFastBPETokenizer {
private:
    FastBPETokenizer tokenizer;  ///< Внутренний C++ токенизатор

public:
    /**
     * @brief Конструктор с параметрами из Python
     * @param vocab_size Целевой размер словаря
     * @param byte_level Использовать byte-level режим
     */
    PyFastBPETokenizer(size_t vocab_size = 32000, bool byte_level = true) 
        : tokenizer(TokenizerConfig{
            .vocab_size = vocab_size,
            .cache_size = 10000,
            .byte_level = byte_level,
            .enable_cache = true,
            .enable_profiling = false
        }) {}
    
    /**
     * @brief Загрузить модель из файлов
     * @param vocab_path Путь к файлу словаря
     * @param merges_path Путь к файлу слияний
     * @return true при успешной загрузке
     */
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load(vocab_path, merges_path);
    }
    
    /**
     * @brief Закодировать текст в токены
     * @param text Входной текст
     * @return Список ID токенов
     */
    std::vector<uint32_t> encode(const std::string& text) {
        return tokenizer.encode(text);
    }
    
    /**
     * @brief Декодировать токены обратно в текст
     * @param tokens Список ID токенов
     * @return Восстановленный текст
     */
    std::string decode(const std::vector<uint32_t>& tokens) {
        return tokenizer.decode(tokens);
    }
    
    /**
     * @brief Пакетное кодирование нескольких текстов
     * @param texts Список текстов
     * @return Список списков ID токенов
     * 
     * @note Конвертирует std::string в std::string_view для эффективности
     */
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string>& texts) {
        std::vector<std::string_view> views;
        views.reserve(texts.size());
        
        for (const auto& t : texts) {
            views.push_back(t);
        }
        return tokenizer.encode_batch(views);
    }
    
    /**
     * @brief Получить размер словаря
     */
    size_t vocab_size() const {
        return tokenizer.vocab_size();
    }
    
    /**
     * @brief Получить количество правил слияния
     */
    size_t merges_count() const {
        return tokenizer.merges_count();
    }
    
    /**
     * @brief Получить ID токена для неизвестных символов
     */
    uint32_t unknown_id() const {
        return tokenizer.unknown_id();
    }
    
    /**
     * @brief Версия токенизатора
     */
    std::string version() const {
        return "1.0.0";
    }
    
    /**
     * @brief Получить статистику работы
     * @return Python словарь со статистикой
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
     * @brief Сбросить статистику
     */
    void reset_stats() {
        tokenizer.reset_stats();
    }
};

// ==================== Определение Python модуля ====================

/**
 * @brief Инициализация Python модуля bpe_tokenizer_cpp
 * 
 * Создает модуль со следующими компонентами:
 * - Класс FastBPETokenizer для работы с токенизатором
 * - Функции версионирования и информации о возможностях
 * - Перенаправление C++ вывода в Python
 */
PYBIND11_MODULE(bpe_tokenizer_cpp, m) {
    // Документация модуля
    m.doc() = R"pbdoc(
        Fast BPE Tokenizer for C++ code
        ================================
        
        Высокопроизводительная реализация BPE токенизатора на C++.
        
        Пример использования:
            from bpe_tokenizer_cpp import FastBPETokenizer
            
            # Создание токенизатора
            tokenizer = FastBPETokenizer(vocab_size=32000, byte_level=True)
            
            # Загрузка обученной модели
            tokenizer.load("vocab.json", "merges.txt")
            
            # Токенизация
            tokens = tokenizer.encode("int main() { return 0; }")
            print(f"Токены: {tokens}")
            
            # Декодирование
            text = tokenizer.decode(tokens)
            print(f"Восстановлено: {text}")
            
            # Статистика
            stats = tokenizer.stats
            print(f"Попаданий в кэш: {stats['cache_hit_rate']:.2%}")
    )pbdoc";
    
    // Версия модуля
    m.attr("__version__") = "1.0.0";
    
    // ==================== Класс токенизатора ====================
    
    py::class_<PyFastBPETokenizer>(m, "FastBPETokenizer")
        // Конструкторы
        .def(py::init<size_t, bool>(), 
             py::arg("vocab_size") = 32000,
             py::arg("byte_level") = true,
             R"pbdoc(
             Создает новый токенизатор.
             
             Args:
                 vocab_size: Целевой размер словаря (по умолчанию 32000)
                 byte_level: Использовать byte-level режим (по умолчанию True)
             )pbdoc")
        
        // Загрузка модели
        .def("load", &PyFastBPETokenizer::load,
             py::arg("vocab_path"), 
             py::arg("merges_path"),
             R"pbdoc(
             Загружает словарь и правила слияния из файлов.
             
             Args:
                 vocab_path: Путь к JSON файлу словаря
                 merges_path: Путь к текстовому файлу слияний
             
             Returns:
                 True при успешной загрузке, False при ошибке
             )pbdoc")
        
        // Кодирование
        .def("encode", &PyFastBPETokenizer::encode,
             py::arg("text"),
             R"pbdoc(
             Токенизирует текст в список ID токенов.
             
             Args:
                 text: Входной текст для токенизации
             
             Returns:
                 Список целых чисел - ID токенов
             )pbdoc")
        
        // Декодирование
        .def("decode", &PyFastBPETokenizer::decode,
             py::arg("tokens"),
             R"pbdoc(
             Декодирует токены обратно в текст.
             
             Args:
                 tokens: Список ID токенов
             
             Returns:
                 Восстановленный текст
             )pbdoc")
        
        // Пакетное кодирование
        .def("encode_batch", &PyFastBPETokenizer::encode_batch,
             py::arg("texts"),
             R"pbdoc(
             Токенизирует несколько текстов за один вызов.
             
             Args:
                 texts: Список строк для токенизации
             
             Returns:
                 Список списков ID токенов
             )pbdoc")
        
        // Свойства только для чтения
        .def_property_readonly("vocab_size", &PyFastBPETokenizer::vocab_size,
             "int: Текущий размер словаря")
        
        .def_property_readonly("merges_count", &PyFastBPETokenizer::merges_count,
             "int: Количество правил слияния")
        
        .def_property_readonly("unknown_id", &PyFastBPETokenizer::unknown_id,
             "int: ID токена <UNK>")
        
        .def_property_readonly("stats", &PyFastBPETokenizer::stats,
             "dict: Статистика работы токенизатора")
        
        // Методы
        .def("version", &PyFastBPETokenizer::version,
             "str: Версия токенизатора")
        
        .def("reset_stats", &PyFastBPETokenizer::reset_stats,
             "Сбросить статистику работы")
        
        // Строковое представление
        .def("__repr__", [](const PyFastBPETokenizer& t) {
            return "<FastBPETokenizer vocab_size=" + std::to_string(t.vocab_size()) + 
                   " merges=" + std::to_string(t.merges_count()) + ">";
        });
    
    // ==================== Функции модуля ====================
    
    m.def("version", []() { return "1.0.0"; }, 
          "str: Версия C++ библиотеки");
    
    m.def("supported_features", []() {
        std::vector<std::string> features = {
            "byte_level",
            "caching",
            "batch_processing",
            "parallel_encode",
            "simd_optimizations",
            "thread_safe"
        };
        return features;
    }, R"pbdoc(
        list: Список поддерживаемых возможностей
        
        Returns:
            Список строк с названиями поддерживаемых функций
    )pbdoc");
    
    m.def("has_avx2", []() {
        return SIMDUtils::has_avx2();
    }, "bool: Проверяет, доступны ли AVX2 оптимизации");
    
    // ==================== Обработка вывода ====================
    
    // Перенаправление std::cout и std::cerr в Python stdout/stderr
    py::add_ostream_redirect(m, "ostream_redirect");
}

/**
 * @example examples/python_example.py
 * Пример использования из Python:
 * @code
 * from bpe_tokenizer_cpp import FastBPETokenizer, supported_features, has_avx2
 * 
 * # Проверяем доступные функции
 * print("Поддерживаемые функции:", supported_features())
 * print("AVX2 доступен:", has_avx2())
 * 
 * # Создаем токенизатор
 * tok = FastBPETokenizer(vocab_size=5000, byte_level=True)
 * 
 * # Загружаем модель
 * if tok.load("vocab.json", "merges.txt"):
 *     print(f"Загружен словарь размером {tok.vocab_size}")
 *     
 *     # Тестируем
 *     code = "std::vector<int> v;"
 *     tokens = tok.encode(code)
 *     decoded = tok.decode(tokens)
 *     
 *     print(f"Исходный код: {code}")
 *     print(f"Токены: {tokens}")
 *     print(f"Восстановлено: {decoded}")
 *     print(f"Статистика: {tok.stats}")
 * @endcode
 */