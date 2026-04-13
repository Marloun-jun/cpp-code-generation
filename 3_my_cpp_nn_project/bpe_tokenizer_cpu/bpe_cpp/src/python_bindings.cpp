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
 *          **Архитектура биндингов:**
 *          ┌──────────────────┐    ┌───────────┐    ┌──────────────────┐
 *          │    Python код    │ -> │  pybind11 │ -> │     C++ класс    │
 *          │ tokenizer.encode │    │  обертка  │    │ FastBPETokenizer │
 *          └──────────────────┘    └───────────┘    └──────────────────┘
 * 
 *          **Предоставляемый API:**
 *          @code{.py}
 *          from bpe_tokenizer_cpp import FastBPETokenizer
 *          
 *          # Создание с параметрами по умолчанию
 *          tokenizer = FastBPETokenizer()
 *          
 *          # Или с кастомными параметрами
 *          tokenizer = FastBPETokenizer(
 *              vocab_size=10000,
 *              byte_level=True,
 *              cache_size=10000,
 *              enable_cache=True
 *          )
 *          
 *          # Загрузка модели
 *          tokenizer.load("vocab.json", "merges.txt")
 *          
 *          # Использование
 *          tokens = tokenizer.encode("int main() { return 0; }")
 *          text = tokenizer.decode(tokens)
 *          stats = tokenizer.stats
 *          
 *          print(f"Hit rate: {stats['cache_hit_rate']:.2%}")
 *          @endcode
 * 
 *          **Преимущества использования C++ из Python:**
 *          - Скорость        - До 50x быстрее чистой Python реализации
 *          - Память          - Эффективное использование без копирования
 *          - Многопоточность - Полная поддержка из Python
 * 
 * @note Требует pybind11 (устанавливается через CMake)
 * @see FastBPETokenizer
 * @see https://pybind11.readthedocs.io/
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "fast_tokenizer.hpp"
#include "optimized_types.hpp"
#include "simd_utils.hpp"

namespace py = pybind11;
using namespace bpe;

// ============================================================================
// Python-совместимая обертка для FastBPETokenizer
// ============================================================================

/**
 * @brief Обертка для использования FastBPETokenizer из Python
 * 
 * Основные функции:
 * - Конвертация C++ исключений в Python исключения
 * - Преобразование типов (std::vector ↔ list, std::string ↔ str)
 * - Управление временем жизни объектов
 * - Предоставление удобного Python API
 */
class PyFastBPETokenizer {
private:
    FastBPETokenizer tokenizer;    ///< Внутренний C++ токенизатор

public:
    // ========================================================================
    // Конструкторы
    // ========================================================================

    /**
     * @brief Конструктор по умолчанию (с настройками по умолчанию)
     */
    PyFastBPETokenizer() : tokenizer(TokenizerConfig{}) {}

    /**
     * @brief Конструктор только с размером словаря
     * @param vocab_size Целевой размер словаря
     */
    explicit PyFastBPETokenizer(size_t vocab_size)
        : tokenizer(create_config(vocab_size, true, 10000, true, false, true, 0,
                                   "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>")) {}

    /**
     * @brief Конструктор с размером словаря и режимом byte-level
     * @param vocab_size Целевой размер словаря
     * @param byte_level Использовать byte-level режим
     */
    PyFastBPETokenizer(size_t vocab_size, bool byte_level)
        : tokenizer(create_config(vocab_size, byte_level, 10000, true, false, true, 0,
                                   "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>")) {}

    /**
     * @brief Полный конструктор со всеми параметрами
     * 
     * @param vocab_size Размер словаря
     * @param byte_level Byte-level режим
     * @param cache_size Размер кэша
     * @param enable_cache Включить кэширование
     * @param enable_profiling Включить профилирование
     * @param use_memory_pool Использовать пул памяти
     * @param num_threads Количество потоков (0 = auto)
     * @param unknown_token Токен неизвестных символов
     * @param pad_token Токен паддинга
     * @param bos_token Токен начала последовательности
     * @param eos_token Токен конца последовательности
     * @param mask_token Токен маски
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

    // ========================================================================
    // Основные методы токенизации
    // ========================================================================

    /**
     * @brief Загрузить модель из файлов
     * @param vocab_path Путь к JSON словарю
     * @param merges_path Путь к TXT слияниям
     * @return true при успешной загрузке
     */
    bool load(const std::string& vocab_path, const std::string& merges_path) {
        return tokenizer.load(vocab_path, merges_path);
    }

    /**
     * @brief Кодировать текст в последовательность ID токенов
     * @param text Входной текст
     * @return std::vector<uint32_t> Вектор ID токенов
     */
    std::vector<uint32_t> encode(const std::string& text) {
        return tokenizer.encode(text);
    }

    /**
     * @brief Декодировать последовательность ID обратно в текст
     * @param tokens Вектор ID токенов
     * @return std::string Восстановленный текст
     */
    std::string decode(const std::vector<uint32_t>& tokens) {
        return tokenizer.decode(tokens);
    }

    /**
     * @brief Пакетное кодирование нескольких текстов
     * @param texts Список текстов для кодирования
     * @return std::vector<std::vector<uint32_t>> Список результатов
     * 
     * @note Использует параллельную обработку (OpenMP) для ускорения
     */
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string>& texts) {
        std::vector<std::string_view> views;
        views.reserve(texts.size());
        for (const auto& t : texts) {
            views.push_back(t);
        }
        return tokenizer.encode_batch(views);
    }

    // ========================================================================
    // Геттеры для доступа к информации о модели
    // ========================================================================

    /// @return Текущий размер словаря
    size_t vocab_size() const { return tokenizer.vocab_size(); }

    /// @return Количество правил слияния
    size_t merges_count() const { return tokenizer.merges_count(); }

    /// @return ID токена <UNK>
    uint32_t unknown_id() const { return tokenizer.unknown_id(); }

    /// @return ID токена <PAD>
    uint32_t pad_id() const { return tokenizer.pad_id(); }

    /// @return ID токена <BOS>
    uint32_t bos_id() const { return tokenizer.bos_id(); }

    /// @return ID токена <EOS>
    uint32_t eos_id() const { return tokenizer.eos_id(); }

    /// @return ID токена <MASK>
    uint32_t mask_id() const { return tokenizer.mask_id(); }

    /**
     * @brief Получить статистику работы токенизатора
     * @return py::dict Словарь со статистикой
     * 
     * **Ключи словаря:**
     * - encode_calls         - Количество вызовов encode
     * - decode_calls         - Количество вызовов decode
     * - cache_hits           - Попадания в кэш
     * - cache_misses         - Промахи кэша
     * - total_tokens         - Всего обработано токенов
     * - total_encode_time_ms - Общее время encode (мс)
     * - total_decode_time_ms - Общее время decode (мс)
     * - avg_encode_time_ms   - Среднее время encode (мс)
     * - avg_decode_time_ms   - Среднее время decode (мс)
     * - cache_hit_rate       - Процент попаданий в кэш (0-1)
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
     * @return std::string Версия в формате "major.minor.patch"
     */
    std::string version() const { return "3.4.0"; }

    /**
     * @brief Получить информацию о модели
     * @return std::string Многострочное описание
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

// ============================================================================
// Определение Python модуля
// ============================================================================

/**
 * @brief Инициализация Python модуля bpe_tokenizer_cpp
 * 
 * Эта функция вызывается при импорте модуля в Python:
 * @code{.py}
 * import bpe_tokenizer_cpp
 * @endcode
 */
PYBIND11_MODULE(bpe_tokenizer_cpp, m) {
    // Документация модуля
    m.doc() = R"pbdoc(
        Fast BPE Tokenizer for C++ code
        ================================
        
        Высокопроизводительная реализация BPE токенизатора на C++ с Python биндингами.
        
        Пример использования:
            from bpe_tokenizer_cpp import FastBPETokenizer
            
            # Создание токенизатора
            tokenizer = FastBPETokenizer(vocab_size=10000, byte_level=True)
            
            # Загрузка обученной модели
            tokenizer.load("vocab.json", "merges.txt")
            
            # Кодирование текста
            tokens = tokenizer.encode("int main() { return 0; }")
            print(f"Токены: {tokens}")
            
            # Декодирование обратно
            text = tokenizer.decode(tokens)
            print(f"Декодировано: {text}")
            
            # Статистика производительности
            stats = tokenizer.stats
            print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
    )pbdoc";

    // Версия модуля
    m.attr("__version__") = "3.4.0";

    // ========================================================================
    // Определение класса FastBPETokenizer
    // ========================================================================
    
    py::class_<PyFastBPETokenizer>(m, "FastBPETokenizer", R"pbdoc(
        Высокопроизводительный BPE токенизатор на C++.
        
        Поддерживает:
        - Byte-level режим для корректной работы с Unicode
        - Кэширование результатов для повторяющихся текстов
        - SIMD оптимизации (AVX2, AVX, SSE4.2)
        - Параллельную обработку батчей
        - Сбор статистики производительности
    )pbdoc")

        // ====================================================================
        // Конструкторы
        // ====================================================================
        
        .def(py::init<>(), R"pbdoc(
            Создает токенизатор с настройками по умолчанию:
            - vocab_size:      10000
            - byte_level:      True
            - cache_size:      10000
            - enable_cache:    True
            - use_memory_pool: True
            - num_threads:     0 (auto)
        )pbdoc")
        
        .def(py::init<size_t>(), py::arg("vocab_size"),
             R"pbdoc(Создает токенизатор с указанным размером словаря.
             
             Args:
                 vocab_size: Целевой размер словаря (количество токенов)
             
             Остальные параметры - по умолчанию:
             - byte_level:   True
             - cache_size:   10000
             - enable_cache: True
        )pbdoc")
        
        .def(py::init<size_t, bool>(), 
             py::arg("vocab_size"), py::arg("byte_level"),
             R"pbdoc(Создает токенизатор с размером словаря и режимом byte-level.
             
             Args:
                 vocab_size: Целевой размер словаря
                 byte_level: True  - обрабатывать UTF-8 как байты (рекомендуется),
                             False - как символы (только для ASCII)
        )pbdoc")
        
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
             py::arg("unknown_token") = "<UNK>",
             py::arg("pad_token") = "<PAD>",
             py::arg("bos_token") = "<BOS>",
             py::arg("eos_token") = "<EOS>",
             py::arg("mask_token") = "<MASK>",
             R"pbdoc(Создает токенизатор с полной конфигурацией.
             
             Args:
                 vocab_size:       Размер словаря
                 byte_level:       Byte-level режим
                 cache_size:       Размер кэша (количество записей)
                 enable_cache:     Включить кэширование
                 enable_profiling: Включить профилирование
                 use_memory_pool:  Использовать пул памяти
                 num_threads:      Количество потоков (0 = auto)
                 unknown_token:    Токен для неизвестных символов
                 pad_token:        Токен для паддинга
                 bos_token:        Токен начала последовательности
                 eos_token:        Токен конца последовательности
                 mask_token:       Токен маски
        )pbdoc")

        // ====================================================================
        // Методы
        // ====================================================================
        
        .def("load", &PyFastBPETokenizer::load,
             py::arg("vocab_path"), py::arg("merges_path"),
             R"pbdoc(Загружает модель из файлов.
             
             Args:
                 vocab_path:  Путь к JSON файлу словаря
                 merges_path: Путь к TXT файлу слияний
             
             Returns:
                 True при успешной загрузке, False при ошибке
             
             Пример:
                 success = tokenizer.load("models/vocab.json", "models/merges.txt")
                 if not success:
                     print("Ошибка загрузки модели!")
        )pbdoc")
        
        .def("encode", &PyFastBPETokenizer::encode,
             py::arg("text"),
             R"pbdoc(Кодирует текст в список ID токенов.
             
             Args:
                 text: Входной текст (строка)
             
             Returns:
                 Список целых чисел - ID токенов
             
             Пример:
                 tokens = tokenizer.encode("int main() { return 0; }")
                 print(tokens)    # [42, 17, 35, ...]
        )pbdoc")
        
        .def("decode", &PyFastBPETokenizer::decode,
             py::arg("tokens"),
             R"pbdoc(Декодирует список ID токенов обратно в текст.
             
             Args:
                 tokens: Список ID токенов
             
             Returns:
                 Восстановленный текст
             
             Пример:
                 text = tokenizer.decode([42, 17, 35])
                 print(text)    # "int main() { return 0; }"
        )pbdoc")
        
        .def("encode_batch", &PyFastBPETokenizer::encode_batch,
             py::arg("texts"),
             R"pbdoc(Кодирует несколько текстов за один вызов.
             
             Args:
                 texts: Список строк для кодирования
             
             Returns:
                 Список списков ID токенов
             
             Примечание:
                 Использует параллельную обработку (OpenMP) для ускорения.
                 Для больших батчей (более 100 текстов) значительно быстрее
                 последовательных вызовов encode().
        )pbdoc")

        // ====================================================================
        // Свойства
        // ====================================================================
        
        .def_property_readonly("vocab_size", &PyFastBPETokenizer::vocab_size,
             R"pbdoc(Текущий размер словаря (количество токенов).)pbdoc")
        
        .def_property_readonly("merges_count", &PyFastBPETokenizer::merges_count,
             R"pbdoc(Количество правил слияния.)pbdoc")
        
        .def_property_readonly("unknown_id", &PyFastBPETokenizer::unknown_id,
             R"pbdoc(ID токена <UNK> (неизвестный символ).)pbdoc")
        
        .def_property_readonly("pad_id", &PyFastBPETokenizer::pad_id,
             R"pbdoc(ID токена <PAD> (padding для батчей).)pbdoc")
        
        .def_property_readonly("bos_id", &PyFastBPETokenizer::bos_id,
             R"pbdoc(ID токена <BOS> (beginning of sequence).)pbdoc")
        
        .def_property_readonly("eos_id", &PyFastBPETokenizer::eos_id,
             R"pbdoc(ID токена <EOS> (end of sequence).)pbdoc")
        
        .def_property_readonly("mask_id", &PyFastBPETokenizer::mask_id,
             R"pbdoc(ID токена <MASK> (masked language modeling).)pbdoc")
        
        .def_property_readonly("stats", &PyFastBPETokenizer::stats,
             R"pbdoc(Статистика работы токенизатора.
             
             Returns:
                 Словарь со следующими ключами:
                 - encode_calls         - Количество вызовов encode
                 - decode_calls         - Количество вызовов decode
                 - cache_hits           - Попадания в кэш
                 - cache_misses         - Промахи кэша
                 - total_tokens         - Всего обработано токенов
                 - total_encode_time_ms - Общее время encode (мс)
                 - total_decode_time_ms - Общее время decode (мс)
                 - avg_encode_time_ms   - Среднее время encode (мс)
                 - avg_decode_time_ms   - Среднее время decode (мс)
                 - cache_hit_rate       - Процент попаданий в кэш (0-1)
        )pbdoc")

        // ====================================================================
        // Дополнительные методы
        // ====================================================================
        
        .def("reset_stats", &PyFastBPETokenizer::reset_stats,
             R"pbdoc(Сбрасывает статистику работы в ноль.)pbdoc")
        
        .def("version", &PyFastBPETokenizer::version,
             R"pbdoc(Возвращает версию токенизатора в формате "major.minor.patch".)pbdoc")
        
        .def("get_model_info", &PyFastBPETokenizer::get_model_info,
             R"pbdoc(Возвращает подробную информацию о модели в читаемом виде.)pbdoc")
        
        // ====================================================================
        // Строковое представление
        // ====================================================================
        
        .def("__repr__", [](const PyFastBPETokenizer& t) {
            return "<FastBPETokenizer vocab_size=" + std::to_string(t.vocab_size()) +
                   " merges=" + std::to_string(t.merges_count()) + ">";
        });

    // ========================================================================
    // Функции модуля
    // ========================================================================
    
    m.def("version", []() { return "3.4.0"; },
          R"pbdoc(Возвращает версию библиотеки.)pbdoc");
    
    m.def("has_avx2", []() { return SIMDUtils::has_avx2(); },
          R"pbdoc(Проверяет, доступны ли AVX2 оптимизации при компиляции.)pbdoc");
    
    m.def("has_avx", []() { return SIMDUtils::has_avx(); },
          R"pbdoc(Проверяет, доступны ли AVX оптимизации при компиляции.)pbdoc");
    
    m.def("has_sse42", []() { return SIMDUtils::has_sse42(); },
          R"pbdoc(Проверяет, доступны ли SSE4.2 оптимизации при компиляции.)pbdoc");
    
    m.def("simd_level", []() { return SIMDUtils::get_simd_level(); },
          R"pbdoc(Возвращает текущий уровень SIMD оптимизаций (строка).)pbdoc");

    // ========================================================================
    // Перенаправление вывода C++ в Python
    // ========================================================================
    
    // Позволяет перенаправить std::cout и std::cerr в Python sys.stdout/sys.stderr
    py::add_ostream_redirect(m, "ostream_redirect");

    // Логирование загрузки модуля
    m.attr("_loaded") = true;
    
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}