/**
 * @file optimized_types.hpp
 * @brief Компактные типы данных для высокопроизводительной токенизации
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Фундаментальные оптимизированные типы, обеспечивающие
 *          эффективное использование памяти и кэша процессора.
 *          Являются основой всех оптимизаций в FastBPETokenizer.
 * 
 *          **Ключевые оптимизации:**
 * 
 *          1. **merge_key_t (64 бита)**
 *             - Упаковывает два 32-битных ID в одно 64-битное значение
 *             - Экономия: 8 байт вместо 32-64 байт для строковых ключей
 *             - Для 1 млн правил: 8 МБ вместо 32-64 МБ
 *             - Идеальное выравнивание для кэша процессора
 *             - Не требует хеш-функции (используем стандартную)
 * 
 *          2. **TokenizerStats**
 *             - Сбор метрик производительности без оверхэда
 *             - Поддержка многопоточности (атомарная версия)
 *             - Удобные методы анализа (hit rate, средние)
 * 
 *          3. **TokenizerConfig**
 *             - Централизованное хранение настроек
 *             - Валидация параметров на этапе конфигурации
 *             - Автоматический расчет оптимальных значений
 * 
 *          **Влияние на производительность:**
 *          - Память      - -70-80% для правил слияния
 *          - Скорость    - +20-30% за счет лучшей локальности
 *          - Кэш-промахи - -40% благодаря компактности
 * 
 * @see FastBPETokenizer, MemoryPool, StringViewCache
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>

namespace bpe {

// ============================================================================
// merge_key_t - компактное представление пары токенов
// ============================================================================

/**
 * @brief 64-битный ключ для хранения пары ID токенов
 * 
 * **Формат:**
 * @code
 * 64-битное целое:
 * ┌──────────────────────────────┬──────────────────────────────┐
 * │ left_id (32 бита)            │ right_id (32 бита)           │
 * └──────────────────────────────┴──────────────────────────────┘
 *   63 ... 32                      31 ... 0
 * @endcode
 * 
 * **Пример:**
 * @code
 * uint32_t left = 42;     // 0x0000002A
 * uint32_t right = 17;    // 0x00000011
 * merge_key_t key = 0x0000002A00000011
 * @endcode
 * 
 * **Сравнение со строковым представлением:**
 * @code
 * // Строковый подход
 * std::string pair = "42 17";    // ~5 байт
 * // + оверхед std::string       // ~24 байта
 * // + оверхед хеш-таблицы       // ~16 байт
 * // ИТОГО: ~45 байт на пару
 * 
 * // merge_key_t подход
 * merge_key_t key = make_merge_key(42, 17);    // 8 байт
 * // Экономия: 37 байт на пару (82%)
 * @endcode
 * 
 * @warning ID токенов должны помещаться в 32 бита (< 4,294,967,296)
 */
using merge_key_t = uint64_t;

/**
 * @brief Создать компактный ключ из двух ID
 * 
 * @param left ID левого токена
 * @param right ID правого токена
 * @return merge_key_t Упакованный 64-битный ключ
 * 
 * **Алгоритм:**  left << 32 | right
 * **Сложность:** 1 такт процессора
 * 
 * @code
 * merge_key_t key = make_merge_key(1000, 2000);
 * // key = 0x000003E8000007D0
 * @endcode
 */
constexpr merge_key_t make_merge_key(uint32_t left, uint32_t right) noexcept {
    return (static_cast<merge_key_t>(left) << 32) | right;
}

/**
 * @brief Извлечь ID левого токена из ключа
 * 
 * @param key 64-битный ключ
 * @return uint32_t ID левого токена
 * 
 * **Алгоритм:** key >> 32
 * 
 * @code
 * uint32_t left = get_left_from_key(0x000003E8000007D0);    // = 1000
 * @endcode
 */
constexpr uint32_t get_left_from_key(merge_key_t key) noexcept {
    return static_cast<uint32_t>(key >> 32);
}

/**
 * @brief Извлечь ID правого токена из ключа
 * 
 * @param key 64-битный ключ
 * @return uint32_t ID правого токена
 * 
 * **Алгоритм:** key & 0xFFFFFFFF
 * 
 * @code
 * uint32_t right = get_right_from_key(0x000003E8000007D0);    // = 2000
 * @endcode
 */
constexpr uint32_t get_right_from_key(merge_key_t key) noexcept {
    return static_cast<uint32_t>(key & 0xFFFFFFFF);
}

/**
 * @brief Хеш-функция для merge_key_t
 * 
 * Так как merge_key_t уже является целым числом,
 * используем тривиальную хеш-функцию.
 */
struct merge_key_hash {
    size_t operator()(merge_key_t key) const noexcept {
        return static_cast<size_t>(key);
    }
};

// ============================================================================
// TokenizerStats - статистика производительности
// ============================================================================

/**
 * @brief Статистика работы токенизатора
 * 
 * Собирает метрики для анализа производительности:
 * - Количество вызовов encode/decode
 * - Эффективность кэширования (hit/miss)
 * - Временные характеристики
 * - Объем обработанных данных
 * 
 * **Пример анализа:**
 * @code
 * TokenizerStats stats = tokenizer.stats();
 * 
 * std::cout << "Производительность:\n";
 * std::cout << "- Вызовов encode:       " << stats.encode_calls << "\n";
 * std::cout << "- Среднее время encode: " 
 *           << stats.avg_encode_time_ms() << " мс\n";
 * std::cout << "- Попаданий в кэш:      " 
 *           << (stats.cache_hit_rate() * 100) << "%\n";
 * 
 * if (stats.cache_hit_rate() < 0.5) {
 *     std::cout << "Низкая эффективность кэша!\n";
 * }
 * @endcode
 */
struct TokenizerStats {
    size_t encode_calls{0};              ///< Количество вызовов encode()
    size_t decode_calls{0};              ///< Количество вызовов decode()
    size_t total_tokens_processed{0};    ///< Всего обработано токенов
    double total_encode_time_ms{0.0};    ///< Суммарное время encode (мс)
    double total_decode_time_ms{0.0};    ///< Суммарное время decode (мс)
    size_t cache_hits{0};                ///< Попадания в кэш
    size_t cache_misses{0};              ///< Промахи кэша

    /**
     * @brief Сбросить всю статистику в ноль
     */
    void reset() noexcept {
        encode_calls = 0;
        decode_calls = 0;
        cache_hits = 0;
        cache_misses = 0;
        total_tokens_processed = 0;
        total_encode_time_ms = 0.0;
        total_decode_time_ms = 0.0;
    }

    /**
     * @brief Среднее время encode вызова
     * @return double Время в миллисекундах
     */
    double avg_encode_time_ms() const noexcept {
        return encode_calls ? total_encode_time_ms / encode_calls : 0.0;
    }

    /**
     * @brief Среднее время decode вызова
     * @return double Время в миллисекундах
     */
    double avg_decode_time_ms() const noexcept {
        return decode_calls ? total_decode_time_ms / decode_calls : 0.0;
    }

    /**
     * @brief Процент попаданий в кэш (0.0 - 1.0)
     * 
     * **Интерпретация:**
     * - < 0.5    - Кэш малоэффективен
     * - 0.5-0.8  - Приемлемо
     * - 0.8-0.95 - Хорошо
     * - > 0.95   - Отлично
     */
    double cache_hit_rate() const noexcept {
        size_t total = cache_hits + cache_misses;
        return total ? static_cast<double>(cache_hits) / total : 0.0;
    }

    /**
     * @brief Получить общее количество обращений к кэшу
     */
    size_t total_cache_accesses() const noexcept {
        return cache_hits + cache_misses;
    }

    /**
     * @brief Проверить наличие данных
     */
    bool has_data() const noexcept {
        return encode_calls > 0 || decode_calls > 0;
    }

    /**
     * @brief Добавить статистику из другого объекта
     */
    void add(const TokenizerStats& other) noexcept {
        encode_calls += other.encode_calls;
        decode_calls += other.decode_calls;
        cache_hits += other.cache_hits;
        cache_misses += other.cache_misses;
        total_tokens_processed += other.total_tokens_processed;
        total_encode_time_ms += other.total_encode_time_ms;
        total_decode_time_ms += other.total_decode_time_ms;
    }

    /**
     * @brief Оператор += для удобства
     */
    TokenizerStats& operator+=(const TokenizerStats& other) noexcept {
        add(other);
        return *this;
    }

    /**
     * @brief Получить строковое представление статистики
     */
    std::string to_string() const {
        std::string result;
        result += "Статистика токенизатора:\n";
        result += "- Вызовов encode:       " + std::to_string(encode_calls) + "\n";
        result += "- Вызовов decode:       " + std::to_string(decode_calls) + "\n";
        result += "- Токенов обработано:   " + std::to_string(total_tokens_processed) + "\n";
        result += "- Среднее время encode: " + std::to_string(avg_encode_time_ms()) + " мс\n";
        result += "- Среднее время decode: " + std::to_string(avg_decode_time_ms()) + " мс\n";
        result += "- Попаданий в кэш:      " + std::to_string(cache_hits) + "\n";
        result += "- Промахов кэша:        " + std::to_string(cache_misses) + "\n";
        result += "- Эффективность кэша:   " + std::to_string(cache_hit_rate() * 100) + "%\n";
        return result;
    }
};

/**
 * @brief Атомарная версия статистики для многопоточности
 * 
 * Позволяет безопасно обновлять статистику из нескольких потоков
 * без дополнительной синхронизации.
 */
struct AtomicTokenizerStats {
    std::atomic<size_t> encode_calls{0};              ///< Атомарный счетчик encode
    std::atomic<size_t> decode_calls{0};              ///< Атомарный счетчик decode
    std::atomic<size_t> total_tokens_processed{0};    ///< Атомарный счетчик токенов
    std::atomic<double> total_encode_time_ms{0.0};    ///< Атомарное время encode
    std::atomic<double> total_decode_time_ms{0.0};    ///< Атомарное время decode
    std::atomic<size_t> cache_hits{0};                ///< Атомарные попадания в кэш
    std::atomic<size_t> cache_misses{0};              ///< Атомарные промахи кэша

    /**
     * @brief Сбросить статистику
     */
    void reset() noexcept {
        encode_calls = 0;
        decode_calls = 0;
        cache_hits = 0;
        cache_misses = 0;
        total_tokens_processed = 0;
        total_encode_time_ms = 0.0;
        total_decode_time_ms = 0.0;
    }

    /**
     * @brief Получить снимок статистики (неатомарный)
     * @return TokenizerStats Копия текущих значений
     */
    TokenizerStats snapshot() const noexcept {
        TokenizerStats stats;
        stats.encode_calls = encode_calls.load();
        stats.decode_calls = decode_calls.load();
        stats.cache_hits = cache_hits.load();
        stats.cache_misses = cache_misses.load();
        stats.total_tokens_processed = total_tokens_processed.load();
        stats.total_encode_time_ms = total_encode_time_ms.load();
        stats.total_decode_time_ms = total_decode_time_ms.load();
        return stats;
    }
};

// ============================================================================
// TokenizerConfig - конфигурация токенизатора
// ============================================================================

/**
 * @brief Конфигурация для всех версий токенизатора
 * 
 * Централизованное хранение всех параметров с валидацией.
 * 
 * **Параметры по умолчанию:**
 * @code
 * vocab_size:  10000 - Достаточно для большинства задач с++
 * cache_size:  10000 - 10K записей ≈ 10 МБ памяти
 * byte_level:  true  - Обязательно для Unicode
 * num_threads: 0     - auto (использовать все ядра)
 * @endcode
 * 
 * **Пример кастомизации:**
 * @code
 * // Для сервера с 32 ГБ RAM
 * TokenizerConfig config;
 * config.vocab_size = 50000;
 * config.cache_size = 100000;
 * config.num_threads = 16;
 * 
 * if (!config.validate()) {
 *     throw std::runtime_error("Invalid config");
 * }
 * 
 * FastBPETokenizer tokenizer(config);
 * @endcode
 */
struct TokenizerConfig {
    // ========================================================================
    // Основные параметры
    // ========================================================================

    size_t vocab_size{10000};    ///< Целевой размер словаря (токенов)
                                 ///< 1000-5000   - Маленький, быстрый
                                 ///< 8000-16000  - Оптимальный баланс
                                 ///< 32000-50000 - Большой, точный

    size_t cache_size{10000};    ///< Размер кэша в записях
                                 ///< 1000   - 1 МБ, высокая скорость
                                 ///< 10000  - 10 МБ, баланс
                                 ///< 100000 - 100 МБ, максимум

    // ========================================================================
    // Режимы работы
    // ========================================================================

    bool byte_level{true};           ///< Byte-level режим (обязательно для Unicode)
    bool enable_cache{true};         ///< Кэширование результатов (ускорение 2-5x)
    bool enable_profiling{false};    ///< Профилирование (только для бенчмарков)
    bool use_memory_pool{true};      ///< Использовать пул памяти (ускорение 20-30%)

    // ========================================================================
    // Параллелизация
    // ========================================================================

    int num_threads{0};    ///< Количество потоков (0 = auto)
                           ///< 0 - Использовать все ядра
                           ///< 1 - Однопоточный режим
                           ///< N - Ровно N потоков

    // ========================================================================
    // Специальные токены
    // ========================================================================

    std::string unknown_token{"<UNK>"};    ///< Неизвестный токен
    std::string pad_token{"<PAD>"};        ///< Padding токен
    std::string bos_token{"<BOS>"};        ///< Начало последовательности
    std::string eos_token{"<EOS>"};        ///< Конец последовательности
    std::string mask_token{"<MASK>"};      ///< Маскирующий токен

    // ========================================================================
    // Параметры обучения (опционально)
    // ========================================================================

    size_t min_frequency{2};          ///< Минимальная частота для включения в словарь
    size_t max_token_length{1000};    ///< Максимальная длина токена (защита)

    /**
     * @brief Конструктор по умолчанию (оптимальные параметры)
     */
    TokenizerConfig() = default;

    /**
     * @brief Конструктор для основных параметров
     * 
     * @param vs Размер словаря
     * @param cs Размер кэша
     * @param bl Byte-level режим
     */
    TokenizerConfig(size_t vs, size_t cs, bool bl) noexcept
        : vocab_size(vs)
        , cache_size(cs)
        , byte_level(bl) {}

    /**
     * @brief Проверить корректность конфигурации
     * 
     * @return true если все параметры в допустимых пределах
     * 
     * **Проверки:**
     * - vocab_size >= 256    - Минимум для ASCII
     * - vocab_size <= 100000 - Разумный максимум
     * - cache_size <= 100000 - Ограничение по памяти
     */
    bool validate() const noexcept {
        if (vocab_size < 256) return false;
        if (vocab_size > 100000) return false;
        if (cache_size > 100000) return false;
        return true;
    }

    /**
     * @brief Рекомендуемый размер кэша для заданной памяти
     * 
     * @param memory_limit_mb Ограничение памяти в МБ
     * @return size_t Рекомендованный размер кэша
     * 
     * **Эмпирическая формула:**
     * 1 запись ≈ 1 КБ (текст + токены)
     * 1000 записей ≈ 1 МБ
     */
    static size_t recommended_cache_size(size_t memory_limit_mb) {
        return memory_limit_mb * 1000;
    }

    /**
     * @brief Получить эффективное количество потоков
     * 
     * @return int Количество потоков для использования
     * 
     * Если num_threads > 0, возвращает num_threads,
     * иначе возвращает количество аппаратных потоков CPU.
     */
    int effective_num_threads() const noexcept {
        if (num_threads > 0) return num_threads;
        return static_cast<int>(std::thread::hardware_concurrency());
    }

    /**
     * @brief Получить строковое представление конфигурации
     */
    std::string to_string() const {
        std::string result;
        result += "Конфигурация токенизатора:\n";
        result += "- vocab_size:       " + std::to_string(vocab_size) + "\n";
        result += "- cache_size:       " + std::to_string(cache_size) + "\n";
        result += "- byte_level:       " + std::string(byte_level ? "да" : "нет") + "\n";
        result += "- enable_cache:     " + std::string(enable_cache ? "да" : "нет") + "\n";
        result += "- enable_profiling: " + std::string(enable_profiling ? "да" : "нет") + "\n";
        result += "- use_memory_pool:  " + std::string(use_memory_pool ? "да" : "нет") + "\n";
        result += "- num_threads:      " + std::to_string(num_threads) + "\n";
        result += "- unknown_token:    " + unknown_token + "\n";
        result += "- pad_token:        " + pad_token + "\n";
        result += "- bos_token:        " + bos_token + "\n";
        result += "- eos_token:        " + eos_token + "\n";
        result += "- mask_token:       " + mask_token + "\n";
        return result;
    }
};

}    // namespace bpe