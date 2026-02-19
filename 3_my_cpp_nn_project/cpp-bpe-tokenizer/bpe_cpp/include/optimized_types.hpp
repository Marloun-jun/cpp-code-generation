/**
 * @file optimized_types.hpp
 * @brief Оптимизированные типы данных для высокопроизводительного BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Компактные и эффективные типы данных, специально разработанные
 *          для минимизации использования памяти и максимизации скорости:
 * 
 *          - merge_key_t: 64-битный ключ для хранения пары ID токенов
 *            (экономит память по сравнению с хранением строк)
 *          
 *          - TokenizerStats: структура для сбора статистики производительности
 *          
 *          - TokenizerConfig: централизованная конфигурация токенизатора
 * 
 * @note Все функции помечены noexcept для лучшей оптимизации
 * @see FastBPETokenizer
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace bpe {

// ==================== Компактное представление пары для слияний ====================

/**
 * @brief 64-битный ключ для хранения пары ID токенов
 * 
 * Упаковывает два 32-битных ID в одно 64-битное значение.
 * Это позволяет:
 * - Хранить пары в 8 байтах вместо ~50-100 байт для строк
 * - Использовать как ключ в unordered_map без хеш-функции
 * - Уменьшить использование памяти на 80-90%
 * 
 * Формат: [left_id (32 бита)] [right_id (32 бита)]
 */
using merge_key_t = uint64_t;

/**
 * @brief Создать компактный ключ из двух ID токенов
 * @param left ID левого токена
 * @param right ID правого токена
 * @return 64-битный ключ
 * 
 * @code
 * merge_key_t key = make_merge_key(42, 17);
 * // key = 0x0000002A00000011 (42 << 32 | 17)
 * @endcode
 */
inline merge_key_t make_merge_key(uint32_t left, uint32_t right) noexcept {
    return (static_cast<merge_key_t>(left) << 32) | right;
}

/**
 * @brief Извлечь ID левого токена из ключа
 * @param key 64-битный ключ
 * @return ID левого токена
 */
inline uint32_t get_left_from_key(merge_key_t key) noexcept {
    return static_cast<uint32_t>(key >> 32);
}

/**
 * @brief Извлечь ID правого токена из ключа
 * @param key 64-битный ключ
 * @return ID правого токена
 */
inline uint32_t get_right_from_key(merge_key_t key) noexcept {
    return static_cast<uint32_t>(key & 0xFFFFFFFF);
}

// ==================== Статистика производительности ====================

/**
 * @brief Структура для сбора статистики работы токенизатора
 * 
 * Собирает метрики:
 * - Количество вызовов encode/decode
 * - Попадания в кэш (hit/miss)
 * - Время выполнения (для профилирования)
 * - Количество обработанных токенов
 * 
 * Используется для бенчмарков и оптимизации
 */
struct TokenizerStats {
    size_t encode_calls{0};          ///< Количество вызовов encode()
    size_t decode_calls{0};          ///< Количество вызовов decode()
    size_t cache_hits{0};             ///< Попадания в кэш
    size_t cache_misses{0};           ///< Промахи кэша
    size_t total_tokens_processed{0}; ///< Всего обработано токенов
    double total_encode_time_ms{0.0}; ///< Суммарное время encode (мс)
    double total_decode_time_ms{0.0}; ///< Суммарное время decode (мс)

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
     * @brief Среднее время одного encode вызова
     * @return Среднее время в миллисекундах
     */
    double avg_encode_time_ms() const noexcept {
        return encode_calls ? total_encode_time_ms / encode_calls : 0.0;
    }

    /**
     * @brief Среднее время одного decode вызова
     * @return Среднее время в миллисекундах
     */
    double avg_decode_time_ms() const noexcept {
        return decode_calls ? total_decode_time_ms / decode_calls : 0.0;
    }

    /**
     * @brief Процент попаданий в кэш
     * @return Значение от 0.0 до 1.0
     */
    double cache_hit_rate() const noexcept {
        size_t total = cache_hits + cache_misses;
        return total ? static_cast<double>(cache_hits) / total : 0.0;
    }
};

// ==================== Конфигурация токенизатора ====================

/**
 * @brief Структура конфигурации для всех версий токенизатора
 * 
 * Централизованное хранение всех настроек позволяет:
 * - Легко передавать конфигурацию между компонентами
 * - Сериализовать настройки
 * - Валидировать параметры перед использованием
 * 
 * @note Все поля инициализированы значениями по умолчанию
 */
struct TokenizerConfig {
    // Основные параметры
    size_t vocab_size{32000};        ///< Целевой размер словаря
    size_t cache_size{10000};         ///< Размер кэша (количество записей)
    
    // Режимы работы
    bool byte_level{true};            ///< Byte-level режим (по умолчанию включен)
    bool enable_cache{true};          ///< Включить кэширование результатов
    bool enable_profiling{false};     ///< Включить сбор статистики
    
    // Специальные токены
    std::string unknown_token{"<UNK>"};  ///< Токен для неизвестных символов
    std::string pad_token{"<PAD>"};      ///< Токен для паддинга
    std::string bos_token{"<BOS>"};      ///< Токен начала последовательности
    std::string eos_token{"<EOS>"};      ///< Токен конца последовательности

    /**
     * @brief Проверить корректность настроек
     * @return true если конфигурация валидна
     * 
     * Проверяет:
     * - vocab_size >= 256 (минимальный словарь)
     * - vocab_size <= 100000 (разумный максимум)
     * - cache_size <= 100000
     */
    bool validate() const noexcept {
        if (vocab_size < 256) return false;
        if (vocab_size > 100000) return false;
        if (cache_size > 100000) return false;
        return true;
    }
};

} // namespace bpe

/**
 * @example examples/stats_example.cpp
 * Пример использования статистики:
 * @code
 * #include "optimized_types.hpp"
 * 
 * bpe::TokenizerStats stats;
 * 
 * // После нескольких encode вызовов
 * std::cout << "Среднее время encode: " 
 *           << stats.avg_encode_time_ms() << " мс" << std::endl;
 * std::cout << "Попаданий в кэш: " 
 *           << (stats.cache_hit_rate() * 100) << "%" << std::endl;
 * @endcode
 */

/**
 * @example examples/merge_key_example.cpp
 * Пример использования merge_key_t:
 * @code
 * // Создание ключа из пары
 * merge_key_t key = make_merge_key(100, 200);
 * 
 * // Хранение в unordered_map (не нужна хеш-функция!)
 * std::unordered_map<merge_key_t, int> merge_ranks;
 * merge_ranks[key] = 42;
 * 
 * // Извлечение ID обратно
 * uint32_t left = get_left_from_key(key);   // = 100
 * uint32_t right = get_right_from_key(key); // = 200
 * @endcode
 */