/**
 * @file config.hpp
 * @brief Конфигурационные параметры и настройки BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Этот файл содержит глобальные конфигурационные параметры,
 *          управляющие поведением токенизатора. Настройки разделены на категории:
 *          - Параметры токенизации по умолчанию
 *          - Пути к файлам моделей
 *          - Настройки производительности
 *          - Опции кэширования
 * 
 *          Файл обрабатывается CMake для условной компиляции:
 *          - #cmakedefine USE_AVX2 - включает AVX2 оптимизации
 *          - #cmakedefine USE_SSE42 - включает SSE4.2 оптимизации
 * 
 * @note Все параметры являются constexpr для максимальной производительности
 * @warning Изменение параметров требует перекомпиляции проекта
 * 
 * @see FastBPETokenizer
 * @see BPETokenizer
 */

#pragma once

#include <cstddef>

// Опции оптимизации, определяемые CMake
#cmakedefine USE_AVX2   ///< Поддержка AVX2 инструкций
#cmakedefine USE_SSE42  ///< Поддержка SSE4.2 инструкций

namespace bpe {
namespace config {

/**
 * @name Параметры токенизации по умолчанию
 * Базовые настройки, определяющие поведение токенизатора
 */
/**@{*/
constexpr bool DEFAULT_BYTE_LEVEL = true;           ///< Использовать byte-level режим по умолчанию
constexpr size_t DEFAULT_MAX_TOKEN_LENGTH = 1000;   ///< Максимальная длина токена в символах
/**@}*/

/**
 * @name Настройки кэширования
 * Параметры для оптимизации через кэширование результатов
 */
/**@{*/
constexpr bool ENABLE_CACHE = true;                 ///< Включить кэширование результатов
constexpr size_t CACHE_SIZE_LIMIT = 10000;          ///< Максимальный размер кэша (количество записей)
/**@}*/

/**
 * @name Пути к файлам моделей
 * Стандартные пути для сохранения и загрузки моделей
 * @note Пути указаны относительно корня проекта
 */
/**@{*/
constexpr const char* DEFAULT_VOCAB_PATH = "models/cpp_vocab.json";   ///< Путь к файлу словаря
constexpr const char* DEFAULT_MERGES_PATH = "models/cpp_merges.txt";  ///< Путь к файлу слияний
/**@}*/

/**
 * @name Настройки производительности
 * Параметры для оптимизации скорости обработки
 */
/**@{*/
constexpr size_t ENCODE_BATCH_SIZE = 32;            ///< Размер пакета при пакетной обработке
constexpr bool USE_PARALLEL_ENCODE = true;          ///< Использовать параллельное кодирование
/**@}*/

} // namespace config
} // namespace bpe

/**
 * @example examples/simple_example.cpp
 * Пример использования конфигурации:
 * 
 * @code
 * #include "config.hpp"
 * 
 * TokenizerConfig cfg;
 * cfg.byte_level = bpe::config::DEFAULT_BYTE_LEVEL;
 * cfg.cache_size = bpe::config::CACHE_SIZE_LIMIT;
 * FastBPETokenizer tokenizer(cfg);
 * @endcode
 */