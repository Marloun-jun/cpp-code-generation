/**
 * @file fast_tokenizer.hpp
 * @brief Оптимизированная версия BPE токенизатора с поддержкой SIMD и пулов памяти
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Высокопроизводительная реализация BPE токенизатора с фокусом на скорость:
 *          - StringView для избегания копирования строк
 *          - Пул памяти для уменьшения количества аллокаций
 *          - Кэширование результатов для часто встречающихся слов
 *          - SIMD-оптимизации (AVX2, SSE4.2) для массовых операций
 *          - OpenMP поддержка для параллельной обработки
 *          - Потокобезопасное кодирование через shared_mutex
 *          - Сбор статистики производительности
 *          - Параллельное обучение на нескольких ядрах
 * 
 * @note Требует C++17 и поддержки как минимум SSE4.2 для полной функциональности
 * @warning Класс некопируемый, но перемещаемый для безопасности ресурсов
 * 
 * @see BPETokenizer
 * @see MemoryPool
 * @see StringViewCache
 */

#pragma once

#ifdef HAS_CONFIG_H
    #include "config.h"
#endif

#include "memory_pool.hpp"
#include "optimized_types.hpp"
#include "thread_safe_cache.hpp"

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {

/**
 * @brief Оптимизированная версия BPE токенизатора
 * 
 * FastBPETokenizer обеспечивает значительный прирост производительности
 * по сравнению с базовой реализацией за счет:
 * - Использования современных возможностей C++17 (string_view)
 * - Оптимизации работы с памятью (пулы, кэши)
 * - Векторизации критических участков (SIMD)
 * - Параллельной обработки (OpenMP)
 */
class FastBPETokenizer {
public:
    /**
     * @brief Конструктор с конфигурацией
     * @param config Настройки токенизатора (размер словаря, режимы работы и т.д.)
     */
    explicit FastBPETokenizer(const TokenizerConfig& config = TokenizerConfig{});

    /**
     * @brief Деструктор
     */
    ~FastBPETokenizer();

    // Запрещаем копирование (RAII для уникальных ресурсов)
    FastBPETokenizer(const FastBPETokenizer&) = delete;
    FastBPETokenizer& operator=(const FastBPETokenizer&) = delete;

    // Разрешаем перемещение
    FastBPETokenizer(FastBPETokenizer&&) noexcept = default;
    FastBPETokenizer& operator=(FastBPETokenizer&&) noexcept = default;

    // ==================== Основные методы ====================

    /**
     * @brief Закодировать текст в последовательность токенов
     * @param text Входной текст (string_view для избегания копирования)
     * @return Вектор идентификаторов токенов
     */
    std::vector<uint32_t> encode(std::string_view text);

    /**
     * @brief Декодировать последовательность токенов обратно в текст
     * @param tokens Вектор идентификаторов токенов
     * @return Восстановленный текст
     */
    std::string decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Пакетное кодирование нескольких текстов
     * @param texts Вектор входных текстов
     * @return Вектор векторов идентификаторов токенов
     */
    std::vector<std::vector<uint32_t>> encode_batch(
        const std::vector<std::string_view>& texts);

    // ==================== Обучение ====================

    /**
     * @brief Обучить токенизатор на корпусе текстов
     * @param corpus Вектор строк для обучения
     */
    void train(const std::vector<std::string>& corpus);

    /**
     * @brief Параллельное обучение с использованием нескольких ядер
     * @param corpus Вектор строк для обучения
     * @param num_merges Количество операций слияния (по умолчанию 32000)
     */
    void parallel_train(const std::vector<std::string>& corpus, size_t num_merges = 32000);

    // ==================== Загрузка/сохранение ====================

    /**
     * @brief Загрузить модель из текстовых файлов
     * @param vocab_path Путь к файлу словаря
     * @param merges_path Путь к файлу слияний
     * @return true при успешной загрузке, false при ошибке
     */
    bool load(const std::string& vocab_path, const std::string& merges_path);

    /**
     * @brief Сохранить модель в текстовые файлы
     * @param vocab_path Путь для сохранения словаря
     * @param merges_path Путь для сохранения слияний
     * @return true при успешном сохранении, false при ошибке
     */
    bool save(const std::string& vocab_path, const std::string& merges_path) const;

    /**
     * @brief Сохранить модель в единый бинарный файл (быстрая загрузка)
     * @param path Путь для сохранения
     * @return true при успешном сохранении, false при ошибке
     */
    bool save_binary(const std::string& path) const;

    /**
     * @brief Загрузить модель из бинарного файла
     * @param path Путь к бинарному файлу
     * @return true при успешной загрузке, false при ошибке
     */
    bool load_binary(const std::string& path);

    // ==================== Геттеры ====================

    /**
     * @brief Получить размер словаря
     * @return Количество токенов
     */
    size_t vocab_size() const { return id_to_token_.size(); }

    /**
     * @brief Получить количество правил слияния
     * @return Количество пар в merges_
     */
    size_t merges_count() const { return merges_.size(); }

    /**
     * @brief Получить статистику производительности
     * @return Константная ссылка на TokenizerStats
     */
    const TokenizerStats& stats() const { return stats_; }

    /**
     * @brief Сбросить статистику производительности
     */
    void reset_stats() { stats_.reset(); }

    // ==================== Специальные токены ====================

    /**
     * @brief Получить ID токена для неизвестных символов
     * @return ID токена <UNK>
     */
    uint32_t unknown_id() const { return unknown_id_; }

    /**
     * @brief Получить ID токена для паддинга
     * @return ID токена <PAD>
     */
    uint32_t pad_id() const { return pad_id_; }

    /**
     * @brief Получить ID токена начала последовательности
     * @return ID токена <BOS>
     */
    uint32_t bos_id() const { return bos_id_; }

    /**
     * @brief Получить ID токена конца последовательности
     * @return ID токена <EOS>
     */
    uint32_t eos_id() const { return eos_id_; }

private:
    // ==================== Приватные методы ====================

    /**
     * @brief Токенизация отдельного слова
     * @param word Слово для токенизации
     * @return Вектор ID токенов
     */
    std::vector<uint32_t> tokenize_word(std::string_view word);

    /**
     * @brief Byte-level кодирование текста
     * @param text Входной текст
     * @return Вектор ID токенов
     */
    std::vector<uint32_t> byte_level_encode(std::string_view text);

    /**
     * @brief Обычное кодирование текста (с предтокенизацией)
     * @param text Входной текст
     * @return Вектор ID токенов
     */
    std::vector<uint32_t> normal_encode(std::string_view text);

    /**
     * @brief Byte-level декодирование токенов
     * @param tokens Вектор ID токенов
     * @return Восстановленный текст
     */
    std::string byte_level_decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Обычное декодирование токенов
     * @param tokens Вектор ID токенов
     * @return Восстановленный текст
     */
    std::string normal_decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Инициализировать специальные токены в словаре
     */
    void initialize_special_tokens();

    /**
     * @brief Построить обратную карту token -> ID
     */
    void build_token_to_id_map();

    /**
     * @brief Подсчет частот символов в параллельном режиме
     * @param corpus Корпус текстов
     * @return Карта частот символов
     */
    std::unordered_map<std::string, int> count_char_frequencies_parallel(
        const std::vector<std::string>& corpus);

    /**
     * @brief Построение начального словаря на основе частот символов
     * @param char_freq Карта частот символов
     */
    void build_initial_vocabulary(
        const std::unordered_map<std::string, int>& char_freq);

#ifdef USE_AVX2
    /**
     * @brief AVX2-оптимизированная версия токенизации слова
     * @param word Слово для токенизации
     * @return Вектор ID токенов
     */
    std::vector<uint32_t> tokenize_word_avx2(std::string_view word);
#endif

    // ==================== Поля класса ====================

    // Основные структуры данных
    std::vector<std::string> id_to_token_;                    ///< Отображение ID -> токен
    std::unordered_map<std::string, uint32_t> token_to_id_;   ///< Отображение токен -> ID
    std::unordered_map<merge_key_t, uint32_t> merges_;        ///< Правила слияния с рангами

    // Оптимизации
    std::unique_ptr<StringViewCache> cache_;                  ///< Кэш для частых слов
    mutable MemoryPool<4096> memory_pool_;                    ///< Пул памяти для строк

    // Конфигурация и состояние
    TokenizerConfig config_;                                   ///< Настройки токенизатора
    mutable TokenizerStats stats_;                             ///< Статистика производительности
    mutable std::shared_mutex mutex_;                          ///< Мьютекс для потокобезопасности

    // ID специальных токенов
    uint32_t unknown_id_{0};    ///< ID токена <UNK>
    uint32_t pad_id_{0};        ///< ID токена <PAD>
    uint32_t bos_id_{0};        ///< ID токена <BOS>
    uint32_t eos_id_{0};        ///< ID токена <EOS>
};

} // namespace bpe