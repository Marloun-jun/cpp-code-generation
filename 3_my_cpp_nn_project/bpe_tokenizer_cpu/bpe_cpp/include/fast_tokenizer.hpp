/**
 * @file fast_tokenizer.hpp
 * @brief Высокопроизводительная реализация BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.7.0
 * 
 * @details Оптимизированная версия BPE токенизатора с поддержкой:
 *          - Byte-level режима для корректной обработки Unicode
 *          - Кэширования результатов encode (hit rate 50-99%)
 *          - Параллельной обработки батчей через OpenMP
 *          - Компактных структур данных (uint64_t ключи)
 *          - Потокобезопасности через shared_mutex
 *          - Параллельного обучения через ParallelTrainer
 *          - Предварительно вычисленных правил слияния для encode (ускорение 10-15%)
 * 
 *          **Поддержка языков:**
 *          - ASCII (1 байт)          - Максимальная скорость
 *          - Русские буквы (2 байта) - Корректная обработка
 *          - Эмодзи (4 байта)        - Корректная обработка
 * 
 * @note Для получения информации о модели используйте get_model_info()
 * @see FastBPETokenizer, TokenizerConfig, TokenizerStats
 */

#ifndef BPE_FAST_TOKENIZER_HPP
#define BPE_FAST_TOKENIZER_HPP

#ifdef HAS_CONFIG_H
    #include "config.h"
#endif

#include "memory_pool.hpp"
#include "optimized_types.hpp"
#include "thread_safe_cache.hpp"
#include "bpe_export.hpp"
#include "vocabulary.hpp"

#include <atomic>
#include <array>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <climits>

namespace bpe {

// ============================================================================
// Основной класс оптимизированного токенизатора
// ============================================================================

/**
 * @brief Высокопроизводительный BPE токенизатор для production-использования
 * 
 * @details Обеспечивает максимальную скорость токенизации за счет:
 *          - Компактного хранения правил слияния (uint64_t)
 *          - Таблицы быстрого доступа байт -> ID (O(1))
 *          - Потокобезопасного кэширования результатов
 *          - Корректной обработки UTF-8 любой длины
 *          - Предварительно вычисленных правил слияния для encode
 * 
 * @code
 * // Минимальный пример использования
 * FastBPETokenizer tokenizer;
 * tokenizer.load("vocab.json", "merges.txt");
 * auto tokens = tokenizer.encode("// русский комментарий");
 * std::string decoded = tokenizer.decode(tokens);
 * @endcode
 */
class FastBPETokenizer {
public:
    // ========================================================================
    // Публичные структуры данных
    // ========================================================================

    /**
     * @brief Правило слияния для быстрого encode
     * 
     * Содержит предварительно вычисленные ID токенов для применения слияния.
     * Позволяет избежать поиска в token_to_id_ во время encode.
     */
    struct MergeRule {
        uint32_t left;      ///< ID левого токена
        uint32_t right;     ///< ID правого токена
        uint32_t result_id; ///< ID результирующего токена после слияния
    };

    // ========================================================================
    // Конструкторы и управление ресурсами
    // ========================================================================

    /**
     * @brief Конструктор с пользовательской конфигурацией
     * @param config Настройки токенизатора (см. TokenizerConfig)
     */
    explicit FastBPETokenizer(const TokenizerConfig& config = TokenizerConfig{});
    
    /**
     * @brief Деструктор
     */
    ~FastBPETokenizer();

    // Запрет копирования
    FastBPETokenizer(const FastBPETokenizer&) = delete;
    FastBPETokenizer& operator=(const FastBPETokenizer&) = delete;

    // Разрешение перемещения
    FastBPETokenizer(FastBPETokenizer&&) noexcept = default;
    FastBPETokenizer& operator=(FastBPETokenizer&&) noexcept = default;

    // ========================================================================
    // Основные операции токенизации
    // ========================================================================

    /**
     * @brief Закодировать текст в последовательность ID токенов
     * @param text Входной текст (string_view для избегания копирования)
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * @note Поддерживает любые UTF-8 символы (1-4 байта)
     * @note Потокобезопасно для параллельного чтения
     */
    std::vector<uint32_t> encode(std::string_view text);
    
    /**
     * @brief Перегрузка encode для std::string
     */
    std::vector<uint32_t> encode(const std::string& text) {
        return encode(std::string_view(text));
    }
    
    /**
     * @brief Оптимизированная версия для ASCII-текстов
     * @warning Не использовать с Unicode!
     */
    std::vector<uint32_t> encode_ascii(std::string_view text);
    
    /**
     * @brief Декодировать последовательность ID обратно в текст
     * @param tokens Вектор ID токенов
     * @return std::string Восстановленный текст
     */
    std::string decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Пакетное кодирование с параллельной обработкой
     * @param texts Вектор входных текстов
     * @return std::vector<std::vector<uint32_t>> Вектор результатов
     * 
     * @note Требует компиляции с -fopenmp для максимальной скорости
     */
    std::vector<std::vector<uint32_t>> encode_batch(
        const std::vector<std::string_view>& texts);
    
    /**
     * @brief Перегрузка encode_batch для std::string
     */
    std::vector<std::vector<uint32_t>> encode_batch(
        const std::vector<std::string>& texts);

    // ========================================================================
    // Обучение модели
    // ========================================================================

    /**
     * @brief Последовательное обучение (для небольших корпусов)
     */
    void train(const std::vector<std::string>& corpus);

    /**
     * @brief Параллельное обучение с использованием всех ядер
     * @param corpus Вектор строк для обучения
     * @param num_merges Количество операций слияния
     */
    void parallel_train(const std::vector<std::string>& corpus, size_t num_merges = 10000);

    // ========================================================================
    // Сериализация модели
    // ========================================================================

    /**
     * @brief Загрузить модель из текстовых файлов
     * @param vocab_path Путь к vocab.json
     * @param merges_path Путь к merges.txt
     * @return true при успешной загрузке
     */
    bool load(const std::string& vocab_path, const std::string& merges_path);
    
    /**
     * @brief Сохранить модель в текстовые файлы
     * @param vocab_path Путь для vocab.json
     * @param merges_path Путь для merges.txt
     * @return true при успешном сохранении
     */
    bool save(const std::string& vocab_path, const std::string& merges_path) const;
    
    /**
     * @brief Сохранить в бинарном формате (TODO)
     */
    bool save_binary(const std::string& path) const;
    
    /**
     * @brief Загрузить из бинарного формата (TODO)
     */
    bool load_binary(const std::string& path);

    // ========================================================================
    // Геттеры для внутренних структур
    // ========================================================================

    size_t vocab_size() const;
    size_t merges_count() const;
    const TokenizerStats& stats() const;
    void reset_stats();

    // ========================================================================
    // ID специальных токенов
    // ========================================================================

    uint32_t unknown_id() const { return unknown_id_; }
    uint32_t pad_id()     const { return pad_id_; }
    uint32_t bos_id()     const { return bos_id_; }
    uint32_t eos_id()     const { return eos_id_; }
    uint32_t mask_id()    const { return mask_id_; }

    /**
     * @brief Получить карту быстрого поиска правил слияния
     * @return const std::unordered_map<uint64_t, uint32_t>& Ссылка на карту правил
     */
    const std::unordered_map<uint64_t, uint32_t>& merge_rule_map() const { return merge_rule_map_; }
    
    // ========================================================================
    // Информация о модели
    // ========================================================================

    std::string get_model_info() const;

private:
    // ========================================================================
    // Приватные вспомогательные методы
    // ========================================================================

    // UTF-8 вспомогательные функции
    static inline int utf8_char_length(unsigned char first_byte);
    static inline std::string_view get_utf8_char(std::string_view str, size_t pos, int& len);
    
    // Основные методы токенизации
    std::vector<uint32_t> tokenize_word(std::string_view word);
    std::vector<uint32_t> byte_level_encode(std::string_view text);
    std::vector<uint32_t> normal_encode(std::string_view text);
    std::string byte_level_decode(const std::vector<uint32_t>& tokens);
    std::string normal_decode(const std::vector<uint32_t>& tokens);

    // Инициализация и построение
    void initialize_special_tokens();
    void build_byte_to_id_table();
    void sort_merges_by_rank();
    
    // Синхронизация Vocabulary с внутренними структурами
    void sync_vocab_from_maps();    // vocab_ <- id_to_token_
    void sync_maps_from_vocab();    // id_to_token_, token_to_id_ <- vocab_

    // Обучение
    std::unordered_map<std::string, int> count_char_frequencies_parallel(
        const std::vector<std::string>& corpus);
    void build_initial_vocabulary(
        const std::unordered_map<std::string, int>& char_freq);

    // ========================================================================
    // Поля класса
    // ========================================================================

    // Основные структуры данных
    std::vector<std::string> id_to_token_;                     ///< Отображение ID -> токен
    std::unordered_map<std::string, uint32_t> token_to_id_;    ///< Отображение токен -> ID
    
    // Vocabulary для совместимости с ParallelTrainer
    Vocabulary vocab_;    ///< Словарь (обертка)
    
    // Правила слияния (упакованы в uint64_t)
    std::unordered_map<uint64_t, uint32_t> merges_;              ///< Правила слияния (ключ -> ранг)
    std::vector<std::pair<uint64_t, uint32_t>> sorted_merges_;   ///< Отсортированные по рангу
    
    // Быстрые правила для encode (предварительно вычисленные)
    std::unordered_map<uint64_t, uint32_t> merge_rule_map_;    // key -> result_id
    
    // Оптимизации производительности
    std::unique_ptr<StringViewCache> cache_;
    mutable MemoryPool<4096> memory_pool_;

    // Таблица быстрого доступа байт -> ID
    std::array<uint32_t, 256> byte_to_id_;

    // Конфигурация и состояние
    TokenizerConfig config_;
    mutable TokenizerStats stats_;
    mutable std::shared_mutex mutex_;

    // Кэшированные ID специальных токенов
    uint32_t unknown_id_{0};
    uint32_t pad_id_{0};
    uint32_t bos_id_{0};
    uint32_t eos_id_{0};
    uint32_t mask_id_{0};
};

}    // namespace bpe

#endif    // BPE_FAST_TOKENIZER_HPP