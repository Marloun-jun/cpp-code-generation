/**
 * @file parallel_trainer.hpp
 * @brief Параллельное обучение BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 2.0.0
 * 
 * @details Класс для многопоточного обучения BPE токенизатора.
 *          Реализует параллельные алгоритмы для:
 *          - Подсчета частот символов
 *          - Поиска наиболее частых пар
 *          - Применения слияний к корпусу
 * 
 * @note Использует OpenMP для параллелизации циклов
 * @warning Требует потокобезопасных структур данных
 * 
 * @see FastBPETokenizer
 */

#pragma once

#include "optimized_types.hpp"
#include "vocabulary.hpp"
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

namespace bpe {

/**
 * @brief Класс для параллельного обучения BPE токенизатора
 * 
 * Реализует многопоточную версию алгоритма BPE:
 * 1. Параллельный подсчет частот символов
 * 2. Параллельный подсчет частот пар
 * 3. Параллельное применение слияний
 * 
 * @note Все внутренние структуры данных защищены мьютексами
 */
class ParallelTrainer {
public:
    /**
     * @brief Конструктор
     * @param num_threads Количество потоков (0 = автоопределение)
     */
    explicit ParallelTrainer(size_t num_threads = 0);

    /**
     * @brief Деструктор
     */
    ~ParallelTrainer();

    // Запрещаем копирование
    ParallelTrainer(const ParallelTrainer&) = delete;
    ParallelTrainer& operator=(const ParallelTrainer&) = delete;

    // Разрешаем перемещение
    ParallelTrainer(ParallelTrainer&&) noexcept = default;
    ParallelTrainer& operator=(ParallelTrainer&&) noexcept = default;

    /**
     * @brief Обучить токенизатор на корпусе текстов
     * @param corpus Вектор строк для обучения
     * @param target_size Целевой размер словаря
     * @param vocab [out] Словарь для заполнения
     * @param merges [out] Карта слияний для заполнения
     * @return true при успешном обучении
     * 
     * @details Основной метод обучения:
     *          - Разбивает корпус на чанки для параллельной обработки
     *          - Выполняет итеративные слияния до достижения target_size
     *          - Собирает статистику производительности
     */
    bool train(const std::vector<std::string>& corpus,
               size_t target_size,
               Vocabulary& vocab,
               std::unordered_map<merge_key_t, int>& merges);

    /**
     * @brief Получить прогресс обучения (0.0 - 1.0)
     */
    float progress() const { return progress_.load(std::memory_order_relaxed); }

    /**
     * @brief Прервать обучение
     */
    void cancel() { cancel_.store(true, std::memory_order_relaxed); }

    /**
     * @brief Получить статистику обучения
     */
    struct Stats {
        size_t total_merges{0};           ///< Всего выполнено слияний
        double total_time_sec{0.0};        ///< Общее время обучения
        double freq_time_sec{0.0};         ///< Время подсчета частот
        double merge_time_sec{0.0};        ///< Время применения слияний
        size_t peak_memory_bytes{0};        ///< Пиковое использование памяти
    };

    const Stats& stats() const { return stats_; }

private:
    // ==================== Внутренние структуры ====================

    /**
     * @brief Чанк корпуса для параллельной обработки
     */
    struct CorpusChunk {
        size_t start_idx;                   ///< Начальный индекс
        size_t end_idx;                      ///< Конечный индекс
        std::vector<std::string> texts;      ///< Тексты в чанке
    };

    // ==================== Приватные методы ====================

    /**
     * @brief Разбить корпус на чанки
     */
    std::vector<CorpusChunk> split_corpus(const std::vector<std::string>& corpus);

    /**
     * @brief Параллельный подсчет частот символов
     */
    std::unordered_map<std::string, size_t> count_char_frequencies_parallel(
        const std::vector<CorpusChunk>& chunks);

    /**
     * @brief Параллельный подсчет частот пар
     */
    std::unordered_map<merge_key_t, size_t> count_pair_frequencies_parallel(
        const std::vector<CorpusChunk>& chunks,
        const Vocabulary& vocab);

    /**
     * @brief Применить слияние к корпусу параллельно
     */
    void apply_merge_parallel(
        std::vector<CorpusChunk>& chunks,
        merge_key_t merge_pair,
        const std::string& new_token,
        Vocabulary& vocab);

    /**
     * @brief Обновить прогресс
     */
    void update_progress(size_t current, size_t total);

    // ==================== Поля класса ====================

    size_t num_threads_;                     ///< Количество потоков
    std::atomic<float> progress_{0.0f};       ///< Текущий прогресс
    std::atomic<bool> cancel_{false};         ///< Флаг отмены
    Stats stats_;                             ///< Статистика обучения
    mutable std::mutex stats_mutex_;           ///< Мьютекс для статистики
};

} // namespace bpe