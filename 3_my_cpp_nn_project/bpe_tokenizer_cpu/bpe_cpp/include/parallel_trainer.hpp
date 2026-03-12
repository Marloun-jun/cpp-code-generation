/**
 * @file parallel_trainer.hpp
 * @brief Параллельное обучение BPE токенизатора с использованием нескольких ядер
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Этот класс реализует высокопроизводительное параллельное обучение
 *          BPE токенизатора на многоядерных процессорах. Основные возможности:
 * 
 *          **Автоматическое определение числа ядер**    - использует все доступные
 *          **Разбиение корпуса на чанки**               - равномерное распределение нагрузки
 *          **Параллельный подсчет частот**              - для символов и пар
 *          **Безопасное применение слияний**            - синхронизация через мьютексы
 *          **Прогресс обучения**                        - атомарный счетчик для UI/CLI
 *          **Возможность отмены**                       - безопасное прерывание долгого обучения
 *          **Сбор статистики**                          - время, память, количество операций
 * 
 *          **Производительность:**
 *          - Ускорение:          до 8x на 8-ядерном процессоре
 *          - Масштабирование:    почти линейное до 16 ядер
 *          - Память:             O(corpus_size / num_threads) на поток
 * 
 * @note Требует C++17 и поддержки многопоточности
 * @warning Класс некопируемый для безопасности ресурсов
 * 
 * @see FastBPETokenizer::parallel_train()
 * @see Vocabulary
 * @see merge_key_t
 */

#ifndef BPE_PARALLEL_TRAINER_HPP
#define BPE_PARALLEL_TRAINER_HPP

#include "optimized_types.hpp"
#include "vocabulary.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace bpe {

// ======================================================================
// ParallelTrainer - основной класс для параллельного обучения
// ======================================================================

/**
 * @brief Класс для параллельного обучения BPE токенизатора
 * 
 * Реализует алгоритм BPE обучения с распараллеливанием критических секций:
 * 1. Подсчет частот символов (embarrassingly parallel)
 * 2. Подсчет частот пар (разделение корпуса)
 * 3. Применение слияний (с синхронизацией)
 * 
 * \include examples/parallel_train_example.cpp
 * Пример использования:
 * \code
 * ParallelTrainer trainer(std::thread::hardware_concurrency());
 * Vocabulary vocab;
 * std::unordered_map<merge_key_t, int> merges;
 * 
 * std::vector<std::string> corpus = load_corpus("data.txt");
 * 
 * // Запуск обучения с отображением прогресса
 * trainer.train(corpus, 8000, vocab, merges);
 * 
 * // Получение статистики
 * auto stats = trainer.stats();
 * std::cout << "Обучение заняло " << stats.total_time_sec << " с\n";
 * \endcode
 */
class ParallelTrainer {
public:
    // ==================== Публичные типы ====================

    /**
     * @brief Чанк корпуса для параллельной обработки
     */
    struct CorpusChunk {
        size_t start_idx;                  ///< Начальный индекс в исходном корпусе
        size_t end_idx;                    ///< Конечный индекс (не включая)
        std::vector<std::string> texts;    ///< Тексты в чанке (для локальной модификации)
        
        /**
         * @brief Получить размер чанка в байтах
         */
        size_t total_bytes() const;
        
        /**
         * @brief Получить количество текстов в чанке
         */
        size_t size() const { return texts.size(); }
        
        /**
         * @brief Проверить, пуст ли чанк
         */
        bool empty() const { return texts.empty(); }
    };

    /**
     * @brief Структура со статистикой обучения
     */
    struct Stats {
        size_t total_merges{0};         ///< Количество выполненных слияний
        double total_time_sec{0.0};     ///< Общее время обучения (с)
        double freq_time_sec{0.0};      ///< Время подсчета частот (с)
        double merge_time_sec{0.0};     ///< Время применения слияний (с)
        size_t peak_memory_bytes{0};    ///< Пиковое использование памяти (байт)
        
        /**
         * @brief Получить скорость обучения (слияний в секунду)
         */
        double merges_per_second() const;
        
        /**
         * @brief Получить эффективность использования памяти
         */
        double memory_per_merge() const;
        
        /**
         * @brief Сбросить статистику
         */
        void reset();
    };

    // ==================== Конструкторы и деструктор ====================

    /**
     * @brief Конструктор с указанием количества потоков
     * 
     * @param num_threads Количество потоков для параллельной обработки.
     *                    Если <= 0, используется количество аппаратных потоков.
     */
    explicit ParallelTrainer(int num_threads = 0);

    /**
     * @brief Деструктор
     */
    ~ParallelTrainer() = default;

    // Запрещаем копирование (RAII для ресурсов)
    ParallelTrainer(const ParallelTrainer&) = delete;
    ParallelTrainer& operator=(const ParallelTrainer&) = delete;

    // Разрешаем перемещение
    ParallelTrainer(ParallelTrainer&&) noexcept = default;
    ParallelTrainer& operator=(ParallelTrainer&&) noexcept = default;

    // ==================== Основные методы ====================

    /**
     * @brief Запустить параллельное обучение
     * 
     * @param corpus Исходный корпус текстов
     * @param target_size Целевой размер словаря
     * @param vocab [in,out] Словарь (будет заполнен)
     * @param merges [in,out] Правила слияния (будут заполнены)
     * @return true если обучение успешно завершено
     */
    bool train(const std::vector<std::string>& corpus,
               size_t target_size,
               Vocabulary& vocab,
               std::unordered_map<merge_key_t, int>& merges);

    // ==================== Методы для тестов и мониторинга ====================

    /**
     * @brief Получить количество используемых потоков
     */
    size_t num_threads() const { return num_threads_; }

    /**
     * @brief Разбить корпус на указанное количество чанков
     */
    std::vector<CorpusChunk> split_corpus(
        const std::vector<std::string>& corpus, 
        size_t num_chunks);

    /**
     * @brief Подсчет частот символов в параллельном режиме
     */
    std::unordered_map<std::string, size_t> count_char_frequencies_parallel(
        const std::vector<std::string>& corpus);

    /**
     * @brief Получить текущий прогресс обучения (0.0 - 1.0)
     */
    float progress() const { return progress_.load(std::memory_order_relaxed); }

    /**
     * @brief Отменить обучение
     */
    void cancel() { cancel_.store(true, std::memory_order_relaxed); }

    /**
     * @brief Проверить, была ли запрошена отмена
     */
    bool is_cancelled() const { return cancel_.load(std::memory_order_relaxed); }

    /**
     * @brief Сбросить статистику
     */
    void reset_stats();

    // ==================== Статистика ====================

    /**
     * @brief Получить статистику обучения
     */
    Stats stats() const;

private:
    // ==================== Приватные методы ====================

    /**
     * @brief Подсчет частот пар в параллельном режиме
     */
    std::unordered_map<merge_key_t, size_t> count_pair_frequencies_parallel(
        const std::vector<CorpusChunk>& chunks,
        const Vocabulary& vocab);

    /**
     * @brief Применить слияние пары ко всем чанкам параллельно
     */
    void apply_merge_parallel(
        std::vector<CorpusChunk>& chunks,
        merge_key_t merge_pair,
        const std::string& new_token,
        Vocabulary& vocab);

    /**
     * @brief Найти самую частую пару из локальных карт частот
     */
    std::pair<merge_key_t, size_t> find_best_merge(
        const std::vector<std::unordered_map<merge_key_t, size_t>>& local_freqs);

    /**
     * @brief Обновить прогресс обучения
     */
    void update_progress(size_t current, size_t total);

    /**
     * @brief Обновить статистику использования памяти
     */
    void update_memory_usage(size_t bytes);

    /**
     * @brief Получить текущее использование памяти
     */
    size_t get_current_memory_usage() const;

    // ==================== Поля класса ====================

    size_t num_threads_;                   ///< Количество потоков
    std::atomic<float> progress_{0.0f};    ///< Прогресс обучения (0-1)
    std::atomic<bool> cancel_{false};      ///< Флаг отмены
    
    mutable Stats stats_;                  ///< Статистика обучения
    mutable std::mutex stats_mutex_;       ///< Мьютекс для статистики
    mutable std::mutex vocab_mutex_;       ///< Мьютекс для словаря
    mutable std::mutex merges_mutex_;      ///< Мьютекс для правил слияния
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;    ///< Время старта
};

} // namespace bpe

#endif // BPE_PARALLEL_TRAINER_HPP

/**
 * @example examples/parallel_train_example.cpp
 * Полный пример использования ParallelTrainer
 */