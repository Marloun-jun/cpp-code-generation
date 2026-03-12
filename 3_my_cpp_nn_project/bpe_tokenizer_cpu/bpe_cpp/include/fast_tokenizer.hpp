/**
 * @file fast_tokenizer.hpp
 * @brief Оптимизированная версия BPE токенизатора с поддержкой SIMD и пулов памяти
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Высокопроизводительная реализация BPE токенизатора с фокусом на скорость.
 *          Этот класс является оптимизированной альтернативой базовому BPETokenizer
 *          и обеспечивает значительный прирост производительности за счет:
 * 
 *          **StringView**               - избегание копирования строк при передаче параметров
 *          **Пул памяти**               - уменьшение количества аллокаций для временных строк
 *          **Кэширование**              - хранение результатов для часто встречающихся слов
 *          **SIMD-оптимизации**         - векторные инструкции (AVX2, SSE4.2)
 *          **OpenMP**                   - параллельная обработка на нескольких ядрах
 *          **Thread-safe**              - поддержка многопоточного доступа через shared_mutex
 *          **Профилирование**           - встроенный сбор статистики производительности
 *          **Параллельное обучение**    - эффективное использование многоядерных процессоров
 * 
 *          **Производительность:**
 *          - Ускорение encode:             до 5-10x по сравнению с BPETokenizer
 *          - Ускорение batch encode:       до 20x для больших батчей
 *          - Снижение аллокаций памяти:    до 90%
 *          - Hit rate кэша:                60-80% для типичных текстов
 * 
 * @note Требует C++17 и поддержки как минимум SSE4.2 для полной функциональности
 * @warning Класс некопируемый, но перемещаемый для безопасности ресурсов
 * 
 * @see BPETokenizer
 * @see MemoryPool
 * @see StringViewCache
 * @see TokenizerConfig
 * @see TokenizerStats
 */

#ifndef BPE_FAST_TOKENIZER_HPP
#define BPE_FAST_TOKENIZER_HPP

#ifdef HAS_CONFIG_H
    #include "config.h"    // Генерируется CMake из config.h.in
#endif

#include "memory_pool.hpp"
#include "optimized_types.hpp"
#include "thread_safe_cache.hpp"
#include "bpe_export.hpp"

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <functional>
#include <optional>
#include <utility>  // для std::pair
#include <sstream>  // для std::stringstream в get_model_info()
#include <iomanip>  // для форматирования

namespace bpe {

// ======================================================================
// Основной класс токенизатора
// ======================================================================

/**
 * @brief Оптимизированная версия BPE токенизатора
 * 
 * FastBPETokenizer обеспечивает значительный прирост производительности
 * по сравнению с базовой реализацией за счет современных оптимизаций.
 * 
 * \include examples/fast_tokenizer_demo.cpp
 * 
 * Пример использования:
 * \code
 * // Создание с конфигурацией по умолчанию
 * FastBPETokenizer tokenizer;
 * 
 * // Загрузка обученной модели
 * tokenizer.load("vocab.json", "merges.txt");
 * 
 * // Кодирование текста
 * std::string_view code = "int main() { return 0; }";
 * auto tokens = tokenizer.encode(code);
 * 
 * // Декодирование обратно
 * std::string decoded = tokenizer.decode(tokens);
 * 
 * // Пакетная обработка
 * std::vector<std::string_view> batch = {"text1", "text2"};
 * auto batch_results = tokenizer.encode_batch(batch);
 * 
 * // Статистика производительности
 * auto stats = tokenizer.stats();
 * std::cout << "Cache hit rate: " << stats.cache_hit_rate() << "%\n";
 * \endcode
 */
class FastBPETokenizer {
public:
    // ==================== Конструкторы и деструктор ====================

    /**
     * @brief Конструктор с конфигурацией
     * 
     * @param config Настройки токенизатора (размер словаря, режимы работы и т.д.)
     * 
     * Если конфигурация не указана, используются значения по умолчанию:
     * - vocab_size:          8000
     * - cache_size:          10000
     * - byte_level:          true
     * - enable_cache:        true
     * - enable_profiling:    false
     * - use_memory_pool:     true
     * - num_threads:         0 (auto)
     * 
     * @throws std::invalid_argument если конфигурация невалидна
     */
    explicit FastBPETokenizer(const TokenizerConfig& config = TokenizerConfig{});

    /**
     * @brief Деструктор
     * 
     * При включенном профилировании сохраняет отчет в файл.
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
     * 
     * @param text Входной текст (string_view для избегания копирования)
     * @return std::vector<uint32_t> Вектор идентификаторов токенов
     * 
     * Алгоритм работы:
     * 1. Проверка кэша (если включен)
     * 2. Byte-level преобразование (если включено)
     * 3. Применение BPE слияний
     * 4. Сохранение результата в кэш
     * 
     * Потокобезопасность:    использует shared_lock для параллельного чтения
     * 
     * @note Для максимальной производительности используйте string_view,
     *       избегая создания временных строк.
     */
    std::vector<uint32_t> encode(std::string_view text);

    /**
     * @brief Закодировать текст в последовательность токенов (константная версия)
     * 
     * @param text Входной текст
     * @return std::vector<uint32_t> Вектор идентификаторов токенов
     */
    std::vector<uint32_t> encode(const std::string& text) {
        return encode(std::string_view(text));
    }

    /**
     * @brief Декодировать последовательность токенов обратно в текст
     * 
     * @param tokens Вектор идентификаторов токенов
     * @return std::string Восстановленный текст
     * 
     * Алгоритм:
     * 1. Конкатенация строковых представлений токенов
     * 2. Byte-level декодирование (если включено)
     * 3. Удаление специальных токенов
     */
    std::string decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Пакетное кодирование нескольких текстов
     * 
     * @param texts Вектор входных текстов (string_view)
     * @return std::vector<std::vector<uint32_t>> Вектор результатов
     * 
     * Преимущества перед последовательными вызовами:
     * - Параллельная обработка с OpenMP
     * - Меньше блокировок мьютекса
     * - Лучшая локализация данных
     * 
     * @note Для максимальной производительности размер батча
     *       должен быть кратен количеству ядер процессора.
     */
    std::vector<std::vector<uint32_t>> encode_batch(
        const std::vector<std::string_view>& texts);

    /**
     * @brief Пакетное кодирование нескольких текстов (версия с std::string)
     * 
     * @param texts Вектор входных текстов
     * @return std::vector<std::vector<uint32_t>> Вектор результатов
     */
    std::vector<std::vector<uint32_t>> encode_batch(
        const std::vector<std::string>& texts);

    // ==================== Обучение ====================

    /**
     * @brief Обучить токенизатор на корпусе текстов
     * 
     * @param corpus Вектор строк для обучения
     * 
     * Последовательная версия обучения. Для больших корпусов
     * рекомендуется использовать parallel_train().
     * 
     * Алгоритм:
     * 1. Подсчет частот символов
     * 2. Построение начального словаря
     * 3. Итеративное слияние самых частых пар
     */
    void train(const std::vector<std::string>& corpus);

    /**
     * @brief Параллельное обучение с использованием нескольких ядер
     * 
     * @param corpus Вектор строк для обучения
     * @param num_merges Количество операций слияния
     * 
     * Особенности:
     * - Корпус разбивается на чанки
     * - Каждый чанк обрабатывается в отдельном потоке
     * - Используется OpenMP или std::thread
     * - Отображается прогресс в консоли
     * 
     * @note Требует компиляции с поддержкой OpenMP
     */
    void parallel_train(const std::vector<std::string>& corpus, size_t num_merges = 8000);

    // ==================== Загрузка/сохранение ====================

    /**
     * @brief Загрузить модель из текстовых файлов
     * 
     * @param vocab_path Путь к файлу словаря (JSON формат)
     * @param merges_path Путь к файлу слияний (TXT формат)
     * @return true при успешной загрузке, false при ошибке
     * 
     * Поддерживаемые форматы JSON:
     * 1. Объект с массивом tokens:    {"tokens": ["token1", "token2", ...]}
     * 2. Объект с ID ключами:         {"0": "token1", "1": "token2", ...}
     * 3. Прямой массив:               ["token1", "token2", ...]
     * 
     * Формат merges.txt:
     * token1 token2
     * token3 token4
     * ...
     */
    bool load(const std::string& vocab_path, const std::string& merges_path);

    /**
     * @brief Сохранить модель в текстовые файлы
     * 
     * @param vocab_path Путь для сохранения словаря
     * @param merges_path Путь для сохранения слияний
     * @return true при успешном сохранении, false при ошибке
     * 
     * Сохраняет словарь в формате JSON с ID в качестве ключей
     * и слияния в текстовом формате с рангами.
     */
    bool save(const std::string& vocab_path, const std::string& merges_path) const;

    /**
     * @brief Сохранить модель в единый бинарный файл (быстрая загрузка)
     * 
     * @param path Путь для сохранения (рекомендуется .bin)
     * @return true при успешном сохранении, false при ошибке
     * 
     * @note В текущей версии не реализовано
     */
    bool save_binary(const std::string& path) const;

    /**
     * @brief Загрузить модель из бинарного файла
     * 
     * @param path Путь к бинарному файлу
     * @return true при успешной загрузке, false при ошибке
     * 
     * @note В текущей версии не реализовано
     */
    bool load_binary(const std::string& path);

    // ==================== Геттеры ====================

    /**
     * @brief Получить размер словаря
     * @return size_t Количество токенов
     */
    size_t vocab_size() const { 
        std::shared_lock lock(mutex_);
        return id_to_token_.size(); 
    }

    /**
     * @brief Получить количество правил слияния
     * @return size_t Количество пар в merges_
     */
    size_t merges_count() const { 
        std::shared_lock lock(mutex_);
        return merges_.size(); 
    }

    /**
     * @brief Получить статистику производительности
     * @return const TokenizerStats& Константная ссылка на статистику
     */
    const TokenizerStats& stats() const { 
        return stats_; 
    }

    /**
     * @brief Сбросить статистику производительности
     */
    void reset_stats() { 
        stats_.reset(); 
    }

    // ==================== Специальные токены ====================

    /**
     * @brief Получить ID токена для неизвестных символов
     * @return uint32_t ID токена <UNK>
     */
    uint32_t unknown_id() const { 
        return unknown_id_; 
    }

    /**
     * @brief Получить ID токена для паддинга
     * @return uint32_t ID токена <PAD>
     */
    uint32_t pad_id() const { 
        return pad_id_; 
    }

    /**
     * @brief Получить ID токена начала последовательности
     * @return uint32_t ID токена <BOS>
     */
    uint32_t bos_id() const { 
        return bos_id_; 
    }

    /**
     * @brief Получить ID токена конца последовательности
     * @return uint32_t ID токена <EOS>
     */
    uint32_t eos_id() const { 
        return eos_id_; 
    }

    /**
     * @brief Получить ID токена маски
     * @return uint32_t ID токена <MASK>
     */
    uint32_t mask_id() const { 
        return mask_id_; 
    }

    // ==================== Информация о модели ====================

    /**
     * @brief Получить информацию о модели в читаемом виде
     * @return std::string Многострочное описание модели
     * 
     * Формирует строку с основными характеристиками модели:
     * - Размер словаря
     * - Количество правил слияния
     * - Режим работы (byte-level)
     * - ID специальных токенов
     * - Настройки кэширования
     * 
     * Пример вывода:
     * ===================================================
     * ИНФОРМАЦИЯ О FAST BPE TOKENIZER
     * ===================================================
     * Размер словаря:               8000
     * Количество слияний:           7999
     * Byte-level режим:             включен
     * Неизвестных токенов:          <UNK> (ID: 0)
     * Pad токен:                     <PAD> (ID: 1)
     * BOS токен:                     <BOS> (ID: 2)
     * EOS токен:                     <EOS> (ID: 3)
     * Mask токен:                    <MASK> (ID: 4)
     * Кэширование:                   включено (размер: 10000)
     * Пул памяти:                    включен
     * Количество потоков:            auto
     * ===================================================
     */
    std::string get_model_info() const {
        std::shared_lock lock(mutex_);
        
        std::stringstream ss;
        ss << "\n";
        for (int i = 0; i < 50; ++i) ss << '=';
        ss << "\n";
        ss << "ИНФОРМАЦИЯ О FAST BPE TOKENIZER\n";
        ss << "\n";
        for (int i = 0; i < 50; ++i) ss << '=';
        ss << "\n";
        ss << "Размер словаря:               " << id_to_token_.size() << "\n";
        ss << "Количество слияний:           " << merges_.size() << "\n";
        ss << "Byte-level режим:             " << (config_.byte_level ? "включен" : "отключен") << "\n";
        ss << "Неизвестных токенов:          <UNK> (ID: " << unknown_id_ << ")\n";
        ss << "Pad токен:                    <PAD> (ID: " << pad_id_ << ")\n";
        ss << "BOS токен:                    <BOS> (ID: " << bos_id_ << ")\n";
        ss << "EOS токен:                    <EOS> (ID: " << eos_id_ << ")\n";
        ss << "Mask токен:                   <MASK> (ID: " << mask_id_ << ")\n";
        ss << "Кэширование:                  " << (config_.enable_cache ? "включено" : "отключено") 
           << (config_.enable_cache ? " (размер: " + std::to_string(config_.cache_size) + ")" : "") << "\n";
        ss << "Пул памяти:                   " << (config_.use_memory_pool ? "включен" : "отключен") << "\n";
        ss << "Количество потоков:           " << (config_.num_threads == 0 ? "auto" : std::to_string(config_.num_threads)) << "\n";
        ss << "\n";
        for (int i = 0; i < 50; ++i) ss << '=';
        ss << "\n";
        return ss.str();
    }

private:
    // ==================== Приватные методы ====================

    /**
     * @brief Токенизация отдельного слова
     * 
     * @param word Слово для токенизации
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * Применяет BPE слияния к отдельному слову.
     * Использует кэш для ускорения повторяющихся слов.
     */
    std::vector<uint32_t> tokenize_word(std::string_view word);

    /**
     * @brief Byte-level кодирование текста
     * 
     * @param text Входной текст
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * Преобразует UTF-8 текст в последовательность байтовых токенов.
     * Использует lookup table для O(1) доступа.
     * При наличии AVX2 использует SIMD-оптимизированную версию.
     */
    std::vector<uint32_t> byte_level_encode(std::string_view text);

    /**
     * @brief Обычное кодирование текста (с предтокенизацией)
     * 
     * @param text Входной текст
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * Разбивает текст на слова, затем применяет BPE к каждому слову.
     */
    std::vector<uint32_t> normal_encode(std::string_view text);

    /**
     * @brief Byte-level декодирование токенов
     * 
     * @param tokens Вектор ID токенов
     * @return std::string Восстановленный текст
     * 
     * Преобразует последовательность байтовых токенов обратно в UTF-8.
     */
    std::string byte_level_decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Обычное декодирование токенов
     * 
     * @param tokens Вектор ID токенов
     * @return std::string Восстановленный текст
     * 
     * Конкатенирует строковые представления токенов.
     */
    std::string normal_decode(const std::vector<uint32_t>& tokens);

    /**
     * @brief Инициализировать специальные токены в словаре
     * 
     * Добавляет <UNK>, <PAD>, <BOS>, <EOS> в словарь
     * и сохраняет их ID для быстрого доступа.
     */
    void initialize_special_tokens();

    /**
     * @brief Построить обратную карту token -> ID
     * 
     * Создает token_to_id_ из id_to_token_ для быстрого поиска.
     */
    void build_token_to_id_map();

    /**
     * @brief Подсчет частот символов в параллельном режиме
     * 
     * @param corpus Корпус текстов
     * @return std::unordered_map<std::string, int> Карта частот символов
     * 
     * Использует несколько потоков для ускорения обработки больших корпусов.
     * Отображает прогресс в консоли.
     */
    std::unordered_map<std::string, int> count_char_frequencies_parallel(
        const std::vector<std::string>& corpus);

    /**
     * @brief Построение начального словаря на основе частот символов
     * 
     * @param char_freq Карта частот символов
     * 
     * Создает словарь из всех символов, встречающихся в корпусе.
     * Добавляет специальные токены в начало словаря.
     */
    void build_initial_vocabulary(
        const std::unordered_map<std::string, int>& char_freq);

#ifdef USE_AVX2
    /**
     * @brief AVX2-оптимизированная версия токенизации слова
     * 
     * @param word Слово для токенизации
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * Использует 256-битные векторные инструкции для обработки
     * 32 символов за одну операцию. Обеспечивает ускорение до 4x.
     */
    std::vector<uint32_t> tokenize_word_avx2(std::string_view word);
#endif

    // ==================== Поля класса ====================

    // Основные структуры данных
    std::vector<std::string> id_to_token_;                     ///< Отображение ID -> токен
    std::unordered_map<std::string, uint32_t> token_to_id_;    ///< Отображение токен -> ID
    
    // Используем uint64_t как ключ для слияний (левый ID << 32 | правый ID)
    // Это эффективнее и не требует кастомной хеш-функции
    std::unordered_map<uint64_t, uint32_t> merges_;  ///< Правила слияния с рангами

    // Оптимизации
    std::unique_ptr<StringViewCache> cache_;    ///< Кэш для частых слов
    mutable MemoryPool<4096> memory_pool_;      ///< Пул памяти для строк (4KB блоки)

    // Конфигурация и состояние
    TokenizerConfig config_;             ///< Настройки токенизатора
    mutable TokenizerStats stats_;       ///< Статистика производительности
    mutable std::shared_mutex mutex_;    ///< Мьютекс для потокобезопасности

    // ID специальных токенов
    uint32_t unknown_id_{0};    ///< ID токена <UNK> (неизвестный символ)
    uint32_t pad_id_{0};        ///< ID токена <PAD> (padding для батчей)
    uint32_t bos_id_{0};        ///< ID токена <BOS> (beginning of sequence)
    uint32_t eos_id_{0};        ///< ID токена <EOS> (end of sequence)
    uint32_t mask_id_{0};       ///< ID токена <MASK> (для masked language modeling)
};

} // namespace bpe

/**
 * @example examples/fast_tokenizer_demo.cpp
 * Полный пример использования FastBPETokenizer с демонстрацией всех возможностей.
 * 
 * @include examples/fast_tokenizer_demo.cpp
 */

#endif // BPE_FAST_TOKENIZER_HPP