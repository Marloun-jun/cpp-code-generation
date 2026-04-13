/**
 * @file bpe_tokenizer.hpp
 * @brief Основной заголовочный файл BPE токенизатора (базовая версия)
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Объявление класса BPETokenizer - эталонной реализации алгоритма Byte Pair Encoding.
 *          Эта версия служит референсной для сравнения с оптимизированной FastBPETokenizer.
 * 
 *          **Ключевые возможности:**
 *          - Полная реализация BPE алгоритма (обучение, кодирование, декодирование)
 *          - Поддержка всех основных форматов сериализации (JSON, бинарный, HuggingFace)
 *          - Потокобезопасность через shared_mutex (множество читателей / один писатель)
 *          - Корректная обработка UTF-8 через byte-level режим
 *          - Специальные токены для NLP задач (<UNK>, <PAD>, <BOS>, <EOS>, <MASK>)
 * 
 *          **Архитектурные решения:**
 *          - Разделение на интерфейс (ModelExport) и реализацию
 *          - Инкапсуляция словаря в отдельный класс Vocabulary
 *          - Метаданные для версионирования моделей
 *          - RAII подход с управлением ресурсами
 * 
 * @see FastBPETokenizer (оптимизированная версия)
 * @see Vocabulary, ModelExport, ModelMetadata
 */

#pragma once

#include "vocabulary.hpp"
#include "bpe_export.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace bpe {

// ============================================================================
// Типы-алиасы для улучшения читаемости кода
// ============================================================================

///< Тип идентификатора токена (32 бита достаточно для 4B токенов)
using token_id_t = uint32_t;
///< Карта частот для сбора статистики
using freq_map_t = std::unordered_map<std::string, size_t>;

// ============================================================================
// Структуры данных BPE алгоритма
// ============================================================================

/**
 * @brief Представляет пару для слияния в BPE алгоритме
 * 
 * Является ключевой структурой алгоритма - хранит два элемента,
 * которые объединяются в один токен в процессе обучения.
 * 
 * **Пример:** пара {"a", "b"} означает слияние "a" и "b" в "ab"
 * 
 * \include examples/merge_pair_example.cpp
 * Пример использования:
 * @code
 * MergePair pair{"int", "er"};
 * std::cout << "Слияние: " << pair.to_string() << std::endl;  // "int er"
 * 
 * // Использование в хеш-таблице
 * std::unordered_map<MergePair, int, MergePairHash> frequencies;
 * frequencies[pair] = 42;
 * @endcode
 */
struct MergePair {
    std::string left;     ///< Левый элемент (символ, подслово или токен)
    std::string right;    ///< Правый элемент (символ, подслово или токен)

    /**
     * @brief Оператор равенства для использования в контейнерах
     */
    bool operator==(const MergePair& other) const {
        return left == other.left && right == other.right;
    }

    /**
     * @brief Получить строковое представление для отладки
     * @return Строка формата "left right"
     */
    std::string to_string() const {
        return left + " " + right;
    }
};

/**
 * @brief Хеш-функция для MergePair (для unordered_map)
 * 
 * Комбинирует хеши левой и правой частей через разделитель '|',
 * чтобы избежать коллизий типа ("a", "bc") vs ("ab", "c").
 */
struct MergePairHash {
    size_t operator()(const MergePair& p) const {
        return std::hash<std::string>()(p.left + "|" + p.right);
    }
};

// ============================================================================
// Основной класс токенизатора
// ============================================================================

/**
 * @brief Базовая реализация BPE токенизатора
 * 
 * **Назначение:** Эталонная реализация, демонстрирующая алгоритм BPE.
 * 
 * **Алгоритмическая сложность:**
 * - encode: O(n * log(m)) где n - длина текста, m - количество слияний
 * - decode: O(n) линейная от количества токенов
 * - train:  O(k * n * log(n)) где k - размер словаря, n - размер корпуса
 * 
 * **Потокобезопасность:**
 * - encode() / decode() - Разделяемая блокировка (shared_lock)
 * - train() / load():   - Исключительная блокировка (unique_lock)
 * 
 * **Жизненный цикл:**
 * 1. Создание (конструктор)
 * 2. Обучение на корпусе ИЛИ загрузка готовой модели
 * 3. Использование (encode/decode)
 * 4. Сохранение (опционально)
 * 
 * \include examples/bpe_tokenizer_example.cpp
 * @code
 * // Инициализация
 * BPETokenizer tokenizer(10000, true);
 * 
 * // Обучение
 * std::vector<std::string> corpus = load_corpus("data.txt");
 * tokenizer.train_with_progress(corpus);
 * 
 * // Сохранение модели
 * tokenizer.save_to_files("vocab.json", "merges.txt");
 * tokenizer.save_binary("model.bin");
 * 
 * // Использование
 * auto tokens = tokenizer.encode("int main() { return 0; }");
 * std::string code = tokenizer.decode(tokens);
 * 
 * // Экспорт для Python
 * tokenizer.export_to_huggingface("hf_model/");
 * @endcode
 */
class BPETokenizer : public ModelExport {
public:
    // ========================================================================
    // Конструкторы и управление ресурсами
    // ========================================================================

    /**
     * @brief Конструктор по умолчанию (vocab_size=10000, byte_level=true)
     * 
     * Создает токенизатор с параметрами по умолчанию.
     * Требуется либо загрузить модель, либо обучить новую.
     */
    BPETokenizer();

    /**
     * @brief Конструктор с явными параметрами
     * 
     * @param vocab_size Целевой размер словаря (обычно 1000-50000)
     * @param byte_level Режим обработки UTF-8 как байтов (рекомендуется true)
     * 
     * @note Byte-level режим критически важен для работы с Unicode.
     *       Без него токенизатор не сможет обрабатывать символы за пределами ASCII.
     */
    explicit BPETokenizer(size_t vocab_size, bool byte_level = true);

    /**
     * @brief Деструктор
     */
    virtual ~BPETokenizer() override;

    // Запрет копирования (из-за mutex)
    BPETokenizer(const BPETokenizer&) = delete;
    BPETokenizer& operator=(const BPETokenizer&) = delete;

    // Разрешение перемещения
    BPETokenizer(BPETokenizer&& other) noexcept;
    BPETokenizer& operator=(BPETokenizer&& other) noexcept;

    // ========================================================================
    // Конфигурация токенизатора
    // ========================================================================

    /**
     * @brief Включить/выключить byte-level режим
     * @param enable true - обрабатывать UTF-8 как байты, false - как символы
     * 
     * @warning Изменение режима после обучения приведет к несовместимости
     */
    void set_byte_level(bool enable);

    /**
     * @brief Установить токен для неизвестных символов
     * @param token Строковое представление (например, "<UNK>")
     */
    void set_unknown_token(const std::string& token);

    /**
     * @brief Установить целевой размер словаря
     * @param size Желаемое количество токенов
     * 
     * @note Влияет на:
     *       - Качество токенизации (больше = лучше до определенного предела)
     *       - Скорость работы (больше = медленнее)
     *       - Потребление памяти (больше = больше памяти)
     */
    void set_vocab_size(size_t size);

    /**
     * @brief Установить максимальную длину токена
     * @param length Максимальная длина в символах (по умолчанию 1000)
     * 
     * Защита от слишком длинных токенов, которые могут возникнуть
     * при обучении на специфических данных.
     */
    void set_max_token_length(size_t length);

    // ========================================================================
    // Загрузка/сохранение в базовых форматах (для обратной совместимости)
    // ========================================================================

    /**
     * @brief Загрузить модель из двух файлов (формат Python реализации)
     * 
     * @param vocab_path Путь к vocab.json (словарь)
     * @param merges_path Путь к merges.txt (правила слияния)
     * @return true при успешной загрузке
     * 
     * **Формат vocab.json:**
     * @code
     * {
     *   "token1": 0,
     *   "token2": 1,
     *   ...
     * }
     * @endcode
     * 
     * **Формат merges.txt:**
     * @code
     * int er
     * std ::
     * ...
     * @endcode
     */
    bool load_from_files(const std::string& vocab_path, const std::string& merges_path);

    /**
     * @brief Сохранить модель в два файла (для совместимости с Python)
     * 
     * @param vocab_path Путь для сохранения словаря
     * @param merges_path Путь для сохранения слияний
     * @return true при успешном сохранении
     */
    bool save_to_files(const std::string& vocab_path, const std::string& merges_path) const;

    // ========================================================================
    // Реализация интерфейса ModelExport (форматы сериализации)
    // ========================================================================

    // JSON формат (единый файл)
    bool save_to_json(const std::string& path) const override;
    bool load_from_json(const std::string& path) override;

    // Бинарный формат (максимальная производительность)
    bool save_binary(const std::string& path) const override;
    bool load_binary(const std::string& path) override;

    // Экспорт в форматы других библиотек
    bool export_to_huggingface(const std::string& output_dir) const override;
    bool export_to_sentencepiece(const std::string& path) const override;

    // Информация о модели
    std::string get_model_info() const override;

    // ========================================================================
    // Основные методы токенизации
    // ========================================================================

    /**
     * @brief Закодировать текст в последовательность ID токенов
     * 
     * @param text Входной текст (UTF-8)
     * @return std::vector<token_id_t> Вектор идентификаторов
     * 
     * **Алгоритм:**
     * 1. Пре-токенизация (разбиение на слова и пунктуацию)
     * 2. Для каждого слова:
     *    - Разбиение на символы (или байты в byte-level режиме)
     *    - Последовательное применение слияний от большего ранга к меньшему
     * 3. Конкатенация результатов всех слов
     * 
     * **Сложность:** O(n * log(m)) где n - длина текста, m - количество слияний
     * 
     * @note Потокобезопасно для параллельного чтения
     */
    std::vector<token_id_t> encode(const std::string& text) const;

    /**
     * @brief Декодировать последовательность ID обратно в текст
     * 
     * @param tokens Вектор идентификаторов токенов
     * @return std::string Восстановленный текст (UTF-8)
     * 
     * **Алгоритм:**
     * 1. Конвертация каждого ID в строковое представление
     * 2. Конкатенация с учетом специальных токенов
     * 3. Пост-обработка (удаление артефактов byte-level режима)
     * 
     * **Сложность:** O(n) линейная
     * 
     * @note Потокобезопасно для параллельного чтения
     */
    std::string decode(const std::vector<token_id_t>& tokens) const;

    /**
     * @brief Пакетное кодирование нескольких текстов
     * 
     * @param texts Вектор входных текстов
     * @return std::vector<std::vector<token_id_t>> Вектор результатов
     * 
     * **Оптимизация:** Использует одну блокировку на весь пакет вместо
     * блокировки на каждый текст. Эффективно для батчей размером > 10.
     */
    std::vector<std::vector<token_id_t>> encode_batch(
        const std::vector<std::string>& texts) const;

    // ========================================================================
    // Обучение модели
    // ========================================================================

    /**
     * @brief Обучить токенизатор на корпусе текстов
     * 
     * @param corpus Вектор строк для обучения
     * 
     * **Алгоритм обучения (итеративный):**
     * 1. Инициализация: разбить все слова на символы
     * 2. Подсчет частот всех соседних пар
     * 3. Выбор самой частой пары
     * 4. Слияние пары во всем корпусе
     * 5. Добавление нового токена в словарь
     * 6. Повторение шагов 2-5 до достижения vocab_size
     * 
     * **Сложность:** O(k * n * log(n)) где:
     * - k - Размер словаря (обычно 1000-50000)
     * - n - Размер корпуса в символах
     * 
     * @warning Требует исключительной блокировки, не вызывать параллельно
     */
    void train(const std::vector<std::string>& corpus);

    /**
     * @brief Обучение с отображением прогресса
     * 
     * @param corpus Вектор строк для обучения
     * @param verbose Показывать прогресс-бар (по умолчанию true)
     * 
     * Полезно для длительного обучения на больших корпусах.
     * Выводит:
     * - Текущий шаг / общее количество
     * - Процент выполнения
     * - Самая частая пара на текущем шаге
     */
    void train_with_progress(const std::vector<std::string>& corpus, bool verbose = true);

    // ========================================================================
    // Геттеры для доступа к внутренним структурам
    // ========================================================================

    /**
     * @brief Получить константную ссылку на словарь
     */
    const Vocabulary& vocabulary() const;

    /**
     * @brief Текущий размер словаря
     */
    size_t vocab_size() const;

    /**
     * @brief Количество правил слияния
     */
    size_t merges_count() const;

    /**
     * @brief Максимальная длина токена
     */
    size_t max_token_length() const;

    /**
     * @brief ID токена неизвестных символов
     */
    token_id_t unknown_token_id() const;

    /**
     * @brief Проверить, является ли токен специальным
     * 
     * Специальные токены: <UNK>, <PAD>, <BOS>, <EOS>, <MASK>
     */
    bool is_special_token(token_id_t id) const;

    // ========================================================================
    // Методы для тестирования и отладки
    // ========================================================================

    /**
     * @brief Получить ID токена по строке (для тестов)
     */
    token_id_t token_to_id(const std::string& token) const;

    /**
     * @brief Получить строку по ID токена (для тестов)
     */
    std::string id_to_token(token_id_t id) const;

    /**
     * @brief Проверить наличие токена в словаре
     */
    bool contains_token(const std::string& token) const;

    /**
     * @brief ID паддинг-токена
     */
    token_id_t pad_id() const;

    /**
     * @brief ID токена начала последовательности
     */
    token_id_t bos_id() const;

    /**
     * @brief ID токена конца последовательности
     */
    token_id_t eos_id() const;

    /**
     * @brief ID маскирующего токена (для BERT-like моделей)
     */
    token_id_t mask_id() const;

    /**
     * @brief Добавить токен в словарь (только для тестов)
     */
    token_id_t add_token(const std::string& token);

    /**
     * @brief Сбросить статистику (для тестов)
     */
    void reset_stats();

    /**
     * @brief Структура для статистики (совместимость с FastBPETokenizer)
     */
    struct Stats {
        size_t encode_calls{0};
        size_t decode_calls{0};
        size_t cache_hits{0};
        size_t cache_misses{0};

        double cache_hit_rate() const { return 0.0; }
        double avg_encode_time_ms() const { return 0.0; }
        double avg_decode_time_ms() const { return 0.0; }
    };

    /**
     * @brief Получить статистику (заглушка для совместимости)
     */
    Stats stats() const { return Stats{}; }

private:
    // ========================================================================
    // Приватные вспомогательные методы
    // ========================================================================

    /**
     * @brief Пре-токенизация текста (разделение на слова и пунктуацию)
     * 
     * Использует правила:
     * - Пробелы разделяют слова
     * - Пунктуация отделяется от слов
     * - Числа сохраняются целиком
     */
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    /**
     * @brief Разбить слово на минимальные единицы (символы/байты)
     */
    std::vector<std::string> split_into_chars(const std::string& word) const;

    /**
     * @brief Применить все правила слияния к слову
     * 
     * Проходит по слову и последовательно применяет слияния
     * от наиболее частых к редким (по рангу).
     */
    std::vector<std::string> apply_merges(const std::vector<std::string>& word) const;

    /**
     * @brief Конвертировать UTF-8 строку в байты (для byte-level режима)
     */
    std::vector<uint8_t> utf8_to_bytes(const std::string& str) const;

    /**
     * @brief Конвертировать байты обратно в UTF-8
     */
    std::string bytes_to_utf8(const std::vector<uint8_t>& bytes) const;

    /**
     * @brief Получить частоты всех пар в корпусе
     */
    std::unordered_map<MergePair, int, MergePairHash> get_pair_frequencies(
        const std::vector<std::string>& corpus) const;

    /**
     * @brief Выполнить слияние пары во всем корпусе
     */
    std::vector<std::string> merge_pair(
        const MergePair& pair,
        const std::vector<std::string>& corpus) const;

    /**
     * @brief Обновить метаданные перед сохранением
     */
    void update_metadata() const;

    // ========================================================================
    // Поля класса
    // ========================================================================

    // Основные структуры данных
    Vocabulary vocab_;                                            ///< Словарь токенов (основное хранилище)
    std::vector<MergePair> merges_list_;                          ///< Список слияний в порядке обучения
    std::unordered_map<MergePair, int, MergePairHash> merges_;    ///< Карта слияний для быстрого доступа (ранг)

    // Синхронизация
    mutable std::shared_mutex mutex_;    ///< Мьютекс для потокобезопасности
                                         ///< shared_lock для чтения, unique_lock для записи

    // Параметры конфигурации
    bool byte_level_{false};                ///< Режим работы с UTF-8 (true - как байты, false - как символы)
    std::string unknown_token_{"<UNK>"};    ///< Токен для неизвестных символов
    size_t max_token_length_{1000};         ///< Максимальная длина токена (защита от переполнения)
    size_t vocab_size_{10000};              ///< Целевой размер словаря (параметр обучения)

    // Метаданные для сериализации
    mutable ModelMetadata metadata_;    ///< Метаданные модели (версия, дата создания, описание)
};

}    // namespace bpe

/**
 * @example examples/bpe_tokenizer_example.cpp
 * Полный пример использования BPETokenizer:
 * 
 * @include examples/bpe_tokenizer_example.cpp
 * 
 * Демонстрирует:
 * - Инициализацию токенизатора
 * - Обучение на корпусе C++ кода
 * - Сохранение в различных форматах
 * - Кодирование и декодирование
 * - Экспорт для использования в Python
 */