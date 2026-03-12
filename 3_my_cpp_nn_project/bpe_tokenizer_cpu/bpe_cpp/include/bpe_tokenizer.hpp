/**
 * @file bpe_tokenizer.hpp
 * @brief Основной заголовочный файл BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Этот файл содержит объявление класса BPETokenizer - ядра BPE алгоритма.
 *          Класс реализует полный цикл работы с токенизатором:
 * 
 *          **Обучение**              - построение словаря на основе корпуса текстов
 *          **Кодирование**           - преобразование текста в последовательность токенов
 *          **Декодирование**         - восстановление текста из токенов
 *          **Сериализация**          - сохранение и загрузка модели в различных форматах
 *          **Потокобезопасность**    - поддержка многопоточного доступа через shared_mutex
 *          **Byte-level**            - корректная обработка UTF-8 текста на уровне байтов
 * 
 *          Поддерживаемые форматы моделей:
 *          - Текстовый (vocab.json + merges.txt)    - для отладки
 *          - JSON (единый файл)                     - для совместимости
 *          - Бинарный (.bin)                        - для быстрой загрузки
 *          - HuggingFace                            - для использования с трансформерами
 *          - SentencePiece                          - альтернативный формат
 * 
 * @note Токенизатор оптимизирован для кода на C++ и поддерживает
 *       специальные токены: <UNK>, <PAD>, <BOS>, <EOS>, <MASK>
 * 
 * @see Vocabulary
 * @see MergePair
 * @see ModelExport
 */

#pragma once

#include "vocabulary.hpp"
#include "bpe_export.hpp"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <shared_mutex>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <set>

namespace bpe {

// ======================================================================
// Типы для удобства использования
// ======================================================================

using token_id_t = uint32_t;                     ///< Тип для идентификаторов токенов
using freq_map_t = std::unordered_map<std::string, size_t>;  ///< Карта частот

// ======================================================================
// Структуры для представления слияний
// ======================================================================

/**
 * @brief Структура, представляющая пару для слияния в BPE алгоритме
 * 
 * Хранит два элемента (слова или подслова), которые будут объединены
 * в один токен в процессе обучения.
 * 
 * \include examples/merge_pair_example.cpp
 * Пример использования:
 * \code
 * MergePair pair{"a", "b"};    // Слияние "a" и "b" в "ab"
 * if (pair.left == "a" && pair.right == "b") {
 *     std::cout << "Найдена пара для слияния!" << std::endl;
 * }
 * \endcode
 */
struct MergePair {
    std::string left;     ///< Левый элемент пары (может быть символом или подсловом)
    std::string right;    ///< Правый элемент пары

    /**
     * @brief Оператор сравнения для использования в контейнерах
     */
    bool operator==(const MergePair& other) const {
        return left == other.left && right == other.right;
    }
    
    /**
     * @brief Получить строковое представление пары
     * @return Строка вида "left right"
     */
    std::string to_string() const {
        return left + " " + right;
    }
};

/**
 * @brief Хеш-функция для MergePair (для использования в unordered_map)
 * 
 * Позволяет использовать MergePair в качестве ключа в хеш-таблицах.
 * Комбинирует хеши левой и правой частей через конкатенацию с разделителем.
 */
struct MergePairHash {
    size_t operator()(const MergePair& p) const {
        return std::hash<std::string>()(p.left + "|" + p.right);
    }
};

// ======================================================================
// Основной класс токенизатора
// ======================================================================

/**
 * @brief Основной класс BPE токенизатора
 * 
 * Реализует алгоритм Byte Pair Encoding (BPE) для токенизации текста.
 * Это базовая версия токенизатора, которая служит эталоном для сравнения
 * с оптимизированной версией FastBPETokenizer.
 * 
 * **Алгоритм работы:**
 * 1. Пре-токенизация (разбиение текста на слова)
 * 2. Разбиение слов на символы (начальный словарь)
 * 3. Подсчет частот соседних пар символов
 * 4. Слияние самой частой пары в новый токен
 * 5. Повторение шагов 3-4 до достижения нужного размера словаря
 * 
 * **Потокобезопасность:**
 * - encode() и decode() используют shared_lock для параллельного чтения
 * - train() и load() используют unique_lock для монопольного доступа
 * 
 * \include examples/bpe_tokenizer_example.cpp
 * Пример использования:
 * \code
 * // Создание токенизатора
 * BPETokenizer tokenizer(8000, true);    // vocab_size=8000, byte_level=true
 * 
 * // Обучение на корпусе
 * std::vector<std::string> corpus = {"int main() {}", "std::cout << \"Hi\";"};
 * tokenizer.train(corpus);
 * 
 * // Сохранение модели
 * tokenizer.save_to_files("vocab.json", "merges.txt");
 * 
 * // Использование
 * auto tokens = tokenizer.encode("int x = 42;");
 * std::string decoded = tokenizer.decode(tokens);
 * \endcode
 */
class BPETokenizer : public ModelExport {
public:
    // ==================== Конструкторы и деструктор ====================

    /**
     * @brief Конструктор по умолчанию
     * 
     * Создает токенизатор с размером словаря 8000 и byte-level режимом.
     * После создания требуется загрузить существующую модель или обучить новую.
     */
    BPETokenizer();

    /**
     * @brief Конструктор с параметрами
     * 
     * @param vocab_size Максимальный размер словаря (количество токенов)
     * @param byte_level Использовать byte-level обработку UTF-8 текста
     * 
     * byte-level режим позволяет корректно обрабатывать любые символы Unicode,
     * включая русские буквы и эмодзи, преобразуя их в последовательности байтов.
     */
    explicit BPETokenizer(size_t vocab_size, bool byte_level = true);

    /**
     * @brief Деструктор
     */
    virtual ~BPETokenizer() override;

    // Запрещаем копирование (RAII с мьютексами)
    BPETokenizer(const BPETokenizer&) = delete;
    BPETokenizer& operator=(const BPETokenizer&) = delete;

    // Разрешаем перемещение
    BPETokenizer(BPETokenizer&& other) noexcept;
    BPETokenizer& operator=(BPETokenizer&& other) noexcept;

    // ==================== Настройки ====================

    /**
     * @brief Включить/выключить byte-level режим
     * 
     * @param enable true - включить, false - выключить
     * 
     * Byte-level режим обрабатывает UTF-8 текст на уровне байтов,
     * что гарантирует корректную работу с любыми символами Unicode,
     * но увеличивает длину последовательностей.
     */
    void set_byte_level(bool enable) { 
        std::unique_lock lock(mutex_);
        byte_level_ = enable; 
    }

    /**
     * @brief Установить токен для неизвестных символов
     * 
     * @param token Строка-токен (например, "<UNK>")
     * 
     * Этот токен будет использоваться для символов, отсутствующих в словаре.
     * По умолчанию используется "<UNK>".
     */
    void set_unknown_token(const std::string& token) { 
        std::unique_lock lock(mutex_);
        unknown_token_ = token; 
    }

    /**
     * @brief Установить максимальный размер словаря
     * 
     * @param size Желаемый размер словаря
     * 
     * Размер словаря влияет на:
     * - Качество токенизации (больше = лучше)
     * - Скорость работы (больше = медленнее)
     * - Потребление памяти (больше = больше памяти)
     */
    void set_vocab_size(size_t size) { 
        std::unique_lock lock(mutex_);
        vocab_size_ = size; 
    }

    /**
     * @brief Установить максимальную длину токена
     * 
     * @param length Максимальная длина в символах (по умолчанию 1000)
     */
    void set_max_token_length(size_t length) {
        std::unique_lock lock(mutex_);
        max_token_length_ = length;
    }

    // ==================== Загрузка/сохранение (базовые форматы) ====================

    /**
     * @brief Загрузить модель из текстовых файлов (отдельные файлы)
     * 
     * @param vocab_path Путь к файлу словаря (формат JSON)
     * @param merges_path Путь к файлу слияний (формат TXT)
     * @return true при успешной загрузке, false при ошибке
     * 
     * Формат vocab.json:
     * {
     *   "токен1": id1,
     *   "токен2": id2,
     *   ...
     * }
     * 
     * Формат merges.txt:
     * токен1 токен2
     * токен3 токен4
     * ...
     */
    bool load_from_files(const std::string& vocab_path, const std::string& merges_path);

    /**
     * @brief Сохранить модель в текстовые файлы (отдельные файлы)
     * 
     * @param vocab_path Путь для сохранения словаря
     * @param merges_path Путь для сохранения слияний
     * @return true при успешном сохранении, false при ошибке
     */
    bool save_to_files(const std::string& vocab_path, const std::string& merges_path) const;

    // ==================== Загрузка/сохранение (расширенные форматы из ModelExport) ====================

    /**
     * @brief Сохранить модель в JSON формате (единый файл)
     * 
     * @param path Путь для сохранения
     * @return true при успешном сохранении
     * 
     * Создает единый JSON файл, содержащий как словарь, так и правила слияния.
     * Удобно для распространения модели как одного файла.
     */
    bool save_to_json(const std::string& path) const override;

    /**
     * @brief Загрузить модель из JSON формата (единый файл)
     * 
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load_from_json(const std::string& path) override;

    /**
     * @brief Сохранить модель в бинарном формате (единый файл)
     * 
     * @param path Путь для сохранения (рекомендуется расширение .bin)
     * @return true при успешном сохранении
     * 
     * Бинарный формат обеспечивает:
     * - Минимальный размер файла
     * - Максимальную скорость загрузки
     * - Единый файл для всей модели
     */
    bool save_binary(const std::string& path) const override;

    /**
     * @brief Загрузить модель из бинарного формата (единый файл)
     * 
     * @param path Путь к файлу
     * @return true при успешной загрузке
     * 
     * @see validate_binary_model() для проверки целостности
     */
    bool load_binary(const std::string& path) override;

    /**
     * @brief Экспортировать в формат HuggingFace Tokenizers
     * 
     * @param output_dir Директория для сохранения
     * @return true при успешном экспорте
     * 
     * Создает структуру, совместимую с библиотекой transformers:
     * output_dir/
     * ├-- tokenizer.json
     * ├-- tokenizer_config.json
     * └-- special_tokens_map.json
     */
    bool export_to_huggingface(const std::string& output_dir) const override;

    /**
     * @brief Экспортировать в формат SentencePiece
     * 
     * @param path Путь для сохранения (.model)
     * @return true при успешном экспорте
     */
    bool export_to_sentencepiece(const std::string& path) const override;

    /**
     * @brief Получить информацию о модели в читаемом виде
     * 
     * @return std::string Многострочное описание модели
     * 
     * Пример вывода:
     * ========== МОДЕЛЬ BPE TOKENIZER ==========
     * Тип: BPETokenizer
     * Размер словаря: 8000 токенов
     * Правил слияния: 7999
     * Byte-level режим: да
     * ...
     */
    std::string get_model_info() const override;

    // ==================== Основные методы ====================

    /**
     * @brief Закодировать текст в последовательность токенов
     * 
     * @param text Входной текст для кодирования
     * @return std::vector<token_id_t> Вектор идентификаторов токенов
     * 
     * Алгоритм:
     * 1. Пре-токенизация (разбиение на слова/слова)
     * 2. Разбиение каждого слова на символы (начальные токены)
     * 3. Последовательное применение правил слияния от самого частого к редкому
     * 4. Замена неизвестных символов на <UNK>
     * 
     * Потокобезопасность: использует shared_lock для параллельного чтения
     * 
     * \include examples/encode_example.cpp
     * \code
     * auto tokens = tokenizer.encode("int x = 42;");
     * for (auto id : tokens) {
     *     std::cout << id << " ";
     * }
     * \endcode
     */
    std::vector<token_id_t> encode(const std::string& text) const;

    /**
     * @brief Декодировать последовательность токенов обратно в текст
     * 
     * @param tokens Вектор идентификаторов токенов
     * @return std::string Восстановленный текст
     * 
     * Алгоритм:
     * 1. Конкатенация строковых представлений токенов
     * 2. Удаление специальных токенов (опционально)
     * 3. Восстановление пробелов (для byte-level режима)
     * 
     * Потокобезопасность: использует shared_lock для параллельного чтения
     */
    std::string decode(const std::vector<token_id_t>& tokens) const;

    /**
     * @brief Пакетное кодирование нескольких текстов
     * 
     * @param texts Вектор входных текстов
     * @return std::vector<std::vector<token_id_t>> Вектор результатов
     * 
     * Оптимизирован для обработки множества текстов за один вызов.
     * Может быть быстрее последовательных вызовов encode() за счет
     * меньшего количества блокировок.
     */
    std::vector<std::vector<token_id_t>> encode_batch(
        const std::vector<std::string>& texts) const;

    /**
     * @brief Обучить токенизатор на корпусе текстов
     * 
     * @param corpus Вектор строк для обучения
     * 
     * Алгоритм обучения:
     * 1. Пре-токенизация всего корпуса
     * 2. Инициализация словаря символами
     * 3. Подсчет частот всех соседних пар
     * 4. Выбор самой частой пары
     * 5. Слияние выбранной пары во всем корпусе
     * 6. Добавление нового токена в словарь
     * 7. Повторение шагов 3-6 до достижения vocab_size_
     * 
     * Потокобезопасность: использует unique_lock для монопольного доступа
     * 
     * \include examples/train_example.cpp
     * \code
     * std::vector<std::string> corpus;
     * // ... загрузка корпуса ...
     * tokenizer.train(corpus);
     * tokenizer.save_to_files("trained_vocab.json", "trained_merges.txt");
     * \endcode
     */
    void train(const std::vector<std::string>& corpus);

    /**
     * @brief Обучить токенизатор с отображением прогресса
     * 
     * @param corpus Вектор строк для обучения
     * @param verbose Выводить прогресс в консоль
     */
    void train_with_progress(const std::vector<std::string>& corpus, bool verbose = true);

    // ==================== Геттеры ====================

    /**
     * @brief Получить ссылку на словарь
     * @return const Vocabulary& Константная ссылка на Vocabulary
     */
    const Vocabulary& vocabulary() const { 
        std::shared_lock lock(mutex_);
        return vocab_; 
    }

    /**
     * @brief Получить текущий размер словаря
     * @return size_t Количество токенов в словаре
     */
    size_t vocab_size() const { 
        std::shared_lock lock(mutex_);
        return vocab_.size(); 
    }

    /**
     * @brief Получить количество выполненных слияний
     * @return size_t Количество пар в merges_
     */
    size_t merges_count() const { 
        std::shared_lock lock(mutex_);
        return merges_.size(); 
    }

    /**
     * @brief Получить максимальную длину токена
     * @return size_t Максимальная длина токена в символах
     */
    size_t max_token_length() const { 
        std::shared_lock lock(mutex_);
        return max_token_length_; 
    }

    /**
     * @brief Получить ID токена для неизвестных символов
     * @return token_id_t Идентификатор токена <UNK>
     */
    token_id_t unknown_token_id() const;

    /**
     * @brief Проверить, является ли токен специальным
     * @param id ID токена
     * @return true если токен специальный (<UNK>, <PAD>, <BOS>, <EOS>, <MASK>)
     */
    bool is_special_token(token_id_t id) const;

// ==================== Дополнительные методы для тестов ====================

    /**
     * @brief Получить ID токена по строковому представлению
     * @param token Строковое представление токена
     * @return token_id_t ID токена или static_cast<token_id_t>(-1) если не найден
     */
    token_id_t token_to_id(const std::string& token) const {
        std::shared_lock lock(mutex_);
        return vocab_.token_to_id(token);
    }

    /**
     * @brief Получить строковое представление токена по ID
     * @param id ID токена
     * @return std::string Строковое представление или пустая строка если не найден
     */
    std::string id_to_token(token_id_t id) const {
        std::shared_lock lock(mutex_);
        return vocab_.id_to_token(id);
    }

    /**
     * @brief Проверить, существует ли токен в словаре
     * @param token Строковое представление токена
     * @return true если токен существует
     */
    bool contains_token(const std::string& token) const {
        std::shared_lock lock(mutex_);
        return vocab_.contains(token);
    }

    /**
     * @brief Получить ID токена для паддинга
     * @return token_id_t ID токена <PAD> или unknown_token_id() если не найден
     */
    token_id_t pad_id() const {
        std::shared_lock lock(mutex_);
        if (vocab_.contains("<PAD>")) {
            return vocab_.token_to_id("<PAD>");
        }
        return unknown_token_id();
    }

    /**
     * @brief Получить ID токена начала последовательности
     * @return token_id_t ID токена <BOS> или unknown_token_id() если не найден
     */
    token_id_t bos_id() const {
        std::shared_lock lock(mutex_);
        if (vocab_.contains("<BOS>")) {
            return vocab_.token_to_id("<BOS>");
        }
        return unknown_token_id();
    }

    /**
     * @brief Получить ID токена конца последовательности
     * @return token_id_t ID токена <EOS> или unknown_token_id() если не найден
     */
    token_id_t eos_id() const {
        std::shared_lock lock(mutex_);
        if (vocab_.contains("<EOS>")) {
            return vocab_.token_to_id("<EOS>");
        }
        return unknown_token_id();
    }

    /**
     * @brief Получить ID токена для маски
     * @return token_id_t ID токена <MASK> или unknown_token_id() если не найден
     */
    token_id_t mask_id() const {
        std::shared_lock lock(mutex_);
        if (vocab_.contains("<MASK>")) {
            return vocab_.token_to_id("<MASK>");
        }
        return unknown_token_id();
    }

    /**
     * @brief Добавить токен в словарь
     * @param token Строковое представление токена
     * @return token_id_t ID добавленного токена
     * 
     * @note Используется в основном для тестирования.
     *       В обычной работе словарь формируется при обучении.
     */
    token_id_t add_token(const std::string& token) {
        std::unique_lock lock(mutex_);
        return vocab_.add_token(token);
    }

    /**
     * @brief Сбросить статистику (для тестирования)
     */
    void reset_stats() {
        // Базовая версия не собирает статистику
    }

    /**
     * @brief Получить статистику (заглушка для совместимости)
     */
    struct Stats {
        size_t encode_calls{0};
        size_t decode_calls{0};
        size_t cache_hits{0};
        size_t cache_misses{0};
        
        double cache_hit_rate() const { 
            return 0.0; 
        }
        double avg_encode_time_ms() const { 
            return 0.0; 
        }
        double avg_decode_time_ms() const { 
            return 0.0; 
        }
    };
    
    Stats stats() const { return Stats{}; }

private:
    // ==================== Приватные методы ====================

    /**
     * @brief Предварительная токенизация текста
     * 
     * Разбивает текст на слова, сохраняя пунктуацию как отдельные токены.
     * Использует следующие правила:
     * - Пробельные символы разделяют слова
     * - Пунктуация отделяется от слов
     * - Сохраняются числа как единые токены
     * 
     * @param text Входной текст
     * @return std::vector<std::string> Вектор предварительных токенов
     */
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    /**
     * @brief Разбить слово на символы
     * 
     * @param word Входное слово
     * @return std::vector<std::string> Вектор символов
     */
    std::vector<std::string> split_into_chars(const std::string& word) const;

    /**
     * @brief Применить слияния к слову
     * 
     * Последовательно применяет все правила слияния к слову,
     * начиная с самых частых.
     * 
     * @param word Слово, разбитое на части
     * @return std::vector<std::string> Слово после применения слияний
     */
    std::vector<std::string> apply_merges(const std::vector<std::string>& word) const;

    /**
     * @brief Преобразовать UTF-8 строку в байты
     * 
     * Используется в byte-level режиме для обработки Unicode текста.
     * 
     * @param str Входная строка в UTF-8
     * @return std::vector<uint8_t> Вектор байтов
     */
    std::vector<uint8_t> utf8_to_bytes(const std::string& str) const;

    /**
     * @brief Преобразовать байты обратно в UTF-8 строку
     * 
     * @param bytes Вектор байтов
     * @return std::string Восстановленная UTF-8 строка
     */
    std::string bytes_to_utf8(const std::vector<uint8_t>& bytes) const;

    /**
     * @brief Получить частотность пар для обучения
     * 
     * @param corpus Корпус текстов
     * @return std::unordered_map<MergePair, int, MergePairHash> Карта частотности пар
     */
    std::unordered_map<MergePair, int, MergePairHash> get_pair_frequencies(
        const std::vector<std::string>& corpus) const;

    /**
     * @brief Выполнить слияние пары во всем корпусе
     * 
     * @param pair Пара для слияния
     * @param corpus Текущий корпус
     * @return std::vector<std::string> Обновленный корпус
     */
    std::vector<std::string> merge_pair(
        const MergePair& pair,
        const std::vector<std::string>& corpus) const;

    /**
     * @brief Обновить метаданные перед сохранением
     */
    void update_metadata() const;

    // ==================== Поля класса ====================

    Vocabulary vocab_;                                            ///< Словарь токенов (основная структура)
    std::vector<MergePair> merges_list_;                         ///< Список слияний в порядке обучения
    std::unordered_map<MergePair, int, MergePairHash> merges_;    ///< Карта слияний (для быстрого доступа)
    
    mutable std::shared_mutex mutex_;    ///< Мьютекс для потокобезопасности
                                         ///< shared_lock для чтения, unique_lock для записи

    bool byte_level_{false};                ///< Флаг byte-level режима (UTF-8 как байты)
    std::string unknown_token_{"<UNK>"};    ///< Токен для неизвестных символов
    size_t max_token_length_{1000};         ///< Максимальная длина токена (ограничение)
    size_t vocab_size_{8000};               ///< Желаемый размер словаря (параметр обучения)
    
    // Метаданные для экспорта
    mutable ModelMetadata metadata_;    ///< Метаданные модели (версия, дата, описание)
};

} // namespace bpe

/**
 * @example examples/bpe_tokenizer_example.cpp
 * Полный пример использования BPETokenizer:
 * 
 * \include examples/bpe_tokenizer_example.cpp
 * 
 * Пример кода:
 * \code
 * #include "bpe_tokenizer.hpp"
 * #include <iostream>
 * 
 * int main() {
 *     using namespace bpe;
 *     
 *     // Создание токенизатора
 *     BPETokenizer tokenizer(8000, true);    // словарь 8000 токенов, byte-level
 *     
 *     // Загрузка существующей модели
 *     if (!tokenizer.load_from_files("vocab.json", "merges.txt")) {
 *         std::cerr << "Не удалось загрузить модель!" << std::endl;
 *         return 1;
 *     }
 *     
 *     // Использование
 *     std::string code = "int factorial(int n) { return n <= 1 ? 1 : n * factorial(n-1); }";
 *     
 *     auto tokens = tokenizer.encode(code);
 *     std::cout << "Токенов: " << tokens.size() << std::endl;
 *     
 *     auto decoded = tokenizer.decode(tokens);
 *     std::cout << "Декодировано: " << decoded << std::endl;
 *     
 *     // Информация о модели
 *     std::cout << tokenizer.get_model_info() << std::endl;
 *     
 *     return 0;
 * }
 * \endcode
 */