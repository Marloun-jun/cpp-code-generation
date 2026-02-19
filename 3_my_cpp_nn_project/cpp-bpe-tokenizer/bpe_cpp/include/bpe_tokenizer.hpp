/**
 * @file bpe_tokenizer.hpp
 * @brief Основной заголовочный файл BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Этот файл содержит объявление класса BPETokenizer - ядра BPE алгоритма.
 *          Класс реализует:
 *          - Обучение токенизатора на корпусе текстов
 *          - Кодирование текста в последовательность токенов
 *          - Декодирование токенов обратно в текст
 *          - Сохранение и загрузку модели (текстовый и бинарный форматы)
 *          - Экспорт в форматы HuggingFace и SentencePiece
 *          - Потокобезопасное кодирование через shared_mutex
 *          - Byte-level обработку UTF-8 текста
 * 
 * @note Токенизатор оптимизирован для кода на C++ и поддерживает
 *       специальные токены (<UNK>, <PAD>, <BOS>, <EOS>)
 * 
 * @see Vocabulary
 * @see MergePair
 * @see ModelExport
 */

#pragma once

#include "vocabulary.hpp"
#include "bpe_export.hpp"

#include <cstdint>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

namespace bpe {

/**
 * @brief Структура, представляющая пару для слияния в BPE алгоритме
 */
struct MergePair {
    std::string left;   ///< Левый элемент пары
    std::string right;  ///< Правый элемент пары

    bool operator==(const MergePair& other) const {
        return left == other.left && right == other.right;
    }
};

/**
 * @brief Хеш-функция для MergePair (для использования в unordered_map)
 */
struct MergePairHash {
    size_t operator()(const MergePair& p) const {
        return std::hash<std::string>()(p.left + "|" + p.right);
    }
};

/**
 * @brief Основной класс BPE токенизатора
 * 
 * Реализует алгоритм Byte Pair Encoding для токенизации текста.
 * Поддерживает обучение на корпусе, кодирование и декодирование текста,
 * а также сохранение/загрузку обученной модели в различных форматах.
 */
class BPETokenizer : public ModelExport {
public:
    /**
     * @brief Конструктор по умолчанию
     */
    BPETokenizer();

    /**
     * @brief Конструктор с параметрами
     * @param vocab_size Максимальный размер словаря
     * @param byte_level Использовать byte-level обработку (UTF-8 в байты)
     */
    explicit BPETokenizer(size_t vocab_size, bool byte_level = true);

    /**
     * @brief Деструктор
     */
    ~BPETokenizer() override;

    // ==================== Настройки ====================

    /**
     * @brief Включить/выключить byte-level режим
     * @param enable true - включить, false - выключить
     */
    void set_byte_level(bool enable) { byte_level_ = enable; }

    /**
     * @brief Установить токен для неизвестных символов
     * @param token Строка-токен (например, "<UNK>")
     */
    void set_unknown_token(const std::string& token) { unknown_token_ = token; }

    /**
     * @brief Установить максимальный размер словаря
     * @param size Желаемый размер словаря
     */
    void set_vocab_size(size_t size) { vocab_size_ = size; }

    // ==================== Загрузка/сохранение (базовые форматы) ====================

    /**
     * @brief Загрузить модель из текстовых файлов (отдельные файлы)
     * @param vocab_path Путь к файлу словаря
     * @param merges_path Путь к файлу слияний
     * @return true при успешной загрузке, false при ошибке
     */
    bool load_from_files(const std::string& vocab_path, const std::string& merges_path);

    /**
     * @brief Сохранить модель в текстовые файлы (отдельные файлы)
     * @param vocab_path Путь для сохранения словаря
     * @param merges_path Путь для сохранения слияний
     * @return true при успешном сохранении, false при ошибке
     */
    bool save_to_files(const std::string& vocab_path, const std::string& merges_path) const;

    // ==================== Загрузка/сохранение (расширенные форматы из ModelExport) ====================

    /**
     * @brief Сохранить модель в JSON формате (единый файл)
     * @param path Путь для сохранения
     * @return true при успешном сохранении
     */
    bool save_to_json(const std::string& path) const override;

    /**
     * @brief Загрузить модель из JSON формата (единый файл)
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load_from_json(const std::string& path) override;

    /**
     * @brief Сохранить модель в бинарном формате (единый файл)
     * @param path Путь для сохранения
     * @return true при успешном сохранении
     */
    bool save_binary(const std::string& path) const override;

    /**
     * @brief Загрузить модель из бинарного формата (единый файл)
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load_binary(const std::string& path) override;

    /**
     * @brief Экспортировать в формат HuggingFace Tokenizers
     * @param path Путь для сохранения
     * @return true при успешном экспорте
     */
    bool export_to_huggingface(const std::string& path) const override;

    /**
     * @brief Экспортировать в формат SentencePiece
     * @param path Путь для сохранения
     * @return true при успешном экспорте
     */
    bool export_to_sentencepiece(const std::string& path) const override;

    /**
     * @brief Получить информацию о модели в читаемом виде
     * @return Строка с описанием модели
     */
    std::string get_model_info() const override;

    // ==================== Основные методы ====================

    /**
     * @brief Закодировать текст в последовательность токенов
     * @param text Входной текст для кодирования
     * @return Вектор идентификаторов токенов
     */
    std::vector<token_id_t> encode(const std::string& text) const;

    /**
     * @brief Декодировать последовательность токенов обратно в текст
     * @param tokens Вектор идентификаторов токенов
     * @return Восстановленный текст
     */
    std::string decode(const std::vector<token_id_t>& tokens) const;

    /**
     * @brief Пакетное кодирование нескольких текстов
     * @param texts Вектор входных текстов
     * @return Вектор векторов идентификаторов токенов
     */
    std::vector<std::vector<token_id_t>> encode_batch(
        const std::vector<std::string>& texts) const;

    /**
     * @brief Обучить токенизатор на корпусе текстов
     * @param corpus Вектор строк для обучения
     */
    void train(const std::vector<std::string>& corpus);

    // ==================== Геттеры ====================

    /**
     * @brief Получить ссылку на словарь
     * @return Константная ссылка на Vocabulary
     */
    const Vocabulary& vocabulary() const { return vocab_; }

    /**
     * @brief Получить текущий размер словаря
     * @return Количество токенов в словаре
     */
    size_t vocab_size() const { return vocab_.size(); }

    /**
     * @brief Получить количество выполненных слияний
     * @return Количество пар в merges_
     */
    size_t merges_count() const { return merges_.size(); }

    /**
     * @brief Получить максимальную длину токена
     * @return Максимальная длина токена в символах
     */
    size_t max_token_length() const { return max_token_length_; }

    /**
     * @brief Получить ID токена для неизвестных символов
     * @return Идентификатор токена <UNK>
     */
    token_id_t unknown_token_id() const;

private:
    // ==================== Приватные методы ====================

    /**
     * @brief Предварительная токенизация текста
     * @param text Входной текст
     * @return Вектор предварительных токенов
     */
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    /**
     * @brief Разбить слово на символы
     * @param word Входное слово
     * @return Вектор символов
     */
    std::vector<std::string> split_into_chars(const std::string& word) const;

    /**
     * @brief Применить слияния к слову
     * @param word Слово, разбитое на части
     * @return Слово после применения слияний
     */
    std::vector<std::string> apply_merges(const std::vector<std::string>& word) const;

    /**
     * @brief Преобразовать UTF-8 строку в байты
     * @param str Входная строка
     * @return Вектор байтов
     */
    std::vector<uint8_t> utf8_to_bytes(const std::string& str) const;

    /**
     * @brief Преобразовать байты обратно в UTF-8 строку
     * @param bytes Вектор байтов
     * @return Восстановленная строка
     */
    std::string bytes_to_utf8(const std::vector<uint8_t>& bytes) const;

    /**
     * @brief Получить частотность пар для обучения
     * @param corpus Корпус текстов
     * @return Карта частотности пар
     */
    std::unordered_map<MergePair, int, MergePairHash> get_pair_frequencies(
        const std::vector<std::string>& corpus) const;

    /**
     * @brief Выполнить слияние пары во всем корпусе
     * @param pair Пара для слияния
     * @param corpus Текущий корпус
     * @return Обновленный корпус
     */
    std::vector<std::string> merge_pair(
        const MergePair& pair,
        const std::vector<std::string>& corpus) const;

    // ==================== Поля класса ====================

    Vocabulary vocab_;                                  ///< Словарь токенов
    std::unordered_map<MergePair, int, MergePairHash> merges_;  ///< Карта слияний
    mutable std::shared_mutex mutex_;                   ///< Мьютекс для потокобезопасности

    bool byte_level_{false};                            ///< Флаг byte-level режима
    std::string unknown_token_{"<UNK>"};                ///< Токен для неизвестных
    size_t max_token_length_{1000};                      ///< Максимальная длина токена
    size_t vocab_size_{8000};                            ///< Желаемый размер словаря
    
    // Метаданные для экспорта
    mutable ModelMetadata metadata_;                     ///< Метаданные модели
};

} // namespace bpe