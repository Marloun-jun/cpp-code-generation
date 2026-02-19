/**
 * @file bpe_export.hpp
 * @brief Заголовочный файл для экспорта/импорта моделей BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.1.0
 * 
 * @details Определяет функции для сохранения и загрузки моделей в различных форматах:
 *          - JSON (читаемый формат для отладки)
 *          - Бинарный (компактный формат для быстрой загрузки)
 *          - HuggingFace Tokenizers (совместимость с экосистемой HF)
 *          - SentencePiece (альтернативный формат)
 * 
 * @note Все методы должны быть реализованы в классах BPETokenizer и FastBPETokenizer
 * @see BPETokenizer
 * @see FastBPETokenizer
 */

#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <utility>

namespace bpe {

// ======================================================================
// Интерфейс для экспорта/импорта моделей
// ======================================================================

/**
 * @brief Абстрактный класс для экспорта моделей
 * 
 * Определяет единый интерфейс для всех токенизаторов
 */
class ModelExport {
public:
    virtual ~ModelExport() = default;
    
    /**
     * @brief Сохранить модель в JSON формате
     * @param path Путь для сохранения
     * @return true при успешном сохранении
     */
    virtual bool save_to_json(const std::string& path) const = 0;
    
    /**
     * @brief Загрузить модель из JSON формата
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    virtual bool load_from_json(const std::string& path) = 0;
    
    /**
     * @brief Сохранить модель в бинарном формате (единый файл)
     * @param path Путь для сохранения
     * @return true при успешном сохранении
     */
    virtual bool save_binary(const std::string& path) const = 0;
    
    /**
     * @brief Загрузить модель из бинарного формата (единый файл)
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    virtual bool load_binary(const std::string& path) = 0;
    
    /**
     * @brief Экспортировать в формат HuggingFace Tokenizers
     * @param path Путь для сохранения
     * @return true при успешном экспорте
     */
    virtual bool export_to_huggingface(const std::string& path) const = 0;
    
    /**
     * @brief Экспортировать в формат SentencePiece
     * @param path Путь для сохранения
     * @return true при успешном экспорте
     */
    virtual bool export_to_sentencepiece(const std::string& path) const = 0;
    
    /**
     * @brief Получить информацию о модели в читаемом виде
     * @return Строка с описанием модели
     */
    virtual std::string get_model_info() const = 0;
};

// ======================================================================
// Структуры для обмена данными
// ======================================================================

/**
 * @brief Структура для хранения метаданных модели
 */
struct ModelMetadata {
    std::string model_type{"BPE"};           ///< Тип модели (BPE, Unigram, etc)
    std::string version{"1.0.0"};             ///< Версия формата
    size_t vocab_size{0};                     ///< Размер словаря
    size_t merges_count{0};                   ///< Количество правил слияния
    bool byte_level{true};                    ///< Использовать byte-level режим
    std::vector<std::string> special_tokens{  ///< Специальные токены
        "<UNK>", "<PAD>", "<BOS>", "<EOS>"
    };
    std::string creation_date;                 ///< Дата создания
    std::string description;                   ///< Описание модели
    
    /**
     * @brief Сериализовать в JSON
     * @return JSON объект с метаданными
     */
    nlohmann::json to_json() const;
    
    /**
     * @brief Загрузить из JSON
     * @param j JSON объект с метаданными
     */
    void from_json(const nlohmann::json& j);
};

// ======================================================================
// Вспомогательные функции для работы с форматами
// ======================================================================

/**
 * @brief Конвертировать BPE модель в формат HuggingFace
 * @param vocab_path Путь к файлу словаря
 * @param merges_path Путь к файлу слияний
 * @param output_path Путь для сохранения HF модели
 * @return true при успешной конвертации
 */
bool convert_to_huggingface(const std::string& vocab_path, 
                            const std::string& merges_path,
                            const std::string& output_path);

/**
 * @brief Конвертировать BPE модель в формат SentencePiece
 * @param vocab_path Путь к файлу словаря
 * @param output_path Путь для сохранения SentencePiece модели
 * @return true при успешной конвертации
 */
bool convert_to_sentencepiece(const std::string& vocab_path,
                              const std::string& output_path);

/**
 * @brief Проверить целостность бинарного файла модели
 * @param path Путь к бинарному файлу
 * @return true если файл корректен
 */
bool validate_binary_model(const std::string& path);

} // namespace bpe