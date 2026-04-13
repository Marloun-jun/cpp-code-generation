/**
 * @file bpe_export.hpp
 * @brief Унифицированный интерфейс для сериализации моделей BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Предоставляет единый API для сохранения и загрузки моделей в различных форматах:
 *          - JSON          - Человеко-читаемый, для отладки и совместимости
 *          - Бинарный      - Компактный, для быстрой загрузки в production
 *          - HuggingFace   - Интеграция с экосистемой transformers
 *          - SentencePiece - Совместимость с альтернативными реализациями
 * 
 *          Ключевые возможности:
 *          - Множество форматов экспорта без изменения кода токенизатора
 *          - Проверка целостности моделей через контрольные суммы
 *          - Метаданные для обратной совместимости
 *          - Утилиты для конвертации между форматами
 * 
 * @note Все классы токенизаторов должны реализовывать интерфейс ModelExport
 * @see BPETokenizer, FastBPETokenizer, ModelMetadata
 */

#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bpe {

// ============================================================================
// Константы бинарного формата
// ============================================================================

/** @name Константы бинарного формата
 *  @{
 */
constexpr uint32_t BINARY_MAGIC = 0x42504556;      ///< Магическое число "BPEV" для идентификации формата
constexpr uint32_t BINARY_VERSION = 0x00010000;    ///< Версия формата 1.0.0 (major.minor.patch)
constexpr size_t BINARY_HEADER_SIZE = 16;          ///< Размер заголовка в байтах
/** @} */

// ============================================================================
// Интерфейс экспорта моделей
// ============================================================================

/**
 * @brief Абстрактный интерфейс для операций экспорта/импорта моделей
 * 
 * Определяет контракт, который должны реализовывать все токенизаторы для обеспечения
 * единообразной работы с моделями независимо от внутренней реализации.
 * 
 * Особенности:
 * - Поддержка множества форматов сериализации
 * - Единый API для всех производных классов
 * - Возможность расширения новыми форматами
 * 
 * Пример использования:
 * @code
 * std::unique_ptr<ModelExport> tokenizer = std::make_unique<FastBPETokenizer>();
 * 
 * // Загрузка модели
 * if (tokenizer->load_from_json("models/bpe_10000/base")) {
 *     // Работа с токенизатором
 *     
 *     // Экспорт в другой формат
 *     tokenizer->export_to_huggingface("models/hf/");
 *     tokenizer->save_binary("models/bpe_10000.bin");
 * }
 * @endcode
 */
class ModelExport {
public:
    virtual ~ModelExport() = default;

    // ------------------------------------------------------------------------
    // JSON формат (vocab.json + merges.txt)
    // ------------------------------------------------------------------------
    
    /**
     * @brief Сохранить модель в JSON формате (два файла)
     * 
     * Создает два текстовых файла:
     * - [base_path].vocab.json - Словарь токенов с частотами
     * - [base_path].merges.txt - Правила слияния в порядке обучения
     * 
     * @param base_path Базовый путь без расширения (например, "models/bpe_10000/model")
     * @return true если сохранение успешно
     * 
     * @note JSON формат удобен для отладки, но медленнее загружается
     */
    virtual bool save_to_json(const std::string& base_path) const = 0;

    /**
     * @brief Загрузить модель из JSON формата
     * 
     * Ожидает наличие файлов:
     * - [base_path].vocab.json
     * - [base_path].merges.txt
     * 
     * @param base_path Базовый путь без расширения
     * @return true если загрузка успешна
     */
    virtual bool load_from_json(const std::string& base_path) = 0;

    // ------------------------------------------------------------------------
    // Бинарный формат (единый файл .bin)
    // ------------------------------------------------------------------------

    /**
     * @brief Сохранить модель в бинарном формате
     * 
     * Компактное бинарное представление модели в едином файле.
     * Преимущества:
     * - Меньший размер (до 70% экономии по сравнению с JSON)
     * - Быстрая загрузка (без парсинга)
     * - Встроенная проверка целостности
     * 
     * Структура файла:
     * [MAGIC(4)][VERSION(4)][METADATA_SIZE(8)][METADATA][VOCAB][MERGES][CHECKSUM(32)]
     * 
     * @param path Полный путь к файлу (рекомендуется расширение .bin)
     * @return true если сохранение успешно
     */
    virtual bool save_binary(const std::string& path) const = 0;

    /**
     * @brief Загрузить модель из бинарного формата
     * 
     * @param path Полный путь к .bin файлу
     * @return true если загрузка успешна
     * 
     * @see validate_binary_model() для предварительной проверки
     */
    virtual bool load_binary(const std::string& path) = 0;

    // ------------------------------------------------------------------------
    // Экспорт в форматы других библиотек
    // ------------------------------------------------------------------------

    /**
     * @brief Экспорт в формат HuggingFace Tokenizers
     * 
     * Создает структуру директорий, совместимую с библиотекой transformers:
     * - tokenizer.json        - Полная конфигурация
     * - vocab.json            - Словарь
     * - merges.txt            - Правила слияния
     * - tokenizer_config.json - Конфигурация для загрузки
     * 
     * @param output_dir Директория для сохранения
     * @return true если экспорт успешен
     * 
     * @note После экспорта модель можно использовать в Python:
     *       @code{.py}
     *       from transformers import PreTrainedTokenizerFast
     *       tokenizer = PreTrainedTokenizerFast.from_pretrained("output_dir/")
     *       @endcode
     */
    virtual bool export_to_huggingface(const std::string& output_dir) const = 0;

    /**
     * @brief Экспорт в формат SentencePiece
     * 
     * Создает модель в формате SentencePiece (.model), используемую в проектах
     * T5, ALBERT, XLNet и других.
     * 
     * @param path Полный путь к .model файлу
     * @return true если экспорт успешен
     * 
     * @note Требует установленной библиотеки sentencepiece
     */
    virtual bool export_to_sentencepiece(const std::string& path) const = 0;

    // ------------------------------------------------------------------------
    // Информация о модели
    // ------------------------------------------------------------------------

    /**
     * @brief Получить детальную информацию о модели
     * 
     * Возвращает многострочное описание:
     * - Тип и версия модели
     * - Размер словаря и количество слияний
     * - Параметры конфигурации
     * - Специальные токены
     * - Дата создания
     * 
     * @return std::string Форматированное описание
     */
    virtual std::string get_model_info() const = 0;
};

// ============================================================================
// Метаданные модели
// ============================================================================

/**
 * @brief Метаданные модели для сериализации и версионирования
 * 
 * Содержит всю информацию о модели, необходимую для:
 * - Идентификации типа и версии модели
 * - Проверки совместимости при загрузке
 * - Документирования параметров
 * - Обеспечения обратной совместимости
 */
struct ModelMetadata {
    std::string model_type{"BPE"};              ///< Тип токенизатора (BPE/WordPiece/Unigram)
    std::string version{"1.0.0"};               ///< Версия формата (semver)
    size_t vocab_size{0};                       ///< Размер словаря
    size_t merges_count{0};                     ///< Количество правил слияния
    bool byte_level{true};                      ///< Режим byte-level (обработка UTF-8)
    std::vector<std::string> special_tokens{    ///< Стандартные специальные токены
        "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>"
    };
    std::string creation_date;    ///< Дата создания (ISO 8601)
    std::string description;      ///< Описание модели
    std::string hash;             ///< SHA-256 контрольная сумма
    
    /**
     * @brief Конструктор с автоматической генерацией даты
     */
    ModelMetadata() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d");
        creation_date = ss.str();
    }
    
    /**
     * @brief Сериализация в JSON
     * @return nlohmann::json JSON объект с метаданными
     */
    nlohmann::json to_json() const;
    
    /**
     * @brief Десериализация из JSON
     * @param j JSON объект с метаданными
     * @throws nlohmann::json::exception при ошибке парсинга
     */
    void from_json(const nlohmann::json& j);
    
    /**
     * @brief Проверка совместимости версий
     * 
     * @param other Другие метаданные для сравнения
     * @return true если мажорные версии совпадают и тип модели одинаков
     */
    bool is_compatible_with(const ModelMetadata& other) const {
        auto get_major = [](const std::string& v) {
            size_t dot_pos = v.find('.');
            return (dot_pos != std::string::npos) ? v.substr(0, dot_pos) : v;
        };
        
        return get_major(version) == get_major(other.version) &&
               model_type == other.model_type;
    }
    
    /**
     * @brief Получить форматированное строковое представление
     */
    std::string to_string() const {
        std::stringstream ss;
        ss << "Тип модели:     " << model_type << "\n";
        ss << "Версия:         " << version << "\n";
        ss << "Размер словаря: " << vocab_size << "\n";
        ss << "Правил слияния: " << merges_count << "\n";
        ss << "Byte-level:     " << (byte_level ? "да" : "нет") << "\n";
        ss << "Спец. токены:   ";
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << special_tokens[i];
        }
        ss << "\n";
        ss << "Дата создания:  " << creation_date << "\n";
        if (!description.empty()) {
            ss << "Описание:       " << description << "\n";
        }
        if (!hash.empty()) {
            ss << "Контр. сумма:   " << hash << "\n";
        }
        return ss.str();
    }
};

// ============================================================================
// Утилиты для работы с форматами
// ============================================================================

/**
 * @brief Конвертировать JSON модель в формат HuggingFace
 * 
 * Создает полную структуру директорий, совместимую с HuggingFace:
 * - tokenizer.json          - Основной файл конфигурации
 * - tokenizer_config.json   - Конфигурация для загрузки
 * - special_tokens_map.json - Отображение специальных токенов
 * 
 * @param vocab_path Путь к vocab.json
 * @param merges_path Путь к merges.txt
 * @param output_dir Директория для сохранения
 * @return true при успешной конвертации
 */
bool convert_to_huggingface(const std::string& vocab_path,
                            const std::string& merges_path,
                            const std::string& output_dir);

/**
 * @brief Конвертировать JSON модель в формат SentencePiece
 * 
 * @param vocab_path Путь к vocab.json
 * @param output_path Путь для .model файла
 * @return true при успешной конвертации
 * 
 * @note Требует библиотеку sentencepiece-dev
 */
bool convert_to_sentencepiece(const std::string& vocab_path,
                              const std::string& output_path);

/**
 * @brief Валидация бинарного файла модели
 * 
 * Проверяет:
 * - Существование файла и доступность для чтения
 * - Корректность магического числа
 * - Совместимость версии формата
 * - Целостность данных (контрольная сумма)
 * 
 * @param path Путь к .bin файлу
 * @return true если файл корректен
 */
bool validate_binary_model(const std::string& path);

/**
 * @brief Вычислить SHA-256 хеш файла
 * 
 * @param path Путь к файлу
 * @return std::string Хеш в hex формате или пустая строка при ошибке
 */
std::string compute_file_hash(const std::string& path);

/**
 * @brief Получить список поддерживаемых форматов
 * @return std::vector<std::string> Названия форматов
 */
inline std::vector<std::string> get_supported_formats() {
    return {
        "json",            // vocab.json + merges.txt
        "binary",          // Компактный .bin формат
        "huggingface",     // HuggingFace tokenizers
        "sentencepiece"    // SentencePiece .model
    };
}

/**
 * @brief Получить текущую версию формата экспорта
 * @return std::string Версия в формате major.minor.patch
 */
inline std::string get_export_version() {
    return "1.0.0";
}

/**
 * @brief Проверить, является ли файл бинарной моделью
 * 
 * @param path Путь к файлу
 * @return true если файл содержит корректный заголовок BPE модели
 */
inline bool is_binary_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return magic == BINARY_MAGIC;
}

}    // namespace bpe

/**
 * @example examples/export_example.cpp
 * Комплексный пример использования функций экспорта:
 * 
 * @include examples/export_example.cpp
 * 
 * Демонстрирует:
 * - Загрузку модели из JSON
 * - Сохранение в различных форматах
 * - Конвертацию между форматами
 * - Проверку целостности
 */