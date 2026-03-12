/**
 * @file bpe_export.hpp
 * @brief Заголовочный файл для экспорта/импорта моделей BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Этот файл определяет единый интерфейс для сохранения и загрузки моделей
 *          в различных форматах, что обеспечивает:
 * 
 *          **Совместимость**    - работа с разными экосистемами (HuggingFace, SentencePiece)
 *          **Гибкость**         - выбор между читаемыми JSON и компактными бинарными форматами
 *          **Расширяемость**    - легко добавлять новые форматы через наследование
 *          **Целостность**      - проверка корректности сохраненных моделей
 * 
 *          Поддерживаемые форматы:
 *          - **JSON** (текстовый)    - для отладки и совместимости с Python
 *          - **Бинарный**            - для быстрой загрузки в продакшене
 *          - **HuggingFace**         - для использования с трансформерами
 *          - **SentencePiece**       - альтернативная реализация
 * 
 * @note Все методы должны быть реализованы в классах BPETokenizer и FastBPETokenizer
 * @see BPETokenizer
 * @see FastBPETokenizer
 * @see ModelMetadata
 */

#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <memory>
#include <cstdint>

namespace bpe {

// ======================================================================
// Константы для бинарного формата
// ======================================================================

namespace {
    constexpr uint32_t BINARY_MAGIC = 0x42504556;        // "BPEV" в hex
    constexpr uint32_t BINARY_VERSION = 0x00010000;      // версия 1.0.0
    constexpr size_t BINARY_HEADER_SIZE = 16;            // размер заголовка
}

// ======================================================================
// Интерфейс для экспорта/импорта моделей
// ======================================================================

/**
 * @brief Абстрактный класс, определяющий единый интерфейс для экспорта моделей
 * 
 * Все токенизаторы (BPETokenizer, FastBPETokenizer) должны реализовать этот интерфейс,
 * что гарантирует единообразную работу с моделями независимо от внутренней реализации.
 * 
 * \include examples/export_example.cpp
 * Пример использования:
 * \code
 * FastBPETokenizer tokenizer;
 * tokenizer.load("model.json", "merges.txt");
 * 
 * // Сохраняем в разных форматах
 * tokenizer.save_to_json("backup/model");
 * tokenizer.save_binary("model.bin");
 * tokenizer.export_to_huggingface("hf_model/");
 * 
 * // Загружаем обратно
 * tokenizer.load_binary("model.bin");
 * \endcode
 */
class ModelExport {
public:
    virtual ~ModelExport() = default;
    
    /**
     * @brief Сохранить модель в JSON формате (два файла: словарь и слияния)
     * 
     * JSON формат удобен для отладки, так как он читаемый человеком.
     * Создает два файла:
     * - vocab.json     - словарь токенов с их частотами
     * - merges.txt     - правила слияния в порядке обучения
     * 
     * @param base_path Базовый путь для сохранения (без расширения)
     * @return true при успешном сохранении
     * 
     * @note Для совместимости с HuggingFace используйте export_to_huggingface()
     */
    virtual bool save_to_json(const std::string& base_path) const = 0;
    
    /**
     * @brief Загрузить модель из JSON формата (два файла)
     * 
     * Ожидает наличие двух файлов:
     * - vocab.json
     * - merges.txt
     * 
     * @param base_path Базовый путь к файлам (без расширения)
     * @return true при успешной загрузке
     */
    virtual bool load_from_json(const std::string& base_path) = 0;
    
    /**
     * @brief Сохранить модель в бинарном формате (единый файл)
     * 
     * Бинарный формат:
     * - Компактный (меньше места на диске)
     * - Быстрая загрузка (нет парсинга JSON)
     * - Единый файл (удобно для распространения)
     * 
     * Формат файла:
     * [MAGIC][VERSION][METADATA_SIZE][METADATA][VOCAB_SIZE][VOCAB_DATA][MERGES_SIZE][MERGES_DATA]
     * 
     * @param path Путь для сохранения (рекомендуется расширение .bin)
     * @return true при успешном сохранении
     */
    virtual bool save_binary(const std::string& path) const = 0;
    
    /**
     * @brief Загрузить модель из бинарного формата (единый файл)
     * 
     * @param path Путь к бинарному файлу (.bin)
     * @return true при успешной загрузке
     * 
     * @see validate_binary_model() для проверки целостности
     */
    virtual bool load_binary(const std::string& path) = 0;
    
    /**
     * @brief Экспортировать в формат HuggingFace Tokenizers
     * 
     * Создает структуру директорий, совместимую с библиотекой transformers:
     * - tokenizer.json    - полная конфигурация токенизатора
     * - vocab.json        - словарь
     * - merges.txt        - правила слияния
     * - config.json       - конфигурация модели (опционально)
     * 
     * @param output_dir Директория для сохранения HF модели
     * @return true при успешном экспорте
     * 
     * @note После экспорта модель можно использовать с transformers:
     *       from transformers import PreTrainedTokenizerFast
     *       tokenizer = PreTrainedTokenizerFast.from_pretrained("output_dir/")
     */
    virtual bool export_to_huggingface(const std::string& output_dir) const = 0;
    
    /**
     * @brief Экспортировать в формат SentencePiece
     * 
     * Создает модель в формате SentencePiece (.model), которую можно
     * использовать с библиотекой sentencepiece.
     * 
     * @param path Путь для сохранения (.model)
     * @return true при успешном экспорте
     * 
     * @note Формат SentencePiece используется во многих проектах,
     *       включая T5, ALBERT, XLNet.
     */
    virtual bool export_to_sentencepiece(const std::string& path) const = 0;
    
    /**
     * @brief Получить информацию о модели в читаемом виде
     * 
     * Возвращает строку с основными характеристиками модели:
     * - Тип модели
     * - Размер словаря
     * - Количество правил слияния
     * - Режим работы (byte-level)
     * - Специальные токены
     * 
     * @return std::string Многострочное описание модели
     */
    virtual std::string get_model_info() const = 0;
};

// ======================================================================
// Структуры для обмена данными
// ======================================================================

/**
 * @brief Структура для хранения метаданных модели
 * 
 * Содержит всю информацию о модели, не зависящую от реализации:
 * - Версию формата
 * - Размер словаря
 * - Специальные токены
 * - Дату создания
 * 
 * Используется при сериализации для обеспечения обратной совместимости.
 */
struct ModelMetadata {
    std::string model_type{"BPE"};              ///< Тип модели (BPE, Unigram, WordPiece)
    std::string version{"1.0.0"};               ///< Версия формата (semver)
    size_t vocab_size{0};                        ///< Размер словаря
    size_t merges_count{0};                      ///< Количество правил слияния
    bool byte_level{true};                       ///< Использовать byte-level режим
    std::vector<std::string> special_tokens{     ///< Специальные токены
        "<UNK>", "<PAD>", "<BOS>", "<EOS>", "<MASK>"
    };
    std::string creation_date;                    ///< Дата создания в формате ISO 8601
    std::string description;                      ///< Описание модели (опционально)
    std::string hash;                             ///< Контрольная сумма (опционально)
    
    /**
     * @brief Конструктор по умолчанию с автоматической генерацией даты
     */
    ModelMetadata() {
        // Генерируем текущую дату в формате YYYY-MM-DD
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d");
        creation_date = ss.str();
    }
    
    /**
     * @brief Сериализовать метаданные в JSON
     * @return nlohmann::json JSON объект с метаданными
     */
    nlohmann::json to_json() const;
        
    /**
     * @brief Загрузить метаданные из JSON
     * @param j JSON объект с метаданными
     * @throws nlohmann::json::exception при ошибке парсинга
     */
    void from_json(const nlohmann::json& j);
    
    /**
     * @brief Проверить совместимость версий
     * @param other Другие метаданные для сравнения
     * @return true если версии совместимы
     */
    bool is_compatible_with(const ModelMetadata& other) const {
        // Проверяем мажорную версию (должна совпадать)
        auto get_major = [](const std::string& v) -> std::string {
            size_t dot_pos = v.find('.');
            return (dot_pos != std::string::npos) ? v.substr(0, dot_pos) : v;
        };
        
        return get_major(version) == get_major(other.version) &&
               model_type == other.model_type;
    }
    
    /**
     * @brief Получить строковое представление метаданных
     */
    std::string to_string() const {
        std::stringstream ss;
        ss << "Тип модели:      " << model_type << "\n";
        ss << "Версия:          " << version << "\n";
        ss << "Размер словаря:  " << vocab_size << "\n";
        ss << "Правил слияния:  " << merges_count << "\n";
        ss << "Byte-level:      " << (byte_level ? "да" : "нет") << "\n";
        ss << "Спец. токены:    ";
        for (size_t i = 0; i < special_tokens.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << special_tokens[i];
        }
        ss << "\n";
        ss << "Дата создания:   " << creation_date << "\n";
        if (!description.empty()) {
            ss << "Описание:        " << description << "\n";
        }
        if (!hash.empty()) {
            ss << "Контр. сумма:    " << hash << "\n";
        }
        return ss.str();
    }
};

// ======================================================================
// Вспомогательные функции для работы с форматами
// ======================================================================

/**
 * @brief Конвертировать BPE модель в формат HuggingFace
 * 
 * Создает полную структуру директорий, совместимую с HuggingFace transformers:
 * - tokenizer.json             - основной файл с конфигурацией
 * - tokenizer_config.json      - конфигурация для PreTrainedTokenizerFast
 * - special_tokens_map.json    - отображение специальных токенов
 * 
 * @param vocab_path Путь к файлу словаря (vocab.json)
 * @param merges_path Путь к файлу слияний (merges.txt)
 * @param output_dir Директория для сохранения HF модели
 * @return true при успешной конвертации
 * 
 * @note Функция создает директорию output_dir, если она не существует
 * 
 * Пример использования:
 * \code
 * convert_to_huggingface("models/bpe_8000/vocab.json", 
 *                        "models/bpe_8000/merges.txt", 
 *                        "models/hf_model/");
 * \endcode
 */
bool convert_to_huggingface(const std::string& vocab_path, 
                            const std::string& merges_path,
                            const std::string& output_dir);

/**
 * @brief Конвертировать BPE модель в формат SentencePiece
 * 
 * Создает модель в формате SentencePiece, который представляет собой
 * бинарный файл с протобуферной структурой.
 * 
 * @param vocab_path Путь к файлу словаря
 * @param output_path Путь для сохранения SentencePiece модели (.model)
 * @return true при успешной конвертации
 * 
 * @note Требует наличия библиотеки sentencepiece
 * 
 * Пример использования:
 * \code
 * convert_to_sentencepiece("models/bpe_8000/vocab.json", 
 *                          "sp_model.model");
 * \endcode
 */
bool convert_to_sentencepiece(const std::string& vocab_path,
                              const std::string& output_path);

/**
 * @brief Проверить целостность бинарного файла модели
 * 
 * Выполняет проверки:
 * - Существование файла
 * - Корректность магического числа
 * - Совместимость версии
 * - Целостность данных (контрольная сумма)
 * 
 * @param path Путь к бинарному файлу
 * @return true если файл корректен и может быть загружен
 * 
 * Пример использования:
 * \code
 * if (validate_binary_model("model.bin")) {
 *     tokenizer.load_binary("model.bin");
 * } else {
 *     std::cerr << "Модель повреждена!" << std::endl;
 * }
 * \endcode
 */
bool validate_binary_model(const std::string& path);

/**
 * @brief Вычислить контрольную сумму файла (SHA-256)
 * 
 * @param path Путь к файлу
 * @return std::string Контрольная сумма в hex формате, пустая строка при ошибке
 */
std::string compute_file_hash(const std::string& path);

/**
 * @brief Получить список поддерживаемых форматов экспорта
 * @return std::vector<std::string> Список форматов
 */
inline std::vector<std::string> get_supported_formats() {
    return {
        "json",            // Текстовый формат (vocab.json + merges.txt)
        "binary",          // Компактный бинарный формат
        "huggingface",     // HuggingFace tokenizers
        "sentencepiece"    // SentencePiece model
    };
}

/**
 * @brief Получить версию формата экспорта
 * @return std::string Версия в формате semver (major.minor.patch)
 */
inline std::string get_export_version() {
    return "1.0.0";
}

/**
 * @brief Проверить, является ли файл бинарной моделью
 * @param path Путь к файлу
 * @return true если файл содержит корректную бинарную модель
 */
inline bool is_binary_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    return magic == BINARY_MAGIC;
}

} // namespace bpe

/**
 * @example examples/export_example.cpp
 * Пример использования функций экспорта:
 * 
 * \include examples/export_example.cpp
 * 
 * Пример кода:
 * \code
 * #include "bpe_export.hpp"
 * #include "fast_tokenizer.hpp"
 * #include <iostream>
 * 
 * int main() {
 *     using namespace bpe;
 *     
 *     // Загружаем обученную модель
 *     FastBPETokenizer tokenizer;
 *     if (!tokenizer.load("model.json", "merges.txt")) {
 *         std::cerr << "Не удалось загрузить модель!" << std::endl;
 *         return 1;
 *     }
 *     
 *     // Получаем информацию о модели
 *     std::cout << tokenizer.get_model_info() << std::endl;
 *     
 *     // Сохраняем в разных форматах
 *     tokenizer.save_to_json("backup/model");
 *     tokenizer.save_binary("backup/model.bin");
 *     tokenizer.export_to_huggingface("hf_model/");
 *     
 *     // Конвертируем существующие файлы
 *     convert_to_huggingface("vocab.json", "merges.txt", "hf_model/");
 *     
 *     // Проверяем бинарный файл
 *     if (validate_binary_model("model.bin")) {
 *         std::cout << "Модель корректна" << std::endl;
 *     }
 *     
 *     // Проверяем тип файла
 *     if (is_binary_model("model.bin")) {
 *         std::cout << "Это бинарная модель" << std::endl;
 *     }
 *     
 *     return 0;
 * }
 * \endcode
 */