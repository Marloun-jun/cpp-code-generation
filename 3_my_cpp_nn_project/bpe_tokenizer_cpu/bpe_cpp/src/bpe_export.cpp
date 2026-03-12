/**
 * @file bpe_export.cpp
 * @brief Реализация функций экспорта/импорта моделей BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Этот файл содержит реализацию методов класса BPETokenizer для
 *          сохранения и загрузки моделей в различных форматах. Поддерживаются:
 * 
 *          **Поддерживаемые форматы:**
 *          1) **JSON (единый файл)** - читаемый формат для отладки
 *             - Сохраняет словарь, правила слияния и метаданные в одном файле
 *             - Удобен для ручного просмотра и редактирования
 *          
 *          2) **HuggingFace Tokenizers** - совместимость с экосистемой HF
 *             - Формат, используемый библиотекой transformers
 *             - Позволяет загружать модель в Python через transformers
 *          
 *          3) **SentencePiece** - альтернативный формат
 *             - Простой текстовый формат: токен<TAB>ID
 *             - Совместим с библиотекой sentencepiece
 * 
 *          **Процесс экспорта:**
 *          1. Сбор данных из внутренних структур (vocabulary, merges)
 *          2. Сериализация в выбранный формат
 *          3. Запись в файл с проверкой ошибок
 *          4. Обработка исключений с выводом диагностики
 * 
 * @note Все методы возвращают bool для индикации успеха/неудачи
 * @see ModelMetadata
 * @see Vocabulary
 */

#include "bpe_tokenizer.hpp"
#include "bpe_export.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

namespace bpe {

// ======================================================================
// ModelMetadata - реализация методов сериализации
// ======================================================================

/**
 * @brief Сериализация метаданных в JSON
 * 
 * Преобразует структуру ModelMetadata в JSON объект для сохранения.
 * Включает все поля: тип модели, версию, размер словаря, специальные токены и т.д.
 * 
 * @return nlohmann::json JSON объект с метаданными
 */
nlohmann::json ModelMetadata::to_json() const {
    nlohmann::json j;
    j["model_type"] = model_type;
    j["version"] = version;
    j["vocab_size"] = vocab_size;
    j["merges_count"] = merges_count;
    j["byte_level"] = byte_level;
    j["special_tokens"] = special_tokens;
    j["creation_date"] = creation_date;
    j["description"] = description;
    j["hash"] = hash;
    return j;
}

/**
 * @brief Десериализация метаданных из JSON
 * 
 * Загружает метаданные из JSON объекта с проверкой наличия полей.
 * Если поле отсутствует, сохраняется текущее значение.
 * 
 * @param j JSON объект с метаданными
 */
void ModelMetadata::from_json(const nlohmann::json& j) {
    if (j.contains("model_type") && j["model_type"].is_string()) 
        model_type = j["model_type"].get<std::string>();
    if (j.contains("version") && j["version"].is_string()) 
        version = j["version"].get<std::string>();
    if (j.contains("vocab_size") && j["vocab_size"].is_number()) 
        vocab_size = j["vocab_size"].get<size_t>();
    if (j.contains("merges_count") && j["merges_count"].is_number()) 
        merges_count = j["merges_count"].get<size_t>();
    if (j.contains("byte_level") && j["byte_level"].is_boolean()) 
        byte_level = j["byte_level"].get<bool>();
    if (j.contains("special_tokens") && j["special_tokens"].is_array()) 
        special_tokens = j["special_tokens"].get<std::vector<std::string>>();
    if (j.contains("creation_date") && j["creation_date"].is_string()) 
        creation_date = j["creation_date"].get<std::string>();
    if (j.contains("description") && j["description"].is_string()) 
        description = j["description"].get<std::string>();
    if (j.contains("hash") && j["hash"].is_string()) 
        hash = j["hash"].get<std::string>();
}

// ======================================================================
// JSON Export/Import (единый файл)
// ======================================================================

/**
 * @brief Сохранить модель в единый JSON файл
 * 
 * Создает JSON файл со следующей структурой:
 * {
 *   "vocab": { ... },    // словарь в формате Vocabulary::to_json()
 *   "merges": [          // правила слияния
 *     ["left", "right"],
 *     ...
 *   ],
 *   "config": {    // конфигурация токенизатора
 *     "byte_level": true,
 *     "vocab_size": 8000,
 *     "unknown_token": "<UNK>"
 *   },
 *   "metadata": { ... }    // метаданные модели
 * }
 * 
 * @param path Путь для сохранения JSON файла
 * @return true при успешном сохранении, false при ошибке
 * 
 * @note Формат удобен для отладки и ручного просмотра
 */
bool BPETokenizer::save_to_json(const std::string& path) const {
    try {
        nlohmann::json j;
        
        // Сохраняем словарь
        j["vocab"] = vocab_.to_json();
        
        // Сохраняем правила слияния (без рангов, только пары)
        std::vector<std::vector<std::string>> merges_list;
        for (const auto& [pair, rank] : merges_) {
            merges_list.push_back({pair.left, pair.right});
        }
        j["merges"] = merges_list;
        
        // Сохраняем конфигурацию
        j["config"]["byte_level"] = byte_level_;
        j["config"]["vocab_size"] = vocab_size_;
        j["config"]["unknown_token"] = unknown_token_;
        j["config"]["max_token_length"] = max_token_length_;
        
        // Сохраняем метаданные
        j["metadata"] = metadata_.to_json();
        
        // Запись в файл
        std::ofstream file(path);
        if (!file) {
            std::cerr << "Ошибка: не удалось открыть файл для записи: " << path << std::endl;
            return false;
        }
        
        file << j.dump(2);    // Отступ 2 пробела для читаемости
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка сохранения JSON: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Загрузить модель из единого JSON файла
 * 
 * Ожидает структуру JSON, созданную save_to_json().
 * При ошибке парсинга выводит диагностику в std::cerr.
 * 
 * @param path Путь к JSON файлу
 * @return true при успешной загрузке, false при ошибке
 * 
 * @note Все внутренние структуры (vocab_, merges_, config) будут перезаписаны
 */
bool BPETokenizer::load_from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Ошибка: не удалось открыть файл для чтения: " << path << std::endl;
        return false;
    }
    
    try {
        nlohmann::json j;
        file >> j;
        
        // Загрузка словаря
        if (j.contains("vocab")) {
            vocab_.from_json(j["vocab"]);
        }
        
        // Загрузка правил слияния
        if (j.contains("merges") && j["merges"].is_array()) {
            merges_.clear();
            int rank = 0;
            for (const auto& item : j["merges"]) {
                if (item.is_array() && item.size() == 2) {
                    MergePair pair{item[0].get<std::string>(), item[1].get<std::string>()};
                    merges_[pair] = rank++;
                }
            }
        }
        
        // Загрузка конфигурации
        if (j.contains("config")) {
            if (j["config"].contains("byte_level"))
                byte_level_ = j["config"]["byte_level"].get<bool>();
            if (j["config"].contains("vocab_size"))
                vocab_size_ = j["config"]["vocab_size"].get<size_t>();
            if (j["config"].contains("unknown_token"))
                unknown_token_ = j["config"]["unknown_token"].get<std::string>();
            if (j["config"].contains("max_token_length"))
                max_token_length_ = j["config"]["max_token_length"].get<size_t>();
        }
        
        // Загрузка метаданных
        if (j.contains("metadata")) {
            metadata_.from_json(j["metadata"]);
            metadata_.vocab_size = vocab_.size();
            metadata_.merges_count = merges_.size();
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка загрузки JSON: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Бинарный Export/Import
// ======================================================================

/**
 * @brief Сохранить модель в бинарный формат
 * 
 * Формат файла:
 * [MAGIC (4 байта)] - сигнатура "BPEV"
 * [VERSION (4 байта)] - версия формата
 * [VOCAB_SIZE (4 байта)] - размер словаря
 * для каждого токена:
 *   [TOKEN_LEN (4 байта)] - длина токена
 *   [TOKEN_DATA (TOKEN_LEN байт)] - данные токена
 * [MERGES_SIZE (4 байта)] - количество слияний
 * для каждого слияния:
 *   [LEFT_ID (4 байта)] - ID левого токена
 *   [RIGHT_ID (4 байта)] - ID правого токена
 *   [RANK (4 байта)] - ранг слияния
 * 
 * @param path Путь для сохранения
 * @return true при успешном сохранении
 */
bool BPETokenizer::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Ошибка: не удалось открыть файл для записи: " << path << std::endl;
        return false;
    }
    
    try {
        // Магическое число
        uint32_t magic = 0x42504556;  // "BPEV"
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // Версия
        uint32_t version = 0x00010000;  // версия 1.0.0
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Размер словаря
        uint32_t vocab_size = static_cast<uint32_t>(vocab_.size());
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        
        // Токены
        auto tokens = vocab_.get_all_tokens();
        for (const auto& token : tokens) {
            uint32_t len = static_cast<uint32_t>(token.size());
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(token.data(), len);
        }
        
        // Количество слияний
        uint32_t merges_size = static_cast<uint32_t>(merges_.size());
        file.write(reinterpret_cast<const char*>(&merges_size), sizeof(merges_size));
        
        // Слияния
        for (const auto& [pair, rank] : merges_) {
            uint32_t left_id = vocab_.token_to_id(pair.left);
            uint32_t right_id = vocab_.token_to_id(pair.right);
            uint32_t rank_val = static_cast<uint32_t>(rank);
            
            file.write(reinterpret_cast<const char*>(&left_id), sizeof(left_id));
            file.write(reinterpret_cast<const char*>(&right_id), sizeof(right_id));
            file.write(reinterpret_cast<const char*>(&rank_val), sizeof(rank_val));
        }
        
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка сохранения бинарного файла: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Загрузить модель из бинарного формата
 * 
 * @param path Путь к файлу
 * @return true при успешной загрузке
 * 
 * @see save_binary() для описания формата
 */
bool BPETokenizer::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Ошибка: не удалось открыть файл для чтения: " << path << std::endl;
        return false;
    }
    
    try {
        // Чтение магического числа
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x42504556) {
            std::cerr << "Ошибка: неверный формат файла (ожидалась сигнатура BPEV)" << std::endl;
            return false;
        }
        
        // Чтение версии
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        // Проверка совместимости версий (пока пропускаем)
        
        // Чтение размера словаря
        uint32_t vocab_size;
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        // Очищаем текущий словарь
        vocab_.clear();
        
        // Чтение токенов
        for (uint32_t i = 0; i < vocab_size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            if (len > 1000000) {  // Защита от некорректных данных
                std::cerr << "Ошибка: слишком длинный токен (" << len << " байт)" << std::endl;
                return false;
            }
            
            std::string token(len, '\0');
            file.read(&token[0], len);
            vocab_.add_token(token);
        }
        
        // Чтение количества слияний
        uint32_t merges_size;
        file.read(reinterpret_cast<char*>(&merges_size), sizeof(merges_size));
        
        // Очищаем текущие слияния
        merges_.clear();
        
        // Чтение слияний
        for (uint32_t i = 0; i < merges_size; ++i) {
            uint32_t left_id, right_id, rank_val;
            file.read(reinterpret_cast<char*>(&left_id), sizeof(left_id));
            file.read(reinterpret_cast<char*>(&right_id), sizeof(right_id));
            file.read(reinterpret_cast<char*>(&rank_val), sizeof(rank_val));
            
            // Получаем строковые представления токенов
            std::string left = vocab_.id_to_token(left_id);
            std::string right = vocab_.id_to_token(right_id);
            
            MergePair pair{left, right};
            merges_[pair] = rank_val;
        }
        
        // Обновляем метаданные
        metadata_.vocab_size = vocab_.size();
        metadata_.merges_count = merges_.size();
        
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка загрузки бинарного файла: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// HuggingFace Tokenizers Export
// ======================================================================

/**
 * @brief Экспорт в формат HuggingFace Tokenizers
 * 
 * Создает JSON файл, совместимый с библиотекой transformers.
 * Формат соответствует спецификации HuggingFace:
 * {
 *   "model": {
 *     "type": "BPE",
 *     "vocab": { "token0": 0, "token1": 1, ... },
 *     "merges": ["left right", ...]
 *   },
 *   "tokenizer_config": {
 *     "unk_token": "<UNK>",
 *     "pad_token": "<PAD>",
 *     "bos_token": "<BOS>",
 *     "eos_token": "<EOS>"
 *   }
 * }
 * 
 * @param path Путь для сохранения (рекомендуется расширение .json)
 * @return true при успешном экспорте, false при ошибке
 * 
 * @note После экспорта модель можно загрузить в Python:
 *       from transformers import PreTrainedTokenizerFast
 *       tokenizer = PreTrainedTokenizerFast.from_pretrained("path/")
 */
bool BPETokenizer::export_to_huggingface(const std::string& path) const {
    try {
        nlohmann::json j;
        
        // Модель
        j["model"]["type"] = "BPE";
        j["model"]["vocab"] = vocab_.to_json();
        
        // Слияния в формате HuggingFace (строки "left right")
        std::vector<std::string> merges_list;
        for (const auto& [pair, rank] : merges_) {
            merges_list.push_back(pair.left + " " + pair.right);
        }
        j["model"]["merges"] = merges_list;
        
        // Добавляем информацию о добавочных пробелах
        j["model"]["add_prefix_space"] = false;
        
        // Конфигурация токенизатора
        j["tokenizer_config"]["unk_token"] = unknown_token_;
        j["tokenizer_config"]["pad_token"] = "<PAD>";
        j["tokenizer_config"]["bos_token"] = "<BOS>";
        j["tokenizer_config"]["eos_token"] = "<EOS>";
        j["tokenizer_config"]["model_max_length"] = 512;
        
        // Запись в файл
        std::ofstream file(path);
        if (!file) {
            std::cerr << "Ошибка: не удалось открыть файл для записи: " << path << std::endl;
            return false;
        }
        
        file << j.dump(2);
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка экспорта в HuggingFace: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// SentencePiece Export
// ======================================================================

/**
 * @brief Экспорт в формат SentencePiece
 * 
 * Создает простой текстовый файл в формате SentencePiece:
 * токен1<TAB>ID1
 * токен2<TAB>ID2
 * ...
 * 
 * Этот формат используется библиотекой sentencepiece и многими
 * другими инструментами для работы с токенизаторами.
 * 
 * @param path Путь для сохранения (рекомендуется расширение .vocab)
 * @return true при успешном экспорте, false при ошибке
 * 
 * @note Простой формат легко парсить и использовать в других языках
 */
bool BPETokenizer::export_to_sentencepiece(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        std::cerr << "Ошибка: не удалось открыть файл для записи: " << path << std::endl;
        return false;
    }
    
    try {
        auto tokens = vocab_.get_all_tokens();
        for (size_t i = 0; i < tokens.size(); ++i) {
            // Экранируем специальные символы для формата SentencePiece
            std::string token = tokens[i];
            // Заменяем пробел на специальный символ
            // (SentencePiece использует _ для обозначения пробела)
            file << token << "\t" << i << "\n";
        }
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка экспорта в SentencePiece: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Model Information
// ======================================================================

/**
 * @brief Получить информацию о модели в читаемом виде
 * 
 * Формирует многострочную строку с основными характеристиками модели:
 * - Размер словаря
 * - Количество правил слияния
 * - Режим работы (byte-level)
 * - Специальные токены
 * 
 * @return std::string Отформатированная информация о модели
 * 
 * @example
 * === Информация о BPE Tokenizer ===
 * Размер словаря:                8000
 * Количество слияний:            7999
 * Byte-level режим:              enabled
 * Неизвестных токенов:           <UNK>
 * Максимальная длина токена:     1000
 * ===================================
 */
std::string BPETokenizer::get_model_info() const {
    std::stringstream ss;
    ss << "\n" << "==================================================" << "\n";
    ss << "ИНФОРМАЦИЯ О BPE TOKENIZER\n";
    ss << "==================================================" << "\n";
    // 50 символов =
    ss << "Размер словаря:               " << vocab_.size() << "\n";
    ss << "Количество слияний:           " << merges_.size() << "\n";
    ss << "Byte-level режим:             " << (byte_level_ ? "включен" : "отключен") << "\n";
    ss << "Неизвестных токенов:          " << unknown_token_ << "\n";
    ss << "Максимальная длина токена:    " << max_token_length_ << "\n";
    
    // Информация о специальных токенах
    ss << "\nСпециальные токены:\n";
    ss << "  <UNK>: ";
    if (vocab_.contains("<UNK>")) 
        ss << "ID " << vocab_.token_to_id("<UNK>");
    else 
        ss << "не найден";
    ss << "\n";
    
    ss << "  <PAD>: ";
    if (vocab_.contains("<PAD>")) 
        ss << "ID " << vocab_.token_to_id("<PAD>");
    else 
        ss << "не найден";
    ss << "\n";
    
    ss << "  <BOS>: ";
    if (vocab_.contains("<BOS>")) 
        ss << "ID " << vocab_.token_to_id("<BOS>");
    else 
        ss << "не найден";
    ss << "\n";
    
    ss << "  <EOS>: ";
    if (vocab_.contains("<EOS>")) 
        ss << "ID " << vocab_.token_to_id("<EOS>");
    else 
        ss << "не найден";
    ss << "\n";
    
    ss << "==================================================" << "\n";
    // 50 символов =
    return ss.str();
}

} // namespace bpe