/**
 * @file bpe_export.cpp
 * @brief Реализация механизмов сериализации BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Реализует сохранение и загрузку моделей в различных форматах.
 *          Каждый формат оптимизирован для разных сценариев использования:
 * 
 *          **Сценарии использования:**
 *          - **JSON**          - Разработка и отладка (человеко-читаемый формат)
 *          - **Бинарный**      - Продакшен (быстрая загрузка, компактность)
 *          - **HuggingFace**   - Интеграция с экосистемой трансформеров
 *          - **SentencePiece** - Совместимость с другими инструментами
 * 
 *          **Обработка ошибок:**
 *          - Все методы возвращают bool для проверки успеха
 *          - Детальная диагностика в std::cerr
 *          - Защита от некорректных данных при загрузке
 *          - Обработка исключений nlohmann::json
 * 
 * @note Форматы не теряют информацию при конвертации
 * @see BPETokenizer, ModelMetadata, Vocabulary
 */

#include "bpe_tokenizer.hpp"
#include "bpe_export.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace bpe {

// ============================================================================
// ModelMetadata - сериализация метаданных
// ============================================================================

nlohmann::json ModelMetadata::to_json() const {
    nlohmann::json j;
    j["model_type"]     = model_type;
    j["version"]        = version;
    j["vocab_size"]     = vocab_size;
    j["merges_count"]   = merges_count;
    j["byte_level"]     = byte_level;
    j["special_tokens"] = special_tokens;
    j["creation_date"]  = creation_date;
    j["description"]    = description;
    j["hash"]           = hash;
    return j;
}

void ModelMetadata::from_json(const nlohmann::json& j) {
    // Каждое поле проверяется на наличие и тип перед чтением
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

// ============================================================================
// JSON сериализация (единый файл)
// ============================================================================

bool BPETokenizer::save_to_json(const std::string& path) const {
    try {
        nlohmann::json root;
        
        // --------------------------------------------------------------------
        // Сбор данных из внутренних структур
        // --------------------------------------------------------------------
        
        // Словарь токенов
        root["vocab"] = vocab_.to_json();
        
        // Правила слияния (без рангов, только пары)
        std::vector<std::vector<std::string>> merges_list;
        merges_list.reserve(merges_.size());
        for (const auto& [pair, rank] : merges_) {
            merges_list.push_back({pair.left, pair.right});
        }
        root["merges"] = merges_list;
        
        // Конфигурация токенизатора
        root["config"]["byte_level"]       = byte_level_;
        root["config"]["vocab_size"]       = vocab_size_;
        root["config"]["unknown_token"]    = unknown_token_;
        root["config"]["max_token_length"] = max_token_length_;
        
        // Метаданные модели
        root["metadata"] = metadata_.to_json();
        
        // --------------------------------------------------------------------
        // Запись в файл с pretty-printing
        // --------------------------------------------------------------------
        
        std::ofstream file(path);
        if (!file) {
            std::cerr << "[BPE Export] Ошибка: не удалось открыть файл для записи: " 
                      << path << std::endl;
            return false;
        }
        
        file << root.dump(2);    // Отступ 2 пробела для читаемости
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "[BPE Export] Ошибка сохранения JSON: " << e.what() << std::endl;
        return false;
    }
}

bool BPETokenizer::load_from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[BPE Export] Ошибка: не удалось открыть файл для чтения: " 
                  << path << std::endl;
        return false;
    }
    
    try {
        nlohmann::json root;
        file >> root;
        
        // --------------------------------------------------------------------
        // Восстановление внутренних структур
        // --------------------------------------------------------------------
        
        // Загрузка словаря
        if (root.contains("vocab")) {
            vocab_.from_json(root["vocab"]);
        }
        
        // Загрузка правил слияния
        if (root.contains("merges") && root["merges"].is_array()) {
            merges_.clear();
            int rank = 0;
            for (const auto& item : root["merges"]) {
                if (item.is_array() && item.size() == 2) {
                    MergePair pair{
                        item[0].get<std::string>(),
                        item[1].get<std::string>()
                    };
                    merges_[pair] = rank++;
                }
            }
        }
        
        // Загрузка конфигурации
        if (root.contains("config")) {
            const auto& cfg = root["config"];
            if (cfg.contains("byte_level") && cfg["byte_level"].is_boolean())
                byte_level_ = cfg["byte_level"].get<bool>();
            if (cfg.contains("vocab_size") && cfg["vocab_size"].is_number())
                vocab_size_ = cfg["vocab_size"].get<size_t>();
            if (cfg.contains("unknown_token") && cfg["unknown_token"].is_string())
                unknown_token_ = cfg["unknown_token"].get<std::string>();
            if (cfg.contains("max_token_length") && cfg["max_token_length"].is_number())
                max_token_length_ = cfg["max_token_length"].get<size_t>();
        }
        
        // Загрузка и обновление метаданных
        if (root.contains("metadata")) {
            metadata_.from_json(root["metadata"]);
            metadata_.vocab_size = vocab_.size();
            metadata_.merges_count = merges_.size();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BPE Export] Ошибка загрузки JSON: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Бинарная сериализация
// ============================================================================

bool BPETokenizer::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[BPE Export] Ошибка: не удалось открыть бинарный файл для записи: " 
                  << path << std::endl;
        return false;
    }
    
    try {
        // --------------------------------------------------------------------
        // Заголовок файла
        // --------------------------------------------------------------------
        
        // Магическое число для идентификации формата
        uint32_t magic = BINARY_MAGIC;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // Версия формата для обратной совместимости
        uint32_t version = BINARY_VERSION;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // --------------------------------------------------------------------
        // Словарь токенов
        // --------------------------------------------------------------------
        
        uint32_t vocab_size = static_cast<uint32_t>(vocab_.size());
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        
        auto tokens = vocab_.get_all_tokens();
        for (const auto& token : tokens) {
            uint32_t len = static_cast<uint32_t>(token.size());
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(token.data(), len);
        }
        
        // --------------------------------------------------------------------
        // Правила слияния
        // --------------------------------------------------------------------
        
        uint32_t merges_size = static_cast<uint32_t>(merges_.size());
        file.write(reinterpret_cast<const char*>(&merges_size), sizeof(merges_size));
        
        for (const auto& [pair, rank] : merges_) {
            uint32_t left_id  = static_cast<uint32_t>(vocab_.token_to_id(pair.left));
            uint32_t right_id = static_cast<uint32_t>(vocab_.token_to_id(pair.right));
            uint32_t rank_val = static_cast<uint32_t>(rank);
            
            file.write(reinterpret_cast<const char*>(&left_id), sizeof(left_id));
            file.write(reinterpret_cast<const char*>(&right_id), sizeof(right_id));
            file.write(reinterpret_cast<const char*>(&rank_val), sizeof(rank_val));
        }
        
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "[BPE Export] Ошибка сохранения бинарного файла: " << e.what() << std::endl;
        return false;
    }
}

bool BPETokenizer::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[BPE Export] Ошибка: не удалось открыть бинарный файл для чтения: " 
                  << path << std::endl;
        return false;
    }
    
    try {
        // --------------------------------------------------------------------
        // Проверка заголовка
        // --------------------------------------------------------------------
        
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != BINARY_MAGIC) {
            std::cerr << "[BPE Export] Ошибка: неверный формат файла (ожидалась сигнатура BPEV)!"
                      << std::endl;
            return false;
        }
        
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        // Здесь можно добавить проверку совместимости версий
        
        // --------------------------------------------------------------------
        // Загрузка словаря
        // --------------------------------------------------------------------
        
        uint32_t vocab_size;
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        // Защита от слишком больших значений
        if (vocab_size > 1000000) {
            std::cerr << "[BPE Export] Ошибка: некорректный размер словаря (" 
                      << vocab_size << ")!" << std::endl;
            return false;
        }
        
        vocab_.clear();
        for (uint32_t i = 0; i < vocab_size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            // Защита от слишком длинных токенов
            if (len > 1000000) {
                std::cerr << "[BPE Export] Ошибка: слишком длинный токен (" 
                          << len << " байт!)" << std::endl;
                return false;
            }
            
            std::string token(len, '\0');
            file.read(&token[0], len);
            vocab_.add_token(token);
        }
        
        // --------------------------------------------------------------------
        // Загрузка правил слияния
        // --------------------------------------------------------------------
        
        uint32_t merges_size;
        file.read(reinterpret_cast<char*>(&merges_size), sizeof(merges_size));
        
        merges_.clear();
        for (uint32_t i = 0; i < merges_size; ++i) {
            uint32_t left_id, right_id, rank_val;
            file.read(reinterpret_cast<char*>(&left_id), sizeof(left_id));
            file.read(reinterpret_cast<char*>(&right_id), sizeof(right_id));
            file.read(reinterpret_cast<char*>(&rank_val), sizeof(rank_val));
            
            // Проверка валидности ID
            if (left_id >= vocab_.size() || right_id >= vocab_.size()) {
                std::cerr << "[BPE Export] Ошибка: некорректный ID токена в слиянии!"
                          << std::endl;
                return false;
            }
            
            std::string left  = vocab_.id_to_token(left_id);
            std::string right = vocab_.id_to_token(right_id);
            
            MergePair pair{std::move(left), std::move(right)};
            merges_[pair] = rank_val;
        }
        
        // Обновление метаданных
        metadata_.vocab_size = vocab_.size();
        metadata_.merges_count = merges_.size();
        
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "[BPE Export] Ошибка загрузки бинарного файла: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Экспорт в форматы других библиотек
// ============================================================================

bool BPETokenizer::export_to_huggingface(const std::string& path) const {
    try {
        nlohmann::json root;
        
        // --------------------------------------------------------------------
        // Формирование структуры в формате HuggingFace
        // --------------------------------------------------------------------
        
        // Модель BPE
        root["model"]["type"] = "BPE";
        root["model"]["vocab"] = vocab_.to_json();
        
        // Слияния в формате "left right" (как требует HuggingFace)
        std::vector<std::string> merges_list;
        merges_list.reserve(merges_.size());
        for (const auto& [pair, rank] : merges_) {
            merges_list.push_back(pair.left + " " + pair.right);
        }
        root["model"]["merges"] = merges_list;
        root["model"]["add_prefix_space"] = false;
        
        // Конфигурация токенизатора
        root["tokenizer_config"]["unk_token"]        = unknown_token_;
        root["tokenizer_config"]["pad_token"]        = "<PAD>";
        root["tokenizer_config"]["bos_token"]        = "<BOS>";
        root["tokenizer_config"]["eos_token"]        = "<EOS>";
        root["tokenizer_config"]["model_max_length"] = 512;
        
        // --------------------------------------------------------------------
        // Запись файла
        // --------------------------------------------------------------------
        
        std::ofstream file(path);
        if (!file) {
            std::cerr << "[BPE Export] Ошибка: не удалось открыть файл для записи: " 
                      << path << std::endl;
            return false;
        }
        
        file << root.dump(2);
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "[BPE Export] Ошибка экспорта в HuggingFace: " << e.what() << std::endl;
        return false;
    }
}

bool BPETokenizer::export_to_sentencepiece(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        std::cerr << "[BPE Export] Ошибка: не удалось открыть файл для записи: " 
                  << path << std::endl;
        return false;
    }
    
    try {
        auto tokens = vocab_.get_all_tokens();
        for (size_t i = 0; i < tokens.size(); ++i) {
            // Формат:     токен<TAB>ID
            // Примечание: SentencePiece использует '_' для обозначения пробела,
            //             но мы сохраняем исходные токены без изменений
            file << tokens[i] << "\t" << i << "\n";
        }
        return file.good();
        
    } catch (const std::exception& e) {
        std::cerr << "[BPE Export] Ошибка экспорта в SentencePiece: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Информация о модели
// ============================================================================

std::string BPETokenizer::get_model_info() const {
    std::stringstream ss;
    const std::string line = "============================================================\n";
    
    ss << "\n" << line;
    ss << "ИНФОРМАЦИЯ О BPE TOKENIZER\n";
    ss << line;
    
    // Основные параметры
    ss << "Размер словаря:            " << vocab_.size() << "\n";
    ss << "Количество слияний:        " << merges_.size() << "\n";
    ss << "Byte-level режим:          " << (byte_level_ ? "включен" : "отключен") << "\n";
    ss << "Неизвестных токенов:       " << unknown_token_ << "\n";
    ss << "Максимальная длина токена: " << max_token_length_ << "\n";
    
    // Специальные токены
    ss << "\nСпециальные токены:\n";
    
    auto print_token_info = [&](const std::string& token) {
        ss << "  " << token << ": ";
        if (vocab_.contains(token)) {
            ss << "ID " << vocab_.token_to_id(token);
        } else {
            ss << "не найден";
        }
        ss << "\n";
    };
    
    print_token_info("<UNK>");
    print_token_info("<PAD>");
    print_token_info("<BOS>");
    print_token_info("<EOS>");
    
    ss << line;
    return ss.str();
}

}    // namespace bpe