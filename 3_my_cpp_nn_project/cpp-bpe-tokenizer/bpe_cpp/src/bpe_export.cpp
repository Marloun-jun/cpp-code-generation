/**
 * @file bpe_export.cpp
 * @brief Экспорт/импорт моделей BPE токенизатора в различные форматы
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Поддержка форматов:
 *          - JSON (читаемый, для отладки)
 *          - Бинарный (компактный, быстрая загрузка)
 *          - HuggingFace Tokenizers (совместимость)
 *          - SentencePiece (альтернативный формат)
 */

#include "bpe_tokenizer.hpp"
#include "bpe_export.hpp"
#include <fstream>
#include <iostream>
#include <sstream>      // Добавлено для std::stringstream
#include <nlohmann/json.hpp>

namespace bpe {

// ======================================================================
// Экспорт в JSON
// ======================================================================

bool BPETokenizer::save_to_json(const std::string& path) const {
    nlohmann::json j;
    
    // Сохраняем словарь
    j["vocab"] = vocab_.to_json();
    
    // Сохраняем мерджи
    std::vector<std::pair<std::string, std::string>> merges_list;
    for (const auto& [pair, rank] : merges_) {
        merges_list.push_back({pair.left, pair.right});
    }
    j["merges"] = merges_list;
    
    // Сохраняем конфигурацию
    j["config"]["byte_level"] = byte_level_;
    j["config"]["vocab_size"] = vocab_size_;
    j["config"]["unknown_token"] = unknown_token_;
    
    // Сохраняем метаданные
    j["metadata"]["version"] = "1.0.0";
    j["metadata"]["creation_date"] = metadata_.creation_date;
    j["metadata"]["description"] = metadata_.description;
    
    // Запись в файл
    std::ofstream file(path);
    if (!file) return false;
    file << j.dump(2);
    
    return true;
}

bool BPETokenizer::load_from_json(const std::string& path) {
    std::ifstream file(path);
    if (!file) return false;
    
    try {
        nlohmann::json j;
        file >> j;
        
        // Загружаем словарь
        if (j.contains("vocab")) {
            vocab_.from_json(j["vocab"]);
        }
        
        // Загружаем мерджи
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
        
        // Загружаем конфигурацию
        if (j.contains("config")) {
            if (j["config"].contains("byte_level"))
                byte_level_ = j["config"]["byte_level"].get<bool>();
            if (j["config"].contains("vocab_size"))
                vocab_size_ = j["config"]["vocab_size"].get<size_t>();
            if (j["config"].contains("unknown_token"))
                unknown_token_ = j["config"]["unknown_token"].get<std::string>();
        }
        
        // Загружаем метаданные
        if (j.contains("metadata")) {
            metadata_.from_json(j["metadata"]);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка загрузки JSON: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Сохранение в единый бинарный файл (совместимость с ModelExport)
// ======================================================================

bool BPETokenizer::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    
    try {
        // Магическое число для проверки формата
        uint32_t magic = 0x42504542;  // "BPEB"
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // Версия формата
        uint32_t version = 1;
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
        
        // Количество мерджей
        uint32_t merges_count = static_cast<uint32_t>(merges_.size());
        file.write(reinterpret_cast<const char*>(&merges_count), sizeof(merges_count));
        
        // Мерджи (сначала левый, потом правый токен)
        for (const auto& [pair, rank] : merges_) {
            uint32_t left_len = static_cast<uint32_t>(pair.left.size());
            uint32_t right_len = static_cast<uint32_t>(pair.right.size());
            
            file.write(reinterpret_cast<const char*>(&left_len), sizeof(left_len));
            file.write(pair.left.data(), left_len);
            file.write(reinterpret_cast<const char*>(&right_len), sizeof(right_len));
            file.write(pair.right.data(), right_len);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка бинарного сохранения: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Загрузка из единого бинарного файла (совместимость с ModelExport)
// ======================================================================

bool BPETokenizer::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    try {
        // Проверка магического числа
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x42504542) return false;
        
        // Проверка версии
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) return false;
        
        // Очищаем текущие данные
        vocab_.clear();
        merges_.clear();
        
        // Загрузка словаря
        uint32_t vocab_size;
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        for (uint32_t i = 0; i < vocab_size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            std::string token(len, '\0');
            file.read(&token[0], len);
            
            vocab_.add_token(token);
        }
        
        // Загрузка мерджей
        uint32_t merges_count;
        file.read(reinterpret_cast<char*>(&merges_count), sizeof(merges_count));
        
        for (uint32_t i = 0; i < merges_count; ++i) {
            uint32_t left_len, right_len;
            
            file.read(reinterpret_cast<char*>(&left_len), sizeof(left_len));
            std::string left(left_len, '\0');
            file.read(&left[0], left_len);
            
            file.read(reinterpret_cast<char*>(&right_len), sizeof(right_len));
            std::string right(right_len, '\0');
            file.read(&right[0], right_len);
            
            MergePair pair{left, right};
            merges_[pair] = i;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка бинарной загрузки: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Экспорт в формат HuggingFace Tokenizers
// ======================================================================

bool BPETokenizer::export_to_huggingface(const std::string& path) const {
    try {
        nlohmann::json j;
        
        // HuggingFace формат
        j["model"]["type"] = "BPE";
        j["model"]["vocab"] = vocab_.to_json();
        
        std::vector<std::string> merges_list;
        for (const auto& [pair, rank] : merges_) {
            merges_list.push_back(pair.left + " " + pair.right);
        }
        j["model"]["merges"] = merges_list;
        
        j["tokenizer_config"]["unk_token"] = unknown_token_;
        j["tokenizer_config"]["pad_token"] = "<PAD>";
        j["tokenizer_config"]["bos_token"] = "<BOS>";
        j["tokenizer_config"]["eos_token"] = "<EOS>";
        
        std::ofstream file(path);
        if (!file) return false;
        file << j.dump(2);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка экспорта в HuggingFace: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Экспорт в формат SentencePiece
// ======================================================================

bool BPETokenizer::export_to_sentencepiece(const std::string& path) const {
    std::ofstream file(path);
    if (!file) return false;
    
    try {
        // SentencePiece формат (простой текстовый)
        auto tokens = vocab_.get_all_tokens();
        for (size_t i = 0; i < tokens.size(); ++i) {
            file << tokens[i] << "\t" << i << "\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка экспорта в SentencePiece: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Информация о модели
// ======================================================================

std::string BPETokenizer::get_model_info() const {
    std::stringstream ss;
    ss << "=== BPE Tokenizer Model Info ===\n";
    ss << "Vocabulary size: " << vocab_.size() << "\n";
    ss << "Merges count: " << merges_.size() << "\n";
    ss << "Byte-level mode: " << (byte_level_ ? "enabled" : "disabled") << "\n";
    ss << "Unknown token: " << unknown_token_ << "\n";
    ss << "Max token length: " << max_token_length_ << "\n";
    ss << "================================";
    return ss.str();
}

} // namespace bpe