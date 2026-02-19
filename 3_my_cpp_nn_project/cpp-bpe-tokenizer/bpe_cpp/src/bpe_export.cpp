/**
 * @file bpe_export.cpp
 * @brief Экспорт/импорт моделей BPE токенизатора в различные форматы
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.1.0
 * 
 * @details Поддержка форматов:
 *          - JSON (читаемый, для отладки)
 *          - Бинарный (компактный, быстрая загрузка)
 *          - HuggingFace Tokenizers (совместимость)
 *          - SentencePiece (альтернативный формат)
 */

#include "bpe_tokenizer.hpp"
#include "bpe_export.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

namespace bpe {

// ======================================================================
// Реализация ModelMetadata
// ======================================================================

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
    return j;
}

void ModelMetadata::from_json(const nlohmann::json& j) {
    if (j.contains("model_type")) model_type = j["model_type"].get<std::string>();
    if (j.contains("version")) version = j["version"].get<std::string>();
    if (j.contains("vocab_size")) vocab_size = j["vocab_size"].get<size_t>();
    if (j.contains("merges_count")) merges_count = j["merges_count"].get<size_t>();
    if (j.contains("byte_level")) byte_level = j["byte_level"].get<bool>();
    if (j.contains("special_tokens")) special_tokens = j["special_tokens"].get<std::vector<std::string>>();
    if (j.contains("creation_date")) creation_date = j["creation_date"].get<std::string>();
    if (j.contains("description")) description = j["description"].get<std::string>();
}

// ======================================================================
// Экспорт в JSON (единый файл)
// ======================================================================

bool BPETokenizer::save_to_json(const std::string& path) const {
    nlohmann::json j;
    
    j["vocab"] = vocab_.to_json();
    
    std::vector<std::pair<std::string, std::string>> merges_list;
    for (const auto& [pair, rank] : merges_) {
        merges_list.push_back({pair.left, pair.right});
    }
    j["merges"] = merges_list;
    
    j["config"]["byte_level"] = byte_level_;
    j["config"]["vocab_size"] = vocab_size_;
    j["config"]["unknown_token"] = unknown_token_;
    
    j["metadata"] = metadata_.to_json();
    
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
        
        if (j.contains("vocab")) {
            vocab_.from_json(j["vocab"]);
        }
        
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
        
        if (j.contains("config")) {
            if (j["config"].contains("byte_level"))
                byte_level_ = j["config"]["byte_level"].get<bool>();
            if (j["config"].contains("vocab_size"))
                vocab_size_ = j["config"]["vocab_size"].get<size_t>();
            if (j["config"].contains("unknown_token"))
                unknown_token_ = j["config"]["unknown_token"].get<std::string>();
        }
        
        if (j.contains("metadata")) {
            metadata_.from_json(j["metadata"]);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка загрузки JSON: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Экспорт в формат HuggingFace Tokenizers
// ======================================================================

bool BPETokenizer::export_to_huggingface(const std::string& path) const {
    try {
        nlohmann::json j;
        
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
        std::cerr << "Ошибка экспорта в HuggingFace: " << e.what() << std::endl;
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
        auto tokens = vocab_.get_all_tokens();
        for (size_t i = 0; i < tokens.size(); ++i) {
            file << tokens[i] << "\t" << i << "\n";
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка экспорта в SentencePiece: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Информация о модели
// ======================================================================

std::string BPETokenizer::get_model_info() const {
    std::stringstream ss;
    ss << "=== Информация о BPE Tokenizer ===\n";
    ss << "Размер словаря: " << vocab_.size() << "\n";
    ss << "Количество слияний: " << merges_.size() << "\n";
    ss << "Byte-level режим: " << (byte_level_ ? "enabled" : "disabled") << "\n";
    ss << "Неизвестных токенов: " << unknown_token_ << "\n";
    ss << "Максимальная длина токена: " << max_token_length_ << "\n";
    ss << "===================================";
    return ss.str();
}

} // namespace bpe