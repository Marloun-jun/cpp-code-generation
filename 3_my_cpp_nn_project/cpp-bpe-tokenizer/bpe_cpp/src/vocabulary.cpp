/**
 * @file vocabulary.cpp
 * @brief Реализация класса Vocabulary для управления словарём токенов
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Реализация двустороннего отображения между токенами и их ID.
 *          Поддерживает два формата сериализации:
 *          - JSON (текстовый, читаемый)
 *          - Бинарный (быстрый, компактный)
 * 
 * @note Бинарный формат рекомендуется для продакшн использования
 * @warning При загрузке из JSON выполняется валидация формата
 * 
 * @see Vocabulary
 */

#include "vocabulary.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace bpe {

// ==================== Добавление токенов ====================

token_id_t Vocabulary::add_token(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;  // Токен уже существует
    }
    
    token_id_t id = static_cast<token_id_t>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(token);
    return id;
}

void Vocabulary::add_special_tokens(const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        if (token_to_id_.find(token) == token_to_id_.end()) {
            add_token(token);
        }
    }
}

// ==================== Поиск ====================

token_id_t Vocabulary::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : INVALID_TOKEN;
}

const std::string& Vocabulary::id_to_token(token_id_t id) const {
    if (id < id_to_token_.size()) {
        return id_to_token_[id];
    }
    
    static const std::string empty_string;
    return empty_string;
}

// ==================== Проверки ====================

bool Vocabulary::contains(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

bool Vocabulary::contains_id(token_id_t id) const {
    return id < id_to_token_.size();
}

// ==================== Сериализация в JSON ====================

nlohmann::json Vocabulary::to_json() const {
    nlohmann::json j;
    j["size"] = id_to_token_.size();
    j["tokens"] = nlohmann::json::array();
    
    for (const auto& token : id_to_token_) {
        j["tokens"].push_back(token);
    }
    
    return j;
}

void Vocabulary::from_json(const nlohmann::json& j) {
    token_to_id_.clear();
    id_to_token_.clear();
    
    std::cout << "📦 Загрузка JSON словаря, тип: " << j.type_name() << std::endl;
    
    // Формат 1: Прямой массив ["a", "b", "c"]
    if (j.is_array()) {
        std::cout << "  Формат: массив, элементов: " << j.size() << std::endl;
        
        id_to_token_.reserve(j.size());
        
        for (size_t i = 0; i < j.size(); ++i) {
            std::string token = j[i].get<std::string>();
            token_to_id_[token] = static_cast<token_id_t>(i);
            id_to_token_.push_back(token);
            
            if (i < 5) {  // Показываем первые 5 для отладки
                std::cout << "    Токен " << i << ": '" << token << "'" << std::endl;
            }
        }
    }
    // Формат 2: Объект {"size": 9, "tokens": ["a", "b", "c"]}
    else if (j.is_object() && j.contains("tokens")) {
        std::cout << "  Формат: объект с tokens" << std::endl;
        
        const auto& tokens = j["tokens"];
        if (tokens.is_array()) {
            id_to_token_.reserve(tokens.size());
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::string token = tokens[i].get<std::string>();
                token_to_id_[token] = static_cast<token_id_t>(i);
                id_to_token_.push_back(token);
                
                if (i < 5) {
                    std::cout << "    Токен " << i << ": '" << token << "'" << std::endl;
                }
            }
        }
    }
    else {
        std::cerr << "❌ Неподдерживаемый формат JSON!" << std::endl;
        return;
    }
    
    std::cout << "  ✅ Загружено токенов: " << id_to_token_.size() << std::endl;
}

// ==================== Сохранение/загрузка (текстовый формат) ====================

bool Vocabulary::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "❌ Не удалось открыть файл для записи: " << path << std::endl;
        return false;
    }
    
    try {
        auto j = to_json();
        file << j.dump(2);  // Отступ 2 пробела для читаемости
        std::cout << "💾 Словарь сохранен в текстовый файл: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка при сохранении словаря: " << e.what() << std::endl;
        return false;
    }
}

bool Vocabulary::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "❌ Не удалось открыть файл: " << path << std::endl;
        return false;
    }
    
    try {
        nlohmann::json j;
        file >> j;
        
        from_json(j);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка загрузки словаря: " << e.what() << std::endl;
        return false;
    }
}

// ==================== Сохранение/загрузка (бинарный формат) ====================

bool Vocabulary::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ Не удалось открыть бинарный файл для записи: " << path << std::endl;
        return false;
    }
    
    try {
        // Записываем размер словаря (4 байта)
        uint32_t size = static_cast<uint32_t>(id_to_token_.size());
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        
        // Записываем каждый токен
        for (const auto& token : id_to_token_) {
            uint32_t len = static_cast<uint32_t>(token.length());
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(token.c_str(), len);
        }
        
        std::cout << "💾 Словарь сохранен в бинарный файл: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка при бинарном сохранении: " << e.what() << std::endl;
        return false;
    }
}

bool Vocabulary::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ Не удалось открыть бинарный файл: " << path << std::endl;
        return false;
    }
    
    try {
        token_to_id_.clear();
        id_to_token_.clear();
        
        // Читаем размер
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        
        if (!file) {
            std::cerr << "❌ Ошибка чтения размера словаря" << std::endl;
            return false;
        }
        
        id_to_token_.reserve(size);
        
        // Читаем каждый токен
        for (uint32_t i = 0; i < size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            if (!file || len > 10000) {  // Защита от некорректных данных
                std::cerr << "❌ Ошибка чтения длины токена" << std::endl;
                return false;
            }
            
            std::string token(len, '\0');
            file.read(&token[0], len);
            
            if (!file) {
                std::cerr << "❌ Ошибка чтения токена" << std::endl;
                return false;
            }
            
            token_to_id_[token] = i;
            id_to_token_.push_back(std::move(token));
        }
        
        std::cout << "✅ Загружено токенов из бинарного файла: " << id_to_token_.size() << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка при бинарной загрузке: " << e.what() << std::endl;
        return false;
    }
}

// ==================== Вспомогательные методы ====================

std::vector<std::string> Vocabulary::get_all_tokens() const {
    return id_to_token_;
}

} // namespace bpe