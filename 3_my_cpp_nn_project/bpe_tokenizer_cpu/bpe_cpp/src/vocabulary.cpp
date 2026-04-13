/**
 * @file vocabulary.cpp
 * @brief Реализация класса Vocabulary для управления словарём токенов
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Реализация двустороннего отображения между токенами и их ID.
 *          Этот класс является фундаментальным строительным блоком для всех
 *          токенизаторов в проекте.
 * 
 *          **Структуры данных и их обоснование:**
 *          - id_to_token_: std::vector<std::string> - Произвольный доступ O(1)
 *          - token_to_id_: std::unordered_map       - Хеш-таблица O(1) в среднем
 * 
 *          **Форматы сериализации:**
 * 
 *          1. **Текстовый JSON** (читаемый, для отладки)
 *             @code
 *             // Прямой массив
 *             ["hello", "world", "<UNK>"]
 *             
 *             // Объект с метаданными
 *             {"size": 3, "tokens": ["hello", "world", "<UNK>"]}
 *             @endcode
 * 
 *          2. **Бинарный формат** (быстрый, для продакшена)
 *             @code
 *             [MAGIC: "VOCB"] [VERSION: 1] [COUNT: N]
 *             [LEN1] [TOKEN1] [LEN2] [TOKEN2] ...
 *             @endcode
 * 
 * @see Vocabulary (заголовочный файл)
 */

#include "vocabulary.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace bpe {

// ============================================================================
// Конструкторы
// ============================================================================

Vocabulary::Vocabulary(const std::vector<std::string>& tokens) {
    id_to_token_.reserve(tokens.size());
    for (const auto& token : tokens) {
        token_to_id_[token] = static_cast<token_id_t>(id_to_token_.size());
        id_to_token_.push_back(token);
    }
}

Vocabulary::Vocabulary(const std::unordered_map<std::string, size_t>& char_freq,
                       size_t min_freq) {
    // Фильтруем символы по минимальной частоте
    for (const auto& [ch, freq] : char_freq) {
        if (freq >= min_freq) {
            token_to_id_[ch] = static_cast<token_id_t>(id_to_token_.size());
            id_to_token_.push_back(ch);
        }
    }
}

// ============================================================================
// Добавление токенов
// ============================================================================

token_id_t Vocabulary::add_token(const std::string& token) {
    // Проверяем существование токена
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;    // Токен уже есть - возвращаем существующий ID
    }

    // Новый токен - добавляем в конец
    token_id_t id = static_cast<token_id_t>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(token);
    return id;
}

token_id_t Vocabulary::add_token(std::string&& token) {
    // Проверяем существование токена
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }

    // Новый токен - перемещаем строку
    token_id_t id = static_cast<token_id_t>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(std::move(token));
    return id;
}

void Vocabulary::add_special_tokens(const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        if (token_to_id_.find(token) == token_to_id_.end()) {
            add_token(token);
        }
    }
}

// ============================================================================
// Поиск
// ============================================================================

token_id_t Vocabulary::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : INVALID_TOKEN;
}

const std::string& Vocabulary::id_to_token(token_id_t id) const {
    if (id >= id_to_token_.size()) {
        throw std::out_of_range(
            "[Vocabulary] ID " + std::to_string(id) + " вне диапазона [0, " +
            std::to_string(id_to_token_.size() - 1) + "]"
        );
    }
    return id_to_token_[id];
}

const std::string& Vocabulary::id_to_token_unsafe(token_id_t id) const {
    return id_to_token_[id];
}

// ============================================================================
// Проверки
// ============================================================================

bool Vocabulary::contains(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

bool Vocabulary::contains_id(token_id_t id) const {
    return id < id_to_token_.size();
}

// ============================================================================
// Размер и управление
// ============================================================================

size_t Vocabulary::size() const {
    return id_to_token_.size();
}

bool Vocabulary::empty() const {
    return id_to_token_.empty();
}

void Vocabulary::clear() {
    id_to_token_.clear();
    token_to_id_.clear();
}

void Vocabulary::reserve(size_t capacity) {
    id_to_token_.reserve(capacity);
    // Для хеш-таблицы рекомендуем коэффициент загрузки 0.7-0.8
    token_to_id_.rehash(static_cast<size_t>(capacity * 1.5));
}

// ============================================================================
// JSON сериализация
// ============================================================================

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

    std::cout << "[Vocabulary] Загрузка JSON, тип: " << j.type_name() << std::endl;

    // ------------------------------------------------------------------------
    // Формат 1: Прямой массив ["a", "b", "c"]
    // ------------------------------------------------------------------------
    if (j.is_array()) {
        std::cout << "[Vocabulary] Формат: массив, элементов: " << j.size() << std::endl;

        id_to_token_.reserve(j.size());

        for (size_t i = 0; i < j.size(); ++i) {
            std::string token = j[i].get<std::string>();
            token_to_id_[token] = static_cast<token_id_t>(i);
            id_to_token_.push_back(token);

            if (i < 5) {
                std::cout << "Токен " << i << ": '" << token << "'" << std::endl;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Формат 2: Объект {"size": N, "tokens": ["a", "b", "c"]}
    // ------------------------------------------------------------------------
    else if (j.is_object() && j.contains("tokens") && j["tokens"].is_array()) {
        std::cout << "[Vocabulary] Формат: объект с tokens" << std::endl;

        const auto& tokens = j["tokens"];
        id_to_token_.reserve(tokens.size());

        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string token = tokens[i].get<std::string>();
            token_to_id_[token] = static_cast<token_id_t>(i);
            id_to_token_.push_back(token);

            if (i < 5) {
                std::cout << "Токен " << i << ": '" << token << "'" << std::endl;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Формат 3: Объект с ID ключами {"0": "a", "1": "b", "2": "c"}
    // ------------------------------------------------------------------------
    else if (j.is_object()) {
        std::cout << "[Vocabulary] Формат: объект с ID ключами" << std::endl;

        // Находим максимальный ID для резервирования памяти
        size_t max_id = 0;
        for (auto it = j.begin(); it != j.end(); ++it) {
            try {
                size_t id = std::stoull(it.key());
                if (id > max_id) max_id = id;
            } catch (...) {
                // Игнорируем нечисловые ключи (например, "size")
            }
        }

        std::cout << "[Vocabulary] Максимальный ID: " << max_id << std::endl;

        // Резервируем место
        id_to_token_.resize(max_id + 1);

        // Заполняем словарь
        for (auto it = j.begin(); it != j.end(); ++it) {
            try {
                size_t id = std::stoull(it.key());
                std::string token = it.value().get<std::string>();

                if (id < id_to_token_.size()) {
                    id_to_token_[id] = token;
                    token_to_id_[token] = static_cast<token_id_t>(id);
                }

                if (id < 5) {
                    std::cout << "Токен " << id << ": '" << token << "'" << std::endl;
                }
            } catch (...) {
                // Игнорируем нечисловые ключи
            }
        }
    }

    else {
        throw std::runtime_error("[Vocabulary] Неподдерживаемый формат JSON!");
    }

    std::cout << "[Vocabulary] Загружено токенов: " << id_to_token_.size() << std::endl;
}

// ============================================================================
// Сохранение/загрузка в текстовые файлы
// ============================================================================

bool Vocabulary::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Vocabulary] Ошибка: не удалось открыть " << path << std::endl;
        return false;
    }

    try {
        auto j = to_json();
        file << j.dump(2);    // pretty-print с отступами
        std::cout << "[Vocabulary] Сохранено " << id_to_token_.size()
                  << " токенов в " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Vocabulary] Ошибка сохранения: " << e.what() << std::endl;
        return false;
    }
}

bool Vocabulary::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Vocabulary] Ошибка: не удалось открыть " << path << std::endl;
        return false;
    }

    try {
        nlohmann::json j;
        file >> j;
        from_json(j);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Vocabulary] Ошибка загрузки: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Бинарная сериализация
// ============================================================================

bool Vocabulary::save_binary(const std::string& path) const {
    std::cout << "[Vocabulary] Сохранение в бинарный формат: " << path << std::endl;
    std::cout << "[Vocabulary] Размер словаря: " << id_to_token_.size() << std::endl;

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[Vocabulary] Ошибка: не удалось открыть " << path << std::endl;
        return false;
    }

    try {
        // --------------------------------------------------------------------
        // Заголовок файла
        // --------------------------------------------------------------------
        
        // Магическое число для идентификации формата
        uint32_t magic = 0x564F4342;    // "VOCB" (Vocabulary Binary)
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        std::cout << "[Vocabulary] Магическое число: 0x" << std::hex << magic << std::dec << std::endl;

        // Версия формата (для обратной совместимости)
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Количество токенов
        uint32_t count = static_cast<uint32_t>(id_to_token_.size());
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        std::cout << "[Vocabulary] Количество токенов: " << count << std::endl;

        // --------------------------------------------------------------------
        // Данные токенов
        // --------------------------------------------------------------------
        for (const auto& token : id_to_token_) {
            uint32_t len = static_cast<uint32_t>(token.size());
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(token.data(), len);
        }

        std::cout << "[Vocabulary] Бинарное сохранение завершено" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Vocabulary] Ошибка при бинарном сохранении: " << e.what() << std::endl;
        return false;
    }
}

bool Vocabulary::load_binary(const std::string& path) {
    std::cout << "\n[Vocabulary] Бинарная загрузка: " << path << std::endl;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Vocabulary] Ошибка: не удалось открыть файл!" << std::endl;
        std::cerr << "[Vocabulary] Ошибка: " << strerror(errno) << std::endl;
        return false;
    }

    std::cout << "[Vocabulary] Файл открыт успешно!" << std::endl;

    try {
        token_to_id_.clear();
        id_to_token_.clear();

        // --------------------------------------------------------------------
        // Проверка заголовка
        // --------------------------------------------------------------------
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        std::cout << "[Vocabulary] Магическое число: 0x" << std::hex << magic << std::dec << std::endl;

        if (magic != 0x564F4342) {
            std::cerr << "[Vocabulary] Ошибка: неверный формат файла (ожидалось VOCB)!" << std::endl;
            return false;
        }

        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        std::cout << "[Vocabulary] Версия формата: " << version << std::endl;

        if (version != 1) {
            std::cerr << "[Vocabulary] Ошибка: неподдерживаемая версия " << version << std::endl;
            return false;
        }

        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::cout << "[Vocabulary] Количество токенов из файла: " << size << std::endl;

        if (size > 1000000) {
            std::cerr << "[Vocabulary] Ошибка: слишком большой словарь (" << size << ")!" << std::endl;
            return false;
        }

        if (!file) {
            std::cerr << "[Vocabulary] Ошибка чтения заголовка!" << std::endl;
            return false;
        }

        id_to_token_.reserve(size);

        // --------------------------------------------------------------------
        // Чтение токенов
        // --------------------------------------------------------------------
        for (uint32_t i = 0; i < size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));

            if (i < 5) {
                std::cout << "[Vocabulary] Токен " << i << ", длина = " << len << std::endl;
            }

            if (len > 10000) {    // Защита от некорректных данных
                std::cerr << "[Vocabulary] Ошибка: слишком длинный токен (" << len << ")" << std::endl;
                return false;
            }

            if (!file) {
                std::cerr << "[Vocabulary] Ошибка чтения длины токена " << i << std::endl;
                return false;
            }

            std::string token(len, '\0');
            file.read(&token[0], len);

            if (!file) {
                std::cerr << "[Vocabulary] Ошибка чтения токена " << i << std::endl;
                return false;
            }

            token_to_id_[token] = static_cast<token_id_t>(i);
            id_to_token_.push_back(std::move(token));

            if (i < 5) {
                std::cout << "[Vocabulary] Токен " << i << ": '" << id_to_token_.back() << "'" << std::endl;
            }
        }

        std::cout << "[Vocabulary] Загружено токенов: " << id_to_token_.size() << std::endl;
        std::cout << "[Vocabulary] Бинарная загрузка успешно завершена!\n" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[Vocabulary] Ошибка при бинарной загрузке: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Вспомогательные методы
// ============================================================================

std::vector<std::string> Vocabulary::get_all_tokens() const {
    return id_to_token_;    // Возвращаем копию
}

token_id_t Vocabulary::next_id() const {
    return static_cast<token_id_t>(id_to_token_.size());
}

token_id_t Vocabulary::max_id() const {
    return id_to_token_.empty() ? INVALID_TOKEN :
           static_cast<token_id_t>(id_to_token_.size() - 1);
}

const std::vector<std::string>& Vocabulary::tokens() const {
    return id_to_token_;
}

const std::unordered_map<std::string, token_id_t>& Vocabulary::mapping() const {
    return token_to_id_;
}

}    // namespace bpe