/**
 * @file vocabulary.cpp
 * @brief Реализация класса Vocabulary для управления словарём токенов
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Этот файл содержит реализацию двустороннего отображения между токенами
 *          и их числовыми идентификаторами. Vocabulary является фундаментальным
 *          компонентом всех токенизаторов в проекте.
 * 
 *          **Структуры данных:**
 *          - id_to_token_: std::vector<std::string>    для O(1) доступа по ID
 *          - token_to_id_: std::unordered_map          для O(1) поиска по токену
 * 
 *          **Поддерживаемые форматы сериализации:**
 * 
 *          1) **JSON (текстовый, читаемый)**
 *             - Формат 1:    Прямой массив ["токен1", "токен2", ...]
 *             - Формат 2:    Объект {"size": N, "tokens": [...]}
 *             - Удобен для отладки и ручного просмотра
 * 
 *          2) **Бинарный (быстрый, компактный)**
 *             - Магическое число "VOCB" (0x564F4342) для проверки формата
 *             - Версионирование для обратной совместимости
 *             - Прямая запись строк с длиной
 *             - Рекомендуется для продакшн использования
 * 
 *          **Производительность:**
 *          - add_token():      O(1) амортизированно
 *          - token_to_id():    O(1) в среднем
 *          - id_to_token():    O(1) гарантированно
 *          - save()/load():    O(n) где n - размер словаря
 * 
 * @note Бинарный формат рекомендуется для продакшн использования
 * @warning При загрузке из JSON выполняется валидация формата
 * 
 * @see Vocabulary
 */

#include "vocabulary.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cerrno>

namespace bpe {

// ======================================================================
// Конструкторы
// ======================================================================

/**
 * @brief Конструктор по умолчанию
 */
Vocabulary::Vocabulary() = default;

/**
 * @brief Конструктор с вектором токенов
 * 
 * @param tokens Вектор начальных токенов
 */
Vocabulary::Vocabulary(const std::vector<std::string>& tokens) {
    id_to_token_.reserve(tokens.size());
    for (const auto& token : tokens) {
        token_to_id_[token] = static_cast<token_id_t>(id_to_token_.size());
        id_to_token_.push_back(token);
    }
}

/**
 * @brief Конструктор с картой частот символов
 * 
 * @param char_freq Карта частот символов
 * @param min_freq Минимальная частота для включения
 * 
 * Создаёт начальный словарь из символов, встречающихся
 * в корпусе с частотой не ниже min_freq.
 */
Vocabulary::Vocabulary(const std::unordered_map<std::string, size_t>& char_freq, 
                       size_t min_freq) {
    for (const auto& [ch, freq] : char_freq) {
        if (freq >= min_freq) {
            token_to_id_[ch] = static_cast<token_id_t>(id_to_token_.size());
            id_to_token_.push_back(ch);
        }
    }
}

// ======================================================================
// Добавление токенов
// ======================================================================

/**
 * @brief Добавить новый токен в словарь
 * 
 * @param token Строка-токен для добавления
 * @return token_id_t ID назначенный токену
 * 
 * **Алгоритм:**
 * 1. Проверяем существование токена в token_to_id_
 * 2. Если существует - возвращаем существующий ID
 * 3. Если новый:
 *    - Создаём новый ID = текущий размер id_to_token_
 *    - Добавляем токен в конец id_to_token_
 *    - Добавляем отображение в token_to_id_
 *    - Возвращаем новый ID
 * 
 * **Пример:**
 * \code
 * Vocabulary vocab;
 * token_id_t id = vocab.add_token("hello");    // id = 0
 * id = vocab.add_token("hello");               // id = 0 (уже существует)
 * id = vocab.add_token("world");               // id = 1
 * \endcode
 */
token_id_t Vocabulary::add_token(const std::string& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;    // Токен уже существует
    }
    
    token_id_t id = static_cast<token_id_t>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(token);
    return id;
}

/**
 * @brief Добавить токен с перемещением (эффективная версия)
 * 
 * @param token Строка-токен для добавления (будет перемещена)
 * @return token_id_t ID назначенный токену
 */
token_id_t Vocabulary::add_token(std::string&& token) {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) {
        return it->second;
    }
    
    token_id_t id = static_cast<token_id_t>(id_to_token_.size());
    token_to_id_[token] = id;
    id_to_token_.push_back(std::move(token));
    return id;
}

/**
 * @brief Добавить несколько специальных токенов
 * 
 * @param tokens Вектор строк для добавления
 * 
 * Добавляет только те токены, которых ещё нет в словаре.
 * Полезно для добавления стандартных токенов:
 * {"<UNK>", "<PAD>", "<BOS>", "<EOS>"}
 */
void Vocabulary::add_special_tokens(const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        if (token_to_id_.find(token) == token_to_id_.end()) {
            add_token(token);
        }
    }
}

// ======================================================================
// Поиск
// ======================================================================

/**
 * @brief Получить ID токена по строке
 * 
 * @param token Строка-токен для поиска
 * @return token_id_t ID токена или INVALID_TOKEN если не найден
 * 
 * **Сложность:**    O(1) в среднем (хеш-таблица)
 */
token_id_t Vocabulary::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return (it != token_to_id_.end()) ? it->second : INVALID_TOKEN;
}

/**
 * @brief Получить строку токена по ID
 * 
 * @param id ID токена
 * @return const std::string& Ссылка на строку токена
 * @throws std::out_of_range если ID не существует
 */
const std::string& Vocabulary::id_to_token(token_id_t id) const {
    if (id >= id_to_token_.size()) {
        throw std::out_of_range("Token ID out of range: " + std::to_string(id));
    }
    return id_to_token_[id];
}

/**
 * @brief Получить строку токена по ID (без проверки границ)
 * 
 * @param id ID токена
 * @return const std::string& Ссылка на строку токена
 * 
 * @warning Не проверяет границы! Используйте только если уверены в ID.
 */
const std::string& Vocabulary::id_to_token_unsafe(token_id_t id) const {
    return id_to_token_[id];
}

// ======================================================================
// Проверки
// ======================================================================

/**
 * @brief Проверить наличие токена в словаре
 * 
 * @param token Строка для проверки
 * @return true если токен существует
 */
bool Vocabulary::contains(const std::string& token) const {
    return token_to_id_.find(token) != token_to_id_.end();
}

/**
 * @brief Проверить наличие ID в словаре
 * 
 * @param id ID для проверки
 * @return true если ID существует
 */
bool Vocabulary::contains_id(token_id_t id) const {
    return id < id_to_token_.size();
}

// ======================================================================
// Методы для получения информации о словаре
// ======================================================================

/**
 * @brief Получить размер словаря
 * @return size_t Количество токенов в словаре
 */
size_t Vocabulary::size() const {
    return id_to_token_.size();
}

/**
 * @brief Проверить, пуст ли словарь
 * @return true если словарь пуст
 */
bool Vocabulary::empty() const {
    return id_to_token_.empty();
}

/**
 * @brief Очистить словарь
 * 
 * Удаляет все токены и очищает все структуры данных.
 */
void Vocabulary::clear() {
    id_to_token_.clear();
    token_to_id_.clear();
}

/**
 * @brief Зарезервировать память под указанное количество токенов
 * @param capacity Желаемая ёмкость
 * 
 * Позволяет оптимизировать производительность при заранее известном
 * количестве токенов, избегая многократных переаллокаций.
 */
void Vocabulary::reserve(size_t capacity) {
    id_to_token_.reserve(capacity);
    token_to_id_.rehash(static_cast<size_t>(capacity * 1.5));
}

// ======================================================================
// Сериализация в JSON
// ======================================================================

/**
 * @brief Преобразовать словарь в JSON
 * 
 * @return nlohmann::json JSON объект в формате:
 * {
 *   "size":      1000,
 *   "tokens":    ["токен0", "токен1", ...]
 * }
 * 
 * @note Формат оптимизирован для чтения человеком
 */
nlohmann::json Vocabulary::to_json() const {
    nlohmann::json j;
    j["size"] = id_to_token_.size();
    j["tokens"] = nlohmann::json::array();
    
    for (const auto& token : id_to_token_) {
        j["tokens"].push_back(token);
    }
    
    return j;
}

/**
 * @brief Загрузить словарь из JSON
 * 
 * @param j JSON объект
 * 
 * **Поддерживаемые форматы:**
 * 1. Прямой массив:            ["токен0", "токен1", ...]
 * 2. Объект с полем tokens:    {"tokens": ["токен0", ...]}
 * 3. Объект с ID ключами:      {"0": "токен0", "1": "токен1", ...}
 * 
 * Выводит отладочную информацию о процессе загрузки.
 * 
 * @throws std::runtime_error при ошибке формата
 */
void Vocabulary::from_json(const nlohmann::json& j) {
    token_to_id_.clear();
    id_to_token_.clear();
    
    std::cout << "Загрузка JSON словаря, тип: " << j.type_name() << std::endl;
    
    // ===== Формат 1: Прямой массив ["a", "b", "c"] =====
    if (j.is_array()) {
        std::cout << "Формат: массив, элементов: " << j.size() << std::endl;
        
        id_to_token_.reserve(j.size());
        
        for (size_t i = 0; i < j.size(); ++i) {
            std::string token = j[i].get<std::string>();
            token_to_id_[token] = static_cast<token_id_t>(i);
            id_to_token_.push_back(token);
            
            if (i < 5) {    // Показываем первые 5 для отладки
                std::cout << "Токен " << i << ": '" << token << "'" << std::endl;
            }
        }
    }
    // ===== Формат 2: Объект {"size": 9, "tokens": ["a", "b", "c"]} =====
    else if (j.is_object() && j.contains("tokens") && j["tokens"].is_array()) {
        std::cout << "Формат: объект с tokens" << std::endl;
        
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
    // ===== Формат 3: Объект с ID в качестве ключей {"0": "token0", "1": "token1"} =====
    else if (j.is_object()) {
        std::cout << "Формат: объект с ID ключами" << std::endl;
        
        // Находим максимальный ID
        size_t max_id = 0;
        for (auto it = j.begin(); it != j.end(); ++it) {
            try {
                size_t id = std::stoull(it.key());
                if (id > max_id) max_id = id;
            } catch (...) {
                // Игнорируем нечисловые ключи
            }
        }
        
        std::cout << "Максимальный ID: " << max_id << std::endl;
        
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
        throw std::runtime_error("Неподдерживаемый формат JSON словаря");
    }
    
    std::cout << "Загружено токенов: " << id_to_token_.size() << std::endl;
}

// ======================================================================
// Сохранение/загрузка (текстовый формат)
// ======================================================================

/**
 * @brief Сохранить словарь в текстовый JSON файл
 * 
 * @param path Путь к файлу
 * @return true при успешном сохранении, false при ошибке
 * 
 * @note Использует to_json() для генерации JSON
 */
bool Vocabulary::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл для записи: " << path << std::endl;
        return false;
    }
    
    try {
        auto j = to_json();
        file << j.dump(2);    // Отступ 2 пробела для читаемости
        std::cout << "Словарь сохранен в текстовый файл: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при сохранении словаря: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Загрузить словарь из текстового JSON файла
 * 
 * @param path Путь к файлу
 * @return true при успешной загрузке, false при ошибке
 * 
 * @note Использует from_json() для парсинга
 */
bool Vocabulary::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл: " << path << std::endl;
        return false;
    }
    
    try {
        nlohmann::json j;
        file >> j;
        
        from_json(j);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Ошибка загрузки словаря: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Сохранение/загрузка (бинарный формат)
// ======================================================================

/**
 * @brief Сохранить словарь в бинарный файл
 * 
 * @param path Путь к файлу
 * @return true при успешном сохранении, false при ошибке
 * 
 * **Бинарный формат:**
 * ```
 * [MAGIC: 4 байта]      - "VOCB" (0x564F4342)
 * [VERSION: 4 байта]    - 1
 * [COUNT: 4 байта]      - количество токенов
 * [для каждого токена:]
 *   [LEN: 4 байта]      - длина токена
 *   [DATA: LEN байт]    - сам токен (UTF-8)
 * ```
 * 
 * **Преимущества:**
 * - Компактность (нет JSON-синтаксиса)
 * - Скорость загрузки (нет парсинга)
 * - Прямое отображение в память
 * 
 * @note Рекомендуется для продакшн использования
 */
bool Vocabulary::save_binary(const std::string& path) const {
    std::cout << "=== Vocabulary::save_binary ===" << std::endl;
    std::cout << "Путь: " << path << std::endl;
    std::cout << "Размер словаря: " << id_to_token_.size() << std::endl;
    
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Не удалось открыть файл для записи: " << path << std::endl;
        return false;
    }
    
    try {
        // Магическое число для проверки формата
        uint32_t magic = 0x564F4342;    // "VOCB" (Vocabulary)
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        std::cout << "Записано магическое число: 0x" << std::hex << magic << std::dec << std::endl;
        
        // Версия формата
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Количество токенов
        uint32_t count = static_cast<uint32_t>(id_to_token_.size());
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));
        std::cout << "Записано токенов: " << count << std::endl;
        
        // Токены
        for (const auto& token : id_to_token_) {
            uint32_t len = static_cast<uint32_t>(token.size());
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(token.data(), len);
        }
        
        std::cout << "Файл успешно записан" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка при сохранении: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Загрузить словарь из бинарного файла
 * 
 * @param path Путь к файлу
 * @return true при успешной загрузке, false при ошибке
 * 
 * **Процесс загрузки:**
 * 1. Проверка магического числа
 * 2. Проверка версии формата
 * 3. Чтение количества токенов
 * 4. Чтение каждого токена с проверкой длины
 * 5. Построение token_to_id_ и id_to_token_
 * 
 * @note Содержит подробную отладочную информацию
 */
bool Vocabulary::load_binary(const std::string& path) {
    std::cout << "НАЧАЛО load_binary!" << std::endl;
    std::cout << "Путь: " << path << std::endl;
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "НЕ УДАЛОСЬ ОТКРЫТЬ ФАЙЛ: " << path << std::endl;
        std::cerr << "Ошибка: " << strerror(errno) << std::endl;
        return false;
    }
    
    std::cout << "Файл успешно открыт!" << std::endl;
    std::cout.flush();
    
    try {
        token_to_id_.clear();
        id_to_token_.clear();
        
        // Читаем магическое число
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        std::cout << ">>> Магическое число: 0x" << std::hex << magic << std::dec << std::endl;
        std::cout.flush();
        
        if (magic != 0x564F4342) {
            std::cerr << ">>> Неверное магическое число! Ожидалось 0x564F4342" << std::endl;
            std::cerr.flush();
            return false;
        }
        
        // Читаем версию
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        std::cout << ">>> Версия: " << version << std::endl;
        std::cout.flush();
        
        if (version != 1) {
            std::cerr << ">>> Неверная версия! Ожидалась 1" << std::endl;
            std::cerr.flush();
            return false;
        }
        
        // Читаем размер словаря
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::cout << ">>> Размер словаря из файла: " << size << std::endl;
        std::cout.flush();
        
        if (size > 1000000) {
            std::cerr << ">>> Некорректный размер словаря: " << size << std::endl;
            std::cerr.flush();
            return false;
        }
        
        if (!file) {
            std::cerr << ">>> Ошибка чтения размера словаря!" << std::endl;
            std::cerr.flush();
            return false;
        }
        
        id_to_token_.reserve(size);
        
        // Читаем каждый токен
        for (uint32_t i = 0; i < size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            if (i < 5) {
                std::cout << ">>> Чтение токена " << i << ", длина = " << len << std::endl;
                std::cout.flush();
            }
            
            if (len > 10000) {    // Максимальная разумная длина токена
                std::cerr << ">>> Некорректная длина токена " << i << ": " << len << std::endl;
                std::cerr.flush();
                return false;
            }
            
            if (!file) {
                std::cerr << ">>> Ошибка чтения длины токена " << i << std::endl;
                std::cerr.flush();
                return false;
            }
            
            std::string token(len, '\0');
            file.read(&token[0], len);
            
            if (!file) {
                std::cerr << ">>> Ошибка чтения токена " << i << std::endl;
                std::cerr.flush();
                return false;
            }
            
            token_to_id_[token] = static_cast<token_id_t>(i);
            id_to_token_.push_back(std::move(token));
            
            if (i < 5) {
                std::cout << ">>> Токен " << i << ": '" << id_to_token_.back() << "' (длина " << len << ")" << std::endl;
                std::cout.flush();
            }
        }
        
        std::cout << ">>> Загружено токенов: " << id_to_token_.size() << std::endl;
        std::cout.flush();
        std::cout << ">>> Vocabulary::load_binary SUCCESS" << std::endl;
        std::cout.flush();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << ">>> Ошибка при бинарной загрузке: " << e.what() << std::endl;
        std::cerr.flush();
        return false;
    }
}

// ======================================================================
// Вспомогательные методы
// ======================================================================

/**
 * @brief Получить все токены словаря
 * @return std::vector<std::string> Вектор всех токенов в порядке их ID
 * 
 * @note Возвращает копию вектора для безопасности
 */
std::vector<std::string> Vocabulary::get_all_tokens() const {
    return id_to_token_;
}

/**
 * @brief Получить следующий свободный ID
 * @return token_id_t ID для нового токена
 */
token_id_t Vocabulary::next_id() const {
    return static_cast<token_id_t>(id_to_token_.size());
}

/**
 * @brief Получить максимальный ID в словаре
 * @return token_id_t Максимальный ID
 */
token_id_t Vocabulary::max_id() const {
    return id_to_token_.empty() ? INVALID_TOKEN : 
           static_cast<token_id_t>(id_to_token_.size() - 1);
}

/**
 * @brief Получить ссылку на внутренний вектор токенов
 * @return const std::vector<std::string>& Ссылка на вектор
 */
const std::vector<std::string>& Vocabulary::tokens() const {
    return id_to_token_;
}

/**
 * @brief Получить ссылку на внутреннюю хеш-таблицу
 * @return const std::unordered_map<std::string, token_id_t>& Ссылка на map
 */
const std::unordered_map<std::string, token_id_t>& Vocabulary::mapping() const {
    return token_to_id_;
}

} // namespace bpe