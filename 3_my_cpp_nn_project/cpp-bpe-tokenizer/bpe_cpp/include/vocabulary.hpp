/**
 * @file vocabulary.hpp
 * @brief Управление словарём токенов BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Класс Vocabulary обеспечивает двустороннее отображение между токенами и их ID:
 *          - token_to_id_ для быстрого поиска ID по токену (unordered_map)
 *          - id_to_token_ для доступа к токену по ID (вектор с произвольным доступом)
 *          
 *          Поддерживаемые операции:
 *          - Добавление новых токенов с автоматическим назначением ID
 *          - Поиск токенов и ID
 *          - Сериализация в JSON и обратно
 *          - Сохранение/загрузка в текстовом и бинарном форматах
 *          - Получение списка всех токенов
 *          - Очистка словаря
 * 
 * @note ID токенов назначаются последовательно, начиная с 0
 * @warning INVALID_TOKEN используется для обозначения отсутствующего токена
 * 
 * @see BPETokenizer
 * @see FastBPETokenizer
 */

#pragma once

#include <nlohmann/json.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bpe {

// Тип для идентификатора токена
using token_id_t = uint32_t;

// Значение для обозначения невалидного/отсутствующего токена
constexpr token_id_t INVALID_TOKEN = std::numeric_limits<token_id_t>::max();

/**
 * @brief Класс для управления словарём токенов
 * 
 * Vocabulary хранит двустороннее отображение между строками-токенами
 * и их числовыми идентификаторами. Обеспечивает эффективный поиск
 * в обоих направлениях и сериализацию словаря.
 */
class Vocabulary {
public:
    /**
     * @brief Конструктор по умолчанию
     */
    Vocabulary() = default;

    // ==================== Добавление токенов ====================

    /**
     * @brief Добавить новый токен в словарь
     * @param token Строка-токен для добавления
     * @return ID назначенный токену
     * @note Если токен уже существует, возвращает его существующий ID
     */
    token_id_t add_token(const std::string& token);

    /**
     * @brief Добавить несколько специальных токенов
     * @param tokens Вектор строк для добавления
     */
    void add_special_tokens(const std::vector<std::string>& tokens);

    // ==================== Поиск ====================

    /**
     * @brief Получить ID токена по строке
     * @param token Строка-токен для поиска
     * @return ID токена или INVALID_TOKEN если не найден
     */
    token_id_t token_to_id(const std::string& token) const;

    /**
     * @brief Получить строку токена по ID
     * @param id ID токена
     * @return Ссылка на строку токена
     * @throws std::out_of_range если ID не существует
     */
    const std::string& id_to_token(token_id_t id) const;

    // ==================== Проверки ====================

    /**
     * @brief Проверить наличие токена в словаре
     * @param token Строка для проверки
     * @return true если токен существует
     */
    bool contains(const std::string& token) const;

    /**
     * @brief Проверить наличие ID в словаре
     * @param id ID для проверки
     * @return true если ID существует
     */
    bool contains_id(token_id_t id) const;

    // ==================== Размер ====================

    /**
     * @brief Получить размер словаря
     * @return Количество токенов в словаре
     */
    size_t size() const { return id_to_token_.size(); }

    /**
     * @brief Проверить, пуст ли словарь
     * @return true если словарь пуст
     */
    bool empty() const { return id_to_token_.empty(); }

    // ==================== Очистка ====================

    /**
     * @brief Очистить словарь (удалить все токены)
     * 
     * Удаляет все токены из словаря, сбрасывая его в пустое состояние.
     * После вызова size() вернет 0, а empty() вернет true.
     * 
     * @note Используется при переинициализации словаря перед обучением
     */
    void clear() {
        token_to_id_.clear();
        id_to_token_.clear();
    }

    // ==================== Сериализация ====================

    /**
     * @brief Преобразовать словарь в JSON
     * @return JSON объект с токенами
     * @note Формат: {"0": "токен0", "1": "токен1", ...}
     */
    nlohmann::json to_json() const;

    /**
     * @brief Загрузить словарь из JSON
     * @param j JSON объект с токенами
     * @throws std::runtime_error при ошибке парсинга
     */
    void from_json(const nlohmann::json& j);

    // ==================== Сохранение/загрузка ====================

    /**
     * @brief Сохранить словарь в текстовый JSON файл
     * @param path Путь к файлу
     * @return true при успешном сохранении
     */
    bool save(const std::string& path) const;

    /**
     * @brief Загрузить словарь из текстового JSON файла
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load(const std::string& path);

    /**
     * @brief Сохранить словарь в бинарный файл
     * @param path Путь к файлу
     * @return true при успешном сохранении
     * @note Бинарный формат обеспечивает более быструю загрузку
     */
    bool save_binary(const std::string& path) const;

    /**
     * @brief Загрузить словарь из бинарного файла
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load_binary(const std::string& path);

    // ==================== Вспомогательные методы ====================

    /**
     * @brief Получить все токены словаря
     * @return Вектор всех токенов в порядке их ID
     */
    std::vector<std::string> get_all_tokens() const;

private:
    std::unordered_map<std::string, token_id_t> token_to_id_;  ///< Отображение токен -> ID
    std::vector<std::string> id_to_token_;                     ///< Отображение ID -> токен
};

} // namespace bpe