/**
 * @file vocabulary.hpp
 * @brief Управление словарём токенов BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Класс Vocabulary обеспечивает двустороннее отображение между токенами и их ID -
 *          это сердце любого токенизатора. Эффективность этого класса критически важна
 *          для производительности всего проекта.
 * 
 *          **Структуры данных:**
 * 
 *          1) **token_to_id_** (std::unordered_map)
 *             - Хеш-таблица для быстрого поиска ID по токену
 *             - Сложность:    O(1) в среднем
 *             - Используется при кодировании (encode)
 * 
 *          2) **id_to_token_** (std::vector)
 *             - Вектор с произвольным доступом для получения токена по ID
 *             - Сложность:    O(1) гарантированно
 *             - Используется при декодировании (decode)
 * 
 *          **Преимущества такого подхода:**
 *          - Максимальная скорость в обоих направлениях
 *          - Компактное хранение (вектор занимает меньше места, чем map)
 *          - ID назначаются последовательно, что удобно для индексации
 * 
 *          **Поддерживаемые операции:**
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

#ifndef BPE_VOCABULARY_HPP
#define BPE_VOCABULARY_HPP

#include <nlohmann/json.hpp>

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace bpe {

// ======================================================================
// Основные определения
// ======================================================================

/**
 * @brief Тип для идентификатора токена
 * 
 * Используется 32-битное беззнаковое целое, что позволяет хранить до 4 миллиардов токенов.
 * Этого более чем достаточно для любых разумных словарей (обычно 32k - 100k токенов).
 */
using token_id_t = uint32_t;

/**
 * @brief Значение для обозначения невалидного/отсутствующего токена
 * 
 * Используется максимальное значение uint32_t (0xFFFFFFFF), так как обычные ID
 * никогда не достигнут этого значения (максимальный разумный словарь ~1M токенов).
 */
constexpr token_id_t INVALID_TOKEN = std::numeric_limits<token_id_t>::max();

// ======================================================================
// Vocabulary - основной класс управления словарём
// ======================================================================

/**
 * @brief Класс для управления словарём токенов
 * 
 * Vocabulary хранит двустороннее отображение между строками-токенами
 * и их числовыми идентификаторами. Обеспечивает эффективный поиск
 * в обоих направлениях и сериализацию словаря.
 * 
 * Пример использования:
 * \code
 * Vocabulary vocab;
 * 
 * // Добавление токенов
 * token_id_t id1 = vocab.add_token("hello");
 * token_id_t id2 = vocab.add_token("world");
 * 
 * // Поиск
 * token_id_t found = vocab.token_to_id("hello");    // = 0
 * std::string token = vocab.id_to_token(1);         // = "world"
 * 
 * // Проверка наличия
 * if (vocab.contains("hello")) {
 *     std::cout << "Токен найден!\n";
 * }
 * 
 * // Размер словаря
 * std::cout << "Словарь содержит " << vocab.size() << " токенов\n";
 * 
 * // Сохранение
 * vocab.save("vocab.json");
 * 
 * // Очистка
 * vocab.clear();
 * \endcode
 */
class Vocabulary {
public:
    /**
     * @brief Конструктор по умолчанию
     * 
     * Создаёт пустой словарь без токенов.
     */
    Vocabulary();

    /**
     * @brief Конструктор с начальными токенами
     * 
     * @param tokens Вектор начальных токенов
     */
    explicit Vocabulary(const std::vector<std::string>& tokens);

    /**
     * @brief Конструктор с картой частот символов
     * 
     * @param char_freq Карта частот символов
     * @param min_freq Минимальная частота для включения
     * 
     * Создаёт начальный словарь из символов, встречающихся
     * в корпусе с частотой не ниже min_freq.
     */
    explicit Vocabulary(const std::unordered_map<std::string, size_t>& char_freq, 
                        size_t min_freq = 2);

    // Запрещаем копирование, разрешаем перемещение
    Vocabulary(const Vocabulary&) = delete;
    Vocabulary& operator=(const Vocabulary&) = delete;
    Vocabulary(Vocabulary&&) = default;
    Vocabulary& operator=(Vocabulary&&) = default;

    /**
     * @brief Виртуальный деструктор
     */
    virtual ~Vocabulary() = default;

    // ==================== Добавление токенов ====================

    /**
     * @brief Добавить новый токен в словарь
     * 
     * @param token Строка-токен для добавления
     * @return token_id_t ID назначенный токену
     */
    token_id_t add_token(const std::string& token);

    /**
     * @brief Добавить несколько специальных токенов
     * 
     * @param tokens Вектор строк для добавления
     */
    void add_special_tokens(const std::vector<std::string>& tokens);

    /**
     * @brief Добавить токен с перемещением (для эффективности)
     * 
     * @param token Строка-токен для добавления (будет перемещена)
     * @return token_id_t ID назначенный токену
     */
    token_id_t add_token(std::string&& token);

    // ==================== Поиск ====================

    /**
     * @brief Получить ID токена по строке
     * 
     * @param token Строка-токен для поиска
     * @return token_id_t ID токена или INVALID_TOKEN если не найден
     */
    token_id_t token_to_id(const std::string& token) const;

    /**
     * @brief Получить строку токена по ID
     * 
     * @param id ID токена
     * @return const std::string& Ссылка на строку токена
     * @throws std::out_of_range если ID не существует
     */
    const std::string& id_to_token(token_id_t id) const;

    /**
     * @brief Получить строку токена по ID (без проверки границ)
     * 
     * @param id ID токена
     * @return const std::string& Ссылка на строку токена
     * 
     * @warning Не проверяет границы! Используйте только если уверены в ID.
     */
    const std::string& id_to_token_unsafe(token_id_t id) const;

    // ==================== Проверки ====================

    /**
     * @brief Проверить наличие токена в словаре
     * 
     * @param token Строка для проверки
     * @return true если токен существует
     */
    bool contains(const std::string& token) const;

    /**
     * @brief Проверить наличие ID в словаре
     * 
     * @param id ID для проверки
     * @return true если ID существует
     */
    bool contains_id(token_id_t id) const;

    // ==================== Размер и управление памятью ====================

    /**
     * @brief Получить размер словаря
     * @return size_t Количество токенов в словаре
     */
    size_t size() const;

    /**
     * @brief Проверить, пуст ли словарь
     * @return true если словарь пуст
     */
    bool empty() const;

    /**
     * @brief Зарезервировать память под указанное количество токенов
     * @param capacity Желаемая ёмкость
     * 
     * Позволяет оптимизировать производительность при заранее известном
     * количестве токенов, избегая многократных переаллокаций.
     */
    void reserve(size_t capacity);

    // ==================== Очистка ====================

    /**
     * @brief Очистить словарь (удалить все токены)
     */
    void clear();

    // ==================== Сериализация ====================

    /**
     * @brief Преобразовать словарь в JSON
     * 
     * @return nlohmann::json JSON объект с токенами
     * 
     * **Формат:**    {"0": "токен0", "1": "токен1", ...}
     */
    nlohmann::json to_json() const;

    /**
     * @brief Загрузить словарь из JSON
     * 
     * @param j JSON объект с токенами
     * @throws std::runtime_error при ошибке парсинга
     */
    void from_json(const nlohmann::json& j);

    // ==================== Сохранение/загрузка ====================

    /**
     * @brief Сохранить словарь в текстовый JSON файл
     * 
     * @param path Путь к файлу
     * @return true при успешном сохранении
     */
    bool save(const std::string& path) const;

    /**
     * @brief Загрузить словарь из текстового JSON файла
     * 
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load(const std::string& path);

    /**
     * @brief Сохранить словарь в бинарный файл
     * 
     * @param path Путь к файлу
     * @return true при успешном сохранении
     */
    bool save_binary(const std::string& path) const;

    /**
     * @brief Загрузить словарь из бинарного файла
     * 
     * @param path Путь к файлу
     * @return true при успешной загрузке
     */
    bool load_binary(const std::string& path);

    // ==================== Вспомогательные методы ====================

    /**
     * @brief Получить все токены словаря
     * @return std::vector<std::string> Вектор всех токенов в порядке их ID
     */
    std::vector<std::string> get_all_tokens() const;

    /**
     * @brief Получить следующий свободный ID
     * @return token_id_t ID для нового токена
     */
    token_id_t next_id() const;

    /**
     * @brief Получить максимальный ID в словаре
     * @return token_id_t Максимальный ID
     */
    token_id_t max_id() const;

    /**
     * @brief Получить ссылку на внутренний вектор токенов
     * @return const std::vector<std::string>& Ссылка на вектор
     */
    const std::vector<std::string>& tokens() const;

    /**
     * @brief Получить ссылку на внутреннюю хеш-таблицу
     * @return const std::unordered_map<std::string, token_id_t>& Ссылка на map
     */
    const std::unordered_map<std::string, token_id_t>& mapping() const;

private:
    std::unordered_map<std::string, token_id_t> token_to_id_;    ///< Отображение токен -> ID (хеш-таблица)
    std::vector<std::string> id_to_token_;                       ///< Отображение ID -> токен (вектор)
};

} // namespace bpe

#endif // BPE_VOCABULARY_HPP

/**
 * @example examples/vocabulary_benchmark.cpp
 * Пример бенчмарка операций со словарём
 */

/**
 * @example examples/vocabulary_example.cpp
 * Полный пример использования Vocabulary
 */