/**
 * @file thread_safe_cache.hpp
 * @brief Потокобезопасные реализации кэша для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Две реализации кэша с разными стратегиями:
 * 
 *          1. ThreadSafeLRUCache - обобщенный LRU кэш с политикой
 *             "наименее недавно использованный" (Least Recently Used)
 *          
 *          2. StringViewCache - специализированный кэш для строк
 *             с оптимизацией под string_view и сбором статистики
 * 
 * @note Обе реализации потокобезопасны через shared_mutex
 * @warning StringViewCache хранит копии строк (std::string) как ключи
 * 
 * @see FastBPETokenizer
 */

#pragma once

#include <algorithm>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace bpe {

/**
 * @brief Потокобезопасный LRU кэш с политикой наименее недавнего использования
 * 
 * @tparam Key Тип ключа (должен быть хешируемым)
 * @tparam Value Тип значения
 * 
 * Реализует алгоритм LRU (Least Recently Used):
 * - При переполнении удаляется элемент, к которому дольше всего не было обращений
 * - Каждое обращение (get/put) перемещает элемент в начало списка
 * - Использует shared_mutex для конкурентного чтения
 * 
 * @note Идеально подходит для кэширования результатов encode()
 */
template<typename Key, typename Value>
class ThreadSafeLRUCache {
private:
    /**
     * @brief Внутренняя структура записи в кэше
     */
    struct CacheEntry {
        Value value;                                ///< Хранимое значение
        typename std::list<Key>::iterator lru_iterator;  ///< Итератор на позицию в LRU списке
    };
    
    std::unordered_map<Key, CacheEntry> cache_;    ///< Основное хранилище
    std::list<Key> lru_list_;                       ///< Список для LRU порядка
    mutable std::shared_mutex mutex_;                ///< Мьютекс для потокобезопасности
    size_t capacity_;                                ///< Максимальный размер кэша

public:
    /**
     * @brief Конструктор с указанием вместимости
     * @param capacity Максимальное количество элементов
     */
    explicit ThreadSafeLRUCache(size_t capacity) : capacity_(capacity) {}

    /**
     * @brief Получить значение по ключу
     * @param key Ключ для поиска
     * @param out [out] Ссылка для сохранения результата
     * @return true если ключ найден, false иначе
     * 
     * @note При успешном поиске элемент перемещается в начало LRU списка
     */
    bool get(const Key& key, Value& out) {
        std::shared_lock lock(mutex_);
        
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return false;
        }
        
        // Для обновления LRU позиции нужна уникальная блокировка
        lock.unlock();
        std::unique_lock unique_lock(mutex_);
        
        // Перемещаем элемент в начало списка (самый свежий)
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
        
        out = it->second.value;
        return true;
    }

    /**
     * @brief Поместить значение в кэш
     * @param key Ключ
     * @param value Значение для хранения
     * 
     * @details
     * - Если ключ существует - обновляет значение и перемещает в начало
     * - Если ключ новый и есть место - добавляет
     * - Если ключ новый и кэш полон - удаляет самый старый элемент
     */
    void put(const Key& key, const Value& value) {
        std::unique_lock lock(mutex_);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Обновляем существующий элемент
            it->second.value = value;
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
            return;
        }
        
        // Добавляем новый элемент
        lru_list_.push_front(key);
        cache_[key] = {value, lru_list_.begin()};
        
        // Проверяем, не превышен ли лимит
        if (cache_.size() > capacity_) {
            // Удаляем самый старый элемент (с конца списка)
            auto last = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(last);
        }
    }

    /**
     * @brief Очистить кэш
     */
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        lru_list_.clear();
    }

    /**
     * @brief Получить текущий размер кэша
     * @return Количество элементов в кэше
     */
    size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }
};

/**
 * @brief Специализированный кэш для строк с оптимизацией под string_view
 * 
 * Особенности:
 * - Хранит std::string как ключи для стабильности
 * - Принимает string_view для интерфейса (без копирования при поиске)
 * - Поддерживает сбор статистики попаданий
 * - Использует политику "наиболее старая запись" при переполнении
 * 
 * @note Создан специально для кэширования результатов tokenize_word()
 */
class StringViewCache {
private:
    /**
     * @brief Структура записи в кэше
     */
    struct Entry {
        std::vector<uint32_t> tokens;  ///< Закэшированные токены
        size_t last_access;             ///< Время последнего доступа (монотонный счетчик)
    };
    
    std::unordered_map<std::string, Entry> cache_;  ///< Хранилище (ключ - скопированная строка)
    mutable std::shared_mutex mutex_;                ///< Мьютекс для потокобезопасности
    size_t capacity_;                                ///< Максимальный размер
    size_t access_counter_{0};                        ///< Глобальный счетчик доступа
    size_t hits_{0};                                  ///< Количество попаданий
    size_t misses_{0};                                ///< Количество промахов

public:
    /**
     * @brief Конструктор с указанием вместимости
     * @param capacity Максимальное количество элементов
     */
    explicit StringViewCache(size_t capacity) : capacity_(capacity) {}

    /**
     * @brief Получить значение по ключу
     * @param key Ключ (string_view, не копируется)
     * @param out [out] Ссылка для сохранения результата
     * @return true если ключ найден
     * 
     * @note При поиске ключ преобразуется в std::string для поиска в unordered_map
     */
    bool get(std::string_view key, std::vector<uint32_t>& out) {
        std::shared_lock lock(mutex_);
        
        auto it = cache_.find(std::string(key));
        if (it == cache_.end()) {
            ++misses_;
            return false;
        }
        
        ++hits_;
        out = it->second.tokens;
        it->second.last_access = ++access_counter_;
        return true;
    }

    /**
     * @brief Поместить значение в кэш
     * @param key Ключ (string_view)
     * @param value Вектор токенов для кэширования
     * 
     * @details При переполнении удаляется самая старая запись
     *          (с наименьшим last_access)
     */
    void put(std::string_view key, const std::vector<uint32_t>& value) {
        std::unique_lock lock(mutex_);
        
        // Проверяем, нужно ли удалять старые записи
        if (cache_.size() >= capacity_ && cache_.find(std::string(key)) == cache_.end()) {
            // Находим самую старую запись
            auto oldest = std::min_element(
                cache_.begin(), cache_.end(),
                [](const auto& a, const auto& b) {
                    return a.second.last_access < b.second.last_access;
                });
            
            if (oldest != cache_.end()) {
                cache_.erase(oldest);
            }
        }
        
        // Добавляем новую запись
        cache_[std::string(key)] = {value, ++access_counter_};
    }

    /**
     * @brief Очистить кэш
     */
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        access_counter_ = 0;
        hits_ = 0;
        misses_ = 0;
    }

    /**
     * @brief Получить текущий размер кэша
     * @return Количество элементов
     */
    size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

    /**
     * @brief Получить процент попаданий в кэш
     * @return Значение от 0.0 до 1.0
     */
    double hit_rate() const {
        std::shared_lock lock(mutex_);
        size_t total = hits_ + misses_;
        return total ? static_cast<double>(hits_) / total : 0.0;
    }

    /**
     * @brief Получить количество попаданий
     */
    size_t hits() const {
        std::shared_lock lock(mutex_);
        return hits_;
    }

    /**
     * @brief Получить количество промахов
     */
    size_t misses() const {
        std::shared_lock lock(mutex_);
        return misses_;
    }

    /**
     * @brief Сбросить статистику
     */
    void reset_stats() {
        std::unique_lock lock(mutex_);
        hits_ = 0;
        misses_ = 0;
    }
};

} // namespace bpe

/**
 * @example examples/cache_example.cpp
 * Пример использования кэша:
 * @code
 * #include "thread_safe_cache.hpp"
 * 
 * // LRU кэш для любых типов
 * bpe::ThreadSafeLRUCache<int, std::string> lru_cache(100);
 * std::string value;
 * 
 * lru_cache.put(42, "answer");
 * if (lru_cache.get(42, value)) {
 *     std::cout << "Найдено: " << value << std::endl;
 * }
 * 
 * // Специализированный кэш для строк
 * bpe::StringViewCache str_cache(1000);
 * std::vector<uint32_t> tokens;
 * 
 * str_cache.put("hello", {1, 2, 3});
 * if (str_cache.get("hello", tokens)) {
 *     std::cout << "Попаданий в кэш: " << str_cache.hit_rate() * 100 << "%" << std::endl;
 * }
 * @endcode
 */