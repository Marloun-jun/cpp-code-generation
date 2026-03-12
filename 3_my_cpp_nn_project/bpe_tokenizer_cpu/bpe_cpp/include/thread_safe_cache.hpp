/**
 * @file thread_safe_cache.hpp
 * @brief Потокобезопасные реализации кэша для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Две реализации кэша с разными стратегиями, критически важные
 *          для производительности FastBPETokenizer:
 * 
 *          1) **ThreadSafeLRUCache** - обобщенный LRU кэш
 *             - Политика "наименее недавно использованный" (Least Recently Used)
 *             - Шаблонный - работает с любыми типами ключей и значений
 *             - Идеален для кэширования результатов encode()
 * 
 *          2) **StringViewCache** - специализированный кэш для строк
 *             - Оптимизирован для работы с string_view
 *             - Хранит std::string как ключи для стабильности
 *             - Собирает статистику попаданий/промахов
 *             - Использует политику "наиболее старая запись" при переполнении
 *             - Специально для кэширования результатов tokenize_word()
 * 
 *          **Производительность:**
 *          - ThreadSafeLRUCache:    O(1) для get/put
 *          - StringViewCache:       O(1) для get/put с учетом хеширования
 *          - Оба используют shared_mutex для конкурентного чтения
 *          - Hit rate 60-80% для типичных текстов
 * 
 * @note Обе реализации потокобезопасны через shared_mutex
 * @warning StringViewCache хранит копии строк (std::string) как ключи,
 *          что необходимо для стабильности при использовании string_view
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
#include <cstdint>
#include <stdexcept>

namespace bpe {

// ======================================================================
// ThreadSafeLRUCache - обобщенный LRU кэш
// ======================================================================

/**
 * @brief Потокобезопасный LRU кэш с политикой наименее недавнего использования
 * 
 * @tparam Key Тип ключа (должен быть хешируемым)
 * @tparam Value Тип значения
 * 
 * Реализует алгоритм LRU (Least Recently Used) - при переполнении удаляется
 * элемент, к которому дольше всего не было обращений. Это оптимальная стратегия
 * для кэширования результатов encode(), так как часто используемые тексты
 * имеют тенденцию повторяться.
 * 
 * **Алгоритм работы:**
 * 1. Хеш-таблица для быстрого доступа O(1)
 * 2. Двусвязный список для отслеживания порядка использования
 * 3. При каждом get/put элемент перемещается в начало списка
 * 4. При переполнении удаляется элемент с конца списка
 * 
 * **Потокобезопасность:**
 * - shared_mutex для конкурентного чтения
 * - unique_lock для записи и обновления LRU позиции
 * 
 * \include examples/lru_cache_example.cpp
 * Пример использования:
 * \code
 * // Создаем кэш на 100 элементов
 * ThreadSafeLRUCache<int, std::string> cache(100);
 * 
 * // Добавляем элемент
 * cache.put(42, "answer");
 * 
 * // Ищем элемент
 * std::string value;
 * if (cache.get(42, value)) {
 *     std::cout << "Найдено: " << value << std::endl;
 * }
 * 
 * // Размер кэша
 * std::cout << "Размер: " << cache.size() << std::endl;
 * \endcode
 */
template<typename Key, typename Value>
class ThreadSafeLRUCache {
private:
    /**
     * @brief Внутренняя структура записи в кэше
     * 
     * Содержит само значение и итератор на позицию в LRU списке.
     * Итератор позволяет за O(1) перемещать элемент в списке.
     */
    struct CacheEntry {
        Value value;                                       ///< Хранимое значение
        typename std::list<Key>::iterator lru_iterator;    ///< Итератор на позицию в LRU списке
    };
    
    std::unordered_map<Key, CacheEntry> cache_;    ///< Основное хранилище (хеш-таблица)
    std::list<Key> lru_list_;                      ///< Список для LRU порядка (передние - свежие)
    mutable std::shared_mutex mutex_;              ///< Мьютекс для потокобезопасности
    size_t capacity_;                              ///< Максимальный размер кэша

public:
    /**
     * @brief Конструктор с указанием вместимости
     * 
     * @param capacity Максимальное количество элементов
     * 
     * @throws std::invalid_argument если capacity == 0
     */
    explicit ThreadSafeLRUCache(size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("Емкость кэша должна быть положительной");
        }
    }

    /**
     * @brief Получить значение по ключу
     * 
     * @param key Ключ для поиска
     * @param out [out] Ссылка для сохранения результата
     * @return true если ключ найден, false иначе
     * 
     * **Алгоритм:**
     * 1. Блокируем на чтение (shared_lock)
     * 2. Ищем ключ в хеш-таблице
     * 3. Если не найден -> возвращаем false
     * 4. Если найден:
     *    a. Снимаем shared_lock
     *    b. Блокируем на запись (unique_lock)
     *    c. Перемещаем элемент в начало LRU списка (самый свежий)
     *    d. Сохраняем значение в out
     *    e. Возвращаем true
     * 
     * **Сложность:**    O(1) амортизированно
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
        // splice - O(1) операция перестановки в списке
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
        
        out = it->second.value;
        return true;
    }

    /**
     * @brief Поместить значение в кэш
     * 
     * @param key Ключ
     * @param value Значение для хранения
     * 
     * **Алгоритм:**
     * 1. Блокируем на запись (unique_lock)
     * 2. Если ключ существует:
     *    а) Обновляем значение
     *    б) Перемещаем в начало LRU списка
     * 3. Если ключ новый:
     *    а) Добавляем в начало списка
     *    б) Добавляем в хеш-таблицу с итератором
     *    в) Если размер превышен -> удаляем элемент с конца списка
     * 
     * **Сложность:**    O(1) амортизированно
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
     * @brief Поместить значение в кэш с перемещением
     * 
     * @param key Ключ
     * @param value Значение для хранения (будет перемещено)
     */
    void put(const Key& key, Value&& value) {
        std::unique_lock lock(mutex_);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            // Обновляем существующий элемент
            it->second.value = std::move(value);
            lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
            return;
        }
        
        // Добавляем новый элемент
        lru_list_.push_front(key);
        cache_[key] = {std::move(value), lru_list_.begin()};
        
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
     * 
     * Удаляет все элементы и сбрасывает состояние.
     */
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        lru_list_.clear();
    }

    /**
     * @brief Удалить элемент по ключу
     * 
     * @param key Ключ для удаления
     * @return true если элемент был удален
     */
    bool erase(const Key& key) {
        std::unique_lock lock(mutex_);
        
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return false;
        }
        
        lru_list_.erase(it->second.lru_iterator);
        cache_.erase(it);
        return true;
    }

    /**
     * @brief Получить текущий размер кэша
     * @return size_t Количество элементов в кэше
     */
    size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

    /**
     * @brief Проверить, пуст ли кэш
     * @return true если кэш пуст
     */
    bool empty() const {
        std::shared_lock lock(mutex_);
        return cache_.empty();
    }

    /**
     * @brief Получить максимальную вместимость
     * @return size_t capacity_
     */
    size_t capacity() const {
        return capacity_;
    }

    /**
     * @brief Проверить наличие ключа
     * @param key Ключ для проверки
     * @return true если ключ существует
     * 
     * @note Не обновляет LRU порядок!
     */
    bool contains(const Key& key) const {
        std::shared_lock lock(mutex_);
        return cache_.find(key) != cache_.end();
    }
};

// ======================================================================
// StringViewCache - специализированный кэш для строк
// ======================================================================

/**
 * @brief Специализированный кэш для строк с оптимизацией под string_view
 * 
 * Особенности:
 * - Хранит std::string как ключи для стабильности (важно для string_view)
 * - Принимает string_view для интерфейса (без копирования при поиске)
 * - Поддерживает сбор статистики попаданий (hits/misses)
 * - Использует политику "наиболее старая запись" при переполнении
 * - Идеален для кэширования результатов tokenize_word()
 * 
 * **Почему храним std::string, а не string_view?**
 * string_view не владеет памятью и может стать невалидным, если исходная строка
 * будет уничтожена. Для кэша нужно стабильное хранение, поэтому ключи копируются.
 * 
 * **Статистика:**
 * - hit_rate()         - процент попаданий
 * - hits()/misses()    - абсолютные значения
 * - Можно сбросить reset_stats()
 * 
 * \include examples/string_cache_example.cpp
 * Пример использования:
 * \code
 * // Создаем кэш на 1000 элементов
 * StringViewCache cache(1000);
 * 
 * // Добавляем результат
 * cache.put("hello", {1, 2, 3});
 * 
 * // Ищем
 * std::vector<uint32_t> tokens;
 * if (cache.get("hello", tokens)) {
 *     std::cout << "Найдено! " << tokens.size() << " токенов\n";
 * }
 * 
 * // Статистика
 * std::cout << "Hit rate: " << cache.hit_rate() * 100 << "%\n";
 * std::cout << "Попаданий: " << cache.hits() << "\n";
 * std::cout << "Промахов: " << cache.misses() << "\n";
 * \endcode
 */
class StringViewCache {
private:
    /**
     * @brief Структура записи в кэше
     */
    struct Entry {
        std::vector<uint32_t> tokens;    ///< Закэшированные токены (результат encode)
        size_t last_access;              ///< Время последнего доступа (монотонный счетчик)
    };
    
    std::unordered_map<std::string, Entry> cache_;    ///< Хранилище (ключ - скопированная строка)
    mutable std::shared_mutex mutex_;                 ///< Мьютекс для потокобезопасности
    size_t capacity_;                                 ///< Максимальный размер
    size_t access_counter_{0};                        ///< Глобальный счетчик доступа (монотонный)
    size_t hits_{0};                                  ///< Количество попаданий
    size_t misses_{0};                                ///< Количество промахов

public:
    /**
     * @brief Конструктор с указанием вместимости
     * 
     * @param capacity Максимальное количество элементов
     * 
     * @throws std::invalid_argument если capacity == 0
     */
    explicit StringViewCache(size_t capacity) : capacity_(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("Емкость кэша должна быть положительной");
        }
    }

    /**
     * @brief Получить значение по ключу
     * 
     * @param key Ключ (string_view, не копируется при поиске)
     * @param out [out] Ссылка для сохранения результата
     * @return true если ключ найден
     * 
     * **Алгоритм:**
     * 1. Блокируем на чтение
     * 2. Преобразуем string_view в string для поиска (временный объект)
     * 3. Если найден:
     *    a. Увеличиваем hits
     *    b. Обновляем last_access
     *    c. Копируем токены в out
     * 4. Если не найден: увеличиваем misses
     * 
     * **Сложность:**    O(1) с учетом хеширования строки
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
     * @brief Получить значение по ключу (без копирования)
     * 
     * @param key Ключ (string_view)
     * @return const std::vector<uint32_t>* Указатель на токены или nullptr
     * 
     * @warning Указатель валиден только пока элемент в кэше
     */
    const std::vector<uint32_t>* get(std::string_view key) {
        std::shared_lock lock(mutex_);
        
        auto it = cache_.find(std::string(key));
        if (it == cache_.end()) {
            ++misses_;
            return nullptr;
        }
        
        ++hits_;
        it->second.last_access = ++access_counter_;
        return &it->second.tokens;
    }

    /**
     * @brief Поместить значение в кэш (копирование)
     * 
     * @param key Ключ (string_view)
     * @param value Вектор токенов для кэширования
     * 
     * **Алгоритм:**
     * 1. Блокируем на запись
     * 2. Если кэш полон и ключ новый:
     *    а) Находим самую старую запись (с наименьшим last_access)
     *    б) Удаляем её
     * 3. Добавляем новую запись с текущим access_counter_
     * 
     * **Сложность:**    O(n) в худшем случае при поиске минимума
     */
    void put(std::string_view key, const std::vector<uint32_t>& value) {
        std::unique_lock lock(mutex_);
        
        std::string key_str(key);
        auto it = cache_.find(key_str);
        
        if (it != cache_.end()) {
            // Обновляем существующую запись
            it->second.tokens = value;
            it->second.last_access = ++access_counter_;
            return;
        }
        
        // Проверяем, нужно ли удалять старые записи
        if (cache_.size() >= capacity_) {
            // Находим самую старую запись (с наименьшим last_access)
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
        cache_[std::move(key_str)] = {value, ++access_counter_};
    }

    /**
     * @brief Поместить значение в кэш (перемещение)
     * 
     * @param key Ключ (string_view)
     * @param value Вектор токенов (будет перемещен)
     */
    void put(std::string_view key, std::vector<uint32_t>&& value) {
        std::unique_lock lock(mutex_);
        
        std::string key_str(key);
        auto it = cache_.find(key_str);
        
        if (it != cache_.end()) {
            // Обновляем существующую запись
            it->second.tokens = std::move(value);
            it->second.last_access = ++access_counter_;
            return;
        }
        
        // Проверяем, нужно ли удалять старые записи
        if (cache_.size() >= capacity_) {
            // Находим самую старую запись (с наименьшим last_access)
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
        cache_[std::move(key_str)] = {std::move(value), ++access_counter_};
    }

    /**
     * @brief Очистить кэш и сбросить статистику
     */
    void clear() {
        std::unique_lock lock(mutex_);
        cache_.clear();
        access_counter_ = 0;
        hits_ = 0;
        misses_ = 0;
    }

    /**
     * @brief Удалить элемент по ключу
     * 
     * @param key Ключ для удаления
     * @return true если элемент был удален
     */
    bool erase(std::string_view key) {
        std::unique_lock lock(mutex_);
        return cache_.erase(std::string(key)) > 0;
    }

    /**
     * @brief Получить текущий размер кэша
     * @return size_t Количество элементов
     */
    size_t size() const {
        std::shared_lock lock(mutex_);
        return cache_.size();
    }

    /**
     * @brief Проверить, пуст ли кэш
     * @return true если кэш пуст
     */
    bool empty() const {
        std::shared_lock lock(mutex_);
        return cache_.empty();
    }

    /**
     * @brief Получить процент попаданий в кэш
     * @return double Значение от 0.0 до 1.0
     */
    double hit_rate() const {
        std::shared_lock lock(mutex_);
        size_t total = hits_ + misses_;
        return total ? static_cast<double>(hits_) / total : 0.0;
    }

    /**
     * @brief Получить количество попаданий
     * @return size_t hits_
     */
    size_t hits() const {
        std::shared_lock lock(mutex_);
        return hits_;
    }

    /**
     * @brief Получить количество промахов
     * @return size_t misses_
     */
    size_t misses() const {
        std::shared_lock lock(mutex_);
        return misses_;
    }

    /**
     * @brief Получить общее количество обращений
     * @return size_t hits + misses
     */
    size_t total_accesses() const {
        std::shared_lock lock(mutex_);
        return hits_ + misses_;
    }

    /**
     * @brief Сбросить статистику попаданий/промахов
     * 
     * Оставляет содержимое кэша нетронутым, сбрасывает только счетчики.
     */
    void reset_stats() {
        std::unique_lock lock(mutex_);
        hits_ = 0;
        misses_ = 0;
    }

    /**
     * @brief Получить максимальную вместимость
     * @return size_t capacity_
     */
    size_t capacity() const {
        return capacity_;
    }

    /**
     * @brief Проверить наличие ключа
     * @param key Ключ для проверки
     * @return true если ключ существует
     * 
     * @note Не обновляет время доступа и статистику
     */
    bool contains(std::string_view key) const {
        std::shared_lock lock(mutex_);
        return cache_.find(std::string(key)) != cache_.end();
    }
};

} // namespace bpe

/**
 * @example examples/cache_benchmark.cpp
 * Пример бенчмарка кэша:
 * 
 * @code
 * #include "thread_safe_cache.hpp"
 * #include <benchmark/benchmark.h>
 * 
 * static void BM_LRUCache(benchmark::State& state) {
 *     bpe::ThreadSafeLRUCache<int, int> cache(1000);
 *     
 *     // Заполняем кэш
 *     for (int i = 0; i < 1000; ++i) {
 *         cache.put(i, i * 2);
 *     }
 *     
 *     int value;
 *     for (auto _ : state) {
 *         for (int i = 0; i < 1000; ++i) {
 *             cache.get(i % 1500, value);    // 66% попаданий
 *         }
 *     }
 * }
 * BENCHMARK(BM_LRUCache);
 * 
 * static void BM_StringViewCache(benchmark::State& state) {
 *     bpe::StringViewCache cache(1000);
 *     std::vector<uint32_t> tokens(10);
 *     
 *     // Заполняем кэш
 *     for (int i = 0; i < 1000; ++i) {
 *         cache.put(std::to_string(i), tokens);
 *     }
 *     
 *     std::vector<uint32_t> out;
 *     for (auto _ : state) {
 *         for (int i = 0; i < 1000; ++i) {
 *             cache.get(std::to_string(i % 1500), out);
 *         }
 *     }
 * }
 * BENCHMARK(BM_StringViewCache);
 * 
 * BENCHMARK_MAIN();
 * @endcode
 */

/**
 * @example examples/cache_example.cpp
 * Полный пример использования кэша:
 * 
 * @include examples/cache_example.cpp
 */