/**
 * @file memory_pool.hpp
 * @brief Пул памяти для оптимизации аллокаций в BPE токенизаторе
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Реализация пула памяти для быстрых аллокаций малых объектов.
 *          Основные особенности:
 *          - Фиксированный размер блока (шаблонный параметр BlockSize)
 *          - Объекты > BlockSize направляются в обычную кучу
 *          - Потокобезопасность через std::mutex
 *          - Поддержка перемещения, запрет копирования
 *          - Интеграция с STL через PoolAllocator
 * 
 * @note Идеально подходит для частых аллокаций небольших строк в токенизаторе
 * @warning Не вызывает деструкторы объектов! Только для POD типов
 * 
 * @see FastBPETokenizer
 * @see PoolAllocator
 */

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace bpe {

/**
 * @brief Пул памяти с фиксированным размером блока
 * 
 * @tparam BlockSize Размер блока памяти (по умолчанию 4096 байт)
 * 
 * Реализует стратегию выделения памяти:
 * 1. Предварительно выделяет блоки фиксированного размера
 * 2. Объекты, помещающиеся в блок, выделяются из пула (быстро)
 * 3. Крупные объекты (> BlockSize) идут в обычную кучу
 * 4. Освобожденные блоки возвращаются в свободный список
 */
template<size_t BlockSize = 4096>
class MemoryPool {
private:
    /**
     * @brief Внутренняя структура блока памяти
     * 
     * Использует технику "embedded free list":
     * - Пока блок свободен, data используется как указатель на следующий свободный блок
     * - Когда блок выделен, data используется для хранения объекта
     */
    struct Block {
        Block* next;                    ///< Указатель на следующий свободный блок
        char data[BlockSize - sizeof(Block*)];  ///< Полезные данные
    };
    
    Block* free_list_{nullptr};          ///< Голова списка свободных блоков
    std::vector<Block*> blocks_;          ///< Все выделенные блоки (для очистки)
    std::mutex mutex_;                    ///< Мьютекс для потокобезопасности

public:
    /**
     * @brief Конструктор - выделяет первый блок
     */
    MemoryPool() {
        allocate_new_block();
    }
    
    /**
     * @brief Деструктор - освобождает все выделенные блоки
     */
    ~MemoryPool() {
        for (auto block : blocks_) {
            ::operator delete(block);
        }
    }
    
    // Запрещаем копирование (RAII для уникальных ресурсов)
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Разрешаем перемещение
    MemoryPool(MemoryPool&& other) noexcept
        : free_list_(other.free_list_)
        , blocks_(std::move(other.blocks_)) {
        other.free_list_ = nullptr;
    }
    
    /**
     * @brief Выделить память
     * @param size Требуемый размер в байтах
     * @return Указатель на выделенную память
     * 
     * @note Для size <= BlockSize использует пул, иначе обычный new
     */
    void* allocate(size_t size) {
        if (size > BlockSize) {
            // Крупные объекты идут в обычную кучу
            return ::operator new(size);
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!free_list_) {
            allocate_new_block();
        }
        
        Block* block = free_list_;
        free_list_ = block->next;
        return block->data;
    }
    
    /**
     * @brief Освободить память
     * @param ptr Указатель на память
     * @param size Размер объекта (должен совпадать с запрошенным при allocate)
     * 
     * @note Возвращает блок в свободный список для повторного использования
     */
    void deallocate(void* ptr, size_t size) {
        if (size > BlockSize) {
            // Крупные объекты освобождаем обычным способом
            ::operator delete(ptr);
            return;
        }
        
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Возвращаем блок в свободный список
        Block* block = static_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
    }

private:
    /**
     * @brief Выделить новый блок и добавить его в свободный список
     */
    void allocate_new_block() {
        Block* block = static_cast<Block*>(::operator new(sizeof(Block)));
        blocks_.push_back(block);
        
        // Добавляем блок в начало свободного списка
        block->next = free_list_;
        free_list_ = block;
    }
};

/**
 * @brief STL-совместимый аллокатор, использующий MemoryPool
 * 
 * @tparam T Тип объектов для аллокации
 * 
 * Позволяет использовать MemoryPool с контейнерами STL:
 * @code
 * MemoryPool<> pool;
 * std::vector<int, PoolAllocator<int>> vec(pool);
 * @endcode
 */
template<typename T>
class PoolAllocator {
private:
    MemoryPool<>& pool_;  ///< Ссылка на пул памяти

public:
    using value_type = T;

    /**
     * @brief Конструктор от пула
     * @param pool Ссылка на MemoryPool
     */
    explicit PoolAllocator(MemoryPool<>& pool) noexcept : pool_(pool) {}

    /**
     * @brief Конструктор копирования для rebind
     * @tparam U Другой тип
     */
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) noexcept 
        : pool_(other.pool()) {}

    /**
     * @brief Выделить память для n объектов
     * @param n Количество объектов
     * @return Указатель на память
     * 
     * @note Для n == 1 использует пул, иначе обычный new
     */
    T* allocate(size_t n) {
        if (n > 1) {
            // Массивы идут в обычную кучу
            return static_cast<T*>(::operator new(n * sizeof(T)));
        }
        return static_cast<T*>(pool_.allocate(sizeof(T)));
    }

    /**
     * @brief Освободить память
     * @param ptr Указатель на память
     * @param n Количество объектов (должно совпадать с allocate)
     */
    void deallocate(T* ptr, size_t n) noexcept {
        if (n > 1) {
            ::operator delete(ptr);
        } else {
            pool_.deallocate(ptr, sizeof(T));
        }
    }

    /**
     * @brief Получить ссылку на пул
     * @return Константная ссылка на MemoryPool
     */
    MemoryPool<>& pool() const { return pool_; }

    // ==================== Операторы сравнения ====================

    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return &pool_ == &other.pool();
    }

    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }
};

} // namespace bpe

/**
 * @example examples/pool_example.cpp
 * Пример использования MemoryPool:
 * @code
 * #include "memory_pool.hpp"
 * 
 * // Создаем пул с блоками 1024 байта
 * bpe::MemoryPool<1024> pool;
 * 
 * // Выделяем память
 * void* ptr1 = pool.allocate(100);
 * void* ptr2 = pool.allocate(500);
 * 
 * // Освобождаем
 * pool.deallocate(ptr1, 100);
 * pool.deallocate(ptr2, 500);
 * 
 * // Использование с STL
 * std::vector<int, bpe::PoolAllocator<int>> vec{pool};
 * vec.push_back(42);
 * @endcode
 */