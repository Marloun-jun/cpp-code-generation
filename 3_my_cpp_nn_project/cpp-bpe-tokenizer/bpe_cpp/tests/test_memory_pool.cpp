/**
 * @file test_memory_pool.cpp
 * @brief Модульные тесты для класса MemoryPool
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.1.0
 * 
 * @details Набор тестов для проверки функциональности пула памяти:
 *          - Базовое выделение и освобождение памяти
 *          - Выделение памяти больше размера блока
 *          - Повторное использование освобожденных блоков
 *          - Множественные выделения
 * 
 * @see MemoryPool
 */

#include <gtest/gtest.h>

#include "memory_pool.hpp"

#include <vector>

using namespace bpe;

// ======================================================================
// Тесты базовых операций
// ======================================================================

/**
 * @test Проверка базового выделения и освобождения памяти
 */
TEST(MemoryPoolTest, BasicAllocation) {
    MemoryPool<1024> pool;
    
    void* ptr1 = pool.allocate(100);
    void* ptr2 = pool.allocate(200);
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    
    pool.deallocate(ptr1, 100);
    pool.deallocate(ptr2, 200);
}

/**
 * @test Проверка повторного использования освобожденных блоков
 */
TEST(MemoryPoolTest, ReuseAfterDeallocation) {
    MemoryPool<1024> pool;
    
    void* ptr1 = pool.allocate(100);
    pool.deallocate(ptr1, 100);
    
    void* ptr2 = pool.allocate(100);
    
    // Должен переиспользовать тот же блок
    EXPECT_EQ(ptr1, ptr2);
    
    pool.deallocate(ptr2, 100);
}

/**
 * @test Проверка выделения памяти больше размера блока
 */
TEST(MemoryPoolTest, LargeAllocation) {
    MemoryPool<1024> pool;
    
    // Больше размера блока - должно уйти в обычную кучу
    void* ptr = pool.allocate(2048);
    EXPECT_NE(ptr, nullptr);
    
    pool.deallocate(ptr, 2048);
}

/**
 * @test Проверка множественных выделений
 */
TEST(MemoryPoolTest, MultipleAllocations) {
    MemoryPool<1024> pool;
    std::vector<void*> ptrs;
    
    // Выделяем много маленьких блоков
    const int NUM_ALLOCATIONS = 100;
    const int BLOCK_SIZE = 64;
    
    for (int i = 0; i < NUM_ALLOCATIONS; ++i) {
        void* ptr = pool.allocate(BLOCK_SIZE);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Проверяем, что все указатели уникальны
    for (size_t i = 0; i < ptrs.size(); ++i) {
        for (size_t j = i + 1; j < ptrs.size(); ++j) {
            EXPECT_NE(ptrs[i], ptrs[j]);
        }
    }
    
    // Освобождаем
    for (void* ptr : ptrs) {
        pool.deallocate(ptr, BLOCK_SIZE);
    }
}

/**
 * @test Проверка выделения памяти разного размера
 */
TEST(MemoryPoolTest, DifferentSizes) {
    MemoryPool<1024> pool;
    std::vector<std::pair<void*, size_t>> allocations;
    
    std::vector<size_t> sizes = {16, 32, 64, 128, 256, 512};
    
    // Выделяем блоки разного размера
    for (size_t size : sizes) {
        void* ptr = pool.allocate(size);
        EXPECT_NE(ptr, nullptr);
        allocations.emplace_back(ptr, size);
    }
    
    // Освобождаем в обратном порядке
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        pool.deallocate(it->first, it->second);
    }
}

/**
 * @test Проверка граничных значений
 */
TEST(MemoryPoolTest, EdgeCases) {
    MemoryPool<1024> pool;
    
    // Выделение нулевого размера
    void* ptr = pool.allocate(0);
    EXPECT_NE(ptr, nullptr);  // Должен вернуть валидный указатель
    
    // Освобождение nullptr
    pool.deallocate(nullptr, 0);  // Не должно упасть
    
    // Освобождение с неправильным размером (не проверяется)
    void* ptr2 = pool.allocate(100);
    pool.deallocate(ptr2, 200);  // Технически неправильно, но не должно упасть
}
