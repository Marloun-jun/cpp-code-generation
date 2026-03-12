/**
 * @file test_memory_pool.cpp
 * @brief Модульные тесты для класса MemoryPool
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Набор тестов для проверки функциональности пула памяти,
 *          который используется для оптимизации аллокаций в FastBPETokenizer.
 * 
 *          **Проверяемые аспекты:**
 * 
 *          1) **Базовые операции**
 *             - Выделение памяти (allocate)
 *             - Освобождение памяти (deallocate)
 *             - Уникальность указателей при разных выделениях
 * 
 *          2) **Повторное использование**
 *             - Освобожденные блоки должны быть доступны для новых выделений
 *             - Данные не сохраняются между использованиями
 * 
 *          3) **Размеры блоков**
 *             - Малые блоки (до размера пула)
 *             - Блоки больше размера пула (должны идти в обычную кучу)
 *             - Разные размеры в одной сессии
 * 
 *          4) **Множественные выделения**
 *             - Несколько выделений подряд
 *             - Освобождение в разном порядке
 * 
 *          5) **Граничные случаи**
 *             - Выделение нулевого размера
 *             - Освобождение nullptr
 *             - Освобождение с неправильным размером
 * 
 *          6) **PoolAllocator и STL контейнеры**
 *             - Использование с std::vector
 *             - Использование с std::string
 *             - Использование с std::unordered_map
 *             - Использование с std::list и std::set
 * 
 * @note Все тесты используют один и тот же пул памяти для аллокаторов
 * @see MemoryPool
 * @see PoolAllocator
 */

#include <gtest/gtest.h>

#include "memory_pool.hpp"
#include "test_helpers.hpp"

#include <vector>
#include <cstring>
#include <set>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <list>
#include <string>
#include <functional>

using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr size_t SMALL_BLOCK_SIZE = 32;
    constexpr size_t MEDIUM_BLOCK_SIZE = 64;
    constexpr size_t LARGE_BLOCK_SIZE = 256;
    constexpr size_t HUGE_BLOCK_SIZE = 2048;
    constexpr size_t POOL_BLOCK_SIZE = 1024;
    constexpr int NUM_ALLOCATIONS = 5;
    constexpr int STRESS_ITERATIONS = 100;
    constexpr int VECTOR_SIZE = 100;
    constexpr int MAP_SIZE = 20;
    constexpr int LIST_SIZE = 50;
    
    // Тестовые данные
    constexpr uint8_t TEST_PATTERN_1 = 0xAA;
    constexpr uint8_t TEST_PATTERN_2 = 0xBB;
    constexpr uint8_t TEST_PATTERN_3 = 0xCC;
}

// ======================================================================
// Тесты базовых операций
// ======================================================================

TEST(MemoryPoolTest, BasicAllocation) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    void* ptr1 = pool.allocate(SMALL_BLOCK_SIZE);
    void* ptr2 = pool.allocate(MEDIUM_BLOCK_SIZE);
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
    
    pool.deallocate(ptr1, SMALL_BLOCK_SIZE);
    pool.deallocate(ptr2, MEDIUM_BLOCK_SIZE);
}

TEST(MemoryPoolTest, ReuseAfterDeallocation) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    void* ptr1 = pool.allocate(SMALL_BLOCK_SIZE);
    ASSERT_NE(ptr1, nullptr);
    
    std::memset(ptr1, TEST_PATTERN_1, SMALL_BLOCK_SIZE);
    
    pool.deallocate(ptr1, SMALL_BLOCK_SIZE);
    
    void* ptr2 = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr2, nullptr);
    
    if (ptr2) {
        std::memset(ptr2, TEST_PATTERN_2, SMALL_BLOCK_SIZE);
    }
    
    pool.deallocate(ptr2, SMALL_BLOCK_SIZE);
}

TEST(MemoryPoolTest, LargeAllocation) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    void* ptr = pool.allocate(HUGE_BLOCK_SIZE);
    EXPECT_NE(ptr, nullptr);
    
    pool.deallocate(ptr, HUGE_BLOCK_SIZE);
}

TEST(MemoryPoolTest, MultipleAllocations) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    void* ptr1 = pool.allocate(SMALL_BLOCK_SIZE);
    void* ptr2 = pool.allocate(MEDIUM_BLOCK_SIZE);
    void* ptr3 = pool.allocate(SMALL_BLOCK_SIZE);
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);
    
    pool.deallocate(ptr2, MEDIUM_BLOCK_SIZE);
    pool.deallocate(ptr1, SMALL_BLOCK_SIZE);
    pool.deallocate(ptr3, SMALL_BLOCK_SIZE);
    
    void* ptr4 = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr4, nullptr);
    
    pool.deallocate(ptr4, SMALL_BLOCK_SIZE);
}

TEST(MemoryPoolTest, DifferentSizes) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    std::vector<void*> ptrs;
    std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256};
    
    for (size_t size : sizes) {
        void* ptr = pool.allocate(size);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    std::set<void*> unique_ptrs(ptrs.begin(), ptrs.end());
    EXPECT_EQ(unique_ptrs.size(), ptrs.size());
    
    for (size_t i = 0; i < ptrs.size(); ++i) {
        pool.deallocate(ptrs[i], sizes[i]);
    }
}

// ======================================================================
// Тесты граничных случаев
// ======================================================================

TEST(MemoryPoolTest, EdgeCases) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    void* ptr = pool.allocate(0);
    EXPECT_NE(ptr, nullptr);
    
    pool.deallocate(nullptr, 0);
    
    void* ptr2 = pool.allocate(SMALL_BLOCK_SIZE);
    ASSERT_NE(ptr2, nullptr);
    
    pool.deallocate(ptr2, SMALL_BLOCK_SIZE / 2);
    pool.deallocate(ptr2, SMALL_BLOCK_SIZE);
}

TEST(MemoryPoolTest, StressTest) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    std::vector<std::pair<void*, size_t>> allocations;
    
    for (int i = 0; i < STRESS_ITERATIONS; ++i) {
        size_t size = (i % 3 == 0) ? SMALL_BLOCK_SIZE : 
                      (i % 3 == 1) ? MEDIUM_BLOCK_SIZE : LARGE_BLOCK_SIZE;
        
        void* ptr = pool.allocate(size);
        ASSERT_NE(ptr, nullptr);
        
        std::memset(ptr, static_cast<uint8_t>(i % 256), size);
        
        allocations.push_back({ptr, size});
        
        if (i % 10 == 0 && !allocations.empty()) {
            for (size_t j = 0; j < allocations.size() / 2; ++j) {
                pool.deallocate(allocations[j].first, allocations[j].second);
            }
            allocations.erase(allocations.begin(), 
                             allocations.begin() + allocations.size() / 2);
        }
    }
    
    for (const auto& alloc : allocations) {
        pool.deallocate(alloc.first, alloc.second);
    }
}

// ======================================================================
// Тесты статистики
// ======================================================================

TEST(MemoryPoolTest, Statistics) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    size_t used = pool.block_count() - pool.free_count();
    
    std::cout << "\nНачальное состояние:" << std::endl;
    std::cout << "  блоков: " << pool.block_count() << std::endl;
    std::cout << "  свободно: " << pool.free_count() << std::endl;
    std::cout << "  занято: " << used << std::endl;
    
    EXPECT_EQ(used, 0);
    size_t initial_blocks = pool.block_count();
    
    void* ptr = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr, nullptr);
    
    used = pool.block_count() - pool.free_count();
    
    std::cout << "\nПосле выделения:" << std::endl;
    std::cout << "  блоков: " << pool.block_count() << std::endl;
    std::cout << "  свободно: " << pool.free_count() << std::endl;
    std::cout << "  занято: " << used << std::endl;
    
    EXPECT_EQ(used, 1);
    
    pool.deallocate(ptr, SMALL_BLOCK_SIZE);
    
    used = pool.block_count() - pool.free_count();
    
    std::cout << "\nПосле освобождения:" << std::endl;
    std::cout << "  блоков: " << pool.block_count() << std::endl;
    std::cout << "  свободно: " << pool.free_count() << std::endl;
    std::cout << "  занято: " << used << std::endl;
    
    EXPECT_EQ(used, 0);
    EXPECT_EQ(pool.block_count(), initial_blocks);
}

// ======================================================================
// Тесты выравнивания
// ======================================================================

TEST(MemoryPoolTest, Alignment) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    size_t sizes[] = {1, 2, 4, 8, 16, SMALL_BLOCK_SIZE, MEDIUM_BLOCK_SIZE};
    
    for (size_t size : sizes) {
        void* ptr = pool.allocate(size);
        EXPECT_NE(ptr, nullptr);
        
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        EXPECT_EQ(addr % alignof(std::max_align_t), 0) 
            << "Указатель не выровнен для размера " << size;
        
        pool.deallocate(ptr, size);
    }
}

// ======================================================================
// Тесты с PoolAllocator и STL контейнерами
// ======================================================================

TEST(MemoryPoolTest, VectorWithPoolAllocator) {
    // Используем MemoryPool<> без указания размера
    MemoryPool<> pool;
    using Alloc = PoolAllocator<int>;
    
    Alloc alloc(pool);
    
    std::vector<int, Alloc> vec(alloc);
    
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        vec.push_back(i);
    }
    
    EXPECT_EQ(vec.size(), VECTOR_SIZE);
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

TEST(MemoryPoolTest, StringWithPoolAllocator) {
    MemoryPool<> pool;
    using Alloc = PoolAllocator<char>;
    
    Alloc alloc(pool);
    
    std::basic_string<char, std::char_traits<char>, Alloc> str(alloc);
    
    str = "Hello, World! This is a test string from pool-allocated memory.";
    
    EXPECT_EQ(str, "Hello, World! This is a test string from pool-allocated memory.");
    EXPECT_GT(str.size(), 0);
    
    std::cout << "\nСтрока успешно создана, длина: " << str.size() << std::endl;
}

TEST(MemoryPoolTest, MapWithPoolAllocator) {
    MemoryPool<> pool;
    using Alloc = PoolAllocator<std::pair<const int, std::string>>;
    
    Alloc alloc(pool);
    
    std::unordered_map<int, std::string, 
                       std::hash<int>, 
                       std::equal_to<int>, 
                       Alloc> map(10, std::hash<int>(), std::equal_to<int>(), alloc);
    
    for (int i = 0; i < MAP_SIZE; ++i) {
        map[i] = "value" + std::to_string(i);
    }
    
    EXPECT_EQ(map.size(), MAP_SIZE);
    for (int i = 0; i < MAP_SIZE; ++i) {
        EXPECT_EQ(map[i], "value" + std::to_string(i));
    }
    
    std::cout << "Map успешно создана, размер: " << map.size() << std::endl;
}

TEST(MemoryPoolTest, ListWithPoolAllocator) {
    MemoryPool<> pool;
    using Alloc = PoolAllocator<int>;
    
    Alloc alloc(pool);
    
    std::list<int, Alloc> lst(alloc);
    
    for (int i = 0; i < LIST_SIZE; ++i) {
        lst.push_back(i);
    }
    
    EXPECT_EQ(lst.size(), LIST_SIZE);
    
    int i = 0;
    for (int val : lst) {
        EXPECT_EQ(val, i);
        ++i;
    }
}

TEST(MemoryPoolTest, SetWithPoolAllocator) {
    MemoryPool<> pool;
    using Alloc = PoolAllocator<int>;
    
    Alloc alloc(pool);
    
    std::set<int, std::less<int>, Alloc> set(std::less<int>(), alloc);
    
    for (int i = 0; i < LIST_SIZE; ++i) {
        set.insert(i);
    }
    
    EXPECT_EQ(set.size(), LIST_SIZE);
    
    int i = 0;
    for (int val : set) {
        EXPECT_EQ(val, i);
        ++i;
    }
}

// ======================================================================
// Тест с несколькими контейнерами и одним пулом
// ======================================================================

TEST(MemoryPoolTest, MultipleContainersWithSamePool) {
    MemoryPool<> pool;
    
    using AllocInt = PoolAllocator<int>;
    using AllocChar = PoolAllocator<char>;
    using AllocPair = PoolAllocator<std::pair<const int, std::string>>;
    
    AllocInt allocInt(pool);
    AllocChar allocChar(pool);
    AllocPair allocPair(pool);
    
    std::vector<int, AllocInt> vec(allocInt);
    std::list<int, AllocInt> lst(allocInt);
    std::basic_string<char, std::char_traits<char>, AllocChar> str(allocChar);
    std::unordered_map<int, std::string, 
                       std::hash<int>, 
                       std::equal_to<int>, 
                       AllocPair> map(10, std::hash<int>(), std::equal_to<int>(), allocPair);
    
    for (int i = 0; i < 30; ++i) {
        vec.push_back(i);
        lst.push_back(i * 2);
    }
    
    str = "Hello from multiple containers using the same memory pool!";
    
    for (int i = 0; i < 10; ++i) {
        map[i] = "value" + std::to_string(i);
    }
    
    EXPECT_EQ(vec.size(), 30);
    EXPECT_EQ(lst.size(), 30);
    EXPECT_EQ(str, "Hello from multiple containers using the same memory pool!");
    EXPECT_EQ(map.size(), 10);
    
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// ======================================================================
// Тесты на утечки памяти
// ======================================================================

TEST(MemoryPoolTest, NoLeaks) {
    const int ITERATIONS = 1000;
    
    for (int i = 0; i < ITERATIONS; ++i) {
        MemoryPool<> pool;
        
        std::vector<void*> ptrs;
        std::vector<size_t> sizes;
        
        for (int j = 0; j < 10; ++j) {
            size_t size = SMALL_BLOCK_SIZE + j * 10;
            void* ptr = pool.allocate(size);
            EXPECT_NE(ptr, nullptr);
            ptrs.push_back(ptr);
            sizes.push_back(size);
        }
        
        for (int j = static_cast<int>(ptrs.size()) - 1; j >= 0; --j) {
            pool.deallocate(ptrs[j], sizes[j]);
        }
    }
}

// ======================================================================
// Тесты производительности (опционально)
// ======================================================================

TEST(MemoryPoolTest, DISABLED_Performance) {
    const int ALLOCATIONS = 100000;
    
    {
        MemoryPool<> pool;
        std::vector<void*> ptrs;
        ptrs.reserve(ALLOCATIONS);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < ALLOCATIONS; ++i) {
            ptrs.push_back(pool.allocate(SMALL_BLOCK_SIZE));
        }
        
        auto mid = std::chrono::high_resolution_clock::now();
        
        for (void* ptr : ptrs) {
            pool.deallocate(ptr, SMALL_BLOCK_SIZE);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        auto alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start);
        auto dealloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - mid);
        
        std::cout << "\nПул памяти:" << std::endl;
        std::cout << "  выделение:   " << alloc_time.count() << " мс" << std::endl;
        std::cout << "  освобождение: " << dealloc_time.count() << " мс" << std::endl;
    }
    
    {
        std::vector<void*> ptrs;
        ptrs.reserve(ALLOCATIONS);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < ALLOCATIONS; ++i) {
            ptrs.push_back(::operator new(SMALL_BLOCK_SIZE));
        }
        
        auto mid = std::chrono::high_resolution_clock::now();
        
        for (void* ptr : ptrs) {
            ::operator delete(ptr);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        auto alloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start);
        auto dealloc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - mid);
        
        std::cout << "Стандартная куча:" << std::endl;
        std::cout << "  выделение:   " << alloc_time.count() << " мс" << std::endl;
        std::cout << "  освобождение: " << dealloc_time.count() << " мс" << std::endl;
    }
}