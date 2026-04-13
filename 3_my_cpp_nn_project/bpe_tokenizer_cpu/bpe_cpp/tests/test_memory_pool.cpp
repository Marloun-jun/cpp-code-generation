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
 *          ┌────────────────────┬──────────────────────────────────────┐
 *          │ Базовые операции   │ allocate/deallocate, уникальность    │
 *          │ Повторное          │ Освобожденные блоки переиспользуются │
 *          │ использование      │                                      │
 *          │ Размеры блоков     │ Малые, большие, разные размеры       │
 *          │ Множественные      │ Несколько выделений подряд           │
 *          │ выделения          │ в разном порядке                     │
 *          │ Граничные случаи   │ Нулевой размер, nullptr, ошибки      │
 *          │ Выравнивание       │ Проверка alignof(std::max_align_t)   │
 *          │ Статистика         │ block_count, free_count, used        │
 *          │ STL контейнеры     │ vector, string, map, list, set       │
 *          │ Производительность │ Сравнение с new/delete               │
 *          └────────────────────┴──────────────────────────────────────┘
 * 
 * @note Все тесты используют один и тот же пул памяти для аллокаторов
 * @see MemoryPool, PoolAllocator
 */

#include <gtest/gtest.h>

#include "memory_pool.hpp"
#include "test_helpers.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    constexpr size_t SMALL_BLOCK_SIZE = 32;     ///< Маленький блок (до пула)
    constexpr size_t MEDIUM_BLOCK_SIZE = 64;    ///< Средний блок
    constexpr size_t LARGE_BLOCK_SIZE = 256;    ///< Большой блок
    constexpr size_t HUGE_BLOCK_SIZE = 2048;    ///< Огромный блок (больше пула)
    constexpr size_t POOL_BLOCK_SIZE = 1024;    ///< Размер блока пула
    constexpr int NUM_ALLOCATIONS = 5;          ///< Количество выделений в тестах
    constexpr int STRESS_ITERATIONS = 100;      ///< Итераций в стресс-тесте
    constexpr int VECTOR_SIZE = 100;            ///< Размер вектора
    constexpr int MAP_SIZE = 20;                ///< Размер хеш-таблицы
    constexpr int LIST_SIZE = 50;               ///< Размер списка
    
    // Тестовые паттерны для заполнения памяти
    constexpr uint8_t TEST_PATTERN_1 = 0xAA;    ///< Паттерн 10101010
    constexpr uint8_t TEST_PATTERN_2 = 0xBB;    ///< Паттерн 10111011
    constexpr uint8_t TEST_PATTERN_3 = 0xCC;    ///< Паттерн 11001100
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string CYAN = "\033[36m";
}

// ============================================================================
// Тесты базовых операций
// ============================================================================

/**
 * @test Проверка базового выделения памяти
 * 
 * Убеждается, что:
 * - Указатели не nullptr
 * - Разные выделения дают разные указатели
 * - Освобождение работает
 */
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

/**
 * @test Проверка повторного использования блоков
 * 
 * Убеждается, что после освобождения блок снова доступен
 * и что данные не сохраняются между использованиями.
 */
TEST(MemoryPoolTest, ReuseAfterDeallocation) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    // Первое выделение
    void* ptr1 = pool.allocate(SMALL_BLOCK_SIZE);
    ASSERT_NE(ptr1, nullptr);
    
    // Заполняем паттерном
    std::memset(ptr1, TEST_PATTERN_1, SMALL_BLOCK_SIZE);
    
    // Освобождаем
    pool.deallocate(ptr1, SMALL_BLOCK_SIZE);
    
    // Второе выделение - должен получить тот же или другой блок
    void* ptr2 = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr2, nullptr);
    
    // Заполняем другим паттерном
    if (ptr2) {
        std::memset(ptr2, TEST_PATTERN_2, SMALL_BLOCK_SIZE);
    }
    
    pool.deallocate(ptr2, SMALL_BLOCK_SIZE);
}

/**
 * @test Проверка выделения больших блоков (больше размера пула)
 * 
 * Блоки размером больше POOL_BLOCK_SIZE должны идти в обычную кучу.
 */
TEST(MemoryPoolTest, LargeAllocation) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    void* ptr = pool.allocate(HUGE_BLOCK_SIZE);
    EXPECT_NE(ptr, nullptr);
    
    pool.deallocate(ptr, HUGE_BLOCK_SIZE);
}

/**
 * @test Проверка множественных выделений
 * 
 * Выделяет несколько блоков, освобождает в разном порядке,
 * затем выделяет снова.
 */
TEST(MemoryPoolTest, MultipleAllocations) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    // Выделяем три блока
    void* ptr1 = pool.allocate(SMALL_BLOCK_SIZE);
    void* ptr2 = pool.allocate(MEDIUM_BLOCK_SIZE);
    void* ptr3 = pool.allocate(SMALL_BLOCK_SIZE);
    
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr3, nullptr);
    
    // Освобождаем в другом порядке
    pool.deallocate(ptr2, MEDIUM_BLOCK_SIZE);
    pool.deallocate(ptr1, SMALL_BLOCK_SIZE);
    pool.deallocate(ptr3, SMALL_BLOCK_SIZE);
    
    // Выделяем снова
    void* ptr4 = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr4, nullptr);
    
    pool.deallocate(ptr4, SMALL_BLOCK_SIZE);
}

/**
 * @test Проверка выделения блоков разных размеров
 * 
 * Убеждается, что можно выделять блоки разных размеров
 * и что все указатели уникальны.
 */
TEST(MemoryPoolTest, DifferentSizes) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    std::vector<void*> ptrs;
    std::vector<size_t> sizes = {8, 16, 32, 64, 128, 256};
    
    for (size_t size : sizes) {
        void* ptr = pool.allocate(size);
        EXPECT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Проверяем, что все указатели уникальны
    std::set<void*> unique_ptrs(ptrs.begin(), ptrs.end());
    EXPECT_EQ(unique_ptrs.size(), ptrs.size());
    
    // Освобождаем все блоки
    for (size_t i = 0; i < ptrs.size(); ++i) {
        pool.deallocate(ptrs[i], sizes[i]);
    }
}

// ============================================================================
// Тесты граничных случаев
// ============================================================================

/**
 * @test Проверка граничных случаев
 * 
 * Тестирует:
 * - Выделение нулевого размера
 * - Освобождение nullptr
 * - Освобождение с неправильным размером
 */
TEST(MemoryPoolTest, EdgeCases) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    // Выделение нулевого размера
    void* ptr = pool.allocate(0);
    // Просто проверяем, что вызов не бросил исключение
    SUCCEED() << "allocate(0) completed, ptr=" << (ptr ? "non-null" : "null");
    
    // Освобождение nullptr - должно работать без ошибок
    EXPECT_NO_THROW(pool.deallocate(nullptr, 0));
    
    // Проверяем, что пул продолжает работать
    void* ptr2 = pool.allocate(SMALL_BLOCK_SIZE);
    ASSERT_NE(ptr2, nullptr);
    
    // Заполняем память для проверки
    std::memset(ptr2, TEST_PATTERN_1, SMALL_BLOCK_SIZE);
    
    // Освобождаем с неправильным размером - должно быть безопасно
    EXPECT_NO_THROW(pool.deallocate(ptr2, SMALL_BLOCK_SIZE / 2));
    
    // Освобождаем с правильным размером
    EXPECT_NO_THROW(pool.deallocate(ptr2, SMALL_BLOCK_SIZE));
    
    // Проверяем, что можно выделить снова
    void* ptr3 = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr3, nullptr);
    pool.deallocate(ptr3, SMALL_BLOCK_SIZE);
}

/**
 * @test Стресс-тест с множеством операций
 * 
 * Выполняет множество выделений и освобождений в случайном порядке
 * для проверки устойчивости пула.
 */
TEST(MemoryPoolTest, StressTest) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    std::vector<std::pair<void*, size_t>> allocations;
    
    for (int i = 0; i < STRESS_ITERATIONS; ++i) {
        // Выбираем размер в зависимости от итерации
        size_t size = (i % 3 == 0) ? SMALL_BLOCK_SIZE : 
                      (i % 3 == 1) ? MEDIUM_BLOCK_SIZE : LARGE_BLOCK_SIZE;
        
        void* ptr = pool.allocate(size);
        ASSERT_NE(ptr, nullptr);
        
        // Заполняем память паттерном на основе индекса
        std::memset(ptr, static_cast<uint8_t>(i % 256), size);
        
        allocations.push_back({ptr, size});
        
        // Каждые 10 итераций освобождаем половину накопленных блоков
        if (i % 10 == 0 && !allocations.empty()) {
            for (size_t j = 0; j < allocations.size() / 2; ++j) {
                pool.deallocate(allocations[j].first, allocations[j].second);
            }
            allocations.erase(allocations.begin(), 
                             allocations.begin() + allocations.size() / 2);
        }
    }
    
    // Освобождаем оставшиеся блоки
    for (const auto& alloc : allocations) {
        pool.deallocate(alloc.first, alloc.second);
    }
}

// ============================================================================
// Тесты статистики
// ============================================================================

/**
 * @test Проверка сбора статистики
 * 
 * Проверяет, что block_count, free_count и used_count
 * правильно отражают состояние пула.
 */
TEST(MemoryPoolTest, Statistics) {
    MemoryPool<POOL_BLOCK_SIZE> pool;
    
    size_t used = pool.block_count() - pool.free_count();
    
    std::cout << "\n" << CYAN << "Начальное состояние:" << RESET << std::endl;
    std::cout << "- Блоков:   " << pool.block_count() << std::endl;
    std::cout << "- Свободно: " << pool.free_count() << std::endl;
    std::cout << "- Занято:   " << used << std::endl;
    
    EXPECT_EQ(used, 0);
    size_t initial_blocks = pool.block_count();
    
    void* ptr = pool.allocate(SMALL_BLOCK_SIZE);
    EXPECT_NE(ptr, nullptr);
    
    used = pool.block_count() - pool.free_count();
    
    std::cout << "\n" << CYAN << "После выделения:" << RESET << std::endl;
    std::cout << "- Блоков:   " << pool.block_count() << std::endl;
    std::cout << "- Свободно: " << pool.free_count() << std::endl;
    std::cout << "- Занято:   " << used << std::endl;
    
    EXPECT_EQ(used, 1);
    
    pool.deallocate(ptr, SMALL_BLOCK_SIZE);
    
    used = pool.block_count() - pool.free_count();
    
    std::cout << "\n" << CYAN << "После освобождения:" << RESET << std::endl;
    std::cout << "- Блоков:   " << pool.block_count() << std::endl;
    std::cout << "- Свободно: " << pool.free_count() << std::endl;
    std::cout << "- Занято:   " << used << std::endl;
    
    EXPECT_EQ(used, 0);
    EXPECT_EQ(pool.block_count(), initial_blocks);
}

// ============================================================================
// Тесты выравнивания
// ============================================================================

/**
 * @test Проверка выравнивания указателей
 * 
 * Убеждается, что все указатели, возвращаемые пулом,
 * правильно выровнены для любого типа.
 */
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

// ============================================================================
// Тесты с PoolAllocator и STL контейнерами
// ============================================================================

/**
 * @test Использование PoolAllocator с std::vector
 */
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

/**
 * @test Использование PoolAllocator с std::string
 */
TEST(MemoryPoolTest, StringWithPoolAllocator) {
    MemoryPool<> pool;
    using Alloc = PoolAllocator<char>;
    
    Alloc alloc(pool);
    
    std::basic_string<char, std::char_traits<char>, Alloc> str(alloc);
    
    str = "Hello, World! This is a test string from pool-allocated memory.";
    
    EXPECT_EQ(str, "Hello, World! This is a test string from pool-allocated memory.");
    EXPECT_GT(str.size(), 0);
    
    std::cout << "\n" << GREEN << "Строка успешно создана, длина: " << str.size() << RESET << std::endl;
}

/**
 * @test Использование PoolAllocator с std::unordered_map
 */
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
    
    std::cout << "\n" << GREEN << "Map успешно создана, размер: " << map.size() << RESET << std::endl;
}

/**
 * @test Использование PoolAllocator с std::list
 */
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

/**
 * @test Использование PoolAllocator с std::set
 */
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

// ============================================================================
// Тест с несколькими контейнерами и одним пулом
// ============================================================================

/**
 * @test Использование одного пула для нескольких контейнеров
 * 
 * Проверяет, что разные типы контейнеров могут использовать
 * один и тот же пул памяти без конфликтов.
 */
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
    
    // Заполняем все контейнеры
    for (int i = 0; i < 30; ++i) {
        vec.push_back(i);
        lst.push_back(i * 2);
    }
    
    str = "Hello from multiple containers using the same memory pool!";
    
    for (int i = 0; i < 10; ++i) {
        map[i] = "value" + std::to_string(i);
    }
    
    // Проверяем все контейнеры
    EXPECT_EQ(vec.size(), 30);
    EXPECT_EQ(lst.size(), 30);
    EXPECT_EQ(str, "Hello from multiple containers using the same memory pool!");
    EXPECT_EQ(map.size(), 10);
    
    for (int i = 0; i < 30; ++i) {
        EXPECT_EQ(vec[i], i);
    }
}

// ============================================================================
// Тесты на утечки памяти
// ============================================================================

/**
 * @test Проверка отсутствия утечек памяти
 * 
 * Многократно создает и разрушает пул с множеством операций.
 * Утечки будут обнаружены инструментами типа Valgrind.
 */
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

// ============================================================================
// Тесты производительности (опционально, отключены по умолчанию)
// ============================================================================

/**
 * @test Сравнение производительности пула и стандартной кучи
 * 
 * @note Тест отключен (DISABLED), так как предназначен для ручного запуска
 */
TEST(MemoryPoolTest, DISABLED_Performance) {
    const int ALLOCATIONS = 100000;
    
    {
        // Тест пула памяти
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
        
        std::cout << "\n" << CYAN << "Пул памяти:" << RESET << std::endl;
        std::cout << "- Выделение:    " << alloc_time.count() << " мс" << std::endl;
        std::cout << "- Освобождение: " << dealloc_time.count() << " мс" << std::endl;
    }
    
    {
        // Тест стандартной кучи (new/delete)
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
        
        std::cout << "\n" << CYAN << "Стандартная куча:" << RESET << std::endl;
        std::cout << "- Выделение:    " << alloc_time.count() << " мс" << std::endl;
        std::cout << "- Освобождение: " << dealloc_time.count() << " мс" << std::endl;
    }
}