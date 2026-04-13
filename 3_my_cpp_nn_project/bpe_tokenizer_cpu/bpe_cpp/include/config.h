#ifndef BPE_TOKENIZER_CONFIG_H
#define BPE_TOKENIZER_CONFIG_H

// ======================================================================
// Информация о версии
// ======================================================================

// Полная версия проекта в формате "major.minor.patch"
#define PROJECT_VERSION "3.0.0"

// Мажорная версия проекта (число)
#define PROJECT_VERSION_MAJOR 3

// Минорная версия проекта (число)
#define PROJECT_VERSION_MINOR 0

// Патч-версия проекта (число)
#define PROJECT_VERSION_PATCH 0

// ======================================================================
// Опции параллелизации
// ======================================================================

// Использовать OpenMP для параллельных циклов
#define USE_OPENMP

// Использовать Intel Threading Building Blocks
/* #undef USE_TBB */

// Максимальное количество потоков по умолчанию
#define DEFAULT_NUM_THREADS 4

// ======================================================================
// SIMD оптимизации
// ======================================================================

// Поддержка AVX2 инструкций
/* #undef USE_AVX2 */

// Поддержка AVX инструкций
/* #undef USE_AVX */

// Поддержка SSE4.2 инструкций
/* #undef USE_SSE42 */

// Поддержка AVX-512 инструкций
/* #undef USE_AVX512 */

// ======================================================================
// Оптимизации памяти
// ======================================================================

// Использовать пулы памяти для частых аллокаций
/* #undef USE_MEMORY_POOL */

// Размер пула памяти по умолчанию (в байтах)
#define DEFAULT_POOL_SIZE 1048576

// Максимальный размер кэша токенов
#define DEFAULT_CACHE_SIZE 10000

// Использовать выравнивание для кэш-линий
/* #undef USE_CACHE_ALIGNMENT */

// ======================================================================
// Инструменты разработчика
// ======================================================================

// Включить встроенный профайлер
/* #undef BUILD_WITH_PROFILING */

// Режим отладки
/* #undef DEBUG_MODE */

// Сборка с санитайзерами
/* #undef USE_SANITIZERS */

// Включить детальное логирование
/* #undef ENABLE_LOGGING */

// ======================================================================
// Совместимость с платформами
// ======================================================================

// Определение операционной системы
#if defined(_WIN32) || defined(_WIN64)
    #define OS_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
    #define OS_MACOS 1
#elif defined(__linux__)
    #define OS_LINUX 1
#elif defined(__unix__)
    #define OS_UNIX 1
#endif

// Поддержка больших файлов (64-битные смещения)
/* #undef _FILE_OFFSET_BITS */

// Определение для POSIX систем
#if defined(OS_LINUX) || defined(OS_MACOS) || defined(OS_UNIX)
    #define OS_POSIX 1
#endif

// ======================================================================
// Версии для числовых сравнений
// ======================================================================

// Версия в числовом формате (major*10000 + minor*100 + patch)
#define PROJECT_VERSION_NUM 30000

// ======================================================================
// Проверки совместимости
// ======================================================================

// Проверка, что версия удовлетворяет минимальным требованиям
#if !defined(PROJECT_VERSION_NUM) || PROJECT_VERSION_NUM < 30000
    #error "Требуется версия проекта >= 3.0.0"
#endif

// ======================================================================
// Экспорт символов для DLL (Windows)
// ======================================================================

#ifdef _WIN32
    #ifdef BPE_TOKENIZER_EXPORTS
        #define BPE_API __declspec(dllexport)
    #else
        #define BPE_API __declspec(dllimport)
    #endif
#else
    #define BPE_API
#endif

// ======================================================================
// Версии компилятора
// ======================================================================

#if defined(__clang__)
    #define COMPILER_CLANG
    #define COMPILER_VERSION __clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__
#elif defined(__GNUC__) || defined(__GNUG__)
    #define COMPILER_GCC
    #define COMPILER_VERSION __GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__
#elif defined(_MSC_VER)
    #define COMPILER_MSVC
    #define COMPILER_VERSION _MSC_VER
#endif

// ======================================================================
// Проверки доступности SIMD
// ======================================================================

// Проверить, доступна ли какая-либо SIMD оптимизация
#if defined(USE_AVX512) || defined(USE_AVX2) || defined(USE_AVX) || defined(USE_SSE42)
    #define HAS_SIMD 1
#else
    #define HAS_SIMD 0
#endif

// ======================================================================
// Значения по умолчанию для неопределенных макросов
// ======================================================================

#ifndef DEFAULT_NUM_THREADS
    #define DEFAULT_NUM_THREADS 1
#endif

#ifndef DEFAULT_POOL_SIZE
    #define DEFAULT_POOL_SIZE 1048576    // 1 МБ
#endif

#ifndef DEFAULT_CACHE_SIZE
    #define DEFAULT_CACHE_SIZE 10000
#endif

#endif    // BPE_TOKENIZER_CONFIG_H
