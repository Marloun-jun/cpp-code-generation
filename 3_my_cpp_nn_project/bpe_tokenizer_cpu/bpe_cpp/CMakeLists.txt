# ======================================================================
# @file CMakeLists.txt
# @brief Основной файл конфигурации CMake для сборки BPE токенизатора
# 
# @author Евгений П.
# @date 2026
# @version 3.5.0
# 
# @details Этот файл является центральным элементом системы сборки проекта.
#          Он управляет всем процессом компиляции, линковки и установки
#          всех компонентов BPE токенизатора.
# 
#          **Управляемые компоненты:**
#          1) **Основная библиотека** (bpe_tokenizer)
#             - Базовая реализация BPE
#             - Оптимизированная версия с SIMD/OpenMP
#             - Утилиты и вспомогательные классы
# 
#          2) **Тестирование**
#             - Модульные тесты (Google Test)
#             - Бенчмарки производительности (Google Benchmark)
#             - Интеграционные тесты
# 
#          3) **Примеры использования**
#             - Простой пример (simple_example.cpp)
#             - Пакетная обработка (batch_example.cpp)
#             - Демо оптимизированной версии (fast_tokenizer_demo.cpp)
#             - Обучение токенизатора (train_example.cpp)
# 
#          4) **Python интеграция**
#             - Python биндинги через pybind11
#             - Интеграция с PyTorch
# 
#          5) **Веб-сервер**
#             - Простой сервер на сокетах
#             - REST API сервер на CrowCpp
# 
#          **Опции конфигурации:**
#          - BUILD_PYTHON_BINDINGS - сборка Python модуля
#          - BUILD_TESTING - сборка тестов
#          - BUILD_BENCHMARKS - сборка бенчмарков
#          - BUILD_EXAMPLES - сборка примеров
#          - BUILD_WEB_SERVER - сборка веб-сервера
# 
#          - USE_OPENMP - использование OpenMP
#          - USE_TBB - использование Intel TBB
#          - USE_SIMD - использование SIMD инструкций
#          - USE_CACHE - использование кэширования
# 
#          - BUILD_WITH_PROFILING - поддержка профилирования
#          - BUILD_WITH_ASAN - AddressSanitizer
#          - BUILD_WITH_UBSAN - UndefinedBehaviorSanitizer
# 
#          - USE_LOCAL_JSON - использовать локальный json.hpp (быстрее)
# 
# @note Требует CMake 3.16 или выше
# @see README.md для инструкций по сборке
# ======================================================================

cmake_minimum_required(VERSION 3.16)
project(CPPBpeTokenizer 
    VERSION 3.5.0 
    LANGUAGES CXX
    DESCRIPTION "Высокопроизводительный BPE токенизатор для C++"
    HOMEPAGE_URL "https://github.com/yourusername/bpe_tokenizer_cpu"
)

# ======================================================================
# Определение переменных для config.h (ВАЖНО: после project())
# ======================================================================

# Разбираем версию на компоненты
string(REPLACE "." ";" VERSION_LIST ${PROJECT_VERSION})
list(GET VERSION_LIST 0 PROJECT_VERSION_MAJOR)
list(GET VERSION_LIST 1 PROJECT_VERSION_MINOR)
list(GET VERSION_LIST 2 PROJECT_VERSION_PATCH)

# Числовая версия для сравнений (major*10000 + minor*100 + patch)
math(EXPR PROJECT_VERSION_NUM "${PROJECT_VERSION_MAJOR}*10000 + ${PROJECT_VERSION_MINOR}*100 + ${PROJECT_VERSION_PATCH}")

# Значения по умолчанию для опций
set(DEFAULT_NUM_THREADS "4" CACHE STRING "Количество потоков по умолчанию")
set(DEFAULT_POOL_SIZE "1048576" CACHE STRING "Размер пула памяти по умолчанию (1 МБ)")
set(DEFAULT_CACHE_SIZE "10000" CACHE STRING "Размер кэша по умолчанию (количество записей)")

# ======================================================================
# Политики CMake
# ======================================================================
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)  # Политика для FetchContent
endif()

# ======================================================================
# Основные настройки компиляции
# ======================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======================================================================
# Проверка компилятора
# ======================================================================
message(STATUS "🔧 Компилятор: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Проверяем минимальную версию компилятора
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0")
        message(FATAL_ERROR "❌ GCC 7.0 или выше требуется для C++17")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.0")
        message(FATAL_ERROR "❌ Clang 5.0 или выше требуется для C++17")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "❌ AppleClang 10.0 или выше требуется для C++17")
    endif()
endif()

# ======================================================================
# CCache для ускорения повторных сборок
# ======================================================================
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    message(STATUS "✅ CCache найден: повторные сборки будут быстрее")
endif()

# ======================================================================
# Опции сборки
# ======================================================================
option(BUILD_PYTHON_BINDINGS "Собрать Python биндинги" ON)
option(BUILD_TESTING "Собрать тесты" ON)
option(BUILD_BENCHMARKS "Собрать бенчмарки" ON)
option(BUILD_EXAMPLES "Собрать примеры" ON)
option(BUILD_WEB_SERVER "Собрать веб-сервер" OFF)

# Оптимизации производительности
option(USE_OPENMP "Использовать OpenMP для параллельного обучения" ON)
option(USE_TBB "Использовать Intel TBB для параллельных алгоритмов" OFF)
option(USE_SIMD "Использовать SIMD оптимизации" ON)
option(USE_CACHE "Использовать кэширование результатов" ON)

# Инструменты разработчика
option(BUILD_WITH_PROFILING "Собрать с поддержкой профилирования" OFF)
option(BUILD_WITH_ASAN "Собрать с AddressSanitizer" OFF)
option(BUILD_WITH_UBSAN "Собрать с UndefinedBehaviorSanitizer" OFF)

# Опция для локального JSON (для быстрой сборки в России)
option(USE_LOCAL_JSON "Использовать локальный json.hpp (быстрее)" OFF)

# ======================================================================
# Пути
# ======================================================================
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTORCC OFF)
set(CMAKE_AUTOUIC OFF)

# ======================================================================
# Режим сборки
# ======================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Тип сборки" FORCE)
endif()

# ======================================================================
# Флаги компиляции по режимам
# ======================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "📦 Режим сборки: Release (максимальная производительность)")
    
    # Базовые оптимизации
    add_compile_options(-O3 -DNDEBUG)
    
    # Архитектурные оптимизации
    if(USE_SIMD)
        add_compile_options(-march=native -mtune=native)
        message(STATUS "   ✅ Архитектурные оптимизации включены (-march=native)")
    endif()
    
    # LTO и оптимизации времени линковки
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED)
    if(IPO_SUPPORTED)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "   ✅ Link Time Optimization включен")
    endif()
    
    # Дополнительные оптимизации
    add_compile_options(
        -funroll-loops        # Развертывание циклов
        -fomit-frame-pointer  # Убрать frame pointer
    )
    message(STATUS "   ✅ Дополнительные оптимизации циклов включены")
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "🔧 Режим сборки: Debug (отладка)")
    add_compile_options(-g -O0 -DDEBUG)
    
    # Санитайзеры для отладки
    if(BUILD_WITH_ASAN)
        add_compile_options(-fsanitize=address)
        add_link_options(-fsanitize=address)
        message(STATUS "   ✅ AddressSanitizer включен")
    endif()
    
    if(BUILD_WITH_UBSAN)
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
        message(STATUS "   ✅ UndefinedBehaviorSanitizer включен")
    endif()
    
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "📊 Режим сборки: RelWithDebInfo (оптимизация с отладочной информацией)")
    add_compile_options(-O2 -g -DNDEBUG)
endif()

# Профилирование
if(BUILD_WITH_PROFILING)
    add_compile_options(-pg -fno-omit-frame-pointer)
    add_link_options(-pg)
    message(STATUS "📈 Поддержка профилирования включена (-pg)")
endif()

# ======================================================================
# Поиск и подключение зависимостей
# ======================================================================
include(FetchContent)

# Поиск Threads (pthread)
find_package(Threads REQUIRED)

# OpenMP для параллельных вычислений
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "✅ OpenMP найден: версия ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "⚠️ OpenMP не найден, параллельное обучение будет отключено")
        set(USE_OPENMP OFF)
    endif()
endif()

# ----------------------------------------------------------------------
# nlohmann/json - с поддержкой локальной копии
# ----------------------------------------------------------------------

# Путь к локальной папке с зависимостями (на уровень выше bpe_cpp)
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/../third_party" CACHE PATH "Путь к папке с третьесторонними библиотеками")

if(USE_LOCAL_JSON)
    message(STATUS "📦 Использую локальный json.hpp из: ${THIRD_PARTY_DIR}")
    
    # Проверяем разные варианты расположения
    if(EXISTS "${THIRD_PARTY_DIR}/nlohmann/json.hpp")
        # Структура: third_party/nlohmann/json.hpp
        set(JSON_INCLUDE_DIR "${THIRD_PARTY_DIR}")
        message(STATUS "   ✅ Найден: ${THIRD_PARTY_DIR}/nlohmann/json.hpp")
        
    elseif(EXISTS "${THIRD_PARTY_DIR}/json.hpp")
        # Структура: third_party/json.hpp (один файл)
        set(JSON_INCLUDE_DIR "${THIRD_PARTY_DIR}")
        message(STATUS "   ✅ Найден: ${THIRD_PARTY_DIR}/json.hpp")
        
        # Создаём симлинк или папку для совместимости с #include <nlohmann/json.hpp>
        if(NOT EXISTS "${THIRD_PARTY_DIR}/nlohmann")
            file(MAKE_DIRECTORY "${THIRD_PARTY_DIR}/nlohmann")
            file(CREATE_LINK "${THIRD_PARTY_DIR}/json.hpp" "${THIRD_PARTY_DIR}/nlohmann/json.hpp" SYMBOLIC)
        endif()
        
    elseif(EXISTS "${THIRD_PARTY_DIR}/json/single_include/nlohmann/json.hpp")
        # Полный репозиторий
        set(JSON_INCLUDE_DIR "${THIRD_PARTY_DIR}/json/single_include")
        message(STATUS "   ✅ Найден репозиторий: ${THIRD_PARTY_DIR}/json/single_include")
        
    else()
        message(WARNING "❌ Локальный json.hpp не найден в ${THIRD_PARTY_DIR}")
        message(STATUS "   Переключаюсь на загрузку из сети...")
        set(USE_LOCAL_JSON OFF)
    endif()
    
    if(USE_LOCAL_JSON AND DEFINED JSON_INCLUDE_DIR)
        # Создаём интерфейсную библиотеку для локального JSON
        add_library(nlohmann_json INTERFACE)
        target_include_directories(nlohmann_json INTERFACE "${JSON_INCLUDE_DIR}")
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
        message(STATUS "   ✅ Локальный JSON настроен")
    endif()
endif()

if(NOT USE_LOCAL_JSON OR NOT DEFINED JSON_INCLUDE_DIR)
    message(STATUS "📦 Загрузка nlohmann/json из сети (shallow clone)...")
    
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE           # Только последний коммит
        GIT_PROGRESS TRUE          # Показываем прогресс
    )
    
    FetchContent_MakeAvailable(nlohmann_json)
    message(STATUS "   ✅ nlohmann/json загружен из сети")
endif()

# ----------------------------------------------------------------------
# fmt
# ----------------------------------------------------------------------
message(STATUS "📦 Загрузка fmt...")
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.1.1
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(fmt)
message(STATUS "   ✅ fmt загружен")

# ----------------------------------------------------------------------
# Google Test
# ----------------------------------------------------------------------
if(BUILD_TESTING)
    message(STATUS "📦 Загрузка Google Test...")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
    message(STATUS "   ✅ Google Test загружен")
endif()

# ----------------------------------------------------------------------
# Google Benchmark
# ----------------------------------------------------------------------
if(BUILD_BENCHMARKS)
    message(STATUS "📦 Загрузка Google Benchmark...")
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googlebenchmark)
    message(STATUS "   ✅ Google Benchmark загружен")
endif()

# ----------------------------------------------------------------------
# Intel TBB
# ----------------------------------------------------------------------
if(USE_TBB)
    message(STATUS "🔍 Поиск Intel TBB...")
    find_package(TBB QUIET)
    if(TBB_FOUND)
        message(STATUS "   ✅ TBB найден в системе")
    else()
        message(STATUS "   📦 TBB не найден, загрузка через Git...")
        FetchContent_Declare(
            tbb
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG v2021.11.0
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(tbb)
        message(STATUS "   ✅ TBB загружен из репозитория")
    endif()
endif()

# ----------------------------------------------------------------------
# pybind11 для Python биндингов
# ----------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "🐍 Настройка Python биндингов...")
    
    # Находим Python
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)
    message(STATUS "   ✅ Python ${Python_VERSION} найден")
    
    # Загружаем pybind11
    message(STATUS "   📦 Загрузка pybind11...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "   ✅ pybind11 загружен")
endif()

# ----------------------------------------------------------------------
# Indicators для прогресс-баров
# ----------------------------------------------------------------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/indicators/include)
    message(STATUS "✅ Indicators library найден локально")
    add_library(indicators INTERFACE)
    target_include_directories(indicators INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/indicators/include
    )
else()
    message(STATUS "ℹ️ Indicators library не найден (прогресс-бары будут упрощенными)")
endif()

# ======================================================================
# Исходные файлы библиотеки
# ======================================================================
file(GLOB LIB_SOURCES 
    CONFIGURE_DEPENDS
    src/*.cpp
)

# Исключаем python_bindings.cpp из основной библиотеки
list(REMOVE_ITEM LIB_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/python_bindings.cpp
)

message(STATUS "📁 Найдено исходных файлов для библиотеки: ${LIB_SOURCES}")

# ======================================================================
# Создание основной библиотеки
# ======================================================================
add_library(bpe_tokenizer ${LIB_SOURCES})

# ======================================================================
# Генерация config.h
# ======================================================================
# Создаем директорию для сгенерированных файлов
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

# Генерируем config.h из шаблона
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
    @ONLY
)
message(STATUS "✅ Сгенерирован config.h")

# ======================================================================
# Настройка путей включения
# ======================================================================
target_include_directories(bpe_tokenizer 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        src
)

# Добавляем пути к зависимостям
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(bpe_tokenizer PUBLIC nlohmann_json::nlohmann_json)
else()
    target_include_directories(bpe_tokenizer PRIVATE ${nlohmann_json_SOURCE_DIR}/include)
    target_link_libraries(bpe_tokenizer PUBLIC nlohmann_json::nlohmann_json)
endif()

target_link_libraries(bpe_tokenizer 
    PUBLIC
        fmt::fmt
    PRIVATE
        Threads::Threads
)

# ======================================================================
# Подключение опциональных библиотек
# ======================================================================

# OpenMP
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(bpe_tokenizer PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "✅ OpenMP подключен")
endif()

# TBB
if(USE_TBB)
    if(TARGET TBB::tbb)
        target_link_libraries(bpe_tokenizer PRIVATE TBB::tbb)
    elseif(TARGET tbb)
        target_link_libraries(bpe_tokenizer PRIVATE tbb)
    endif()
    target_compile_definitions(bpe_tokenizer PRIVATE USE_TBB=1)
    message(STATUS "✅ TBB подключен")
endif()

# Indicators
if(TARGET indicators)
    target_link_libraries(bpe_tokenizer PRIVATE indicators)
    target_compile_definitions(bpe_tokenizer PRIVATE USE_INDICATORS=1)
    message(STATUS "✅ Indicators подключен")
endif()

# ======================================================================
# SIMD оптимизации - проверка отдельных инструкций
# ======================================================================
if(USE_SIMD)
    include(CheckCXXCompilerFlag)
    
    set(SIMD_FLAGS "")
    set(SIMD_DEFINES "")

    message(STATUS "🔍 Проверка поддержки SIMD инструкций...")
    
    # Проверяем поддержку компилятором (без запуска)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    
    # Для x86_64 архитектуры
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # Просто включаем оптимальные флаги для вашего процессора
        # i5-1240P поддерживает AVX2, но не AVX-512
        if(COMPILER_SUPPORTS_AVX2)
            list(APPEND SIMD_FLAGS "-mavx2" "-mbmi2")
            list(APPEND SIMD_DEFINES "USE_AVX2=1" "USE_AVX=1" "USE_SSE42=1")
            message(STATUS "   ✅ AVX2 оптимизации включены")
        elseif(COMPILER_SUPPORTS_AVX)
            list(APPEND SIMD_FLAGS "-mavx")
            list(APPEND SIMD_DEFINES "USE_AVX=1" "USE_SSE42=1")
            message(STATUS "   ✅ AVX оптимизации включены")
        elseif(COMPILER_SUPPORTS_SSE42)
            list(APPEND SIMD_FLAGS "-msse4.2")
            list(APPEND SIMD_DEFINES "USE_SSE42=1")
            message(STATUS "   ✅ SSE4.2 оптимизации включены")
        else()
            message(STATUS "   ℹ️ SIMD оптимизации не поддерживаются компилятором")
        endif()
    endif()

    # Применяем флаги
    if(SIMD_FLAGS)
        target_compile_options(bpe_tokenizer PRIVATE ${SIMD_FLAGS})
        message(STATUS "   🔧 SIMD флаги: ${SIMD_FLAGS}")
    endif()

    if(SIMD_DEFINES)
        target_compile_definitions(bpe_tokenizer PUBLIC ${SIMD_DEFINES})
        message(STATUS "   🔧 SIMD определения: ${SIMD_DEFINES}")
    endif()
endif()

# ======================================================================
# Версионные определения
# ======================================================================
target_compile_definitions(bpe_tokenizer PRIVATE
    DEFAULT_NUM_THREADS=${DEFAULT_NUM_THREADS}
    DEFAULT_POOL_SIZE=${DEFAULT_POOL_SIZE}
    DEFAULT_CACHE_SIZE=${DEFAULT_CACHE_SIZE}
)

# ======================================================================
# Свойства библиотеки
# ======================================================================
set_target_properties(bpe_tokenizer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "bpe_tokenizer"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# ======================================================================
# Поддиректории
# ======================================================================

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_WEB_SERVER)
    add_subdirectory(web)
endif()

# ======================================================================
# Установка
# ======================================================================
install(TARGETS bpe_tokenizer
    EXPORT bpe_tokenizerTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
    DESTINATION include
)

#install(EXPORT bpe_tokenizerTargets
#    FILE bpe_tokenizerConfig.cmake
#    NAMESPACE bpe_tokenizer::
#    DESTINATION lib/cmake/bpe_tokenizer
#)

# ======================================================================
# Итоговая информация о сборке
# ======================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "📋 CPP BPE Tokenizer - Конфигурация сборки")
message(STATUS "========================================")
message(STATUS "📦 Версия: ${PROJECT_VERSION}")
message(STATUS "📦 Тип сборки: ${CMAKE_BUILD_TYPE}")
message(STATUS "⚙️  C++ стандарт: C++${CMAKE_CXX_STANDARD}")
message(STATUS "📁 Директория сборки: ${CMAKE_BINARY_DIR}")
message(STATUS "")
message(STATUS "📦 Компоненты:")
message(STATUS "   • Тесты: ${BUILD_TESTING}")
message(STATUS "   • Бенчмарки: ${BUILD_BENCHMARKS}")
message(STATUS "   • Примеры: ${BUILD_EXAMPLES}")
message(STATUS "   • Python биндинги: ${BUILD_PYTHON_BINDINGS}")
message(STATUS "   • Веб-сервер: ${BUILD_WEB_SERVER}")
message(STATUS "")
message(STATUS "⚡ Оптимизации:")
message(STATUS "   • OpenMP: ${USE_OPENMP} (${OpenMP_CXX_FOUND})")
message(STATUS "   • TBB: ${USE_TBB}")
message(STATUS "   • SIMD: ${USE_SIMD}")
message(STATUS "   • Кэширование: ${USE_CACHE}")
message(STATUS "")
message(STATUS "📦 Зависимости:")
if(USE_LOCAL_JSON)
    message(STATUS "   • nlohmann/json: локальный")
else()
    message(STATUS "   • nlohmann/json: из сети")
endif()
message(STATUS "")
message(STATUS "🛠️  Инструменты разработчика:")
message(STATUS "   • Профилирование: ${BUILD_WITH_PROFILING}")
message(STATUS "   • AddressSanitizer: ${BUILD_WITH_ASAN}")
message(STATUS "   • UBSan: ${BUILD_WITH_UBSAN}")
message(STATUS "")
message(STATUS "✅ Конфигурация завершена")
message(STATUS "   Для сборки выполните: cmake --build .")
message(STATUS "========================================")