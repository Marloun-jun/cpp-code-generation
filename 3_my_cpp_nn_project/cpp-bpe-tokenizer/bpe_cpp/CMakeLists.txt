# ======================================================================
# CMakeLists.txt - Основной файл сборки BPE токенизатора
# ======================================================================
#
# @file CMakeLists.txt
# @brief Конфигурация CMake для сборки C++ BPE токенизатора
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @details Этот файл управляет сборкой всех компонентов проекта:
#          - Основная библиотека bpe_tokenizer
#          - Оптимизированные версии с SIMD/OpenMP
#          - Тесты (Google Benchmark)
#          - Примеры использования
#          - Python биндинги (pybind11)
#          - Веб-сервер (CrowCpp)
#
# @note Требует CMake 3.16 или выше
# @see README.md для инструкций по сборке
# ======================================================================

cmake_minimum_required(VERSION 3.16)
project(CPPBpeTokenizer 
    VERSION 0.1.0 
    LANGUAGES CXX
    DESCRIPTION "Высокопроизводительный BPE токенизатор для C++ кода"
    HOMEPAGE_URL "https://github.com/yourusername/cpp-bpe-tokenizer"
)

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
# Опции сборки
# ======================================================================
option(BUILD_PYTHON_BINDINGS "Собрать Python биндинги" ON)
option(BUILD_TESTING "Собрать тесты" ON)
option(BUILD_BENCHMARKS "Собрать бенчмарки" ON)
option(BUILD_EXAMPLES "Собрать примеры" ON)
option(BUILD_WEB_SERVER "Собрать веб-сервер" ON)

# Оптимизации производительности
option(USE_OPENMP "Использовать OpenMP для параллельного обучения" ON)
option(USE_TBB "Использовать Intel TBB для параллельных алгоритмов" OFF)
option(USE_SIMD "Использовать SIMD оптимизации" ON)
option(USE_CACHE "Использовать кэширование результатов" ON)

# Инструменты разработчика
option(BUILD_WITH_PROFILING "Собрать с поддержкой профилирования" OFF)
option(BUILD_WITH_ASAN "Собрать с AddressSanitizer" OFF)
option(BUILD_WITH_UBSAN "Собрать с UndefinedBehaviorSanitizer" OFF)

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
    message(STATUS "🔧 Режим сборки: Release (максимальная производительность)")
    
    # Базовые оптимизации
    add_compile_options(-O3 -DNDEBUG)
    
    # Архитектурные оптимизации
    if(USE_SIMD)
        add_compile_options(-march=native -mtune=native)
        message(STATUS "   ✓ Архитектурные оптимизации включены (-march=native)")
    endif()
    
    # LTO и оптимизации времени линковки
    add_compile_options(-flto)
    add_link_options(-flto)
    message(STATUS "   ✓ Link Time Optimization включен")
    
    # Дополнительные оптимизации
    add_compile_options(
        -funroll-loops        # Развертывание циклов
        -fomit-frame-pointer  # Убрать frame pointer
    )
    message(STATUS "   ✓ Дополнительные оптимизации циклов включены")
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "🔧 Режим сборки: Debug (отладка)")
    add_compile_options(-g -O0 -DDEBUG)
    
    # Санитайзеры для отладки
    if(BUILD_WITH_ASAN)
        add_compile_options(-fsanitize=address)
        add_link_options(-fsanitize=address)
        message(STATUS "   ✓ AddressSanitizer включен")
    endif()
    
    if(BUILD_WITH_UBSAN)
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
        message(STATUS "   ✓ UndefinedBehaviorSanitizer включен")
    endif()
    
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "🔧 Режим сборки: RelWithDebInfo (оптимизация с отладочной информацией)")
    add_compile_options(-O2 -g -DNDEBUG)
endif()

# Профилирование
if(BUILD_WITH_PROFILING)
    add_compile_options(-pg -fno-omit-frame-pointer)
    add_link_options(-pg)
    message(STATUS "📊 Поддержка профилирования включена (-pg)")
endif()

# ======================================================================
# Поиск и подключение зависимостей
# ======================================================================
include(FetchContent)

# Поиск Threads (pthread)
find_package(Threads REQUIRED)
if(TARGET Threads::Threads)
    message(STATUS "✅ Threads::Threads найден")
else()
    message(WARNING "⚠️ Threads::Threads не найден, используем pthread напрямую")
    add_library(Threads::Threads INTERFACE IMPORTED)
    set_property(TARGET Threads::Threads PROPERTY 
        INTERFACE_LINK_LIBRARIES pthread)
endif()

# OpenMP для параллельных вычислений
find_package(OpenMP QUIET)
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    message(STATUS "✅ OpenMP найден: версия ${OpenMP_CXX_VERSION}")
elseif(USE_OPENMP AND NOT OpenMP_CXX_FOUND)
    message(WARNING "⚠️ OpenMP не найден, параллельное обучение будет отключено")
endif()

# nlohmann/json для работы с JSON
message(STATUS "📦 Загрузка nlohmann/json...")
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)
message(STATUS "   ✓ nlohmann/json загружен")

# fmt для форматирования строк
message(STATUS "📦 Загрузка fmt...")
FetchContent_Declare(
    fmt
    URL https://github.com/fmtlib/fmt/releases/download/9.1.0/fmt-9.1.0.zip
)
FetchContent_MakeAvailable(fmt)
message(STATUS "   ✓ fmt загружен")

# Google Benchmark для тестирования производительности
if(BUILD_BENCHMARKS OR BUILD_TESTING)
    message(STATUS "📦 Загрузка Google Benchmark...")
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.7.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
    message(STATUS "   ✓ Google Benchmark загружен")
endif()

# Intel TBB для параллельных алгоритмов
if(USE_TBB)
    message(STATUS "📦 Поиск Intel TBB...")
    find_package(TBB QUIET)
    if(NOT TBB_FOUND)
        message(STATUS "   TBB не найден в системе, загрузка...")
        FetchContent_Declare(
            tbb
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG v2021.8.0
        )
        FetchContent_MakeAvailable(tbb)
        message(STATUS "   ✓ TBB загружен из репозитория")
    else()
        message(STATUS "   ✓ TBB найден в системе")
    endif()
endif()

# Indicators для прогресс-баров (опционально)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/indicators/include)
    message(STATUS "📊 Indicators library найден")
    add_library(indicators INTERFACE)
    target_include_directories(indicators INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/indicators/include
    )
    
    # Проверяем наличие нужных файлов
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/indicators/include/indicators/progress_bar.hpp)
        message(STATUS "   ✓ progress_bar.hpp найден")
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/indicators/include/indicators/cursor_control.hpp)
        message(STATUS "   ✓ cursor_control.hpp найден")
    endif()
else()
    message(STATUS "ℹ️ Indicators library не найден (прогресс-бары будут упрощенными)")
    message(STATUS "   Для установки: git clone https://github.com/p-ranav/indicators.git")
endif()

# ======================================================================
# Пути к зависимостям из build/_deps/
# ======================================================================
set(DEPS_INCLUDE_DIRS
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/nlohmann_json-src/include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/fmt-src/include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/googletest-src/googletest/include
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/googlebenchmark-src/include
)

# Добавляем пути ко всем таргетам
include_directories(${DEPS_INCLUDE_DIRS})

# ======================================================================
# Исходные файлы библиотеки
# ======================================================================
set(LIB_SOURCES
    src/bpe_tokenizer.cpp      # Базовая версия
    src/vocabulary.cpp          # Словарь
    src/utils.cpp               # Утилиты
    src/fast_tokenizer.cpp      # Оптимизированная версия
    src/parallel_trainer.cpp    # Параллельное обучение
#    src/simd_utils.cpp          # SIMD оптимизации (опционально)
#    src/memory_pool.cpp         # Пул памяти (опционально)
#    src/profiler.cpp            # Профайлер (опционально)
    src/bpe_export.cpp          # Экспорт моделей
    src/python_bindings.cpp     # Python биндинги через pybind11
)

# ======================================================================
# Создание основной библиотеки
# ======================================================================
add_library(bpe_tokenizer ${LIB_SOURCES})

# Интерфейсная библиотека для оптимизированных заголовков
add_library(bpe_optimized INTERFACE)
target_include_directories(bpe_optimized 
    INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

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
message(STATUS "⚙️ Сгенерирован config.h")

# ======================================================================
# Настройка путей включения для библиотеки
# ======================================================================
target_include_directories(bpe_tokenizer 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        src
        ${DEPS_INCLUDE_DIRS}
)

# ======================================================================
# Линковка зависимостей
# ======================================================================
target_link_libraries(bpe_tokenizer 
    PRIVATE
        nlohmann_json::nlohmann_json
        fmt::fmt
        bpe_optimized
)

# Копируем config.h в исходную директорию для удобства
add_custom_command(TARGET bpe_tokenizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_BINARY_DIR}/include/config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/config.h
    COMMENT "📋 Копирование config.h в include директорию"
)

# ======================================================================
# Подключение опциональных библиотек
# ======================================================================

# OpenMP
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(bpe_tokenizer PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bpe_tokenizer PRIVATE USE_OPENMP)
    message(STATUS "✅ OpenMP подключен")
endif()

# TBB
if(USE_TBB AND TARGET TBB::tbb)
    target_link_libraries(bpe_tokenizer PRIVATE TBB::tbb)
    target_compile_definitions(bpe_tokenizer PRIVATE USE_TBB)
    message(STATUS "✅ TBB подключен")
endif()

# Indicators
if(TARGET indicators)
    target_link_libraries(bpe_tokenizer PRIVATE indicators)
    target_compile_definitions(bpe_tokenizer PRIVATE USE_INDICATORS)
    message(STATUS "✅ Indicators подключен")
endif()

# ======================================================================
# SIMD оптимизации
# ======================================================================
if(USE_SIMD)
    include(CheckCXXCompilerFlag)
    
    # Включаем AVX2
    target_compile_options(bpe_tokenizer PRIVATE -mavx2 -mbmi2)
    target_compile_definitions(bpe_tokenizer PRIVATE USE_AVX2)
    message(STATUS "✅ AVX2 оптимизации включены")
    
    # Проверяем и включаем SSE4.2
    check_cxx_compiler_flag("-msse4.2" HAS_SSE42)
    if(HAS_SSE42)
        target_compile_options(bpe_tokenizer PRIVATE -msse4.2)
        target_compile_definitions(bpe_tokenizer PRIVATE USE_SSE42)
        message(STATUS "✅ SSE4.2 оптимизации включены")
    endif()
endif()

# ======================================================================
# Версионные определения
# ======================================================================
target_compile_definitions(bpe_tokenizer PRIVATE
    PROJECT_VERSION="${PROJECT_VERSION}"
    BPE_TOKENIZER_VERSION="${CMAKE_PROJECT_VERSION}"
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

# Тесты
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "🧪 Тесты будут собраны")
endif()

# Бенчмарки
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
    message(STATUS "📊 Бенчмарки будут собраны")
endif()

# Примеры
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
    message(STATUS "📚 Примеры будут собраны")
endif()

# Веб-сервер
if(BUILD_WEB_SERVER)
    add_subdirectory(web)
    message(STATUS "🌐 Веб-сервер будет собран")
endif()

# ======================================================================
# Python биндинги
# ======================================================================

if(BUILD_PYTHON_BINDINGS)
    # Находим Python
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    message(STATUS "🐍 Python найден: ${Python_VERSION}")
    message(STATUS "   • Include пути: ${Python_INCLUDE_DIRS}")
    
    # Путь к pybind11 из виртуального окружения
    set(PYBIND11_INCLUDE_DIR /home/john/Projects/NS/.venv/lib/python3.12/site-packages/pybind11/include)
    
    # Проверяем существование пути
    if(EXISTS ${PYBIND11_INCLUDE_DIR}/pybind11/pybind11.h)
        message(STATUS "   ✅ pybind11 найден в: ${PYBIND11_INCLUDE_DIR}")
    else()
        message(WARNING "   ⚠️ pybind11 не найден по пути: ${PYBIND11_INCLUDE_DIR}")
    endif()
    
    # Подключаем pybind11
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
    
    # Добавляем пути к pybind11 в библиотеку
    target_include_directories(bpe_tokenizer PRIVATE 
        ${PYBIND11_INCLUDE_DIR}
        ${Python_INCLUDE_DIRS}
    )
    
    # Создаем модуль
    pybind11_add_module(bpe_tokenizer_cpp src/python_bindings.cpp)
    
    target_link_libraries(bpe_tokenizer_cpp PRIVATE bpe_tokenizer)
    
    target_include_directories(bpe_tokenizer_cpp PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
        ${PYBIND11_INCLUDE_DIR}
        ${Python_INCLUDE_DIRS}
        ${DEPS_INCLUDE_DIRS}
    )
    
    message(STATUS "🐍 Python биндинги будут собраны")
endif()

# ======================================================================
# Пользовательские цели
# ======================================================================

# Запуск всех бенчмарков
add_custom_target(benchmark-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target benchmarks
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_tokenizer
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_fast_tokenizer
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_comparison
    COMMENT "🏃 Запуск всех бенчмарков"
)

# Профилирование
add_custom_target(profile
    COMMAND perf record -g ${CMAKE_BINARY_DIR}/benchmarks/bench_fast_tokenizer
    COMMAND perf report -g graph --no-children
    COMMENT "📈 Профилирование быстрого токенизатора"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Очистка сгенерированных файлов
add_custom_target(clean-generated
    COMMAND ${CMAKE_COMMAND} -E remove
        ${CMAKE_CURRENT_SOURCE_DIR}/include/config.h
    COMMENT "🧹 Удаление сгенерированных файлов"
)

# ======================================================================
# Установка
# ======================================================================
install(TARGETS bpe_tokenizer
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)

# ======================================================================
# Итоговая информация о сборке
# ======================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "📦 CPP BPE Tokenizer - Конфигурация сборки")
message(STATUS "========================================")
message(STATUS "🔧 Тип сборки: ${CMAKE_BUILD_TYPE}")
message(STATUS "🛠️  C++ стандарт: C++${CMAKE_CXX_STANDARD}")
message(STATUS "📁 Директория сборки: ${CMAKE_BINARY_DIR}")
message(STATUS "")
message(STATUS "📋 Компоненты:")
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
message(STATUS "🔬 Инструменты разработчика:")
message(STATUS "   • Профилирование: ${BUILD_WITH_PROFILING}")
message(STATUS "   • AddressSanitizer: ${BUILD_WITH_ASAN}")
message(STATUS "   • UBSan: ${BUILD_WITH_UBSAN}")
message(STATUS "")
message(STATUS "✅ Конфигурация завершена")
message(STATUS "   Для сборки выполните: cmake --build .")
message(STATUS "========================================")