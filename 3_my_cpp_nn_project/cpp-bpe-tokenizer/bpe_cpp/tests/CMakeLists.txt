# ======================================================================
# tests/CMakeLists.txt - Тесты для BPE токенизатора
# ======================================================================
#
# @file tests/CMakeLists.txt
# @brief Конфигурация сборки модульных тестов
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
# ======================================================================

# Политика для устранения предупреждения о времени
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# ======================================================================
# Подключение Google Test
# ======================================================================

message(STATUS "📦 Загрузка Google Test...")

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Опции для Google Test
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)  # Не нужен GMock
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)  # Не устанавливать в систему

FetchContent_MakeAvailable(googletest)

message(STATUS "   ✅ Google Test загружен")

# ======================================================================
# Общие настройки для всех тестов
# ======================================================================

# Добавляем пути для поиска заголовочных файлов
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include
)

# Функция для добавления теста с общими настройками
function(add_bpe_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    target_link_libraries(${TEST_NAME}
        PRIVATE
            bpe_tokenizer
            gtest_main
            pthread
            benchmark::benchmark
            benchmark::benchmark_main
    )
    
    # Добавляем тест в CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    include(GoogleTest)
    gtest_add_tests(${TEST_NAME} "" AUTO)
    
    message(STATUS "   ✅ Добавлен тест: ${TEST_NAME}")
endfunction()

# ======================================================================
# Базовые тесты
# ======================================================================

message(STATUS "\n📋 Добавление тестов:")

add_bpe_test(bpe_tests
    "test_vocabulary.cpp;test_tokenizer.cpp;test_utils.cpp"
)

# ======================================================================
# Тесты для быстрого токенизатора
# ======================================================================

add_bpe_test(test_fast_tokenizer "test_fast_tokenizer.cpp")

# ======================================================================
# Тесты для параллельной обработки
# ======================================================================

add_bpe_test(test_parallel "test_parallel.cpp")

# Подключаем OpenMP если доступно
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(test_parallel PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(test_parallel PRIVATE USE_OPENMP)
    message(STATUS "   ⚡ OpenMP подключен для test_parallel")
endif()

# ======================================================================
# Тесты для пула памяти
# ======================================================================

add_bpe_test(test_memory_pool "test_memory_pool.cpp")

# ======================================================================
# Тесты совместимости с Python
# ======================================================================

add_bpe_test(test_compatibility "test_compatibility.cpp")

# ======================================================================
# Тест обученной модели
# ======================================================================

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_trained_model.cpp)
    add_bpe_test(test_trained_model "test_trained_model.cpp")
endif()

# ======================================================================
# Копирование тестовых данных
# ======================================================================

# Копируем тестовые примеры C++ кода
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/bench_data/sample_code.txt)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/bench_data/sample_code.txt
        ${CMAKE_CURRENT_BINARY_DIR}/data/sample_code.txt
        COPYONLY
    )
    message(STATUS "   📋 Скопирован sample_code.txt")
endif()

# Копируем тестовые модели если есть
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../models)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../models
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    )
    message(STATUS "   📋 Скопированы тестовые модели")
endif()

# ======================================================================
# Пользовательские цели для запуска тестов
# ======================================================================

# Запуск всех тестов
add_custom_target(run_tests
    COMMAND ctest --output-on-failure
    COMMENT "🏃 Запуск всех тестов"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Запуск конкретной группы тестов
add_custom_target(run_fast_tests
    COMMAND ctest -R fast -V
    COMMENT "🏃 Запуск тестов FastTokenizer"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_memory_tests
    COMMAND ctest -R memory -V
    COMMENT "🏃 Запуск тестов MemoryPool"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_parallel_tests
    COMMAND ctest -R parallel -V
    COMMENT "🏃 Запуск тестов ParallelTrainer"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# ======================================================================
# Информация о сборке
# ======================================================================

message(STATUS "")
message(STATUS "📊 Тесты:")
message(STATUS "   • bpe_tests - базовые тесты (vocabulary, tokenizer, utils)")
message(STATUS "   • test_fast_tokenizer - оптимизированная версия")
message(STATUS "   • test_parallel - параллельная обработка")
message(STATUS "   • test_memory_pool - пул памяти")
message(STATUS "   • test_compatibility - совместимость с Python")
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_trained_model.cpp)
    message(STATUS "   • test_trained_model - проверка обученной модели")
endif()
message(STATUS "")
message(STATUS "🚀 Для запуска тестов:")
message(STATUS "   make run_tests           # все тесты")
message(STATUS "   make run_fast_tests      # только FastTokenizer")
message(STATUS "   make run_memory_tests    # только MemoryPool")
message(STATUS "   make run_parallel_tests  # только ParallelTrainer")
message(STATUS "   или")
message(STATUS "   cd build && ctest -V")
message(STATUS "========================================\n")