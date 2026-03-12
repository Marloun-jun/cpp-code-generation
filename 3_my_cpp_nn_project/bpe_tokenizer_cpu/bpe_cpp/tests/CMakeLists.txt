# ======================================================================
# tests/CMakeLists.txt - Тесты для BPE токенизатора
# ======================================================================
#
# @file tests/CMakeLists.txt
# @brief Конфигурация сборки модульных тестов
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
# ======================================================================

cmake_minimum_required(VERSION 3.16)
project(BPETests LANGUAGES CXX)

# ======================================================================
# Политики CMake
# ======================================================================

# Политика для устранения предупреждения о времени
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# ======================================================================
# Опции сборки
# ======================================================================

option(BUILD_TESTS "Сборка тестов" ON)
option(USE_OPENMP "Использовать OpenMP для параллельных тестов" ON)
option(DOWNLOAD_GTEST "Скачать Google Test автоматически" ON)
option(COPY_TEST_DATA "Копировать тестовые данные" ON)
option(RUN_BENCHMARKS_AS_TESTS "Запускать бенчмарки как тесты" OFF)

# ======================================================================
# Подключение Google Test (через Git - самый надёжный способ)
# ======================================================================

if(BUILD_TESTS AND DOWNLOAD_GTEST)
    message(STATUS "📦 Загрузка Google Test через Git...")
    
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )

    # Опции для Google Test
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)      # Не нужен GMock
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)    # Не устанавливать в систему
    
    # Подавляем предупреждения при сборке Google Test
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()

    FetchContent_MakeAvailable(googletest)
    
    message(STATUS "   ✅ Google Test загружен и собран")
    
    # Создаем псевдонимы для совместимости
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
    
elseif(BUILD_TESTS)
    # Ищем Google Test в системе
    find_package(GTest REQUIRED)
    message(STATUS "✅ Google Test найден в системе")
endif()

# ======================================================================
# Поиск OpenMP
# ======================================================================

if(USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "⚡ OpenMP найден: параллельные тесты будут оптимизированы")
    else()
        message(STATUS "ℹ️ OpenMP не найден: параллельные тесты без оптимизации")
        set(USE_OPENMP OFF)
    endif()
endif()

# ======================================================================
# Общие настройки для всех тестов
# ======================================================================

# Добавляем пути для поиска заголовочных файлов
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include
    ${googletest_SOURCE_DIR}/googletest/include
)

# Функция для добавления теста с общими настройками
function(add_bpe_test TEST_NAME TEST_SOURCES)
    # Разбиваем список файлов
    string(REPLACE ";" " " SOURCE_LIST "${TEST_SOURCES}")
    
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    
    # Линковка с Google Test
    target_link_libraries(${TEST_NAME}
        PRIVATE
            bpe_tokenizer
            gtest_main
            gtest
            pthread
    )
    
    # Добавляем зависимости от Google Test
    add_dependencies(${TEST_NAME} gtest gtest_main)
    
    # Устанавливаем C++17 стандарт
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
    )
    
    # Добавляем тест в CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Устанавливаем таймаут для теста (5 минут)
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 300)
    
    # Добавляем тег для группировки
    string(REGEX REPLACE "^test_" "" TEST_GROUP ${TEST_NAME})
    set_tests_properties(${TEST_NAME} PROPERTIES LABELS "${TEST_GROUP}")
    
    # Регистрируем тест в CTest
    include(GoogleTest)
    gtest_discover_tests(${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        PROPERTIES LABELS "${TEST_GROUP}"
    )
    
    message(STATUS "   ✅ Добавлен тест: ${TEST_NAME}")
endfunction()

# ======================================================================
# Сборка тестов (только если включено)
# ======================================================================

if(BUILD_TESTS)
    message(STATUS "")
    message(STATUS "🧪 Сборка тестов:")
    message(STATUS "========================================")

    # ======================================================================
    # Базовые тесты (объединённые для ускорения сборки)
    # ======================================================================
    
    set(BASE_TEST_SOURCES
        test_vocabulary.cpp
        test_tokenizer.cpp
        test_utils.cpp
    )
    
    # Проверяем существование файлов
    set(EXISTING_BASE_SOURCES "")
    foreach(SOURCE ${BASE_TEST_SOURCES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE})
            list(APPEND EXISTING_BASE_SOURCES ${SOURCE})
        else()
            message(WARNING "⚠️ Файл не найден: ${SOURCE}")
        endif()
    endforeach()
    
    if(EXISTING_BASE_SOURCES)
        add_bpe_test(bpe_tests "${EXISTING_BASE_SOURCES}")
    endif()

    # ======================================================================
    # Тесты для быстрого токенизатора
    # ======================================================================
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_fast_tokenizer.cpp)
        add_bpe_test(test_fast_tokenizer test_fast_tokenizer.cpp)
    endif()

    # ======================================================================
    # Тесты для параллельной обработки
    # ======================================================================
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_parallel.cpp)
        add_bpe_test(test_parallel test_parallel.cpp)
        
        # Подключаем OpenMP если доступно
        if(USE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(test_parallel PRIVATE OpenMP::OpenMP_CXX)
            message(STATUS "      ⚡ OpenMP подключен для test_parallel")
        endif()
    endif()

    # ======================================================================
    # Тесты для пула памяти
    # ======================================================================
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_pool.cpp)
        add_bpe_test(test_memory_pool test_memory_pool.cpp)
    endif()

    # ======================================================================
    # Тесты совместимости с Python
    # ======================================================================

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_compatibility.cpp)
        add_bpe_test(test_compatibility test_compatibility.cpp)
        
        # Добавляем Google Benchmark, если он доступен
        if(TARGET benchmark::benchmark)
            target_link_libraries(test_compatibility PRIVATE benchmark::benchmark)
            target_compile_definitions(test_compatibility PRIVATE HAS_BENCHMARK)
            message(STATUS "      📊 Google Benchmark подключен для test_compatibility")
        elseif(TARGET benchmark)
            target_link_libraries(test_compatibility PRIVATE benchmark)
            target_compile_definitions(test_compatibility PRIVATE HAS_BENCHMARK)
            message(STATUS "      📊 Google Benchmark подключен для test_compatibility")
        else()
            message(STATUS "      ℹ️ Google Benchmark не найден, тест производительности будет пропущен")
        endif()
    endif()

    # ======================================================================
    # Тест обученной модели
    # ======================================================================
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_trained_model.cpp)
        add_bpe_test(test_trained_model test_trained_model.cpp)
    endif()

    # ======================================================================
    # Дополнительные тесты
    # ======================================================================
    
    # Тест производительности как тест (если включено)
    if(RUN_BENCHMARKS_AS_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/bench_tokenizer.cpp)
        add_bpe_test(benchmark_as_test ../benchmarks/bench_tokenizer.cpp)
    endif()

else()
    message(STATUS "ℹ️ Сборка тестов отключена (BUILD_TESTS=OFF)")
endif()

# ======================================================================
# Копирование тестовых данных
# ======================================================================

if(COPY_TEST_DATA)
    message(STATUS "")
    message(STATUS "📁 Копирование тестовых данных:")
    message(STATUS "========================================")
    
    # Создаем директорию для данных
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/models)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python_models)

    # Копируем тестовые примеры C++ кода
    set(SAMPLE_CODE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/bench_data/sample_code.txt)
    if(EXISTS ${SAMPLE_CODE_PATH})
        configure_file(
            ${SAMPLE_CODE_PATH}
            ${CMAKE_CURRENT_BINARY_DIR}/data/sample_code.txt
            COPYONLY
        )
        message(STATUS "   ✅ sample_code.txt скопирован")
    else()
        message(STATUS "   ⚠️ sample_code.txt не найден")
    endif()

    # Копируем тестовые модели
    set(MODELS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../models)
    if(EXISTS ${MODELS_PATH})
        file(COPY ${MODELS_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        message(STATUS "   ✅ Модели скопированы из ../models/")
    else()
        message(STATUS "   ⚠️ Модели не найдены в ../models/")
    endif()

    # Копируем Python модели для тестов совместимости
    set(PYTHON_MODELS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../bpe_python/models)
    if(EXISTS ${PYTHON_MODELS_PATH})
        file(COPY ${PYTHON_MODELS_PATH} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/python_models)
        message(STATUS "   ✅ Python модели скопированы из ../../bpe_python/models/")
    else()
        message(STATUS "   ⚠️ Python модели не найдены")
    endif()
    
    # Создаем тестовый файл, если его нет
    if(NOT EXISTS ${SAMPLE_CODE_PATH})
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/data/sample_code.txt
            "int main() { return 0; }\n"
            "void test() { }\n"
            "class MyClass { };\n"
        )
        message(STATUS "   📝 Создан тестовый sample_code.txt")
    endif()
endif()

# ======================================================================
# Пользовательские цели для запуска тестов
# ======================================================================

if(BUILD_TESTS)
    message(STATUS "")
    message(STATUS "🎯 Цели для запуска:")
    message(STATUS "========================================")

    # Запуск всех тестов
    add_custom_target(run_tests
        COMMAND ctest --output-on-failure --progress
        COMMENT "🧪 Запуск всех тестов"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make run_tests           - все тесты")

    # Запуск конкретной группы тестов
    add_custom_target(run_fast_tests
        COMMAND ctest -L fast -V
        COMMENT "⚡ Запуск тестов FastTokenizer"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make run_fast_tests      - только FastTokenizer")

    add_custom_target(run_memory_tests
        COMMAND ctest -L memory -V
        COMMENT "🗄️ Запуск тестов MemoryPool"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make run_memory_tests    - только MemoryPool")

    add_custom_target(run_parallel_tests
        COMMAND ctest -L parallel -V
        COMMENT "⚙️ Запуск тестов ParallelTrainer"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make run_parallel_tests  - только ParallelTrainer")

    add_custom_target(run_compatibility_tests
        COMMAND ctest -L compatibility -V
        COMMENT "🐍 Запуск тестов совместимости с Python"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make run_compatibility_tests - совместимость")

    add_custom_target(run_base_tests
        COMMAND ctest -L base -V
        COMMENT "📚 Запуск базовых тестов"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make run_base_tests       - базовые тесты")

    # Запуск тестов с отладкой
    add_custom_target(debug_tests
        COMMAND ctest --output-on-failure --verbose
        COMMENT "🔍 Запуск тестов с подробным выводом"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make debug_tests          - подробный вывод")

    # Запуск тестов с valgrind (если доступен)
    find_program(VALGRIND_FOUND valgrind)
    if(VALGRIND_FOUND)
        add_custom_target(memcheck_tests
            COMMAND ctest -T memcheck
            COMMENT "🔎 Запуск тестов с проверкой памяти (valgrind)"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            USES_TERMINAL
        )
        message(STATUS "   make memcheck_tests       - проверка памяти")
    endif()

    # Запуск тестов несколько раз для выявления флагов
    add_custom_target(stress_tests
        COMMAND ctest --repeat until-fail:10
        COMMENT "🔥 Стресс-тестирование (10 повторов)"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        USES_TERMINAL
    )
    message(STATUS "   make stress_tests         - стресс-тесты")

    # Очистка тестовых данных
    add_custom_target(clean_test_data
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/data
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/models
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/python_models
        COMMENT "🧹 Очистка тестовых данных"
    )
    message(STATUS "   make clean_test_data      - очистка данных")
endif()

# ======================================================================
# Итоговая информация
# ======================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "📊 Итоговая информация о тестах:")
message(STATUS "========================================")

if(BUILD_TESTS)
    # Подсчитываем количество тестов
    get_property(TESTS_DIRECTORY DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
    list(LENGTH TESTS_DIRECTORY NUM_TESTS)
    
    message(STATUS "   Собрано тестов: ${NUM_TESTS}")
    message(STATUS "   Google Test: загружен через Git")
    if(OpenMP_CXX_FOUND)
        message(STATUS "   OpenMP: ✅ доступен")
    else()
        message(STATUS "   OpenMP: ❌ не доступен")
    endif()
    
    message(STATUS "")
    message(STATUS "   Директория данных: ${CMAKE_CURRENT_BINARY_DIR}/data")
    message(STATUS "   Директория моделей: ${CMAKE_CURRENT_BINARY_DIR}/models")
    
    message(STATUS "")
    message(STATUS "   Для запуска всех тестов:")
    message(STATUS "   cd ${CMAKE_CURRENT_BINARY_DIR} && ctest -V")
    message(STATUS "   или make run_tests")
else()
    message(STATUS "⚠️ Тесты не собираются (BUILD_TESTS=OFF)")
endif()

message(STATUS "========================================\n")