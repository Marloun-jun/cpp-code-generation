# ======================================================================
# CMakeLists.txt - Бенчмарки для BPE токенизатора
# ======================================================================
#
# @file benchmarks/CMakeLists.txt
# @brief Конфигурация сборки тестов производительности
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @details Сборка бенчмарков для сравнения производительности:
#          - bpe_benchmarks - базовая версия токенизатора
#          - bench_fast_tokenizer - оптимизированная версия
#          - bench_comparison - прямое сравнение версий
#
# @see ../CMakeLists.txt
# ======================================================================

# Подключаем Google Benchmark
include(FetchContent)

# Политика для устранения предупреждения о времени
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

message(STATUS "Загрузка Google Benchmark...")

FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Опции для Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

# ======================================================================
# Базовый бенчмарк
# ======================================================================

add_executable(bpe_benchmarks bench_tokenizer.cpp)

target_link_libraries(bpe_benchmarks 
    PRIVATE 
        bpe_tokenizer 
        benchmark::benchmark
        benchmark::benchmark_main  # Автоматический main()
)

target_compile_definitions(bpe_benchmarks PRIVATE 
    BENCHMARK_STATIC_DEFINE
)

# Копируем тестовые данные в директорию сборки
add_custom_command(TARGET bpe_benchmarks POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_data
        ${CMAKE_CURRENT_BINARY_DIR}/bench_data
    COMMENT "Копирование тестовых данных"
)

# ======================================================================
# Бенчмарк для оптимизированного токенизатора
# ======================================================================

add_executable(bench_fast_tokenizer bench_fast_tokenizer.cpp)

target_link_libraries(bench_fast_tokenizer 
    PRIVATE 
        bpe_tokenizer 
        benchmark::benchmark
        benchmark::benchmark_main
        Threads::Threads  # Лучше чем pthread (кроссплатформенно)
)

# Включаем OpenMP если доступно
if(USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(bench_fast_tokenizer PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(bench_fast_tokenizer PRIVATE USE_OPENMP)
    message(STATUS "   OpenMP подключен для bench_fast_tokenizer")
endif()

# Копируем модели в директорию сборки (если есть)
add_custom_command(TARGET bench_fast_tokenizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../models
        ${CMAKE_CURRENT_BINARY_DIR}/models
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_data
        ${CMAKE_CURRENT_BINARY_DIR}/bench_data
    COMMENT "Копирование моделей и тестовых данных"
)

# ======================================================================
# Бенчмарк сравнения двух версий
# ======================================================================

add_executable(bench_comparison bench_comparison.cpp)

target_link_libraries(bench_comparison 
    PRIVATE 
        bpe_tokenizer 
        benchmark::benchmark
        benchmark::benchmark_main
        Threads::Threads
)

# Копируем данные для сравнения
add_custom_command(TARGET bench_comparison POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../models
        ${CMAKE_CURRENT_BINARY_DIR}/models
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_data
        ${CMAKE_CURRENT_BINARY_DIR}/bench_data
    COMMENT "Копирование данных для сравнения"
)

# ======================================================================
# Дополнительные тесты производительности (опционально)
# ======================================================================

# Бенчмарк для многопоточного обучения
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_parallel_train.cpp)
    add_executable(bench_parallel_train bench_parallel_train.cpp)
    target_link_libraries(bench_parallel_train 
        PRIVATE 
            bpe_tokenizer 
            benchmark::benchmark
            benchmark::benchmark_main
            Threads::Threads
    )
    
    if(USE_OPENMP AND OpenMP_CXX_FOUND)
        target_link_libraries(bench_parallel_train PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    add_custom_command(TARGET bench_parallel_train POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/bench_data
            ${CMAKE_CURRENT_BINARY_DIR}/bench_data
    )
endif()

# ======================================================================
# Общие настройки для всех бенчмарков
# ======================================================================

# Добавляем пути для поиска заголовочных файлов
foreach(BENCH_TARGET 
        bpe_benchmarks 
        bench_fast_tokenizer 
        bench_comparison
        bench_parallel_train)
    if(TARGET ${BENCH_TARGET})
        target_include_directories(${BENCH_TARGET} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
            ${CMAKE_CURRENT_BINARY_DIR}/../include
        )
    endif()
endforeach()

# ======================================================================
# Пользовательские цели для запуска бенчмарков
# ======================================================================

# Запуск всех бенчмарков
add_custom_target(run_benchmarks
    COMMAND $<TARGET_FILE:bpe_benchmarks> --benchmark_out=baseline.json --benchmark_out_format=json
    COMMAND $<TARGET_FILE:bench_fast_tokenizer> --benchmark_out=fast.json --benchmark_out_format=json
    COMMAND $<TARGET_FILE:bench_comparison> --benchmark_out=comparison.json --benchmark_out_format=json
    COMMENT "🏃 Запуск всех бенчмарков и сохранение результатов в JSON"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Запуск с фильтрацией по названию
add_custom_target(run_encode_benchmarks
    COMMAND $<TARGET_FILE:bench_fast_tokenizer> --benchmark_filter=Encode
    COMMENT "🏃 Запуск только encode бенчмарков"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Запуск с кастомизацией
add_custom_target(run_benchmarks_detailed
    COMMAND $<TARGET_FILE:bench_fast_tokenizer> 
        --benchmark_repetitions=5
        --benchmark_display_aggregates_only=true
        --benchmark_counters_tabular=true
    COMMENT "🏃 Запуск с детальной статистикой (5 повторений)"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# ======================================================================
# Информация о сборке
# ======================================================================
message(STATUS "")
message(STATUS "Бенчмарки:")
message(STATUS "   • bpe_benchmarks - базовая версия")
message(STATUS "   • bench_fast_tokenizer - оптимизированная версия")
message(STATUS "   • bench_comparison - сравнение версий")
if(TARGET bench_parallel_train)
    message(STATUS "   • bench_parallel_train - параллельное обучение")
endif()
message(STATUS "")
message(STATUS "Для запуска: make run_benchmarks")
message(STATUS "Или: ./benchmarks/bpe_benchmarks")
message(STATUS "========================================")