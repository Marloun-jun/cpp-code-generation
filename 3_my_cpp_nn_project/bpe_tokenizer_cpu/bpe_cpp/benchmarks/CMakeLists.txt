# ======================================================================
# @file benchmarks/CMakeLists.txt
# @brief Конфигурация сборки для бенчмарков BPE токенизатора
# ======================================================================

cmake_minimum_required(VERSION 3.16)
project(BPEBenchmarks LANGUAGES CXX)

# ======================================================================
# Поиск зависимостей
# ======================================================================

# Проверяем наличие Google Benchmark
if(NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark не найден!")
endif()

# OpenMP (опционально)
find_package(OpenMP QUIET)

# ======================================================================
# Функция для добавления бенчмарка
# ======================================================================

function(add_benchmark name source_file)
    # Создаем исполняемый файл
    add_executable(${name} ${source_file})
    
    # Линковка базовых зависимостей
    target_link_libraries(${name}
        PRIVATE
        bpe_tokenizer
        benchmark::benchmark
        benchmark::benchmark_main
        Threads::Threads
    )
    
    # OpenMP если доступен
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(${name} PRIVATE ${OpenMP_CXX_FLAGS})
        target_compile_definitions(${name} PRIVATE BENCHMARK_USE_OPENMP)
        message(STATUS "OpenMP включен для ${name}")
    endif()
    
    # Устанавливаем выходную директорию
    set_target_properties(${name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
    )
    
    # Копирование моделей - ИСПРАВЛЕННЫЕ ПУТИ (убрал ../)
    add_custom_command(TARGET ${name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/models"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/models/bpe_8000"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/models/bpe_10000"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/models/bpe_12000"
        
        # Копируем модели всех размеров - ПРАВИЛЬНЫЕ ПУТИ
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/bpe_8000/cpp_vocab.json"
            "${CMAKE_BINARY_DIR}/models/bpe_8000/cpp_vocab.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/bpe_8000/cpp_merges.txt"
            "${CMAKE_BINARY_DIR}/models/bpe_8000/cpp_merges.txt"
            
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/bpe_10000/cpp_vocab.json"
            "${CMAKE_BINARY_DIR}/models/bpe_10000/cpp_vocab.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/bpe_10000/cpp_merges.txt"
            "${CMAKE_BINARY_DIR}/models/bpe_10000/cpp_merges.txt"
            
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/bpe_12000/cpp_vocab.json"
            "${CMAKE_BINARY_DIR}/models/bpe_12000/cpp_vocab.json"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/models/bpe_12000/cpp_merges.txt"
            "${CMAKE_BINARY_DIR}/models/bpe_12000/cpp_merges.txt"
        
        COMMENT "Копирование моделей для ${name}"
    )
    
    # Цель для запуска
    add_custom_target(run_${name}
        COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${name}
        DEPENDS ${name}
        COMMENT "Запуск ${name}"
    )
    
    message(STATUS "✅ Добавлен бенчмарк: ${name}")
endfunction()

# ======================================================================
# Добавление бенчмарков
# ======================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_tokenizer.cpp")
    add_benchmark(bench_tokenizer bench_tokenizer.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_fast_tokenizer.cpp")
    add_benchmark(bench_fast_tokenizer bench_fast_tokenizer.cpp)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_comparison.cpp")
    add_benchmark(bench_comparison bench_comparison.cpp)
endif()

# ======================================================================
# Цель для запуска всех бенчмарков
# ======================================================================

add_custom_target(run_benchmarks
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_tokenizer
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_fast_tokenizer
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_comparison
    DEPENDS bench_tokenizer bench_fast_tokenizer bench_comparison
    COMMENT "Запуск всех бенчмарков"
)

# ======================================================================
# Проверка наличия моделей - ИСПРАВЛЕНО
# ======================================================================

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_8000/cpp_vocab.json")
    message(WARNING "Модели не найдены! Сначала выполните:")
    message(WARNING "  cd ${CMAKE_SOURCE_DIR}/tools/")
    message(WARNING "  python convert_vocab.py --model-size 8000")
    message(WARNING "  python convert_vocab.py --model-size 10000")
    message(WARNING "  python convert_vocab.py --model-size 12000")
endif()

# ======================================================================
# Информация
# ======================================================================

message(STATUS "")
message(STATUS "Бенчмарки:")
if(TARGET bench_tokenizer)
    message(STATUS "  - bench_tokenizer")
endif()
if(TARGET bench_fast_tokenizer)
    message(STATUS "  - bench_fast_tokenizer")
endif()
if(TARGET bench_comparison)
    message(STATUS "  - bench_comparison")
endif()
message(STATUS "")
message(STATUS "Цели: run_benchmarks, run_<name>")
message(STATUS "")