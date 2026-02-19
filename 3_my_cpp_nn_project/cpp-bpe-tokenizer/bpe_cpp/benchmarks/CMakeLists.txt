# ======================================================================
# benchmarks/CMakeLists.txt - Бенчмарки производительности
# ======================================================================
#
# @file benchmarks/CMakeLists.txt
# @brief Конфигурация сборки тестов производительности
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
# ======================================================================

# Политика для устранения предупреждения о времени
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# ======================================================================
# Подключение Google Benchmark (уже загружен в корневом CMake)
# ======================================================================

# ======================================================================
# Функция для добавления бенчмарка
# ======================================================================
function(add_benchmark BENCHMARK_NAME BENCHMARK_SOURCE)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${BENCHMARK_SOURCE})
        add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
        
        target_link_libraries(${BENCHMARK_NAME}
            PRIVATE
                bpe_tokenizer
                benchmark::benchmark
                benchmark::benchmark_main
                Threads::Threads
        )
        
        # Подключаем OpenMP если доступно
        if(USE_OPENMP AND OpenMP_CXX_FOUND)
            target_link_libraries(${BENCHMARK_NAME} PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(${BENCHMARK_NAME} PRIVATE USE_OPENMP)
            message(STATUS "OpenMP подключен для ${BENCHMARK_NAME}")
        endif()
        
        message(STATUS "Добавлен бенчмарк: ${BENCHMARK_NAME}")
    else()
        message(STATUS "Файл ${BENCHMARK_SOURCE} не найден, бенчмарк ${BENCHMARK_NAME} пропущен")
    endif()
endfunction()

# ======================================================================
# Добавление бенчмарков
# ======================================================================

message(STATUS "\nобавление бенчмарков:")

# Базовый бенчмарк
add_benchmark(bpe_benchmarks bench_tokenizer.cpp)

# Бенчмарк для оптимизированной версии
add_benchmark(bench_fast_tokenizer bench_fast_tokenizer.cpp)

# Бенчмарк сравнения
add_benchmark(bench_comparison bench_comparison.cpp)

# ======================================================================
# Копирование тестовых данных
# ======================================================================

# Копируем тестовые примеры C++ кода
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bench_data)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_data/sample_code.txt)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_data/sample_code.txt
        ${CMAKE_CURRENT_BINARY_DIR}/bench_data/sample_code.txt
        COPYONLY
    )
    message(STATUS "Скопирован sample_code.txt")
endif()

# ======================================================================
# Пользовательские цели для запуска бенчмарков
# ======================================================================

# Запуск всех бенчмарков (только существующие)
set(BENCHMARK_COMMANDS)
if(TARGET bpe_benchmarks)
    list(APPEND BENCHMARK_COMMANDS COMMAND $<TARGET_FILE:bpe_benchmarks>)
endif()
if(TARGET bench_fast_tokenizer)
    list(APPEND BENCHMARK_COMMANDS COMMAND $<TARGET_FILE:bench_fast_tokenizer>)
endif()
if(TARGET bench_comparison)
    list(APPEND BENCHMARK_COMMANDS COMMAND $<TARGET_FILE:bench_comparison>)
endif()

add_custom_target(run_benchmarks
    ${BENCHMARK_COMMANDS}
    COMMENT "Запуск всех бенчмарков"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Запуск конкретного бенчмарка
if(TARGET bpe_benchmarks)
    add_custom_target(run_basic_benchmark
        COMMAND $<TARGET_FILE:bpe_benchmarks>
        COMMENT "Запуск базового бенчмарка"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if(TARGET bench_fast_tokenizer)
    add_custom_target(run_fast_benchmark
        COMMAND $<TARGET_FILE:bench_fast_tokenizer>
        COMMENT "Запуск бенчмарка FastTokenizer"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if(TARGET bench_comparison)
    add_custom_target(run_comparison_benchmark
        COMMAND $<TARGET_FILE:bench_comparison>
        COMMENT "Запуск сравнительного бенчмарка"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# ======================================================================
# Информация о сборке
# ======================================================================

message(STATUS "")
message(STATUS "Бенчмарки:")
if(TARGET bpe_benchmarks)
    message(STATUS "   • bpe_benchmarks - базовая версия")
endif()
if(TARGET bench_fast_tokenizer)
    message(STATUS "   • bench_fast_tokenizer - оптимизированная версия")
endif()
if(TARGET bench_comparison)
    message(STATUS "   • bench_comparison - сравнение версий")
endif()
message(STATUS "")
message(STATUS "Для запуска бенчмарков:")
if(TARGET run_benchmarks)
    message(STATUS "   make run_benchmarks")
endif()
message(STATUS "========================================\n")