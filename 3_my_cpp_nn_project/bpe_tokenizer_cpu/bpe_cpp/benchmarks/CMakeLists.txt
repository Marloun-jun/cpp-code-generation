# ======================================================================
# @file benchmarks/CMakeLists.txt
# @brief Конфигурация сборки для бенчмарков BPE токенизатора
# @author Евгений П.
# @date 2026
# @version 3.4.0
# 
# @details Управляет сборкой и запуском тестов производительности:
#          - bench_tokenizer      - Тесты базовой версии
#          - bench_fast_tokenizer - Тесты оптимизированной версии
#          - bench_comparison     - Сравнение версий
# 
# @note Модели должны находиться в ${CMAKE_SOURCE_DIR}/models/
#       (относительно корня проекта bpe_cpp/)
# @see https://github.com/google/benchmark
# ======================================================================

cmake_minimum_required(VERSION 3.16)
project(BPEBenchmarks LANGUAGES CXX)

# ======================================================================
# Поиск зависимостей
# ======================================================================

# Проверяем наличие Google Benchmark (обязательная зависимость)
if(NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark не найден!")
endif()

# OpenMP для многопоточных бенчмарков (опционально)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP найден - многопоточные тесты будут оптимизированы!")
else()
    message(STATUS "OpenMP не найден - многопоточные тесты будут использовать std::thread")
endif()

# ======================================================================
# Конфигурация сборки
# ======================================================================

# Стандарт C++17 для всех бенчмарков
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Оптимизации для бенчмарков
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -DBENCHMARK_DEBUG")

# ======================================================================
# Функция для добавления бенчмарка
# ======================================================================

#[[
    Добавляет исполняемый файл бенчмарка с правильными зависимостями
    и настройками копирования моделей.
    
    Аргументы:
        name        - Имя цели и исполняемого файла
        source_file - Исходный файл с бенчмарками
]]
function(add_benchmark name source_file)
    # Создаем исполняемый файл
    add_executable(${name} ${source_file})
    
    # Линковка базовых зависимостей
    target_link_libraries(${name}
        PRIVATE
        bpe_tokenizer                # Основная библиотека токенизатора
        benchmark::benchmark         # Google Benchmark
        benchmark::benchmark_main    # Основная функция main из Google Benchmark
        Threads::Threads             # Потоковая поддержка
    )
    
    # Подавляем отладочный вывод для бенчмарков
    target_compile_definitions(${name} PRIVATE
        NDEBUG                 # Отключает assert и отладочные выводы
        BENCHMARK_NO_OUTPUT    # Отключает вывод предупреждений токенизатора
    )
    
    # Добавляем флаг для подавления вывода в консоль
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wno-unused-variable)
    endif()

    # OpenMP если доступен
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(${name} PRIVATE ${OpenMP_CXX_FLAGS})
        target_compile_definitions(${name} PRIVATE BENCHMARK_USE_OPENMP)
        message(STATUS "OpenMP включен для ${name}")
    endif()
    
    # Добавляем определение для определения корневой директории
    target_compile_definitions(${name} PRIVATE
        BENCHMARK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        BENCHMARK_BINARY_DIR="${CMAKE_BINARY_DIR}"
        BENCHMARK_PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    )
    
    # Устанавливаем выходную директорию
    set_target_properties(${name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
    )
    
    # ================================================================
    # Копирование моделей для бенчмарка
    # ================================================================
    
    # Создаем директории для моделей всех размеров
    set(MODEL_SIZES 8000 10000 12000)
    
    # Генерируем команды копирования для каждого размера
    set(COPY_COMMANDS "")
    foreach(size ${MODEL_SIZES})
        # Пути к исходным моделям (относительно корня проекта bpe_cpp/)
        set(MODEL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/models/bpe_${size}")
        set(MODEL_BINARY_DIR "${CMAKE_BINARY_DIR}/models/bpe_${size}")
        
        list(APPEND COPY_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MODEL_BINARY_DIR}"
            
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MODEL_SOURCE_DIR}/cpp_vocab.json"
                "${MODEL_BINARY_DIR}/cpp_vocab.json"
            
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MODEL_SOURCE_DIR}/cpp_merges.txt"
                "${MODEL_BINARY_DIR}/cpp_merges.txt"
        )
    endforeach()
    
    # Добавляем команду копирования после сборки
    add_custom_command(TARGET ${name} POST_BUILD
        ${COPY_COMMANDS}
        COMMENT "Копирование моделей для ${name}"
    )
    
    # ================================================================
    # Цель для запуска конкретного бенчмарка
    # ================================================================
    
    add_custom_target(run_${name}
        COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${name}
        DEPENDS ${name}
        COMMENT "Запуск ${name}"
        VERBATIM
    )
    
    message(STATUS "Добавлен бенчмарк: ${name}")
endfunction()

# ======================================================================
# Добавление бенчмарков
# ======================================================================

message(STATUS "")
message(STATUS "Настройка бенчмарков:")

# Проверяем существование исходных файлов и добавляем соответствующие цели
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_tokenizer.cpp")
    add_benchmark(bench_tokenizer bench_tokenizer.cpp)
else()
    message(WARNING "Файл bench_tokenizer.cpp не найден в ${CMAKE_CURRENT_SOURCE_DIR}!")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_fast_tokenizer.cpp")
    add_benchmark(bench_fast_tokenizer bench_fast_tokenizer.cpp)
else()
    message(WARNING "Файл bench_fast_tokenizer.cpp не найден в ${CMAKE_CURRENT_SOURCE_DIR}!")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_comparison.cpp")
    add_benchmark(bench_comparison bench_comparison.cpp)
else()
    message(WARNING "Файл bench_comparison.cpp не найден в ${CMAKE_CURRENT_SOURCE_DIR}!")
endif()

# ======================================================================
# Создаем директорию для результатов в bpe_cpp/reports/benchmarks/
# ======================================================================

set(BENCHMARK_RESULTS_DIR "${CMAKE_SOURCE_DIR}/reports/benchmarks")
file(MAKE_DIRECTORY ${BENCHMARK_RESULTS_DIR})

# Получаем текущую дату и время для имени файла
execute_process(
    COMMAND date "+%Y%m%d_%H%M%S"
    OUTPUT_VARIABLE CURRENT_DATETIME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Собираем список всех доступных бенчмарков
set(AVAILABLE_BENCHMARKS "")
if(TARGET bench_tokenizer)
    list(APPEND AVAILABLE_BENCHMARKS bench_tokenizer)
endif()
if(TARGET bench_fast_tokenizer)
    list(APPEND AVAILABLE_BENCHMARKS bench_fast_tokenizer)
endif()
if(TARGET bench_comparison)
    list(APPEND AVAILABLE_BENCHMARKS bench_comparison)
endif()

# ======================================================================
# Создаем скрипт для запуска всех бенчмарков
# ======================================================================

set(RUN_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/run_benchmarks.sh")
file(WRITE ${RUN_SCRIPT} "#!/bin/bash\n")
file(APPEND ${RUN_SCRIPT} "echo ''\n")
file(APPEND ${RUN_SCRIPT} "echo '============================================================'\n")
file(APPEND ${RUN_SCRIPT} "echo 'ЗАПУСК ВСЕХ БЕНЧМАРКОВ (ТИХИЙ РЕЖИМ)'\n")
file(APPEND ${RUN_SCRIPT} "echo '============================================================'\n")
file(APPEND ${RUN_SCRIPT} "echo ''\n")
file(APPEND ${RUN_SCRIPT} "echo 'Результаты будут сохранены в: ${BENCHMARK_RESULTS_DIR}/'\n")
file(APPEND ${RUN_SCRIPT} "echo ''\n")

foreach(bench ${AVAILABLE_BENCHMARKS})
    set(JSON_FILE "${BENCHMARK_RESULTS_DIR}/${bench}_${CURRENT_DATETIME}.json")
    file(APPEND ${RUN_SCRIPT} "echo 'Запуск ${bench}...'\n")
    file(APPEND ${RUN_SCRIPT} "${CMAKE_BINARY_DIR}/benchmarks/${bench} --benchmark_out=${JSON_FILE} --benchmark_out_format=json > /dev/null 2>&1\n")
    file(APPEND ${RUN_SCRIPT} "if [ \$? -eq 0 ]; then\n")
    file(APPEND ${RUN_SCRIPT} "  echo '${bench} завершен, результаты: ${JSON_FILE}'\n")
    file(APPEND ${RUN_SCRIPT} "else\n")
    file(APPEND ${RUN_SCRIPT} "  echo '${bench} завершился с ошибкой!'\n")
    file(APPEND ${RUN_SCRIPT} "fi\n")
endforeach()

file(APPEND ${RUN_SCRIPT} "echo ''\n")
file(APPEND ${RUN_SCRIPT} "echo '------------------------------------------------------------'\n")
file(APPEND ${RUN_SCRIPT} "echo 'Все бенчмарки завершены!'\n")
file(APPEND ${RUN_SCRIPT} "echo 'Результаты сохранены в: ${BENCHMARK_RESULTS_DIR}/'\n")
file(APPEND ${RUN_SCRIPT} "echo '------------------------------------------------------------'\n")

# Делаем скрипт исполняемым
add_custom_target(run_benchmarks_silent
    COMMAND chmod +x ${RUN_SCRIPT}
    COMMAND ${RUN_SCRIPT}
    COMMENT "Запуск всех бенчмарков (тихий режим)!"
    VERBATIM
)

# ======================================================================
# Для обратной совместимости оставляем старую цель с выводом
# ======================================================================

add_custom_target(run_benchmarks_verbose
    DEPENDS ${AVAILABLE_BENCHMARKS}
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
    COMMAND ${CMAKE_COMMAND} -E echo "ЗАПУСК ВСЕХ БЕНЧМАРКОВ (ПОДРОБНЫЙ РЕЖИМ)"
    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
)

foreach(bench ${AVAILABLE_BENCHMARKS})
    add_custom_command(TARGET run_benchmarks_verbose POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "Запуск ${bench}..."
        COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${bench}
        COMMENT "Запуск ${bench}"
        VERBATIM
    )
endforeach()

add_custom_command(TARGET run_benchmarks_verbose POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------------------"
    COMMAND ${CMAKE_COMMAND} -E echo "Все бенчмарки завершены!"
    COMMAND ${CMAKE_COMMAND} -E echo "------------------------------------------------------------"
    VERBATIM
)

# ======================================================================
# Делаем run_benchmarks по умолчанию тихим
# ======================================================================

add_custom_target(run_benchmarks
    DEPENDS run_benchmarks_silent
    COMMENT "Запуск всех бенчмарков (тихий режим, результаты в JSON)!"
)

# ======================================================================
# Создаем скрипт для показа последних результатов
# ======================================================================

set(SHOW_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/show_results.sh")
file(WRITE ${SHOW_SCRIPT} "#!/bin/bash\n")
file(APPEND ${SHOW_SCRIPT} "echo ''\n")
file(APPEND ${SHOW_SCRIPT} "echo '============================================================'\n")
file(APPEND ${SHOW_SCRIPT} "echo 'ПОСЛЕДНИЕ РЕЗУЛЬТАТЫ БЕНЧМАРКОВ'\n")
file(APPEND ${SHOW_SCRIPT} "echo '============================================================'\n")
file(APPEND ${SHOW_SCRIPT} "echo ''\n")
file(APPEND ${SHOW_SCRIPT} "ls -lt '${BENCHMARK_RESULTS_DIR}'/*.json 2>/dev/null | head -5 || echo 'Нет результатов в ${BENCHMARK_RESULTS_DIR}/'\n")
file(APPEND ${SHOW_SCRIPT} "echo ''\n")

add_custom_target(show_last_results
    COMMAND chmod +x ${SHOW_SCRIPT}
    COMMAND ${SHOW_SCRIPT}
    COMMENT "Показать последние результаты"
    VERBATIM
)

# ======================================================================
# Создаем скрипт для просмотра последнего JSON
# ======================================================================

set(VIEW_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/view_results.sh")
file(WRITE ${VIEW_SCRIPT} "#!/bin/bash\n")
file(APPEND ${VIEW_SCRIPT} "echo ''\n")
file(APPEND ${VIEW_SCRIPT} "echo '============================================================'\n")
file(APPEND ${VIEW_SCRIPT} "echo 'СОДЕРЖИМОЕ ПОСЛЕДНЕГО JSON ФАЙЛА'\n")
file(APPEND ${VIEW_SCRIPT} "echo '============================================================'\n")
file(APPEND ${VIEW_SCRIPT} "echo ''\n")
file(APPEND ${VIEW_SCRIPT} "LATEST=\$(ls -t '${BENCHMARK_RESULTS_DIR}'/*.json 2>/dev/null | head -1)\n")
file(APPEND ${VIEW_SCRIPT} "if [ -n \"\$LATEST\" ]; then\n")
file(APPEND ${VIEW_SCRIPT} "echo \"Файл: \$LATEST\"\n")
file(APPEND ${VIEW_SCRIPT} "echo \"\"\n")
file(APPEND ${VIEW_SCRIPT} "python3 -m json.tool \"\$LATEST\" | head -50\n")
file(APPEND ${VIEW_SCRIPT} "else\n")
file(APPEND ${VIEW_SCRIPT} "echo 'Нет результатов в ${BENCHMARK_RESULTS_DIR}/'\n")
file(APPEND ${VIEW_SCRIPT} "fi\n")
file(APPEND ${VIEW_SCRIPT} "echo ''\n")

add_custom_target(view_last_results
    COMMAND chmod +x ${VIEW_SCRIPT}
    COMMAND ${VIEW_SCRIPT}
    COMMENT "Показать содержимое последнего JSON файла"
    VERBATIM
)
# ======================================================================
# ЦЕЛЬ ДЛЯ ПРОВЕРКИ МОДЕЛЕЙ
# ======================================================================

# Проверяем существование моделей во время конфигурации
set(MODEL_8000_EXISTS FALSE)
set(MODEL_10000_EXISTS FALSE)
set(MODEL_12000_EXISTS FALSE)

if(EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_8000/cpp_vocab.json" AND EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_8000/cpp_merges.txt")
    set(MODEL_8000_EXISTS TRUE)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_10000/cpp_vocab.json" AND EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_10000/cpp_merges.txt")
    set(MODEL_10000_EXISTS TRUE)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_12000/cpp_vocab.json" AND EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_12000/cpp_merges.txt")
    set(MODEL_12000_EXISTS TRUE)
endif()

# Создаем команды для вывода результатов
set(CHECK_MODELS_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
    COMMAND ${CMAKE_COMMAND} -E echo "ПРОВЕРКА НАЛИЧИЯ МОДЕЛЕЙ"
    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Проверка моделей в ${CMAKE_SOURCE_DIR}/models/:"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)

if(MODEL_8000_EXISTS)
    list(APPEND CHECK_MODELS_COMMANDS COMMAND ${CMAKE_COMMAND} -E echo "Модель 8000: найдена!")
else()
    list(APPEND CHECK_MODELS_COMMANDS COMMAND ${CMAKE_COMMAND} -E echo "Модель 8000: НЕ НАЙДЕНА!")
endif()

if(MODEL_10000_EXISTS)
    list(APPEND CHECK_MODELS_COMMANDS COMMAND ${CMAKE_COMMAND} -E echo "Модель 10000: найдена!")
else()
    list(APPEND CHECK_MODELS_COMMANDS COMMAND ${CMAKE_COMMAND} -E echo "Модель 10000: НЕ НАЙДЕНА!")
endif()

if(MODEL_12000_EXISTS)
    list(APPEND CHECK_MODELS_COMMANDS COMMAND ${CMAKE_COMMAND} -E echo "Модель 12000: найдена!")
else()
    list(APPEND CHECK_MODELS_COMMANDS COMMAND ${CMAKE_COMMAND} -E echo "Модель 12000: НЕ НАЙДЕНА!")
endif()

list(APPEND CHECK_MODELS_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
)

add_custom_target(check_models
    ${CHECK_MODELS_COMMANDS}
    COMMENT "Проверка наличия моделей для бенчмарков"
    VERBATIM
)

# ======================================================================
# Проверка наличия моделей (во время конфигурации)
# ======================================================================

message(STATUS "")
message(STATUS "Проверка наличия моделей:")

foreach(size 8000 10000 12000)
    if(EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_${size}/cpp_vocab.json" AND EXISTS "${CMAKE_SOURCE_DIR}/models/bpe_${size}/cpp_merges.txt")
        message(STATUS "Модель ${size}: найдена!")
    else()
        message(STATUS "Модель ${size}: НЕ НАЙДЕНА!")
    endif()
endforeach()

# ======================================================================
# Цель для очистки результатов бенчмарков
# ======================================================================

add_custom_target(clean_benchmark_results
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${BENCHMARK_RESULTS_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BENCHMARK_RESULTS_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo "Результаты бенчмарков в ${BENCHMARK_RESULTS_DIR}/ очищены!"
    COMMENT "Очистка результатов бенчмарков"
)

# ======================================================================
# Цель для справки
# ======================================================================

add_custom_target(benchmarks_help
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Доступные цели для бенчмарков:"
    COMMAND ${CMAKE_COMMAND} -E echo "    run_benchmarks          - запустить все бенчмарки (тихий режим, JSON в reports/benchmarks/)"
    COMMAND ${CMAKE_COMMAND} -E echo "    run_benchmarks_verbose  - запустить все бенчмарки (подробный вывод в консоль)"
    COMMAND ${CMAKE_COMMAND} -E echo "    run_<name>              - запустить конкретный бенчмарк"
    COMMAND ${CMAKE_COMMAND} -E echo "    check_models            - проверить наличие моделей"
    COMMAND ${CMAKE_COMMAND} -E echo "    show_last_results       - показать последние JSON файлы"
    COMMAND ${CMAKE_COMMAND} -E echo "    view_last_results       - показать содержимое последнего JSON"
    COMMAND ${CMAKE_COMMAND} -E echo "    clean_benchmark_results - удалить результаты"
    COMMAND ${CMAKE_COMMAND} -E echo "    benchmarks_help         - показать эту справку"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Результаты сохраняются в: ${BENCHMARK_RESULTS_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Опции запуска бенчмарков:"
    COMMAND ${CMAKE_COMMAND} -E echo "    --benchmark_filter=<regex>  - фильтр тестов"
    COMMAND ${CMAKE_COMMAND} -E echo "    --benchmark_list_tests      - список тестов"
    COMMAND ${CMAKE_COMMAND} -E echo "    --benchmark_min_time=<N>    - мин. время (сек)"
    COMMAND ${CMAKE_COMMAND} -E echo "    --benchmark_repetitions=<N> - повторения"
    COMMAND ${CMAKE_COMMAND} -E echo "    --benchmark_out=<file>      - выходной файл"
    COMMAND ${CMAKE_COMMAND} -E echo "    --benchmark_out_format={json|console|csv}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMENT "Показать справку по бенчмаркам"
)

# ======================================================================
# Итоговая информация
# ======================================================================

message(STATUS "")
message(STATUS "Бенчмарки готовы к сборке:")

set(BENCHMARK_LIST "")
if(TARGET bench_tokenizer)
    list(APPEND BENCHMARK_LIST "- bench_tokenizer (базовая версия)")
endif()
if(TARGET bench_fast_tokenizer)
    list(APPEND BENCHMARK_LIST "- bench_fast_tokenizer (оптимизированная)")
endif()
if(TARGET bench_comparison)
    list(APPEND BENCHMARK_LIST "- bench_comparison (сравнение)")
endif()

foreach(bench ${BENCHMARK_LIST})
    message(STATUS "${bench}")
endforeach()

message(STATUS "")
message(STATUS "Цели для запуска:")
if(TARGET run_benchmarks)
    message(STATUS "- run_benchmarks - все бенчмарки (тихий режим)")
endif()
if(TARGET run_benchmarks_verbose)
    message(STATUS "- run_benchmarks_verbose - подробный режим")
endif()
foreach(bench bench_tokenizer bench_fast_tokenizer bench_comparison)
    if(TARGET run_${bench})
        message(STATUS "- run_${bench}")
    endif()
endforeach()
message(STATUS "- check_models            - проверить наличие моделей")
message(STATUS "- benchmarks_help         - справка")
message(STATUS "- show_last_results       - показать последние результаты")
message(STATUS "- view_last_results       - показать содержимое последнего JSON")
message(STATUS "- clean_benchmark_results - очистка")
message(STATUS "")
message(STATUS "Результаты сохраняются в: ${BENCHMARK_RESULTS_DIR}/")
message(STATUS "")

# ======================================================================
# Настройки для IDE
# ======================================================================

# Добавляем папки для группировки в IDE (Visual Studio, Xcode)
if(TARGET bench_tokenizer OR TARGET bench_fast_tokenizer OR TARGET bench_comparison)
    set_target_properties(
        bench_tokenizer bench_fast_tokenizer bench_comparison
        PROPERTIES
        FOLDER "benchmarks"
    )
endif()

set_target_properties(
    run_benchmarks 
    run_benchmarks_silent
    run_benchmarks_verbose
    clean_benchmark_results 
    benchmarks_help
    check_models
    show_last_results
    view_last_results
    PROPERTIES
    FOLDER "benchmarks/targets"
)

# Добавляем индивидуальные цели запуска
foreach(bench bench_tokenizer bench_fast_tokenizer bench_comparison)
    if(TARGET run_${bench})
        set_target_properties(run_${bench} PROPERTIES FOLDER "benchmarks/targets")
    endif()
endforeach()

# ======================================================================
# Конец файла
# ======================================================================