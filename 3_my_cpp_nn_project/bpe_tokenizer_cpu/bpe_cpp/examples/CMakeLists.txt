# ============================================================================
# examples/CMakeLists.txt
# ============================================================================
#
# @file examples/CMakeLists.txt
# @brief Сборка примеров использования BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.5.0
#
# @details Сборка демонстрационных программ, показывающих различные аспекты
#          использования токенизатора:
#
#          **Доступные примеры:**
#          ┌─────────────────────┬──────────────────────────────────┐
#          │ simple_example      │ Базовый пример для начинающих    │
#          │ batch_example       │ Пакетная обработка текстов       │
#          │ fast_tokenizer_demo │ Демо оптимизированной версии     │
#          │ train_example       │ Обучение токенизатора на корпусе │
#          │ compare_with_python │ Валидация C++ токенизатора       │
#          │                     │ сравнением с Python эталоном     │
#          └─────────────────────┴──────────────────────────────────┘
#
#          Для сравнения с Python реализацией используйте скрипт:
#          python ../scripts/validate_cpp_tokenizer.py
#
# @note Все примеры линкуются с библиотекой bpe_tokenizer
# @see ../CMakeLists.txt
# @see ../scripts/validate_cpp_tokenizer.py
# ============================================================================

# ============================================================================
# Проверка наличия необходимых компонентов
# ============================================================================

# Проверяем, что основная библиотека существует
if(NOT TARGET bpe_tokenizer)
    message(FATAL_ERROR "Библиотека bpe_tokenizer не найдена! Убедитесь, что корневой CMakeLists.txt собран правильно.")
endif()

# Находим компоненты
find_package(OpenMP QUIET)
find_package(Threads REQUIRED)

# ============================================================================
# Общие настройки для всех примеров
# ============================================================================

# Добавляем пути для поиска заголовочных файлов
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include
)

# Функция для добавления примера с общими настройками
function(add_example EXAMPLE_NAME EXAMPLE_SOURCE)
    # Добавляем пример только если исходный файл существует
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_SOURCE})
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})

        target_link_libraries(${EXAMPLE_NAME}
            PRIVATE
                bpe_tokenizer
                Threads::Threads
        )

        # ======================================================================
        # Особые настройки для разных примеров
        # ======================================================================

        # Пример обучения - требует OpenMP
        if(${EXAMPLE_NAME} STREQUAL "train_example")
            if(OpenMP_CXX_FOUND)
                target_link_libraries(${EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
                target_compile_options(${EXAMPLE_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
                message(STATUS "${EXAMPLE_NAME}: OpenMP включен")
            else()
                message(STATUS "${EXAMPLE_NAME}: OpenMP не найден (будет использована последовательная версия)")
            endif()
        endif()

        # Демо быстрого токенизатора - может использовать OpenMP
        if(${EXAMPLE_NAME} STREQUAL "fast_tokenizer_demo")
            if(OpenMP_CXX_FOUND)
                target_link_libraries(${EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
                target_compile_options(${EXAMPLE_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
                message(STATUS "${EXAMPLE_NAME}: OpenMP включен!")
            endif()

            # Определение для включения демо SIMD
            target_compile_definitions(${EXAMPLE_NAME} PRIVATE USE_FAST_TOKENIZER_DEMO)
        endif()

        # ======================================================================
        # Установка свойств
        # ======================================================================

        set_target_properties(${EXAMPLE_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )

        # ======================================================================
        # Копирование необходимых данных в директорию сборки
        # ======================================================================

        add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Копирование моделей и данных для ${EXAMPLE_NAME}..."

            # Копируем модели
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../models
                ${CMAKE_CURRENT_BINARY_DIR}/models

            # Копируем данные (если есть)
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../../data
                ${CMAKE_CURRENT_BINARY_DIR}/data

            # Создаем директорию для результатов
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_CURRENT_BINARY_DIR}/results

            COMMENT "Копирование моделей и данных для ${EXAMPLE_NAME}"
            VERBATIM
        )

        # ======================================================================
        # Цель для запуска примера
        # ======================================================================

        add_custom_target(run_${EXAMPLE_NAME}
            COMMAND ${CMAKE_COMMAND} -E echo ""
            COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
            COMMAND ${CMAKE_COMMAND} -E echo "Запуск ${EXAMPLE_NAME}..."
            COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
            COMMAND $<TARGET_FILE:${EXAMPLE_NAME}>
            DEPENDS ${EXAMPLE_NAME}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Запуск ${EXAMPLE_NAME}"
            VERBATIM
        )

        message(STATUS "Добавлен пример: ${EXAMPLE_NAME}")
    else()
        message(STATUS "Файл ${EXAMPLE_SOURCE} не найден, пример ${EXAMPLE_NAME} пропущен!")
    endif()
endfunction()

# ============================================================================
# Базовые примеры
# ============================================================================

message(STATUS "")
message(STATUS "Добавление примеров:")

# Простой пример
add_example(simple_example simple_example.cpp)

# Пример пакетной обработки
add_example(batch_example batch_example.cpp)

# ============================================================================
# Примеры для оптимизированной версии
# ============================================================================

# Демо быстрого токенизатора
add_example(fast_tokenizer_demo fast_tokenizer_demo.cpp)

# Пример обучения
add_example(train_example train_example.cpp)

# ============================================================================
# Пример для сравнения с Python (исполняемый файл)
# ============================================================================

# Проверяем наличие исходного файла
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compare_with_python.cpp)
    add_executable(compare_with_python compare_with_python.cpp)

    target_link_libraries(compare_with_python
        PRIVATE
            bpe_tokenizer
            Threads::Threads
    )

    target_include_directories(compare_with_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

    set_target_properties(compare_with_python PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    # Копирование моделей для compare_with_python
    add_custom_command(TARGET compare_with_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Копирование моделей для compare_with_python..."
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/../models
            ${CMAKE_CURRENT_BINARY_DIR}/models
        COMMENT "Копирование моделей для compare_with_python"
        VERBATIM
    )

    message(STATUS "Добавлен пример: compare_with_python (исполняемый файл)")
else()
    message(STATUS "Файл compare_with_python.cpp не найден, исполняемый файл не будет собран")
endif()

# ============================================================================
# Дополнительные примеры (если есть)
# ============================================================================

# Пример с многопоточностью
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/parallel_example.cpp)
    add_example(parallel_example parallel_example.cpp)
    if(TARGET parallel_example AND OpenMP_CXX_FOUND)
        target_link_libraries(parallel_example PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(parallel_example PRIVATE USE_OPENMP)
    endif()
endif()

# Пример с профилированием
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/profile_example.cpp)
    add_example(profile_example profile_example.cpp)
    if(TARGET profile_example)
        target_compile_definitions(profile_example PRIVATE BUILD_WITH_PROFILING)
    endif()
endif()

# ============================================================================
# Цель для запуска всех примеров
# ============================================================================

# Динамически формируем список команд для run_all_examples
set(ALL_EXAMPLES_COMMANDS)

# Добавляем разделители и команды для каждого существующего примера
if(TARGET simple_example)
    list(APPEND ALL_EXAMPLES_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "simple_example - базовый пример"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:simple_example>
    )
endif()

if(TARGET batch_example)
    list(APPEND ALL_EXAMPLES_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "batch_example - пакетная обработка"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:batch_example>
    )
endif()

if(TARGET fast_tokenizer_demo)
    list(APPEND ALL_EXAMPLES_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "fast_tokenizer_demo - демо оптимизированной версии"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:fast_tokenizer_demo>
    )
endif()

if(TARGET train_example)
    list(APPEND ALL_EXAMPLES_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "train_example - обучение токенизатора"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:train_example>
    )
endif()

if(TARGET compare_with_python)
    list(APPEND ALL_EXAMPLES_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "compare_with_python - сравнение с Python"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:compare_with_python> --quiet
    )
endif()

# Добавляем финальный разделитель
if(ALL_EXAMPLES_COMMANDS)
    list(APPEND ALL_EXAMPLES_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Все примеры выполнены!"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
    )
endif()

add_custom_target(run_all_examples
    ${ALL_EXAMPLES_COMMANDS}
    COMMENT "Запуск всех доступных примеров"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM
)

# ============================================================================
# Цель для сравнения с Python реализацией
# ============================================================================

# Проверяем наличие Python
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    # Добавляем цель для запуска Python скрипта сравнения
    add_custom_target(compare_with_python_script
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Сравнение с Python реализацией"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_cpp_tokenizer.py
        COMMENT "Сравнение C++ и Python реализаций токенизатора"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        VERBATIM
    )

    add_custom_target(run_compare_with_python
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Сравнение с Python реализацией"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_cpp_tokenizer.py
        COMMENT "Запуск сравнения C++ и Python реализаций"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS compare_with_python
        VERBATIM
    )
else()
    message(STATUS "Python не найден, цель сравнения недоступна!")
endif()

# ============================================================================
# Проверка наличия необходимых файлов
# ============================================================================

# Проверяем наличие моделей и создаем предупреждение если их нет
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../models/bpe_10000/cpp_vocab.json)
    message(STATUS "")
    message(STATUS " ВНИМАНИЕ: Модели не найдены!")
    message(STATUS "Для работы примеров необходимы файлы моделей в:")
    message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/../models/bpe_10000/")
    message(STATUS "")
    message(STATUS "Чтобы сконвертировать Python модели в C++ формат:")
    message(STATUS "cd ${CMAKE_CURRENT_SOURCE_DIR}/../tools/")
    message(STATUS "python convert_vocab.py --model-size 8000")
    message(STATUS "python convert_vocab.py --model-size 10000")
    message(STATUS "python convert_vocab.py --model-size 12000")
endif()

# Проверяем наличие Python скрипта для сравнения
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/validate_cpp_tokenizer.py)
    message(STATUS "Найден скрипт для сравнения с Python: ../scripts/validate_cpp_tokenizer.py")
else()
    message(STATUS "")
    message(STATUS "ВНИМАНИЕ: Скрипт validate_cpp_tokenizer.py не найден!")
endif()

# ============================================================================
# Информация о доступных целях
# ============================================================================

message(STATUS "")
message(STATUS "Доступные цели:")

# Выводим информацию о существующих примерах
message(STATUS "")
message(STATUS "Примеры:")
if(TARGET simple_example)
    message(STATUS "make simple_example     - сборка базового примера")
    message(STATUS "make run_simple_example - запуск базового примера")
endif()
if(TARGET batch_example)
    message(STATUS "make batch_example     - сборка примера пакетной обработки")
    message(STATUS "make run_batch_example - запуск примера пакетной обработки")
endif()
if(TARGET fast_tokenizer_demo)
    message(STATUS "make fast_tokenizer_demo     - сборка демо оптимизированной версии")
    message(STATUS "make run_fast_tokenizer_demo - запуск демо оптимизированной версии")
endif()
if(TARGET train_example)
    message(STATUS "make train_example     - сборка примера обучения")
    message(STATUS "make run_train_example - запуск примера обучения")
endif()
if(TARGET compare_with_python)
    message(STATUS "make compare_with_python - сборка бинарника для сравнения")
endif()

message(STATUS "")
message(STATUS "Все примеры:")
if(TARGET run_all_examples)
    message(STATUS "make run_all_examples - запустить все примеры")
endif()

message(STATUS "")
message(STATUS "Сравнение с Python:")
if(TARGET compare_with_python_script)
    message(STATUS "make compare_with_python_script - запустить сравнение (без сборки)")
endif()
if(TARGET run_compare_with_python)
    message(STATUS "make run_compare_with_python - сборка + сравнение")
endif()