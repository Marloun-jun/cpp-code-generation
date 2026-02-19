# ======================================================================
# examples/CMakeLists.txt - Примеры использования BPE токенизатора
# ======================================================================
#
# @file examples/CMakeLists.txt
# @brief Сборка примеров использования токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Сборка демонстрационных программ:
#          - simple_example      - базовый пример
#          - batch_example       - пакетная обработка
#          - compare_with_python - сравнение с Python
#          - fast_tokenizer_demo - демо оптимизированной версии
#          - train_example       - обучение токенизатора
#
# @note Все примеры линкуются с библиотекой bpe_tokenizer
# @see ../CMakeLists.txt
# ======================================================================

# ======================================================================
# Общие настройки для всех примеров
# ======================================================================

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
                ${CMAKE_THREAD_LIBS_INIT}  # pthread
        )
        
        # Если нужны дополнительные библиотеки для конкретного примера
        if(${EXAMPLE_NAME} STREQUAL "train_example")
            target_link_libraries(${EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(${EXAMPLE_NAME} PRIVATE USE_OPENMP)
        endif()
        
        # Особые настройки для fast_tokenizer_demo
        if(${EXAMPLE_NAME} STREQUAL "fast_tokenizer_demo")
            target_link_libraries(${EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(${EXAMPLE_NAME} PRIVATE 
                USE_OPENMP
                USE_FAST_TOKENIZER
            )
        endif()
        
        # Устанавливаем свойства
        set_target_properties(${EXAMPLE_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        )
        
        # Копируем необходимые данные в директорию сборки
        add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../models
                ${CMAKE_CURRENT_BINARY_DIR}/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../../data
                ${CMAKE_CURRENT_BINARY_DIR}/data
            COMMENT "📋 Копирование моделей и данных для ${EXAMPLE_NAME}"
        )
        
        message(STATUS "   Добавлен пример: ${EXAMPLE_NAME}")
    else()
        message(STATUS "   Файл ${EXAMPLE_SOURCE} не найден, пример ${EXAMPLE_NAME} пропущен")
    endif()
endfunction()

# ======================================================================
# Базовые примеры
# ======================================================================

message(STATUS "\nДобавление примеров:")

# Простой пример
add_example(simple_example simple_example.cpp)

# Пример пакетной обработки
add_example(batch_example batch_example.cpp)

# Пример сравнения с Python
add_example(compare_with_python compare_with_python.cpp)

# ======================================================================
# Примеры для оптимизированной версии
# ======================================================================

# Демо быстрого токенизатора (без условий - добавляем если файл есть)
add_example(fast_tokenizer_demo fast_tokenizer_demo.cpp)

# Пример обучения
add_example(train_example train_example.cpp)

# ======================================================================
# Дополнительные примеры (если есть)
# ======================================================================

# Пример с многопоточностью
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/parallel_example.cpp)
    add_example(parallel_example parallel_example.cpp)
endif()

# Пример с профилированием
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/profile_example.cpp)
    add_example(profile_example profile_example.cpp)
    if(TARGET profile_example)
        target_compile_definitions(profile_example PRIVATE BUILD_WITH_PROFILING)
    endif()
endif()

# ======================================================================
# Проверка наличия необходимых файлов
# ======================================================================

# Проверяем наличие моделей и создаем предупреждение если их нет
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../models/cpp_vocab.json)
    message(WARNING "Модели не найдены в ${CMAKE_CURRENT_SOURCE_DIR}/../models/")
    message(STATUS "Запустите для конвертации: python tools/convert_vocab.py")
endif()

# ======================================================================
# Пользовательские цели для запуска примеров
# ======================================================================

# Динамически формируем список команд для run_all_examples
set(ALL_EXAMPLES_COMMANDS)
if(TARGET simple_example)
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND $<TARGET_FILE:simple_example>)
endif()
if(TARGET batch_example)
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND $<TARGET_FILE:batch_example>)
endif()
if(TARGET compare_with_python)
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND $<TARGET_FILE:compare_with_python>)
endif()
if(TARGET fast_tokenizer_demo)
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND echo "\n=== Fast Tokenizer Demo ===")
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND $<TARGET_FILE:fast_tokenizer_demo>)
endif()
if(TARGET train_example)
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND echo "\n=== Training Example ===")
    list(APPEND ALL_EXAMPLES_COMMANDS COMMAND $<TARGET_FILE:train_example>)
endif()

add_custom_target(run_all_examples
    ${ALL_EXAMPLES_COMMANDS}
    COMMENT "Запуск всех доступных примеров"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Запуск конкретного примера (с проверкой существования)
if(TARGET simple_example)
    add_custom_target(run_simple_example
        COMMAND $<TARGET_FILE:simple_example>
        COMMENT "Запуск simple_example"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if(TARGET batch_example)
    add_custom_target(run_batch_example
        COMMAND $<TARGET_FILE:batch_example>
        COMMENT "Запуск batch_example"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if(TARGET fast_tokenizer_demo)
    add_custom_target(run_fast_demo
        COMMAND $<TARGET_FILE:fast_tokenizer_demo>
        COMMENT "Запуск fast_tokenizer_demo"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

if(TARGET train_example)
    add_custom_target(run_train_example
        COMMAND $<TARGET_FILE:train_example>
        COMMENT "Запуск train_example"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# ======================================================================
# Информация о сборке
# ======================================================================

message(STATUS "")
message(STATUS "Примеры:")
if(TARGET simple_example)
    message(STATUS "   • simple_example - базовый пример")
endif()
if(TARGET batch_example)
    message(STATUS "   • batch_example - пакетная обработка")
endif()
if(TARGET compare_with_python)
    message(STATUS "   • compare_with_python - сравнение с Python")
endif()
if(TARGET fast_tokenizer_demo)
    message(STATUS "   • fast_tokenizer_demo - демо оптимизированной версии")
endif()
if(TARGET train_example)
    message(STATUS "   • train_example - обучение токенизатора")
endif()
message(STATUS "")
message(STATUS "Для запуска примеров:")
if(TARGET run_simple_example)
    message(STATUS "   make run_simple_example")
endif()
if(TARGET run_batch_example)
    message(STATUS "   make run_batch_example")
endif()
if(TARGET run_fast_demo)
    message(STATUS "   make run_fast_demo")
endif()
if(TARGET run_all_examples)
    message(STATUS "   make run_all_examples")
endif()
message(STATUS "========================================\n")