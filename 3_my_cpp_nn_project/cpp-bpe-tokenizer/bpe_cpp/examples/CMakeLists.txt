# ======================================================================
# CMakeLists.txt - Примеры использования BPE токенизатора
# ======================================================================
#
# @file examples/CMakeLists.txt
# @brief Сборка примеров использования токенизатора
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
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
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    
    target_link_libraries(${EXAMPLE_NAME}
        PRIVATE
            bpe_tokenizer
            Threads::Threads
    )
    
    # Если нужны дополнительные библиотеки для конкретного примера
    if(${EXAMPLE_NAME} STREQUAL "fast_tokenizer_demo" OR 
       ${EXAMPLE_NAME} STREQUAL "train_example")
        target_link_libraries(${EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE USE_OPENMP)
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
    
    message(STATUS "   ✅ Добавлен пример: ${EXAMPLE_NAME}")
endfunction()

# ======================================================================
# Базовые примеры
# ======================================================================

message(STATUS "\n📦 Добавление примеров:")

# Простой пример
add_example(simple_example simple_example.cpp)

# Пример пакетной обработки
add_example(batch_example batch_example.cpp)

# Пример сравнения с Python
add_example(compare_with_python compare_with_python.cpp)

# ======================================================================
# Примеры для оптимизированной версии
# ======================================================================

# Демо быстрого токенизатора
if(TARGET bpe_tokenizer AND USE_FAST_TOKENIZER)
    add_example(fast_tokenizer_demo fast_tokenizer_demo.cpp)
    
    # Добавляем определение для условной компиляции
    target_compile_definitions(fast_tokenizer_demo PRIVATE USE_FAST_TOKENIZER)
endif()

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
    target_compile_definitions(profile_example PRIVATE BUILD_WITH_PROFILING)
endif()

# ======================================================================
# Проверка наличия необходимых файлов
# ======================================================================

# Проверяем наличие моделей и создаем предупреждение если их нет
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../models/cpp_vocab.json)
    message(WARNING "⚠️  Модели не найдены в ${CMAKE_CURRENT_SOURCE_DIR}/../models/")
    message(STATUS "   Запустите для конвертации: python tools/convert_vocab.py")
endif()

# ======================================================================
# Пользовательские цели для запуска примеров
# ======================================================================

# Запуск всех примеров
add_custom_target(run_all_examples
    COMMAND $<TARGET_FILE:simple_example>
    COMMAND $<TARGET_FILE:batch_example>
    COMMAND $<TARGET_FILE:compare_with_python>
    COMMAND echo "\n=== Fast Tokenizer Demo ==="
    COMMAND $<TARGET_FILE:fast_tokenizer_demo>
    COMMAND echo "\n=== Training Example ==="
    COMMAND $<TARGET_FILE:train_example>
    COMMENT "🏃 Запуск всех примеров"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Запуск конкретного примера
add_custom_target(run_simple_example
    COMMAND $<TARGET_FILE:simple_example>
    COMMENT "🏃 Запуск simple_example"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_batch_example
    COMMAND $<TARGET_FILE:batch_example>
    COMMENT "🏃 Запуск batch_example"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_fast_demo
    COMMAND $<TARGET_FILE:fast_tokenizer_demo>
    COMMENT "🏃 Запуск fast_tokenizer_demo"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_train_example
    COMMAND $<TARGET_FILE:train_example>
    COMMENT "🏃 Запуск train_example"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# ======================================================================
# Информация о сборке
# ======================================================================

message(STATUS "")
message(STATUS "📊 Примеры:")
message(STATUS "   • simple_example - базовый пример")
message(STATUS "   • batch_example - пакетная обработка")
message(STATUS "   • compare_with_python - сравнение с Python")
if(TARGET fast_tokenizer_demo)
    message(STATUS "   • fast_tokenizer_demo - демо оптимизированной версии")
endif()
message(STATUS "   • train_example - обучение токенизатора")
message(STATUS "")
message(STATUS "🚀 Для запуска примеров:")
message(STATUS "   make run_simple_example")
message(STATUS "   make run_batch_example")
message(STATUS "   make run_fast_demo")
message(STATUS "   make run_all_examples")
message(STATUS "========================================\n")