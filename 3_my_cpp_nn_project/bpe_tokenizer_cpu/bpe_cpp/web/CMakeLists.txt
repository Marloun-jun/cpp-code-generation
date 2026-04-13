# ============================================================================
# web/CMakeLists.txt
# ============================================================================
#
# @file web/CMakeLists.txt
# @brief Конфигурация сборки веб-серверов для BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Этот файл управляет сборкой двух вариантов веб-сервера:
#
#          **Сервер на базе CrowCpp** (рекомендуемый)
#          ┌──────────────────────────────────────────────────────────┐
#          │ Современный REST API сервер                              │
#          │ Поддержка CORS, Swagger UI, Prometheus метрик            │
#          │ Требует библиотеку CrowCpp                               │
#          └──────────────────────────────────────────────────────────┘
#
#          **Простой сервер на сокетах** (fallback)
#          ┌──────────────────────────────────────────────────────────┐
#          │ Легковесный, без внешних зависимостей                    │
#          │ Только стандартная библиотека C++                        │
#          │ Работает везде, где есть компилятор C++17                │
#          └──────────────────────────────────────────────────────────┘
#
#          **Зависимости:**
#          - CrowCpp       - Для tokenizer_server_crow
#          - Zlib          - Для сжатия ответов в Crow (опционально)
#          - pthread       - Для многопоточности
#          - bpe_tokenizer - Наша библиотека
#
#          **Цели сборки:**
#          ┌───────────────────────┬──────────────────────────────────┐
#          │ tokenizer_server_crow │ Сборка Crow сервера              │
#          │ server                │ Сборка простого сервера          │
#          │ run_crow_server       │ Сборка и запуск Crow сервера     │
#          │ run_simple_server     │ Сборка и запуск простого сервера │
#          │ run_all_servers       │ Сборка и запуск обоих серверов   │
#          │ clean_servers         │ Очистка артефактов сборки        │
#          └───────────────────────┴──────────────────────────────────┘
#
# @see server.cpp
# @see server_crow.cpp
# ============================================================================

cmake_minimum_required(VERSION 3.15)
project(BPEWebServers VERSION 3.3.0 LANGUAGES CXX)

# ============================================================================
# Опции сборки
# ============================================================================

option(BUILD_CROW_SERVER "Сборка сервера на базе CrowCpp" ON)
option(BUILD_SIMPLE_SERVER "Сборка простого сервера на сокетах" ON)
option(COPY_MODELS "Копировать модели в директорию сборки" ON)
option(REQUIRE_CROW "Требовать наличие CrowCpp (иначе ошибка)" OFF)
option(ENABLE_COMPRESSION "Включить сжатие ответов в Crow сервере" ON)

# ============================================================================
# Поиск CrowCpp
# ============================================================================

# Возможные пути к CrowCpp (в порядке приоритета)
set(CROW_POSSIBLE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/crow/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/crow/include
    ${CMAKE_SOURCE_DIR}/third_party/crow/include
    /usr/local/include/crow
    /usr/include/crow
    /opt/local/include/crow
    $ENV{HOME}/crow/include
    $ENV{HOME}/local/include/crow
)

set(CROW_INCLUDE_DIR "" CACHE PATH "Путь к заголовочным файлам CrowCpp")

# Если путь не указан явно, ищем в стандартных местах
if(NOT CROW_INCLUDE_DIR AND BUILD_CROW_SERVER)
    message(STATUS "Поиск CrowCpp...")
    foreach(PATH ${CROW_POSSIBLE_PATHS})
        if(EXISTS ${PATH}/crow/app.h)
            set(CROW_INCLUDE_DIR ${PATH})
            message(STATUS "Найден в: ${PATH}")
            break()
        endif()
    endforeach()
endif()

# Проверка наличия CrowCpp
if(BUILD_CROW_SERVER)
    if(EXISTS ${CROW_INCLUDE_DIR}/crow/app.h)
        message(STATUS "CrowCpp найден: ${CROW_INCLUDE_DIR}")
        set(CROW_FOUND TRUE)
    else()
        if(REQUIRE_CROW)
            message(FATAL_ERROR 
                "CrowCpp не найден! Установите CrowCpp:\n"
                "git clone https://github.com/CrowCpp/Crow.git ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/crow\n"
                "Или укажите путь через -DCROW_INCLUDE_DIR=..."
            )
        else()
            message(WARNING 
                "CrowCpp не найден! Сервер tokenizer_server_crow не будет собран.\n"
                "Установите CrowCpp для полноценного REST API сервера."
            )
            set(BUILD_CROW_SERVER OFF)
        endif()
    endif()
endif()

# ============================================================================
# Поиск ZLIB (для сжатия ответов в Crow)
# ============================================================================

if(BUILD_CROW_SERVER AND ENABLE_COMPRESSION)
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        message(STATUS "ZLIB найден: ${ZLIB_LIBRARIES}")
        message(STATUS "Сжатие ответов включено!")
        set(HAVE_ZLIB TRUE)
    else()
        message(STATUS "ZLIB не найден: сжатие ответов отключено!")
        message(STATUS "Установите zlib для улучшения производительности:")
        message(STATUS "- Ubuntu/Debian: sudo apt install zlib1g-dev")
        message(STATUS "- macOS:         brew install zlib")
        message(STATUS "- Windows:       vcpkg install zlib")
        set(HAVE_ZLIB FALSE)
    endif()
endif()

# ============================================================================
# Поиск библиотеки bpe_tokenizer
# ============================================================================

# Предполагаем, что библиотека уже собрана в родительской директории
set(BPE_TOKENIZER_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../build/libbpe_tokenizer.a")
if(NOT EXISTS ${BPE_TOKENIZER_LIB})
    message(WARNING "Библиотека bpe_tokenizer не найдена по пути: ${BPE_TOKENIZER_LIB}")
    message(STATUS "Убедитесь, что сначала собран основной проект")
endif()

# ============================================================================
# Crow Server Executable
# ============================================================================

if(BUILD_CROW_SERVER AND CROW_FOUND)
    message(STATUS "")
    message(STATUS "Сборка Crow сервера...")
    message(STATUS "------------------------------------------------------------")
    
    # Добавление исполняемого файла для Crow сервера
    add_executable(tokenizer_server_crow server_crow.cpp)

    # Настройка путей включения
    target_include_directories(tokenizer_server_crow PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include    # Наши заголовочные файлы
        ${CROW_INCLUDE_DIR}                       # CrowCpp заголовки
    )

    # Подключаем ZLIB если найден
    if(HAVE_ZLIB)
        target_include_directories(tokenizer_server_crow PRIVATE ${ZLIB_INCLUDE_DIRS})
        target_link_libraries(tokenizer_server_crow PRIVATE ZLIB::ZLIB)
        target_compile_definitions(tokenizer_server_crow PRIVATE CROW_ENABLE_COMPRESSION)
    endif()

    # Линковка с необходимыми библиотеками
    target_link_libraries(tokenizer_server_crow PRIVATE 
        bpe_tokenizer       # Наша библиотека токенизатора
        Threads::Threads    # Многопоточность
        pthread             # POSIX threads
    )

    # Установка свойств C++ стандарта
    set_target_properties(tokenizer_server_crow PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        OUTPUT_NAME "bpe_server_crow"
        PREFIX ""
    )

    # Копирование файлов моделей в директорию сборки
    if(COPY_MODELS)
        add_custom_command(TARGET tokenizer_server_crow POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Копирование моделей для Crow сервера..."
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../models
                ${CMAKE_CURRENT_BINARY_DIR}/models
            COMMENT "Копирование моделей для Crow сервера"
        )
        
        # Также копируем примеры для тестирования
        add_custom_command(TARGET tokenizer_server_crow POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Копирование примеров кода..."
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/bench_data
                ${CMAKE_CURRENT_BINARY_DIR}/examples
            COMMENT "Копирование примеров кода"
        )
    endif()

    # Цель для запуска Crow сервера
    add_custom_target(run_crow_server
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Запуск Crow сервера на http://localhost:8080"
        COMMAND ${CMAKE_COMMAND} -E echo "Swagger UI: http://localhost:8080/swagger"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:tokenizer_server_crow> --port 8080
        COMMENT "Запуск Crow сервера"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS tokenizer_server_crow
        USES_TERMINAL
    )

    # Цель для запуска Crow сервера с пользовательским портом
    add_custom_target(run_crow_server_custom
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Запуск Crow сервера на http://localhost:8000 (4 потока)"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:tokenizer_server_crow> --port 8000 --threads 4
        COMMENT "Запуск Crow сервера на порту 8000"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS tokenizer_server_crow
        USES_TERMINAL
    )
    
    message(STATUS "tokenizer_server_crow - REST API сервер с Swagger UI")
    message(STATUS "Для запуска: make run_crow_server")
endif()

# ============================================================================
# Simple Socket Server (Fallback)
# ============================================================================

# Проверяем наличие файла простого сервера
set(SIMPLE_SERVER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/server.cpp)
if(BUILD_SIMPLE_SERVER AND EXISTS ${SIMPLE_SERVER_SRC})
    message(STATUS "")
    message(STATUS "Сборка простого сервера на сокетах...")
    message(STATUS "------------------------------------------------------------")
    
    # Добавление исполняемого файла простого сервера
    add_executable(server ${SIMPLE_SERVER_SRC})

    # Настройка путей включения для простого сервера
    target_include_directories(server PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include    # Наши заголовочные файлы
    )

    # Линковка с библиотеками для простого сервера
    target_link_libraries(server PRIVATE 
        bpe_tokenizer       # Наша библиотека токенизатора
        Threads::Threads    # Многопоточность (для accept)
        pthread             # POSIX threads
        dl                  # Dynamic linking (иногда требуется)
    )

    # Установка свойств C++ стандарта
    set_target_properties(server PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        OUTPUT_NAME "bpe_server_simple"
        PREFIX ""
    )

    # Копирование моделей для простого сервера
    if(COPY_MODELS)
        add_custom_command(TARGET server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Копирование моделей для простого сервера..."
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_CURRENT_SOURCE_DIR}/../models
                ${CMAKE_CURRENT_BINARY_DIR}/models
            COMMENT "Копирование моделей для простого сервера"
        )
    endif()

    # Цель для запуска простого сервера
    add_custom_target(run_simple_server
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Запуск простого сервера на http://localhost:8080"
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND $<TARGET_FILE:server> --port 8080
        COMMENT "Запуск простого сервера"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS server
        USES_TERMINAL
    )
    
    message(STATUS "server - простой сокетный сервер (без внешних зависимостей)")
    message(STATUS "Для запуска: make run_simple_server")
elseif(BUILD_SIMPLE_SERVER)
    message(STATUS "Простой сервер (server.cpp) не найден, пропускаем сборку")
endif()

# ============================================================================
# Цели для запуска обоих серверов
# ============================================================================

# Цель для запуска всех серверов (если оба доступны)
set(ALL_SERVER_TARGETS)
if(TARGET tokenizer_server_crow)
    list(APPEND ALL_SERVER_TARGETS run_crow_server)
endif()
if(TARGET server)
    list(APPEND ALL_SERVER_TARGETS run_simple_server)
endif()

if(ALL_SERVER_TARGETS)
    add_custom_target(run_all_servers
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Запуск всех доступных серверов..."
        COMMAND ${CMAKE_COMMAND} -E echo "============================================================"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        DEPENDS ${ALL_SERVER_TARGETS}
        COMMENT "Запуск всех серверов"
    )
endif()

# Цель для очистки
add_custom_target(clean_servers
    COMMAND ${CMAKE_COMMAND} -E echo "Очистка артефактов серверов..."
    COMMAND ${CMAKE_COMMAND} -E remove -f
        ${CMAKE_CURRENT_BINARY_DIR}/bpe_server_crow
        ${CMAKE_CURRENT_BINARY_DIR}/bpe_server_simple
    COMMAND ${CMAKE_COMMAND} -E echo "Серверы очищены"
    COMMENT "Очистка артефактов серверов"
)

# ============================================================================
# Информация о сборке
# ============================================================================

message(STATUS "")
message(STATUS "============================================================")
message(STATUS "Итоговая информация о сборке:")
message(STATUS "============================================================")

if(TARGET tokenizer_server_crow)
    message(STATUS "")
    message(STATUS "Crow сервер:")
    message(STATUS "- Исполняемый файл: bpe_server_crow")
    message(STATUS "- Запуск:           make run_crow_server")
    message(STATUS "- Адрес:            http://localhost:8080")
    message(STATUS "- Документация:     http://localhost:8080/swagger")
    if(HAVE_ZLIB)
        message(STATUS "- Сжатие ответов: включено!")
    else()
        message(STATUS "- Сжатие ответов: отключено!")
    endif()
endif()

if(TARGET server)
    message(STATUS "")
    message(STATUS "Простой сервер:")
    message(STATUS "- Исполняемый файл: bpe_server_simple")
    message(STATUS "- Запуск:           make run_simple_server")
    message(STATUS "- Адрес:            http://localhost:8080")
endif()

if(NOT TARGET tokenizer_server_crow AND NOT TARGET server)
    message(STATUS "")
    message(STATUS "Ни один сервер не будет собран!")
    message(STATUS "Проверьте наличие исходных файлов и зависимостей.")
endif()

message(STATUS "")
message(STATUS "Дополнительные цели:")
message(STATUS "make run_crow_server_custom - Crow сервер на порту 8000")
if(TARGET run_all_servers)
    message(STATUS "make run_all_servers        - Запуск всех серверов")
endif()
message(STATUS "make clean_servers          - Очистка артефактов")
message(STATUS "============================================================\n")

# ============================================================================
# Дополнительные проверки и инструкции
# ============================================================================

# Проверка наличия библиотеки bpe_tokenizer
if(NOT EXISTS ${BPE_TOKENIZER_LIB})
    message(WARNING "
ВНИМАНИЕ: Библиотека bpe_tokenizer не найдена!
    
    Серверы не будут работать без собранной библиотеки токенизатора.
    
    Чтобы собрать библиотеку:
    1. Перейдите в корневую директорию проекта:
       cd ${CMAKE_CURRENT_SOURCE_DIR}/..
    
    2. Выполните сборку:
       mkdir -p build && cd build
       cmake ..
       make bpe_tokenizer
    
    3. Затем вернитесь сюда и повторите сборку серверов.
    ")
endif()

# Инструкция по установке CrowCpp (если не найден)
if(NOT CROW_FOUND AND BUILD_CROW_SERVER)
    message(STATUS "
Инструкция по установке CrowCpp:

   Способ 1 (рекомендуемый) - клонировать в third_party:
   ------------------------------------------------------------
   git clone https://github.com/CrowCpp/Crow.git ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/crow
   
   Способ 2 (через пакетный менеджер):
   ------------------------------------------------------------
   Ubuntu/Debian: sudo apt install libcrow-dev
   macOS:         brew install crow
   Arch Linux:    sudo pacman -S crow
   
   Способ 3 (указать путь вручную):
   ------------------------------------------------------------
   cmake .. -DCROW_INCLUDE_DIR=/путь/к/crow/include
    ")
endif()

# Инструкция по запуску
message(STATUS "
Быстрый старт:

   1. Собрать библиотеку токенизатора:
      cd .. && mkdir -p build && cd build
      cmake .. && make bpe_tokenizer
   
   2. Собрать серверы:
      cd ../web
      mkdir -p build && cd build
      cmake ..
      make
   
   3. Запустить сервер:
      make run_crow_server      # Для Crow сервера
      # Или
      make run_simple_server    # Для простого сервера
   
   4. Открыть в браузере:
      http://localhost:8080
    ")