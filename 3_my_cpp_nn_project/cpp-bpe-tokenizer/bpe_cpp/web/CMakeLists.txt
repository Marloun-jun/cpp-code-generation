# ======================================================================
# web/CMakeLists.txt - Веб-сервер для BPE токенизатора
# ======================================================================
#
# @file web/CMakeLists.txt
# @brief Конфигурация сборки веб-сервера на базе CrowCpp
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
# ======================================================================

# ======================================================================
# Сервер с Crow
# ======================================================================

# Путь к локальному Crow
set(CROW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/crow/include)

if(EXISTS ${CROW_INCLUDE_DIR}/crow/app.h)
    message(STATUS "✅ Crow найден в: ${CROW_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "❌ Crow не найден! Установите: git clone https://github.com/CrowCpp/Crow.git ../../third_party/crow")
endif()

# Поиск zlib
find_package(ZLIB REQUIRED)

# ======================================================================
# Добавление исполняемого файла сервера (ТОЛЬКО ОДИН РАЗ!)
# ======================================================================
add_executable(tokenizer_server_crow server_crow.cpp)

# ======================================================================
# Настройка путей включения
# ======================================================================
target_include_directories(tokenizer_server_crow PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CROW_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIRS}
)

# ======================================================================
# Линковка с библиотеками
# ======================================================================
target_link_libraries(tokenizer_server_crow PRIVATE 
    bpe_tokenizer
    Threads::Threads
    ZLIB::ZLIB
)

# ======================================================================
# Определения для компиляции (ОПЦИОНАЛЬНО - можно закомментировать если нет zlib)
# ======================================================================
target_compile_definitions(tokenizer_server_crow PRIVATE
    CROW_ENABLE_COMPRESSION
)

# ======================================================================
# Копирование моделей в директорию сборки
# ======================================================================
add_custom_command(TARGET tokenizer_server_crow POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/../models
        ${CMAKE_CURRENT_BINARY_DIR}/models
    COMMENT "📋 Копирование моделей в директорию сборки"
)

# ======================================================================
# Запуск сервера (пользовательская цель)
# ======================================================================
add_custom_target(run_web_server
    COMMAND $<TARGET_FILE:tokenizer_server_crow>
    COMMENT "🌐 Запуск веб-сервера на http://localhost:8080"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tokenizer_server_crow
)

# ======================================================================
# Информация о сборке
# ======================================================================
message(STATUS "")
message(STATUS "🌐 Веб-сервер:")
message(STATUS "   • tokenizer_server_crow - REST API сервер")
message(STATUS "   • Для запуска: make run_web_server")
message(STATUS "   • Адрес: http://localhost:8080")
message(STATUS "========================================\n")