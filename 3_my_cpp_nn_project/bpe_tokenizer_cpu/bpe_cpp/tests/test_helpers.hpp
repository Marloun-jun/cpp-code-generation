// tests/test_helpers.hpp
#pragma once
#include <string>
#include <filesystem>
#include <fstream>      // для std::ofstream
#include <cstdio>

namespace fs = std::filesystem;

namespace bpe_test {

/**
 * @brief Безопасное удаление файла (игнорирует ошибки, если файл не существует)
 */
inline void safe_remove(const std::string& path) {
    std::error_code ec;
    fs::remove(path, ec);
}

/**
 * @brief Создать временный файл с содержимым
 */
inline void create_temp_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << content;
        file.close();
    }
}

/**
 * @brief Проверить существование файла
 */
inline bool file_exists(const std::string& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

} // namespace bpe_test