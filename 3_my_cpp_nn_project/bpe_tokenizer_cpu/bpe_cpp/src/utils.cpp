/**
 * @file utils.cpp
 * @brief Реализация вспомогательных утилит для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Этот файл содержит реализацию набора утилитарных функций,
 *          используемых во всем проекте. Все функции тщательно
 *          оптимизированы и протестированы.
 * 
 *          **Основные категории:**
 * 
 *          1) **Работа с файлами**
 *             - Чтение всего файла в строку (с проверкой ошибок)
 *             - Запись строки в файл (бинарный режим для UTF-8)
 *             - Обработка исключений при ошибках
 * 
 *          2) **Форматирование размеров**
 *             - Конвертация байт в человекочитаемый формат (байт, КБ, МБ, ГБ, ТБ)
 *             - Автоматический выбор подходящей единицы
 *             - Форматирование с двумя знаками после запятой
 * 
 *          3) **Валидация UTF-8**
 *             - Полная проверка согласно спецификации UTF-8
 *             - Определение всех корректных последовательностей (1-4 байта)
 *             - Защита от overlong encoding и surrogate символов
 *             - Проверка границ Unicode (макс. U+10FFFF)
 * 
 *          4) **Экранирование спецсимволов**
 *             - Замена управляющих символов на escape-последовательности
 *             - Экранирование кавычек и обратных слешей
 *             - Преобразование непечатных символов в \xXX формат
 * 
 *          5) **Обработка строк**
 *             - Преобразование регистра (ASCII)
 *             - Обрезка пробелов
 *             - Разделение строк
 *             - Замена подстрок
 *             - Проверка префиксов/суффиксов
 * 
 * @note Все функции потокобезопасны и не используют глобальное состояние
 * @see utils.hpp
 */
/**
 * @file utils.cpp
 * @brief Реализация вспомогательных утилит для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 */

#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <chrono>

namespace bpe {
namespace utils {

// ======================================================================
// Работа с файлами
// ======================================================================

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + path);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (size < 0) {
        throw std::runtime_error("Не удалось определить размер файла: " + path);
    }
    
    std::string buffer(static_cast<size_t>(size), '\0');
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Ошибка чтения файла: " + path);
    }
    
    return buffer;
}

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        return false;
    }
    
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    return file.good();
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

// ======================================================================
// Форматирование размеров
// ======================================================================

std::string format_size(size_t bytes) {
    static const char* units[] = {"байт", "КБ", "МБ", "ГБ", "ТБ"};
    static constexpr int MAX_UNIT = 4;
    
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit < MAX_UNIT) {
        size /= 1024.0;
        ++unit;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return ss.str();
}

// ======================================================================
// Валидация UTF-8
// ======================================================================

bool is_valid_utf8(const std::string& str) {
    if (str.empty()) {
        return true;
    }
    
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    size_t len = str.length();
    size_t i = 0;
    
    while (i < len) {
        if (bytes[i] <= 0x7F) {
            i += 1;
            continue;
        }
        
        if (bytes[i] >= 0xC2 && bytes[i] <= 0xDF) {
            if (i + 1 >= len) return false;
            if ((bytes[i + 1] & 0xC0) != 0x80) return false;
            i += 2;
            continue;
        }
        
        if (bytes[i] >= 0xE0 && bytes[i] <= 0xEF) {
            if (i + 2 >= len) return false;
            if ((bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80) return false;
            
            if (bytes[i] == 0xE0 && bytes[i + 1] < 0xA0) return false;
            if (bytes[i] == 0xED && bytes[i + 1] > 0x9F) return false;
            
            i += 3;
            continue;
        }
        
        if (bytes[i] >= 0xF0 && bytes[i] <= 0xF4) {
            if (i + 3 >= len) return false;
            if ((bytes[i + 1] & 0xC0) != 0x80 || 
                (bytes[i + 2] & 0xC0) != 0x80 || 
                (bytes[i + 3] & 0xC0) != 0x80) return false;
            
            if (bytes[i] == 0xF0 && bytes[i + 1] < 0x90) return false;
            if (bytes[i] == 0xF4 && bytes[i + 1] > 0x8F) return false;
            
            i += 4;
            continue;
        }
        
        return false;
    }
    
    return true;
}

// ======================================================================
// Экранирование спецсимволов
// ======================================================================

std::string escape_string(const std::string& str) {
    std::string escaped;
    escaped.reserve(str.size() * 2);
    
    for (unsigned char c : str) {
        switch (c) {
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\'': escaped += "\\'"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\v': escaped += "\\v"; break;
            case '\a': escaped += "\\a"; break;
            default:
                if (c < 0x20 || c == 0x7F) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\x%02x", c);
                    escaped += buf;
                } else {
                    escaped += static_cast<char>(c);
                }
        }
    }
    
    return escaped;
}

// ======================================================================
// Обработка строк
// ======================================================================

std::string to_lower(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        if (c >= 'A' && c <= 'Z') {
            c += ('a' - 'A');
        }
    }
    return result;
}

std::string to_upper(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        if (c >= 'a' && c <= 'z') {
            c -= ('a' - 'A');
        }
    }
    return result;
}

std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(start, end - start + 1);
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }
    
    return result;
}

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;
    
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ======================================================================
// Timer - реализация
// ======================================================================

Timer::Timer() {
    reset();
}

void Timer::reset() {
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start_).count();
}

double Timer::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
}

double Timer::elapsed_us() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(now - start_).count();
}

int64_t Timer::elapsed_ns() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_).count();
}

} // namespace utils
} // namespace bpe