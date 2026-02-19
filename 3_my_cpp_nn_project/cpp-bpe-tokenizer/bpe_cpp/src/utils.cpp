/**
 * @file utils.cpp
 * @brief Реализация вспомогательных утилит для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Реализация функций для:
 *          - Работы с файлами (чтение/запись)
 *          - Форматирования размеров
 *          - Валидации UTF-8 строк
 *          - Экранирования спецсимволов
 * 
 * @note Все функции потокобезопасны и не используют глобальное состояние
 * @see utils.hpp
 */

#include "utils.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace bpe {
namespace utils {

// ==================== Работа с файлами ====================

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + path);
    }
    
    // Определяем размер файла
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (size < 0) {
        throw std::runtime_error("Не удалось определить размер файла: " + path);
    }
    
    // Читаем весь файл
    std::string buffer(static_cast<size_t>(size), '\0');
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Ошибка чтения файла: " + path);
    }
    
    return buffer;
}

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    return file.good();
}

// ==================== Форматирование ====================

std::string format_size(size_t bytes) {
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    static constexpr int MAX_UNIT = 4;  // Индекс для TB
    
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

// ==================== Работа с UTF-8 ====================

bool is_valid_utf8(const std::string& str) {
    if (str.empty()) {
        return true;
    }
    
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    size_t len = str.length();
    size_t i = 0;
    
    while (i < len) {
        // ASCII (0xxxxxxx)
        if (bytes[i] <= 0x7F) {
            i += 1;
            continue;
        }
        
        // 2-byte sequence (110xxxxx 10xxxxxx)
        if (bytes[i] >= 0xC2 && bytes[i] <= 0xDF) {
            if (i + 1 >= len) {
                return false;  // Недостаточно байт
            }
            if ((bytes[i + 1] & 0xC0) != 0x80) {
                return false;  // Неверный continuation byte
            }
            i += 2;
            continue;
        }
        
        // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
        if (bytes[i] >= 0xE0 && bytes[i] <= 0xEF) {
            if (i + 2 >= len) {
                return false;  // Недостаточно байт
            }
            
            // Проверка continuation bytes
            if ((bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80) {
                return false;
            }
            
            // Проверка на overlong encoding
            if (bytes[i] == 0xE0 && bytes[i + 1] < 0xA0) {
                return false;  // Минимальный valid 3-byte символ
            }
            if (bytes[i] == 0xED && bytes[i + 1] > 0x9F) {
                return false;  // Запрещенные surrogate символы
            }
            
            i += 3;
            continue;
        }
        
        // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
        if (bytes[i] >= 0xF0 && bytes[i] <= 0xF4) {
            if (i + 3 >= len) {
                return false;  // Недостаточно байт
            }
            
            // Проверка continuation bytes
            if ((bytes[i + 1] & 0xC0) != 0x80 || 
                (bytes[i + 2] & 0xC0) != 0x80 || 
                (bytes[i + 3] & 0xC0) != 0x80) {
                return false;
            }
            
            // Проверка на overlong encoding и допустимый диапазон Unicode
            if (bytes[i] == 0xF0 && bytes[i + 1] < 0x90) {
                return false;  // Минимальный valid 4-byte символ
            }
            if (bytes[i] == 0xF4 && bytes[i + 1] > 0x8F) {
                return false;  // Выход за пределы Unicode (макс U+10FFFF)
            }
            
            i += 4;
            continue;
        }
        
        // Невалидный UTF-8 (некорректный стартовый байт)
        return false;
    }
    
    return true;  // Все проверки пройдены
}

std::string escape_string(const std::string& str) {
    std::string escaped;
    escaped.reserve(str.size() * 2);  // Примерная оценка для экранирования
    
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
            default:
                if (c < 0x20 || c == 0x7F) {
                    // Управляющие символы и DEL
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

} // namespace utils
} // namespace bpe