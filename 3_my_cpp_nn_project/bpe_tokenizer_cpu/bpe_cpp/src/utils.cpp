/**
 * @file utils.cpp
 * @brief Реализация утилитарных функций для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Реализация всех вспомогательных функций, объявленных в utils.hpp.
 *          Каждая функция тщательно оптимизирована и протестирована.
 * 
 *          **Ключевые оптимизации:**
 *          - Чтение файлов      - reserve и прямой доступ к буферу
 *          - UTF-8 валидация    - Побитовые операции, минимум ветвлений
 *          - Экранирование      - Предварительное резервирование памяти
 *          - Строковые операции - Избегание лишних копирований
 * 
 *          **Обработка ошибок:**
 *          - Чтение файлов   - Исключения с детальным описанием
 *          - Запись файлов   - Возврат bool с диагностикой в stderr
 *          - UTF-8 валидация - Строгая проверка всех граничных случаев
 * 
 * @see utils.hpp для полного описания интерфейса
 */

#include "utils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace bpe {
namespace utils {

// ============================================================================
// Работа с файлами
// ============================================================================

std::string read_file(const std::string& path) {
    // Открываем файл в бинарном режиме и сразу переходим в конец
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("[utils] Не удалось открыть файл: " + path);
    }

    // Получаем размер файла
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size < 0) {
        throw std::runtime_error("[utils] Не удалось определить размер файла: " + path);
    }

    // Резервируем память и читаем за один вызов
    std::string buffer(static_cast<size_t>(size), '\0');
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("[utils] Ошибка чтения файла: " + path);
    }

    return buffer;
}

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "[utils] Ошибка: не удалось открыть файл для записи: " << path << std::endl;
        return false;
    }

    file.write(content.data(), static_cast<std::streamsize>(content.size()));
    
    if (!file.good()) {
        std::cerr << "[utils] Ошибка при записи в файл: " << path << std::endl;
        return false;
    }

    return true;
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

// ============================================================================
// Форматирование размеров
// ============================================================================

std::string format_size(size_t bytes) {
    static const char* units[] = {"байт", "КБ", "МБ", "ГБ", "ТБ"};
    static constexpr int MAX_UNIT = 4;    // Индекс последней единицы (ТБ)

    int unit = 0;
    double size = static_cast<double>(bytes);

    // Последовательно делим на 1024, пока не получим число < 1024
    while (size >= 1024.0 && unit < MAX_UNIT) {
        size /= 1024.0;
        ++unit;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit];
    return ss.str();
}

// ============================================================================
// Валидация UTF-8 (строгая проверка по спецификации)
// ============================================================================

bool is_valid_utf8(const std::string& str) {
    if (str.empty()) {
        return true;
    }

    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str.data());
    size_t len = str.length();
    size_t i = 0;

    while (i < len) {
        // 1-байтовые символы (ASCII)
        // Диапазон: 0x00 - 0x7F
        if (bytes[i] <= 0x7F) {
            i += 1;
            continue;
        }

        // 2-байтовые символы
        // Диапазон: 0xC2 - 0xDF, затем 0x80 - 0xBF
        if (bytes[i] >= 0xC2 && bytes[i] <= 0xDF) {
            // Проверяем наличие второго байта
            if (i + 1 >= len) return false;
            // Второй байт должен быть 10xxxxxx
            if ((bytes[i + 1] & 0xC0) != 0x80) return false;
            i += 2;
            continue;
        }

        // 3-байтовые символы
        // Диапазон: 0xE0 - 0xEF, затем два байта 10xxxxxx
        if (bytes[i] >= 0xE0 && bytes[i] <= 0xEF) {
            // Проверяем наличие второго и третьего байтов
            if (i + 2 >= len) return false;
            // Оба следующих байта должны быть 10xxxxxx
            if ((bytes[i + 1] & 0xC0) != 0x80 || 
                (bytes[i + 2] & 0xC0) != 0x80) {
                return false;
            }

            // Специальные проверки для граничных случаев
            if (bytes[i] == 0xE0 && bytes[i + 1] < 0xA0) return false;    // Overlong
            if (bytes[i] == 0xED && bytes[i + 1] > 0x9F) return false;    // Surrogate

            i += 3;
            continue;
        }

        // 4-байтовые символы
        // Диапазон: 0xF0 - 0xF4, затем три байта 10xxxxxx
        if (bytes[i] >= 0xF0 && bytes[i] <= 0xF4) {
            // Проверяем наличие второго, третьего и четвертого байтов
            if (i + 3 >= len) return false;
            // Все следующие байты должны быть 10xxxxxx
            if ((bytes[i + 1] & 0xC0) != 0x80 || 
                (bytes[i + 2] & 0xC0) != 0x80 || 
                (bytes[i + 3] & 0xC0) != 0x80) {
                return false;
            }

            // Специальные проверки для граничных случаев
            if (bytes[i] == 0xF0 && bytes[i + 1] < 0x90) return false;    // Overlong
            if (bytes[i] == 0xF4 && bytes[i + 1] > 0x8F) return false;    // > U+10FFFF

            i += 4;
            continue;
        }

        // Если дошли сюда - невалидный первый байт
        return false;
    }

    return true;
}

// ============================================================================
// Экранирование спецсимволов для безопасного вывода
// ============================================================================

std::string escape_string(const std::string& str) {
    // Предварительно резервируем память (эвристика: в 2 раза больше)
    std::string escaped;
    escaped.reserve(str.size() * 2);

    for (unsigned char c : str) {
        switch (c) {
            // Стандартные escape-последовательности
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            case '\\': escaped += "\\\\"; break;
            case '"':  escaped += "\\\""; break;
            case '\'': escaped += "\\'"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\v': escaped += "\\v"; break;
            case '\a': escaped += "\\a"; break;
            
            default:
                // Непечатные символы (коды 0x00-0x1F и 0x7F)
                if (c < 0x20 || c == 0x7F) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\x%02x", c);
                    escaped += buf;
                } else {
                    // Печатные символы оставляем как есть
                    escaped += static_cast<char>(c);
                }
        }
    }

    return escaped;
}

// ============================================================================
// Преобразование регистра (только ASCII)
// ============================================================================

std::string to_lower(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        // Преобразуем только A-Z (0x41-0x5A)
        if (c >= 'A' && c <= 'Z') {
            c += ('a' - 'A');    // Разница между 'a' и 'A' = 32
        }
    }
    return result;
}

std::string to_upper(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        // Преобразуем только a-z (0x61-0x7A)
        if (c >= 'a' && c <= 'z') {
            c -= ('a' - 'A');
        }
    }
    return result;
}

// ============================================================================
// Обрезка пробелов
// ============================================================================

std::string trim(const std::string& str) {
    // Находим первый непробельный символ
    size_t start = str.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return "";    // Строка состоит только из пробелов
    }

    // Находим последний непробельный символ
    size_t end = str.find_last_not_of(" \t\n\r\f\v");
    
    // Возвращаем подстроку без пробелов по краям
    return str.substr(start, end - start + 1);
}

// ============================================================================
// Разделение строк
// ============================================================================

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    // std::getline возвращает строку до разделителя
    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }

    // Обратите внимание: пустые части сохраняются!
    // Например, "a,,c" даст {"a", "", "c"}

    return result;
}

// ============================================================================
// Замена подстрок
// ============================================================================

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return str;    // Замена пустой строки не имеет смысла
    }

    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();    // Продолжаем поиск после замены
    }
    return str;
}

// ============================================================================
// Проверка префиксов/суффиксов
// ============================================================================

bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && 
           str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ============================================================================
// Timer - реализация высокоточного таймера
// ============================================================================

Timer::Timer() {
    reset();    // Автоматически запускаем таймер при создании
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

}    // namespace utils
}    // namespace bpe