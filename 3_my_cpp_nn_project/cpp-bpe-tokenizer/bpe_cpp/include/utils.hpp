/**
 * @file utils.hpp
 * @brief Вспомогательные утилиты для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Набор утилитарных функций и классов, используемых во всем проекте:
 *          - Timer для измерения производительности
 *          - Работа с файлами (чтение/запись)
 *          - Форматирование размеров
 *          - Валидация и экранирование UTF-8 строк
 * 
 * @note Все функции находятся в пространстве имен bpe::utils
 * @see FastBPETokenizer
 */

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace bpe {
namespace utils {

/**
 * @brief Класс для измерения времени выполнения
 * 
 * Использует std::chrono::high_resolution_clock для максимальной точности.
 * Полезен для профилирования и сбора статистики производительности.
 * 
 * Пример использования:
 * @code
 * Timer timer;
 * // ... какой-то код ...
 * std::cout << "Время выполнения: " << timer.elapsed() << " сек" << std::endl;
 * @endcode
 */
class Timer {
public:
    /**
     * @brief Конструктор автоматически запускает таймер
     */
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    /**
     * @brief Получить прошедшее время с момента запуска или последнего сброса
     * @return Время в секундах (как double)
     */
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(now - start_).count();
    }
    
    /**
     * @brief Сбросить таймер (установить начальное время на текущий момент)
     */
    void reset() { 
        start_ = std::chrono::high_resolution_clock::now(); 
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;  ///< Время запуска таймера
};

// ==================== Работа с файлами ====================

/**
 * @brief Прочитать весь файл в строку
 * @param path Путь к файлу
 * @return Содержимое файла в виде строки
 * @throws std::runtime_error если файл не может быть открыт
 * 
 * @note Файл открывается в бинарном режиме для корректного чтения UTF-8
 */
std::string read_file(const std::string& path);

/**
 * @brief Записать строку в файл
 * @param path Путь к файлу
 * @param content Содержимое для записи
 * @return true при успешной записи, false при ошибке
 * 
 * @note Файл будет перезаписан, если существует
 */
bool write_file(const std::string& path, const std::string& content);

// ==================== Форматирование ====================

/**
 * @brief Преобразовать размер в байтах в человекочитаемый формат
 * @param bytes Размер в байтах
 * @return Строка вида "1.23 KB", "45.67 MB", и т.д.
 * 
 * @note Поддерживает единицы: B, KB, MB, GB, TB
 */
std::string format_size(size_t bytes);

// ==================== Работа с UTF-8 ====================

/**
 * @brief Проверить, является ли строка валидной UTF-8
 * @param str Строка для проверки
 * @return true если строка содержит валидную UTF-8 последовательность
 * 
 * @details Проверяет:
 *          - Корректность многобайтовых последовательностей
 *          - Запрещенные символы (overlong encoding, surrogate pairs)
 *          - Выход за допустимые диапазоны Unicode
 */
bool is_valid_utf8(const std::string& str);

/**
 * @brief Экранировать спецсимволы для безопасного вывода
 * @param str Исходная строка
 * @return Строка с экранированными символами
 * 
 * @details Заменяет:
 *          - Управляющие символы на \n, \t, \r и т.д.
 *          - Непечатные символы на \xXX
 *          - Символы вне ASCII на их UTF-8 представление
 * 
 * @example escape_string("Hello\nWorld") -> "Hello\\nWorld"
 */
std::string escape_string(const std::string& str);

} // namespace utils
} // namespace bpe

/**
 * @example examples/timer_example.cpp
 * Пример использования Timer:
 * @code
 * #include "utils.hpp"
 * 
 * bpe::utils::Timer timer;
 * std::vector<int> data(1000000);
 * // заполнение данных...
 * std::cout << "Время инициализации: " << timer.elapsed() << " сек" << std::endl;
 * 
 * timer.reset();
 * std::sort(data.begin(), data.end());
 * std::cout << "Время сортировки: " << timer.elapsed() << " сек" << std::endl;
 * @endcode
 */