/**
 * @file profiler.hpp
 * @brief Простой встроенный профайлер для BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Легковесная система профилирования для поиска узких мест
 *          и оптимизации производительности. Предоставляет:
 * 
 *          **Замер времени**            - точность до наносекунд
 *          **Статистика**               - min/max/avg время, количество вызовов
 *          **Автоматические отчеты**    - таблица с сортировкой по времени
 *          **RAII стиль**               - автоматическое начало/конец замера
 *          **Макросы**                  - удобное добавление профилирования
 *          **Потокобезопасность**       - через std::mutex
 * 
 *          **Типичное использование:**
 *          \code
 *          // Включить профилирование
 *          SimpleProfiler::setEnabled(true);
 *          SimpleProfiler::setOutputFile("profile.txt");
 *          
 *          // В функции, которую хотим измерить
 *          void slow_function() {
 *              PROFILE_FUNCTION();    // Автоматический замер
 *              // ... код ...
 *          }
 *          
 *          // В блоке кода
 *          {
 *              PROFILE_BLOCK("critical_section");
 *              // ... критический код ...
 *          }
 *          
 *          // В конце программы
 *          SimpleProfiler::printReport();    // Вывод в консоль
 *          SimpleProfiler::saveReport();     // Сохранение в файл
 *          \endcode
 * 
 * @note Создан специально для этапа оптимизации производительности
 * @warning Профилирование добавляет небольшие накладные расходы
 * 
 * @see FastBPETokenizer
 */

#pragma once

#include <chrono>
#include <string>
#include <map>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <climits>

namespace bpe {

// ======================================================================
// SimpleProfiler - основной класс профайлера
// ======================================================================

/**
 * @brief Простой профайлер для замера времени выполнения операций
 * 
 * Реализует два способа профилирования:
 * 1. Ручной: start() / stop()
 * 2. RAII: ScopedTimer (рекомендуемый)
 * 
 * **Как это работает:**
 * - Каждый поток имеет свой текущий замер (thread_local)
 * - Статистика собирается глобально с мьютексом
 * - Отчет сортируется по общему времени (самые медленные операции первыми)
 * 
 * **Производительность:**
 * - Накладные расходы:     ~50-100 нс на замер
 * - Потокобезопасность:    минимальные блокировки
 * - Память:                O(количество уникальных операций)
 * 
 * \include examples/profiler_example.cpp
 * Пример использования:
 * \code
 * #include "profiler.hpp"
 * 
 * int main() {
 *     bpe::SimpleProfiler::setEnabled(true);
 *     
 *     for (int i = 0; i < 1000; ++i) {
 *         PROFILE_BLOCK("fast_operation");
 *         // Быстрая операция
 *     }
 *     
 *     {
 *         PROFILE_BLOCK("slow_operation");
 *         std::this_thread::sleep_for(std::chrono::milliseconds(10));
 *     }
 *     
 *     bpe::SimpleProfiler::printReport();
 *     return 0;
 * }
 * \endcode
 */
class SimpleProfiler {
private:
    /**
     * @brief Структура со статистикой по операции
     */
    struct Stats {
        int64_t total_time_ns = 0;          ///< Суммарное время (наносекунды)
        int64_t min_time_ns = INT64_MAX;    ///< Минимальное время
        int64_t max_time_ns = 0;            ///< Максимальное время
        int call_count = 0;                 ///< Количество вызовов
        int current_depth = 0;              ///< Текущая глубина (для вложенных вызовов)
    };
    
    // Статические члены (inline для C++17)
    static inline std::map<std::string, Stats> stats_;                         ///< Глобальная статистика
    static inline std::mutex mutex_;                                           ///< Мьютекс для stats_
    static inline std::string current_output_file_ = "profiler_report.txt";    ///< Файл отчета
    static inline bool enabled_ = true;                                        ///< Флаг активности

public:
    // ==================== Основные методы ====================

    /**
     * @brief Начать замер операции
     * 
     * @param name Имя операции (должно быть уникальным)
     * 
     * **Важно:**    start() и stop() должны вызываться в паре
     *               и из одного потока. Для вложенных замеров 
     *               используйте ScopedTimer или PROFILE_BLOCK.
     * 
     * @see stop()
     */
    static void start(const std::string& name) {
        if (!enabled_) return;
        
        thread_local std::string current_operation;
        thread_local std::chrono::high_resolution_clock::time_point start_time;
        
        if (current_operation.empty()) {
            current_operation = name;
            start_time = std::chrono::high_resolution_clock::now();
        }
    }
    
    /**
     * @brief Закончить замер операции
     * 
     * @param name Имя операции (должно совпадать с start)
     * 
     * Автоматически обновляет статистику:
     * - total_time_ns += длительность
     * - min_time_ns = min(текущий, duration)
     * - max_time_ns = max(текущий, duration)
     * - call_count++
     */
    static void stop(const std::string& name) {
        if (!enabled_) return;
        
        thread_local std::string current_operation;
        thread_local std::chrono::high_resolution_clock::time_point start_time;
        
        if (!current_operation.empty() && current_operation == name) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_time - start_time).count();
            
            std::lock_guard<std::mutex> lock(mutex_);
            auto& s = stats_[name];
            s.total_time_ns += duration;
            s.min_time_ns = std::min(s.min_time_ns, duration);
            s.max_time_ns = std::max(s.max_time_ns, duration);
            s.call_count++;
            
            current_operation.clear();
        }
    }

    // ==================== ScopedTimer (RAII) ====================

    /**
     * @brief RAII класс для автоматического замера времени
     * 
     * Начинает замер в конструкторе и заканчивает в деструкторе.
     * Исключобезопасен и удобен для вложенных замеров.
     * 
     * \code
     * {
     *     SimpleProfiler::ScopedTimer timer("encode_batch");
     *      // код для замера
     * }    // автоматический stop()
     * \endcode
     */
    class ScopedTimer {
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
        
    public:
        /**
         * @brief Конструктор - начинаем замер
         * @param name Имя операции
         */
        explicit ScopedTimer(const std::string& name) : name_(name) {
            if (enabled_) {
                start_ = std::chrono::high_resolution_clock::now();
            }
        }
        
        /**
         * @brief Конструктор перемещения
         */
        ScopedTimer(ScopedTimer&& other) noexcept
            : name_(std::move(other.name_)), start_(other.start_) {
            // После перемещения other не должен вызывать stop в деструкторе
            other.name_.clear();
        }
        
        /**
         * @brief Деструктор - заканчиваем замер и обновляем статистику
         */
        ~ScopedTimer() {
            if (!enabled_ || name_.empty()) return;
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start_).count();
            
            std::lock_guard<std::mutex> lock(mutex_);
            auto& s = stats_[name_];
            s.total_time_ns += duration;
            s.min_time_ns = std::min(s.min_time_ns, duration);
            s.max_time_ns = std::max(s.max_time_ns, duration);
            s.call_count++;
        }
        
        // Запрещаем копирование
        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;
    };

    // ==================== Управление ====================

    /**
     * @brief Сбросить всю статистику
     * 
     * Очищает все накопленные данные. Полезно перед новым
     * циклом профилирования.
     */
    static void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
    }
    
    /**
     * @brief Включить/выключить профилирование
     * 
     * @param enabled true     - профилирование активно
     *                false    - все вызовы игнорируются
     * 
     * Можно отключать профилирование в production сборках
     * для нулевых накладных расходов.
     */
    static void setEnabled(bool enabled) {
        enabled_ = enabled;
    }
    
    /**
     * @brief Проверить, включено ли профилирование
     * @return true если профилирование активно
     */
    static bool isEnabled() {
        return enabled_;
    }
    
    /**
     * @brief Установить файл для отчета
     * 
     * @param filename Путь к файлу (по умолчанию "profiler_report.txt")
     * 
     * @see saveReport()
     */
    static void setOutputFile(const std::string& filename) {
        current_output_file_ = filename;
    }

    // ==================== Отчеты ====================

    /**
     * @brief Вывести отчет в консоль или поток
     * 
     * @param os Выходной поток (по умолчанию std::cout)
     * 
     * Формат отчета:
     * - Таблица с колонками: Операция, Вызовов, Всего (мс),
     *   Среднее (мкс), Мин (мкс), Макс (мкс), % времени
     * - Сортировка по общему времени (убывание)
     * - Итоговая статистика
     * 
     * \code
     * ================================================================================
     * ОТЧЕТ ПРОФИЛИРОВАНИЯ
     * ================================================================================
     * Операция           Вызовов     Всего (мс)    Среднее     Мин     Макс  % времени
     * --------------------------------------------------------------------------------
     * encode_batch            10       123.456       12.3      5.6     45.6      45.2%
     * decode                   5        67.890       13.6      8.9     23.4      24.8%
     * tokenize_word          100        45.678        0.5      0.3      1.2      16.7%
     * --------------------------------------------------------------------------------
     * Всего операций: 3
     * Общее время: 273.024 мс
     * ================================================================================
     * \endcode
     */
    static void printReport(std::ostream& os = std::cout) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (stats_.empty()) {
            os << "Нет данных профилирования\n";
            return;
        }
        
        // Сортируем по общему времени (убывание)
        std::vector<std::pair<std::string, Stats>> sorted(
            stats_.begin(), stats_.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second.total_time_ns > b.second.total_time_ns;
            });
        
        // Верхняя линия (80 символов =)
        os << "\n" << "==============================================================" << "\n";
        os << "ОТЧЕТ ПРОФИЛИРОВАНИЯ\n";
        os << "==============================================================" << "\n\n";

        // Заголовок таблицы (без изменений)
        os << std::left << std::setw(30) << "Операция"
        << std::right << std::setw(12) << "Вызовов"
        << std::setw(15) << "Всего (мс)"
        << std::setw(12) << "Среднее(мкс)"
        << std::setw(12) << "Мин(мкс)"
        << std::setw(12) << "Макс(мкс)"
        << std::setw(10) << "% времени" << "\n";

        // Линия-разделитель
        os << "------------------------------------------------------------" << "\n";
        
        int64_t total_time_all = 0;
        for (const auto& [name, s] : sorted) {
            total_time_all += s.total_time_ns;
        }
        
        for (const auto& [name, s] : sorted) {
            double total_ms = s.total_time_ns / 1'000'000.0;
            double avg_us = (s.call_count > 0) ? 
                (s.total_time_ns / 1000.0 / s.call_count) : 0.0;
            double min_us = s.min_time_ns / 1000.0;
            double max_us = s.max_time_ns / 1000.0;
            double percent = (total_time_all > 0) ? 
                (100.0 * s.total_time_ns / total_time_all) : 0.0;
            
            os << std::left << std::setw(30) << name.substr(0, 29)
               << std::right << std::setw(12) << s.call_count
               << std::fixed << std::setprecision(3)
               << std::setw(15) << total_ms
               << std::setprecision(1)
               << std::setw(12) << avg_us
               << std::setw(12) << min_us
               << std::setw(12) << max_us
               << std::setw(9) << std::setprecision(1) << percent << "%\n";
        }
        
        os << "------------------------------------------------------------" << "\n";
        os << "Всего операций: " << stats_.size() << "\n";
        os << "Общее время: " << std::fixed << std::setprecision(3)
           << total_time_all / 1'000'000.0 << " мс\n";
        os << "==============================================================" << "\n\n";
    }
    
    /**
     * @brief Сохранить отчет в файл
     * 
     * Использует файл, установленный через setOutputFile()
     * 
     * @see setOutputFile()
     */
    static void saveReport() {
        std::ofstream file(current_output_file_);
        if (file.is_open()) {
            printReport(file);
            std::cout << "Отчет сохранен в: " << current_output_file_ << "\n";
        } else {
            std::cerr << "Ошибка: не удалось открыть файл " 
                      << current_output_file_ << " для записи\n";
        }
    }
    
    /**
     * @brief Получить количество зарегистрированных операций
     * @return size_t Количество уникальных операций
     */
    static size_t getOperationCount() {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_.size();
    }
    
    /**
     * @brief Получить общее время выполнения всех операций
     * @return int64_t Общее время в наносекундах
     */
    static int64_t getTotalTime() {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t total = 0;
        for (const auto& [_, s] : stats_) {
            total += s.total_time_ns;
        }
        return total;
    }
};

// ======================================================================
// Удобные макросы для использования
// ======================================================================

/**
 * @def PROFILE_BLOCK(name)
 * @brief Замерить время выполнения блока кода
 * 
 * Использование:
 * \code
 * {
 *     PROFILE_BLOCK("encode_batch");    // Код для замера
 * }    // Автоматическое завершение
 * \endcode
 * 
 * Создает переменную с уникальным именем, используя __LINE__.
 */
#define PROFILE_BLOCK(name) \
    bpe::SimpleProfiler::ScopedTimer CONCAT(_profiler, __LINE__)(name)

/**
 * @def PROFILE_FUNCTION()
 * @brief Замерить время выполнения функции
 * 
 * Использование:
 * \code
 * void myFunction() {
 *     PROFILE_FUNCTION();    // Использует __FUNCTION__ как имя
 *     // Код функции
 * }
 * \endcode
 * 
 * Автоматически использует имя функции как название операции.
 */
#define PROFILE_FUNCTION() \
    bpe::SimpleProfiler::ScopedTimer _profiler(__FUNCTION__)

// Помощник для конкатенации (внутренний макрос)
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

// Версия с условной компиляцией
#ifdef ENABLE_PROFILING
    #define IF_PROFILING(x) x
#else
    #define IF_PROFILING(x)
#endif

} // namespace bpe

/**
 * @example examples/profiler_example.cpp
 * Полный пример использования профайлера:
 * 
 * @code
 * #include "profiler.hpp"
 * #include <thread>
 * #include <vector>
 * 
 * void slow_function() {
 *     PROFILE_FUNCTION();
 *     std::this_thread::sleep_for(std::chrono::milliseconds(10));
 * }
 * 
 * void fast_function() {
 *     PROFILE_FUNCTION();
 *     volatile int sum = 0;
 *     for (int i = 0; i < 10000; ++i) sum += i;
 * }
 * 
 * int main() {
 *     bpe::SimpleProfiler::setEnabled(true);
 *     bpe::SimpleProfiler::setOutputFile("profile_results.txt");
 *     
 *     std::cout << "Профилирование " 
 *               << (bpe::SimpleProfiler::isEnabled() ? "включено" : "отключено") 
 *               << "\n";
 *     
 *     for (int i = 0; i < 5; ++i) {
 *         slow_function();
 *     }
 *     
 *     for (int i = 0; i < 100; ++i) {
 *         fast_function();
 *     }
 *     
 *     {
 *         PROFILE_BLOCK("mixed_work");
 *         for (int i = 0; i < 3; ++i) {
 *             slow_function();
 *         }
 *         for (int i = 0; i < 50; ++i) {
 *             fast_function();
 *         }
 *     }
 *     
 *     std::cout << "Всего операций: " << bpe::SimpleProfiler::getOperationCount() << "\n";
 *     std::cout << "Общее время: " << bpe::SimpleProfiler::getTotalTime() / 1e6 << " мс\n\n";
 *     
 *     bpe::SimpleProfiler::printReport();
 *     bpe::SimpleProfiler::saveReport();
 *     
 *     return 0;
 * }
 * @endcode
 */