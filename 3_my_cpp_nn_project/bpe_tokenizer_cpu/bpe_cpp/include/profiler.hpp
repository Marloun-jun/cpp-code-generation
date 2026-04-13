/**
 * @file profiler.hpp
 * @brief Легковесный профайлер для измерения производительности
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.7.0
 * 
 * @details Встроенная система профилирования для поиска узких мест
 *          и оптимизации производительности токенизатора.
 * 
 *          **Возможности:**
 *          - Замер времени с наносекундной точностью
 *          - Статистика (min/max/avg, количество вызовов)
 *          - Автоматические отчеты с сортировкой
 *          - RAII стиль через ScopedTimer
 *          - Удобные макросы PROFILE_FUNCTION/PROFILE_BLOCK
 *          - Потокобезопасность через std::mutex
 *          - Поддержка вложенных вызовов через стек
 *          - Нулевые накладные расходы при отключении
 * 
 *          **Архитектура:**
 *          - Meyers singleton для глобальных данных (единый экземпляр)
 *          - thread_local стек для поддержки вложенных вызовов
 *          - RAII для автоматического замера
 * 
 *          **Типовой цикл оптимизации:**
 *          @code
 *          // 1. Включаем профилирование
 *          SimpleProfiler::setEnabled(true);
 *          SimpleProfiler::setOutputFile("profile.txt");
 *          
 *          // 2. Запускаем тесты
 *          run_benchmarks();
 *          
 *          // 3. Анализируем отчет
 *          SimpleProfiler::printReport();
 *          
 *          // 4. Оптимизируем самые медленные операции
 *          // 5. Повторяем
 *          @endcode
 * 
 *          **Накладные расходы:**
 *          - Включено  - ~50-100 нс на замер
 *          - Отключено - Проверка одного bool (1-2 такта)
 * 
 * @see FastBPETokenizer (использует этот профайлер)
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace bpe {

// ============================================================================
// SimpleProfiler - основной класс профайлера
// ============================================================================

/**
 * @brief Простой встроенный профайлер
 * 
 * **Архитектура:**
 * - Meyers singleton для глобальных данных (stats, mutex, enabled)
 * - thread_local стек для поддержки вложенных вызовов
 * - RAII для автоматического замера
 * 
 * **Пример использования:**
 * @code
 * // Включение/выключение
 * SimpleProfiler::setEnabled(true);
 * 
 * // Вариант 1: Ручной замер (с поддержкой вложенности)
 * SimpleProfiler::start("encode");
 * // ... код ...
 * SimpleProfiler::start("tokenize_word");    // Вложенный вызов
 * // ... код ...
 * SimpleProfiler::stop("tokenize_word");
 * // ... код ...
 * SimpleProfiler::stop("encode");
 * 
 * // Вариант 2: RAII (рекомендуется)
 * {
 *     SimpleProfiler::ScopedTimer timer("decode");
 *     // ... код ...
 * } // автоматический stop()
 * 
 * // Вариант 3: Макросы (еще удобнее)
 * void my_function() {
 *     PROFILE_FUNCTION();    // Использует имя функции
 *     // ... код ...
 * }
 * 
 * // Вывод результатов
 * SimpleProfiler::printReport();
 * SimpleProfiler::saveReport();
 * @endcode
 */
class SimpleProfiler {
private:
    // ========================================================================
    // Внутренние структуры данных
    // ========================================================================

    /**
     * @brief Статистика по одной операции
     */
    struct Stats {
        int64_t total_time_ns = 0;          ///< Суммарное время (нс)
        int64_t min_time_ns = INT64_MAX;    ///< Минимальное время (нс)
        int64_t max_time_ns = 0;            ///< Максимальное время (нс)
        int call_count = 0;                 ///< Количество вызовов
    };

    /**
     * @brief Элемент стека вызовов
     */
    struct StackFrame {
        std::string name;                                             ///< Имя операции
        std::chrono::high_resolution_clock::time_point start_time;    ///< Время начала
    };

    /**
     * @brief Meyers singleton для глобальных данных
     * 
     * Гарантирует единственность данных во всей программе.
     * Инициализация потокобезопасна (C++11 и выше).
     */
    struct GlobalData {
        std::map<std::string, Stats> stats;                 ///< Глобальная статистика
        std::mutex mutex;                                   ///< Мьютекс для защиты stats
        std::string output_file = "profiler_report.txt";    ///< Файл для отчета
        bool enabled = false;                               ///< Флаг активности профайлера

        /**
         * @brief Получить единственный экземпляр глобальных данных
         * @return GlobalData& Ссылка на статический объект
         */
        static GlobalData& instance() {
            static GlobalData data;
            return data;
        }
    };

    /**
     * @brief Получить thread_local стек вызовов
     * @return std::vector<StackFrame>& Ссылка на стек текущего потока
     */
    static std::vector<StackFrame>& get_call_stack() {
        thread_local std::vector<StackFrame> call_stack;
        return call_stack;
    }

public:
    // ========================================================================
    // Ручное управление замерами (с поддержкой вложенности)
    // ========================================================================

    /**
     * @brief Начать замер операции
     * @param name Имя операции (должно быть уникальным)
     * 
     * @note Поддерживает вложенные вызовы. Каждый start() должен иметь
     *       соответствующий stop() с тем же именем в обратном порядке.
     */
    static void start(const std::string& name) {
        auto& data = GlobalData::instance();
        if (!data.enabled) {
            return;
        }

        auto& call_stack = get_call_stack();
        call_stack.push_back({name, std::chrono::high_resolution_clock::now()});
    }

    /**
     * @brief Закончить замер операции
     * @param name Имя операции (должно совпадать с последним start)
     * 
     * @note Завершает последний запущенный замер с указанным именем.
     *       Если имена не совпадают, выводит предупреждение (в debug режиме).
     */
    static void stop(const std::string& name) {
        auto& data = GlobalData::instance();
        if (!data.enabled) {
            return;
        }

        auto& call_stack = get_call_stack();

        if (call_stack.empty()) {
#ifndef NDEBUG
            std::cerr << "[PROFILER] Внимание: stop(\"" << name
                      << "\") вызывается с пустым стеком!\n";
#endif
            return;
        }

        const auto& frame = call_stack.back();
        if (frame.name != name) {
#ifndef NDEBUG
            std::cerr << "[PROFILER] Внимание: stop(\"" << name
                      << "\") не соответствует start(\"" << frame.name
                      << "\") на вершине стека! Глубина: " << call_stack.size() << "\n";
#endif
            return;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - frame.start_time).count();

        {
            std::lock_guard<std::mutex> lock(data.mutex);
            auto& s = data.stats[name];
            s.total_time_ns += duration;
            s.min_time_ns = std::min(s.min_time_ns, duration);
            s.max_time_ns = std::max(s.max_time_ns, duration);
            s.call_count++;
        }

        call_stack.pop_back();
    }

    // ========================================================================
    // RAII ScopedTimer
    // ========================================================================

    /**
     * @brief RAII класс для автоматического замера времени
     * 
     * Начинает замер в конструкторе, заканчивает в деструкторе.
     * Исключениебезопасен и поддерживает перемещение.
     * 
     * @code
     * {
     *     SimpleProfiler::ScopedTimer timer("encode_batch");
     *     process_batch();    // Время этой функции будет измерено
     * }
     * @endcode
     */
    class ScopedTimer {
    private:
        std::string name_;
        bool active_ = false;

    public:
        /**
         * @brief Конструктор - начало замера
         * @param name Имя операции
         */
        explicit ScopedTimer(const std::string& name) : name_(name) {
            auto& data = GlobalData::instance();
            if (data.enabled) {
                active_ = true;
                start(name_);
            }
        }

        /**
         * @brief Конструктор перемещения
         */
        ScopedTimer(ScopedTimer&& other) noexcept
            : name_(std::move(other.name_)), active_(other.active_) {
            other.active_ = false;
        }

        /**
         * @brief Деструктор - конец замера
         */
        ~ScopedTimer() {
            if (active_) {
                auto& data = GlobalData::instance();
                if (data.enabled) {
                    stop(name_);
                }
            }
        }

        // Запрет копирования
        ScopedTimer(const ScopedTimer&) = delete;
        ScopedTimer& operator=(const ScopedTimer&) = delete;
    };

    // ========================================================================
    // Управление профайлером
    // ========================================================================

    /**
     * @brief Сбросить всю накопленную статистику
     */
    static void reset() {
        auto& data = GlobalData::instance();
        std::lock_guard<std::mutex> lock(data.mutex);
        data.stats.clear();
    }

    /**
     * @brief Включить/выключить профилирование
     * @param enabled true - активен, false - все вызовы игнорируются
     */
    static void setEnabled(bool enabled) {
        auto& data = GlobalData::instance();
        data.enabled = enabled;
    }

    /**
     * @brief Проверить, включено ли профилирование
     */
    static bool isEnabled() {
        auto& data = GlobalData::instance();
        return data.enabled;
    }

    /**
     * @brief Установить файл для сохранения отчета
     * @param filename Путь к файлу
     */
    static void setOutputFile(const std::string& filename) {
        auto& data = GlobalData::instance();
        data.output_file = filename;
    }

    /**
     * @brief Получить количество уникальных операций
     */
    static size_t getOperationCount() {
        auto& data = GlobalData::instance();
        std::lock_guard<std::mutex> lock(data.mutex);
        return data.stats.size();
    }

    /**
     * @brief Получить общее время всех операций
     * @return int64_t Время в наносекундах
     */
    static int64_t getTotalTime() {
        auto& data = GlobalData::instance();
        std::lock_guard<std::mutex> lock(data.mutex);
        int64_t total = 0;
        for (const auto& [_, s] : data.stats) {
            total += s.total_time_ns;
        }
        return total;
    }

    /**
     * @brief Проверить, есть ли незакрытые замеры в текущем потоке
     * @return true если есть незакрытые замеры
     */
    static bool hasPendingMeasurements() {
        auto& data = GlobalData::instance();
        if (!data.enabled) return false;
        auto& call_stack = get_call_stack();
        return !call_stack.empty();
    }

    /**
     * @brief Получить текущую глубину стека вызовов
     */
    static size_t getCallStackDepth() {
        auto& data = GlobalData::instance();
        if (!data.enabled) return 0;
        auto& call_stack = get_call_stack();
        return call_stack.size();
    }

    // ========================================================================
    // Отчеты
    // ========================================================================

    /**
     * @brief Вывести отчет в указанный поток
     * @param os Выходной поток (по умолчанию std::cout)
     * 
     * **Формат отчета:**
     * @code
     * ============================================================
     * ОТЧЕТ ПРОФИЛИРОВАНИЯ
     * ============================================================
     * 
     *   encode
     * - Вызовов:   270
     * - Всего:     22497.988 мс
     * - Среднее:   83325.88 мкс
     * - Мин:       0.10 мкс
     * - Макс:      21615378.70 мкс
     * - % времени: 99.9%
     * ------------------------------------------------------------
     * Всего операций: 11
     * Общее время:    22511.233 мс
     * ============================================================
     * @endcode
     */
    static void printReport(std::ostream& os = std::cout) {
        auto& data = GlobalData::instance();
        std::lock_guard<std::mutex> lock(data.mutex);
        
        if (data.stats.empty()) {
            os << "Нет данных профилирования!\n";
            return;
        }
        
        // Сортируем
        std::vector<std::pair<std::string, Stats>> sorted(
            data.stats.begin(), data.stats.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second.total_time_ns > b.second.total_time_ns;
            });
        
        // Вывод отчета
        os << "\n============================================================\n";
        os << "ОТЧЕТ ПРОФИЛИРОВАНИЯ\n";
        os << "============================================================\n\n";
        
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
            
            os << "  " << name << "\n";
            os << "- Вызовов:   " << s.call_count << "\n";
            os << "- Всего:     " << std::fixed << std::setprecision(3) << total_ms << " мс\n";
            os << "- Среднее:   " << std::fixed << std::setprecision(2) << avg_us << " мкс\n";
            os << "- Мин:       " << std::fixed << std::setprecision(2) << min_us << " мкс\n";
            os << "- Макс:      " << std::fixed << std::setprecision(2) << max_us << " мкс\n";
            os << "- % времени: " << std::fixed << std::setprecision(1) << percent << "%\n";
            os << "\n";
        }
        
        os << "------------------------------------------------------------\n";
        os << "Всего операций: " << data.stats.size() << "\n";
        os << "Общее время:    " << std::fixed << std::setprecision(3)
        << total_time_all / 1'000'000.0 << " мс\n";
        
        if (hasPendingMeasurements()) {
            os << "\nВНИМАНИЕ: Обнаружены незакрытые замеры!\n";
            os << "Глубина стека:  " << getCallStackDepth() << "\n";
        }
        
        os << "============================================================\n\n";
        os.flush();
    }
    
    /**
     * @brief Сохранить отчет в файл
     * 
     * Использует файл, установленный через setOutputFile()
     * 
     * @see setOutputFile()
     */
    static void saveReport() {
        auto& data = GlobalData::instance();
        std::ofstream file(data.output_file);
        if (file.is_open()) {
            printReport(file);
            file.close();
            std::cout << "Отчет сохранен в: " << data.output_file << "\n";
        } else {
            std::cerr << "Ошибка: не удалось открыть файл " 
                      << data.output_file << " для записи!\n";
        }
    }
};

}    // namespace bpe

// ============================================================================
// Удобные макросы
// ============================================================================

// Помощник для конкатенации
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

/**
 * @def PROFILE_BLOCK(name)
 * @brief Замерить время выполнения блока кода
 * 
 * Использование:
 * @code
 * {
 *     PROFILE_BLOCK("encode_batch");
 *     // Код для замера
 * }
 * @endcode
 */
#define PROFILE_BLOCK(name) \
    bpe::SimpleProfiler::ScopedTimer CONCAT(_profiler, __LINE__)(name)

/**
 * @def PROFILE_FUNCTION()
 * @brief Замерить время выполнения функции
 * 
 * Автоматически использует имя функции как название операции.
 * 
 * @code
 * void myFunction() {
 *     PROFILE_FUNCTION();
 *     // Код функции будет измерен
 * }
 * @endcode
 */
#define PROFILE_FUNCTION() \
    bpe::SimpleProfiler::ScopedTimer _profiler(__FUNCTION__)

/**
 * @def IF_PROFILING(x)
 * @brief Условная компиляция для профилирования
 * 
 * Использование с флагом компиляции -DENABLE_PROFILING
 * 
 * @code
 * IF_PROFILING(SimpleProfiler::setEnabled(true));
 * @endcode
 */
#ifdef ENABLE_PROFILING
    #define IF_PROFILING(x) x
#else
    #define IF_PROFILING(x)
#endif

/**
 * @example examples/profiler_demo.cpp
 * Демонстрация использования профайлера с вложенными вызовами
 * 
 * @include examples/profiler_demo.cpp
 * 
 * @code
 * #include "profiler.hpp"
 * #include <thread>
 * #include <chrono>
 * 
 * void worker(int id) {
 *     PROFILE_FUNCTION();
 *     std::this_thread::sleep_for(std::chrono::milliseconds(10 * id));
 * }
 * 
 * void nested_example() {
 *     PROFILE_FUNCTION();
 *     
 *     // Вложенный замер
 *     {
 *         PROFILE_BLOCK("inner_operation");
 *         std::this_thread::sleep_for(std::chrono::milliseconds(2));
 *     }
 *     
 *     std::this_thread::sleep_for(std::chrono::milliseconds(3));
 * }
 * 
 * int main() {
 *     using namespace bpe;
 *     
 *     // Включаем профилирование
 *     SimpleProfiler::setEnabled(true);
 *     SimpleProfiler::setOutputFile("demo_profile.txt");
 *     
 *     // Тестируем разные операции
 *     for (int i = 0; i < 5; ++i) {
 *         PROFILE_BLOCK("fast_operation");
 *         volatile int sum = 0;
 *         for (int j = 0; j < 10000; ++j) sum += j;
 *     }
 *     
 *     for (int i = 0; i < 3; ++i) {
 *         PROFILE_BLOCK("slow_operation");
 *         std::this_thread::sleep_for(std::chrono::milliseconds(5));
 *     }
 *     
 *     // Вложенные вызовы
 *     for (int i = 0; i < 2; ++i) {
 *         nested_example();
 *     }
 *     
 *     for (int i = 1; i <= 3; ++i) {
 *         worker(i);
 *     }
 *     
 *     // Выводим результаты
 *     std::cout << "\nСтатистика:\n";
 *     std::cout << "- Операций:    " << SimpleProfiler::getOperationCount() << "\n";
 *     std::cout << "- Общее время: " << SimpleProfiler::getTotalTime() / 1e6 << " мс\n";
 *     
 *     SimpleProfiler::printReport();
 *     SimpleProfiler::saveReport();
 *     
 *     return 0;
 * }
 * @endcode
 */