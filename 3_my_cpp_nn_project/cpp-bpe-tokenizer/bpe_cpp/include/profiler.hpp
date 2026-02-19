/**
 * @file profiler.hpp
 * @brief Простой встроенный профайлер для BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Легковесная система профилирования для поиска узких мест:
 *          - Замер времени выполнения функций
 *          - Подсчет количества вызовов
 *          - Статистика по операциям
 *          - Автоматические отчеты
 * 
 * @note Создан специально для этапа оптимизации производительности
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

namespace bpe {

/**
 * @brief Простой профайлер для замера времени выполнения операций
 * 
 * Использование:
 *   PROFILE_FUNCTION();  // в начале функции
 *   или
 *   {
 *       ScopedProfiler profiler("encode_batch");
 *       // код для замера
 *   }
 */
class SimpleProfiler {
private:
    struct Stats {
        int64_t total_time_ns = 0;
        int64_t min_time_ns = INT64_MAX;
        int64_t max_time_ns = 0;
        int call_count = 0;
        int current_depth = 0;
    };
    
    static inline std::map<std::string, Stats> stats_;
    static inline std::mutex mutex_;
    static inline std::string current_output_file_ = "profiler_report.txt";
    static inline bool enabled_ = true;

public:
    /**
     * @brief Начать замер операции
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
    
    /**
     * @brief RAII класс для автоматического замера
     */
    class ScopedTimer {
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
        
    public:
        explicit ScopedTimer(const std::string& name) : name_(name) {
            start_ = std::chrono::high_resolution_clock::now();
        }
        
        ~ScopedTimer() {
            if (!enabled_) return;
            
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
    };
    
    /**
     * @brief Сбросить всю статистику
     */
    static void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.clear();
    }
    
    /**
     * @brief Включить/выключить профилирование
     */
    static void setEnabled(bool enabled) {
        enabled_ = enabled;
    }
    
    /**
     * @brief Установить файл для отчета
     */
    static void setOutputFile(const std::string& filename) {
        current_output_file_ = filename;
    }
    
    /**
     * @brief Вывести отчет в консоль
     */
    static void printReport(std::ostream& os = std::cout) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (stats_.empty()) {
            os << "📊 Нет данных профилирования\n";
            return;
        }
        
        // Сортируем по общему времени (убывание)
        std::vector<std::pair<std::string, Stats>> sorted(
            stats_.begin(), stats_.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) {
                return a.second.total_time_ns > b.second.total_time_ns;
            });
        
        os << "\n" << std::string(80, '=') << "\n";
        os << "📊 ОТЧЕТ ПРОФИЛИРОВАНИЯ\n";
        os << std::string(80, '=') << "\n\n";
        
        // Заголовок таблицы
        os << std::left << std::setw(30) << "Операция"
           << std::right << std::setw(12) << "Вызовов"
           << std::setw(15) << "Всего (мс)"
           << std::setw(12) << "Среднее"
           << std::setw(12) << "Мин"
           << std::setw(12) << "Макс"
           << std::setw(10) << "% времени" << "\n";
        
        os << std::string(103, '-') << "\n";
        
        int64_t total_time_all = 0;
        for (const auto& [name, s] : sorted) {
            total_time_all += s.total_time_ns;
        }
        
        for (const auto& [name, s] : sorted) {
            double total_ms = s.total_time_ns / 1'000'000.0;
            double avg_us = s.total_time_ns / 1000.0 / s.call_count;
            double min_us = s.min_time_ns / 1000.0;
            double max_us = s.max_time_ns / 1000.0;
            double percent = 100.0 * s.total_time_ns / total_time_all;
            
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
        
        os << std::string(103, '-') << "\n";
        os << "📈 Всего операций: " << stats_.size() << "\n";
        os << "⏱️  Общее время: " << std::fixed << std::setprecision(3)
           << total_time_all / 1'000'000.0 << " мс\n";
        os << std::string(80, '=') << "\n\n";
    }
    
    /**
     * @brief Сохранить отчет в файл
     */
    static void saveReport() {
        std::ofstream file(current_output_file_);
        if (file.is_open()) {
            printReport(file);
            std::cout << "💾 Отчет сохранен в: " << current_output_file_ << "\n";
        }
    }
};

// ======================================================================
// Удобные макросы для использования
// ======================================================================

/**
 * @brief Замерить время выполнения блока кода
 * 
 * Использование:
 * {
 *     PROFILE_BLOCK("encode_batch");
 *     // код для замера
 * }
 */
#define PROFILE_BLOCK(name) \
    bpe::SimpleProfiler::ScopedTimer CONCAT(_profiler, __LINE__)(name)

/**
 * @brief Замерить время выполнения функции
 * 
 * Использование:
 * void myFunction() {
 *     PROFILE_FUNCTION();
 *     // код функции
 * }
 */
#define PROFILE_FUNCTION() \
    bpe::SimpleProfiler::ScopedTimer _profiler(__FUNCTION__)

// Помощник для конкатенации
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

} // namespace bpe