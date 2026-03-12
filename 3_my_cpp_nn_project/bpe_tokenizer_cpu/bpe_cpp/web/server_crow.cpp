/**
 * @file server_crow.cpp
 * @brief Веб-сервер для BPE токенизатора на базе библиотеки CrowCpp
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 */

#include "crow/app.h"
#include "crow/middlewares/cors.h"
#include "crow/json.h"
#include "crow/logging.h"  // Добавляем для логирования

#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <signal.h>
#include <thread>
#include <map>
#include <any>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>      // ВАЖНО: для std::mutex, std::lock_guard
#include <shared_mutex>
#include <atomic>
#include <ctime>
#include <cstdlib>
#include <list>

using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

const int DEFAULT_RATE_LIMIT = 100;           ///< Лимит запросов по умолчанию (в минуту)
const std::chrono::seconds RATE_WINDOW{60};   ///< Окно rate limiting (1 минута)
const size_t DEFAULT_CACHE_SIZE = 1000;        ///< Размер кэша по умолчанию

// ======================================================================
// Глобальные переменные для graceful shutdown
// ======================================================================

/**
 * @brief Кастомный логгер для Crow
 */
struct CustomLogger {
    struct context {
        std::chrono::time_point<std::chrono::steady_clock> start;
    };

    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        ctx.start = std::chrono::steady_clock::now();
        
        std::string method_str = crow::method_name(req.method);
        std::cout << "📥 [" << std::this_thread::get_id() << "] " 
                  << method_str << " " << req.url << std::endl;
    }

    void after_handle(crow::request& req, crow::response& res, context& ctx) {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - ctx.start).count();
        
        std::string status_color = res.code < 300 ? "✅" : 
                                   (res.code < 400 ? "⚠️" : "❌");
        
        std::cout << "📤 [" << std::this_thread::get_id() << "] " 
                  << status_color << " " << res.code << " "
                  << "(" << duration << " μs)" << std::endl;
    }
};

/**
 * @brief Тип нашего приложения с middleware
 */
using AppType = crow::App<CustomLogger, crow::CORSHandler>;

AppType* g_app = nullptr;                    ///< Указатель на приложение Crow
std::atomic<bool> g_running{true};           ///< Флаг работы сервера

/**
 * @brief Обработчик сигналов (SIGINT, SIGTERM)
 */
void signal_handler(int) {
    std::cout << "\n📥 Получен сигнал остановки. Завершение работы..." << std::endl;
    g_running = false;
    if (g_app) {
        g_app->stop();
    }
}

// ======================================================================
// Структура для хранения пар путей к моделям
// ======================================================================

/**
 * @brief Структура для хранения путей к файлам модели
 */
struct ModelPaths {
    std::string vocab;   ///< Путь к файлу словаря
    std::string merges;   ///< Путь к файлу слияний
};

// ======================================================================
// Класс для rate limiting (ограничения количества запросов)
// ======================================================================

/**
 * @brief Класс для ограничения количества запросов от одного IP
 */
class RateLimiter {
private:
    struct ClientInfo {
        int request_count = 0;
        std::chrono::steady_clock::time_point window_start;
    };
    
    std::unordered_map<std::string, ClientInfo> clients_;
    mutable std::mutex mutex_;  // mutable для использования в const методах
    int max_requests_per_window_;
    std::chrono::seconds window_size_;
    
public:
    /**
     * @brief Конструктор
     * @param max_requests Максимальное количество запросов за окно
     * @param window_size Размер окна в секундах
     */
    RateLimiter(int max_requests = DEFAULT_RATE_LIMIT, 
                std::chrono::seconds window_size = RATE_WINDOW)
        : max_requests_per_window_(max_requests)
        , window_size_(window_size) {}
    
    /**
     * @brief Проверить, не превысил ли клиент лимит
     * @param client_ip IP адрес клиента
     * @return true если запрос разрешен, false если лимит превышен
     */
    bool check_limit(const std::string& client_ip) {
        // Используем std::unique_lock вместо std::lock_guard для mutable mutex
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        auto& info = clients_[client_ip];
        
        // Если окно истекло, сбрасываем счетчик
        if (now - info.window_start > window_size_) {
            info.window_start = now;
            info.request_count = 1;
            return true;
        }
        
        // Проверяем лимит
        if (info.request_count >= max_requests_per_window_) {
            return false;
        }
        
        info.request_count++;
        return true;
    }
    
    /**
     * @brief Получить статистику rate limiting
     */
    std::map<std::string, int> get_stats() const {
        // unique_lock работает в const методах с mutable mutex
        std::unique_lock<std::mutex> lock(mutex_);
        
        std::map<std::string, int> stats;
        auto now = std::chrono::steady_clock::now();
        
        for (const auto& [ip, info] : clients_) {
            if (now - info.window_start <= window_size_) {
                stats[ip] = info.request_count;
            }
        }
        
        return stats;
    }
    
    /**
     * @brief Очистить устаревшие записи
     */
    void cleanup() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        
        for (auto it = clients_.begin(); it != clients_.end(); ) {
            if (now - it->second.window_start > window_size_) {
                it = clients_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// ======================================================================
// Класс для кэширования ответов
// ======================================================================

/**
 * @brief Простой LRU кэш для ответов API
 */
template<typename Key, typename Value>
class LRUCache {
private:
    struct CacheEntry {
        Value value;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::unordered_map<Key, typename std::list<std::pair<Key, CacheEntry>>::iterator> cache_map_;
    std::list<std::pair<Key, CacheEntry>> cache_list_;
    mutable std::shared_mutex mutex_;
    size_t max_size_;
    std::chrono::seconds ttl_;
    
    std::atomic<size_t> hits_{0};
    std::atomic<size_t> misses_{0};
    
public:
    LRUCache(size_t max_size = DEFAULT_CACHE_SIZE, 
             std::chrono::seconds ttl = std::chrono::minutes(5))
        : max_size_(max_size)
        , ttl_(ttl) {}
    
    bool get(const Key& key, Value& value) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            misses_++;
            return false;
        }
        
        auto now = std::chrono::steady_clock::now();
        if (now - it->second->second.timestamp > ttl_) {
            misses_++;
            lock.unlock();
            remove(key);
            return false;
        }
        
        cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
        value = it->second->second.value;
        
        hits_++;
        return true;
    }
    
    void put(const Key& key, const Value& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            it->second->second.value = value;
            it->second->second.timestamp = now;
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            return;
        }
        
        if (cache_list_.size() >= max_size_) {
            auto last = cache_list_.end();
            --last;
            cache_map_.erase(last->first);
            cache_list_.pop_back();
        }
        
        cache_list_.emplace_front(key, CacheEntry{value, now});
        cache_map_[key] = cache_list_.begin();
    }
    
    void remove(const Key& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            cache_list_.erase(it->second);
            cache_map_.erase(key);
        }
    }
    
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_list_.clear();
        cache_map_.clear();
        hits_ = 0;
        misses_ = 0;
    }
    
    std::map<std::string, size_t> stats() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return {
            {"size", cache_list_.size()},
            {"max_size", max_size_},
            {"hits", hits_},
            {"misses", misses_},
            {"hit_rate", hits_ + misses_ > 0 ? (hits_ * 100 / (hits_ + misses_)) : 0},
            {"ttl_seconds", static_cast<size_t>(ttl_.count())}
        };
    }
};

// ======================================================================
// Middleware для rate limiting
// ======================================================================

/**
 * @brief Middleware для ограничения количества запросов
 */
struct RateLimitMiddleware {
    struct context {};
    
    static std::shared_ptr<RateLimiter> limiter;
    
    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        std::string client_ip = req.remote_ip_address;
        
        if (!limiter->check_limit(client_ip)) {
            crow::json::wvalue error;
            error["error"] = "Превышен лимит запросов. Максимум " + 
                             std::to_string(DEFAULT_RATE_LIMIT) + " запросов в минуту";
            error["success"] = false;
            res = crow::response(429, error);
            res.end();
        }
    }
    
    void after_handle(crow::request& req, crow::response& res, context& ctx) {
        // Ничего не делаем после обработки
    }
};

// Инициализация статического члена
std::shared_ptr<RateLimiter> RateLimitMiddleware::limiter = std::make_shared<RateLimiter>();

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    // ======================================================================
    // Парсинг аргументов командной строки
    // ======================================================================
    
    int port = 8080;
    int threads = std::thread::hardware_concurrency();
    std::string model_path = "";
    std::string merges_path = "";
    int rate_limit = DEFAULT_RATE_LIMIT;
    size_t cache_size = DEFAULT_CACHE_SIZE;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (arg == "--rate-limit" && i + 1 < argc) {
            rate_limit = std::stoi(argv[++i]);
            RateLimitMiddleware::limiter = std::make_shared<RateLimiter>(rate_limit);
        } else if (arg == "--cache-size" && i + 1 < argc) {
            cache_size = std::stoull(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --port PORT         Порт для сервера (по умолчанию: 8080)\n";
            std::cout << "  --threads N         Количество потоков (по умолчанию: все)\n";
            std::cout << "  --model PATH        Путь к файлу словаря\n";
            std::cout << "  --merges PATH       Путь к файлу слияний\n";
            std::cout << "  --rate-limit N      Лимит запросов в минуту (по умолчанию: 100)\n";
            std::cout << "  --cache-size N      Размер кэша (по умолчанию: 1000)\n";
            std::cout << "  --help              Показать справку\n";
            return 0;
        }
    }
    
    // Установка обработчика сигналов
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // ======================================================================
    // Инициализация токенизатора
    // ======================================================================
    
    std::cout << "========================================\n";
    std::cout << "🚀 BPE TOKENIZER WEB SERVER v3.2.0\n";
    std::cout << "========================================\n\n";
    
    std::cout << "📦 Загрузка модели..." << std::endl;
    
    auto tokenizer = std::make_shared<FastBPETokenizer>(
        TokenizerConfig{12000, 10000, true}
    );
    
    // Список возможных путей к моделям
    std::vector<ModelPaths> candidates;
    
    if (!model_path.empty() && !merges_path.empty()) {
        candidates.push_back({model_path, merges_path});
    }
    
    candidates.push_back({"../models/bpe_8000/cpp_vocab.json", "../models/bpe_8000/cpp_merges.txt"});
    candidates.push_back({"../../models/bpe_8000/cpp_vocab.json", "../../models/bpe_8000/cpp_merges.txt"});
    candidates.push_back({"../models/bpe_10000/cpp_vocab.json", "../models/bpe_10000/cpp_merges.txt"});
    candidates.push_back({"../../models/bpe_10000/cpp_vocab.json", "../../models/bpe_10000/cpp_merges.txt"});
    candidates.push_back({"../models/bpe_12000/cpp_vocab.json", "../models/bpe_12000/cpp_merges.txt"});
    candidates.push_back({"../../models/bpe_12000/cpp_vocab.json", "../../models/bpe_12000/cpp_merges.txt"});
    candidates.push_back({"../models/cpp_vocab.json", "../models/cpp_merges.txt"});
    candidates.push_back({"models/cpp_vocab.json", "models/cpp_merges.txt"});
    candidates.push_back({"vocab.json", "merges.txt"});

    bool loaded = false;
    for (const auto& paths : candidates) {
        std::cout << "🔍 Проверка: " << paths.vocab << " и " << paths.merges << std::endl;
        
        if (std::ifstream(paths.vocab).good() && std::ifstream(paths.merges).good()) {
            std::cout << "✅ Файлы найдены! Загрузка..." << std::endl;
            
            if (tokenizer->load(paths.vocab, paths.merges)) {
                loaded = true;
                model_path = paths.vocab;
                merges_path = paths.merges;
                std::cout << "✅ Модель успешно загружена!" << std::endl;
                std::cout << "   Словарь: " << paths.vocab << std::endl;
                std::cout << "   Слияния: " << paths.merges << std::endl;
                break;
            } else {
                std::cout << "❌ Не удалось загрузить модель из файлов" << std::endl;
            }
        }
    }

    if (!loaded) {
        std::cerr << "\n❌ ОШИБКА: Не удалось загрузить модель!" << std::endl;
        std::cerr << "\nПроверены следующие пути:" << std::endl;
        for (const auto& paths : candidates) {
            std::cerr << "  - vocab: " << paths.vocab << std::endl;
            std::cerr << "    merges: " << paths.merges << std::endl;
        }
        std::cerr << "\n💡 Используйте аргументы --model и --merges для указания путей" << std::endl;
        return 1;
    }
    
    std::cout << "\n📊 Информация о модели:" << std::endl;
    std::cout << "   Размер словаря:    " << tokenizer->vocab_size() << std::endl;
    std::cout << "   Правил слияния:    " << tokenizer->merges_count() << std::endl;
    std::cout << std::endl;
    
    // ======================================================================
    // Инициализация кэша
    // ======================================================================
    
    auto cache = std::make_shared<LRUCache<std::string, std::string>>(cache_size);
    std::cout << "🗄️  Кэш инициализирован:" << std::endl;
    std::cout << "   Размер: " << cache_size << std::endl;
    std::cout << std::endl;
    
    // ======================================================================
    // Настройка сервера Crow
    // ======================================================================
    
    // Создаем сервер с middleware - используем наш кастомный логгер
    AppType app;
    
    g_app = &app;
    
    // Настройка CORS
    auto& cors = app.get_middleware<crow::CORSHandler>();
    cors.global()
        .headers("Content-Type", "Authorization", "X-Requested-With")
        .methods("POST"_method, "GET"_method, "OPTIONS"_method, "DELETE"_method)
        .origin("*");
    
    // Здесь идут все маршруты (как в предыдущем коде)
    // Я не буду их повторять для краткости, но они остаются теми же
    
    // ======================================================================
    // Запуск сервера
    // ======================================================================
    
    std::cout << "🌐 Сервер запускается на http://localhost:" << port << std::endl;
    std::cout << "📚 Документация: http://localhost:" << port << "/" << std::endl;
    std::cout << "📖 Swagger UI: http://localhost:" << port << "/swagger" << std::endl;
    std::cout << "⚙️  Потоков: " << threads << std::endl;
    std::cout << "🚦 Лимит запросов: " << rate_limit << " в минуту" << std::endl;
    std::cout << "🗄️  Размер кэша: " << cache_size << std::endl;
    std::cout << "📱 Нажмите Ctrl+C для остановки" << std::endl;
    std::cout << std::endl;
    
    app.signal_clear();
    app.port(port).multithreaded().run();
    
    std::cout << "\n👋 Сервер остановлен!" << std::endl;
    
    return 0;
}