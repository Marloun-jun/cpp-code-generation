/**
 * @file server_crow.cpp
 * @brief Веб-сервер для BPE токенизатора на базе библиотеки CrowCpp
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Высокопроизводительный веб-сервер, предоставляющий REST API
 *          для токенизации C++ кода. Поддерживает:
 * 
 *          **Функциональность:**
 *          ┌────────────────────┬──────────────────────────┐
 *          │ Токенизация        │ POST /api/tokenize       │
 *          │ Декодирование      │ POST /api/detokenize     │
 *          │ Пакетная обработка │ POST /api/batch/tokenize │
 *          │ Статистика         │ GET /api/stats           │
 *          │ Сброс статистики   │ POST /api/stats/reset    │
 *          │ Health check       │ GET /api/health          │
 *          │ Swagger UI         │ GET /swagger             │
 *          └────────────────────┴──────────────────────────┘
 * 
 *          **Особенности:**
 *          - Многопоточная обработка запросов
 *          - Rate limiting для защиты от DoS
 *          - LRU кэш для часто запрашиваемых текстов
 *          - Graceful shutdown по сигналам SIGINT/SIGTERM
 *          - Подробное логирование каждого запроса
 *          - Поддержка CORS для кросс-доменных запросов
 * 
 * @compile g++ -std=c++17 -O3 -pthread server_crow.cpp -o server_crow -lcrow -lboost_system
 * @run ./server_crow [--port 8080] [--threads 4] [--rate-limit 100]
 * 
 * @see FastBPETokenizer
 * @see https://crowcpp.org/
 */

#include "crow/app.h"
#include "crow/json.h"
#include "crow/logging.h"
#include "crow/middlewares/cors.h"

#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <atomic>
#include <any>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

const int DEFAULT_RATE_LIMIT = 100;            ///< Лимит запросов по умолчанию (в минуту)
const std::chrono::seconds RATE_WINDOW{60};    ///< Окно rate limiting (1 минута)
const size_t DEFAULT_CACHE_SIZE = 1000;        ///< Размер кэша по умолчанию

// ============================================================================
// Глобальные переменные для graceful shutdown
// ============================================================================

/**
 * @brief Кастомный логгер для Crow с таймингом запросов
 */
struct CustomLogger {
    struct context {
        std::chrono::time_point<std::chrono::steady_clock> start;    ///< Время начала запроса
    };

    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        ctx.start = std::chrono::steady_clock::now();
        
        std::string method_str = crow::method_name(req.method);
        std::cout << "[" << std::this_thread::get_id() << "] " 
                  << method_str << " " << req.url << std::endl;
    }

    void after_handle(crow::request& req, crow::response& res, context& ctx) {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end - ctx.start).count();
        
        std::string status_color = res.code < 300 ? "да" : 
                                   (res.code < 400 ? "[ВНИМАНИЕ]" : "нет");
        
        std::cout << "[" << std::this_thread::get_id() << "] " 
                  << status_color << " " << res.code << " "
                  << "(" << duration << " мs)" << std::endl;
    }
};

/**
 * @brief Тип нашего приложения с middleware
 */
using AppType = crow::App<CustomLogger, crow::CORSHandler>;

AppType* g_app = nullptr;             ///< Указатель на приложение Crow
std::atomic<bool> g_running{true};    ///< Флаг работы сервера

/**
 * @brief Обработчик сигналов (SIGINT, SIGTERM)
 */
void signal_handler(int) {
    std::cout << "\nПолучен сигнал остановки. Завершение работы..." << std::endl;
    g_running = false;
    if (g_app) {
        g_app->stop();
    }
}

// ============================================================================
// Структура для хранения пар путей к моделям
// ============================================================================

/**
 * @brief Структура для хранения путей к файлам модели
 */
struct ModelPaths {
    std::string vocab;     ///< Путь к файлу словаря
    std::string merges;    ///< Путь к файлу слияний
};

// ============================================================================
// Класс для rate limiting (ограничения количества запросов)
// ============================================================================

/**
 * @brief Класс для ограничения количества запросов от одного IP
 * 
 * Реализует алгоритм скользящего окна для защиты от DoS-атак.
 */
class RateLimiter {
private:
    struct ClientInfo {
        int request_count = 0;
        std::chrono::steady_clock::time_point window_start;
    };
    
    std::unordered_map<std::string, ClientInfo> clients_;
    mutable std::mutex mutex_;    // mutable для использования в const методах
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

// ============================================================================
// Класс для кэширования ответов
// ============================================================================

/**
 * @brief Простой LRU кэш для ответов API
 * 
 * @tparam Key Тип ключа
 * @tparam Value Тип значения
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

// ============================================================================
// Middleware для rate limiting
// ============================================================================

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

// ============================================================================
// HTML страница интерфейса
// ============================================================================

/**
 * @brief Возвращает HTML страницу с пользовательским интерфейсом
 * 
 * Содержит встроенный JavaScript для взаимодействия с API и
 * стилизованный CSS для удобного интерфейса.
 */
std::string get_html_page() {
    return R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BPE Tokenizer - C++ Code Tokenization</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 { 
            color: #333; 
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        h1 span {
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            transition: all 0.3s;
            background: #f5f5f5;
        }
        .tab:hover {
            background: #e0e0e0;
        }
        .tab.active {
            background: #4CAF50;
            color: white;
        }
        .tab-content {
            display: none;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .tab-content.active {
            display: block !important;
        }
        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        textarea, input[type="text"] { 
            width: 100%; 
            padding: 15px; 
            margin: 10px 0; 
            font-family: 'Consolas', 'Monaco', monospace; 
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border 0.3s, box-shadow 0.3s;
            white-space: pre;
            wrap: off;
            background: #fafafa;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }
        button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 12px 30px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.3s, box-shadow 0.3s;
            margin: 5px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        button.secondary {
            background: #f44336;
        }
        button.secondary:hover {
            box-shadow: 0 5px 15px rgba(244, 67, 54, 0.4);
        }
        .result { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            margin-top: 20px;
            border-left: 4px solid #4CAF50;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow: auto;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
        }
        .result pre {
            margin: 0;
            padding: 0;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre-wrap;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            color: #f44336;
            font-weight: bold;
        }
        .success {
            color: #4CAF50;
            font-weight: bold;
        }
        .info {
            color: #2196F3;
        }
        .token-list {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }
        .token-item {
            background: #e0e0e0;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            font-family: monospace;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
        .footer a {
            color: #667eea;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        
        /* Уведомления */
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            animation: slideIn 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .notification.success {
            background: linear-gradient(135deg, #4CAF50, #45a049);
        }
        
        .notification.error {
            background: linear-gradient(135deg, #f44336, #d32f2f);
        }
        
        .notification.info {
            background: linear-gradient(135deg, #2196F3, #1976D2);
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Модальное окно подтверждения */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 2000;
            animation: fadeIn 0.3s;
        }
        
        .modal.active {
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 400px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .modal-content p {
            margin: 20px 0;
            font-size: 16px;
            color: #333;
        }
        
        .modal-buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        .modal-buttons button {
            flex: 1;
            margin: 0;
        }
        
        .modal-buttons button.cancel {
            background: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><span>BPE Tokenizer</span> for C++ Code</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('encode')">Токенизация</div>
            <div class="tab" onclick="switchTab('decode')">Декодирование</div>
            <div class="tab" onclick="switchTab('batch')">Пакетная обработка</div>
            <div class="tab" onclick="switchTab('stats')">Статистика</div>
            <div class="tab" onclick="switchTab('about')">О сервере</div>
        </div>
        
        <div id="encode" class="tab-content active">
            <h2>Токенизация C++ кода</h2>
            <div class="two-column">
                <div>
                    <textarea id="code" rows="15" placeholder="Введите C++ код..." style="white-space: pre; wrap: off;">int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}</textarea>
                    <button onclick="tokenize()">
                        Токенизировать <span id="encode-loading" class="loading" style="display:none;"></span>
                    </button>
                    <button class="secondary" onclick="clearResult('encode-result')">
                        Очистить
                    </button>
                </div>
                <div>
                    <div class="result" id="encode-result"></div>
                </div>
            </div>
        </div>
        
        <div id="decode" class="tab-content">
            <h2>Декодирование токенов</h2>
            <div class="two-column">
                <div>
                    <input type="text" id="tokens" placeholder="Введите токены через запятую (например: 42, 43, 44)">
                    <button onclick="detokenize()">
                        Декодировать <span id="decode-loading" class="loading" style="display:none;"></span>
                    </button>
                    <button class="secondary" onclick="clearResult('decode-result')">
                        Очистить
                    </button>
                </div>
                <div>
                    <div class="result" id="decode-result"></div>
                </div>
            </div>
        </div>
        
        <div id="batch" class="tab-content">
            <h2>Пакетная обработка</h2>
            <p>Введите несколько примеров кода (каждый на новой строке)</p>
            <div class="two-column">
                <div>
                    <textarea id="batch-codes" rows="15" placeholder="int main() { return 0; }
void test() { }
class MyClass { };">int main() { return 0; }
void test() { }
class MyClass { };</textarea>
                    <button onclick="batchTokenize()">
                        Обработать пакет <span id="batch-loading" class="loading" style="display:none;"></span>
                    </button>
                    <button class="secondary" onclick="clearBatch()">
                        Очистить
                    </button>
                </div>
                <div>
                    <div class="result" id="batch-result"></div>
                </div>
            </div>
        </div>
        
        <div id="stats" class="tab-content">
            <h2>Детальная статистика</h2>
            <div class="result" id="stats-result"></div>
            <div style="display: flex; gap: 10px; margin-top: 20px;">
                <button onclick="refreshStats()" style="flex: 1;">
                    Обновить статистику
                </button>
                <button class="secondary" onclick="resetStats()" style="flex: 1;">
                    Сбросить статистику
                </button>
            </div>
        </div>
        <div id="about" class="tab-content">
            <h2>О сервере</h2>
            <div class="result">
                <p><strong>BPE Tokenizer Web Service</strong></p><p>Версия: 3.3.0</p><p>API эндпоинты:</p><ul><li><code>POST /api/tokenize</code> - токенизация текста</li><li><code>POST /api/detokenize</code> - декодирование токенов</li><li><code>POST /api/batch/tokenize</code> - пакетная токенизация</li><li><code>GET /api/stats</code> - статистика токенизатора</li><li><code>POST /api/stats/reset</code> - сброс статистики</li><li><code>GET /api/health</code> - проверка работоспособности</li></ul><p><strong>Технические детали:</strong></p><ul><li>Многопоточная обработка запросов</li><li>Защита от DoS-атак</li><li>Поддержка CORS</li>
            </div>
        </div>
        <div class="footer">
            <p>BPE Tokenizer for C++ Code | <a href="https://github.com/yourusername/bpe-tokenizer" target="_blank">GitHub</a> | &copy; 2026</p>
        </div>
    </div>
    
    <!-- Модальное окно подтверждения -->
    <div class="modal" id="confirmModal">
        <div class="modal-content">
            <h3>Подтверждение</h3>
            <p id="confirmMessage">Вы уверены, что хотите сбросить всю статистику?</p>
            <div class="modal-buttons">
                <button onclick="confirmAction(true)">Да</button>
                <button class="cancel" onclick="confirmAction(false)">Нет</button>
            </div>
        </div>
    </div>
    
    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => {
                c.classList.remove('active');
                c.style.display = 'none';
            });
            
            document.querySelector(`.tab[onclick*='${tab}']`).classList.add('active');
            const activeTab = document.getElementById(tab);
            activeTab.classList.add('active');
            activeTab.style.display = 'block';
            
            if (tab === 'stats') {
                refreshStats();
            }
        }
        
        function showLoading(elementId) {
            document.getElementById(elementId).style.display = 'inline-block';
        }
        
        function hideLoading(elementId) {
            document.getElementById(elementId).style.display = 'none';
        }
        
        function clearResult(elementId) {
            document.getElementById(elementId).innerHTML = '';
        }
        
        function clearBatch() {
            document.getElementById('batch-result').innerHTML = '';
            document.getElementById('batch-codes').value = `int main() { return 0; }
void test() { }
class MyClass { };`;
        }
        
        function formatTokens(tokens) {
            let html = '<div class="token-list">';
            for (let i = 0; i < Math.min(tokens.length, 100); i++) {
                html += `<span class="token-item">${tokens[i]}</span>`;
            }
            if (tokens.length > 100) {
                html += `<span class="token-item">... и ещё ${tokens.length - 100}</span>`;
            }
            html += '</div>';
            return html;
        }
        
        // Функция для показа уведомления
        function showNotification(message, type = 'success') {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {
                notification.style.animation = 'slideIn 0.3s reverse';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }
        
        // Функция для подтверждения действия
        let pendingAction = null;
        
        function confirmAction(confirmed) {
            document.getElementById('confirmModal').classList.remove('active');
            if (confirmed && pendingAction) {
                pendingAction();
            }
            pendingAction = null;
        }
        
        function showConfirm(message, onConfirm) {
            document.getElementById('confirmMessage').textContent = message;
            pendingAction = onConfirm;
            document.getElementById('confirmModal').classList.add('active');
        }
        
        function tokenize() {
            const code = document.getElementById('code').value;
            const resultDiv = document.getElementById('encode-result');
            
            if (!code.trim()) {
                resultDiv.innerHTML = '<span class="error">Введите код для токенизации!</span>';
                return;
            }
            
            showLoading('encode-loading');
            resultDiv.innerHTML = 'Обработка...';
            
            fetch('/api/tokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: code})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading('encode-loading');
                
                if (data.success) {
                    const tokens = data.tokens;
                    
                    let resultHtml = `<strong>Токены (${data.count} шт.):</strong><br>`;
                    resultHtml += `<div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; margin: 10px 0; background: #e8f5e8; padding: 10px; border-radius: 5px;">[`;
                    
                    for (let i = 0; i < tokens.length; i++) {
                        if (i > 0) resultHtml += ', ';
                        resultHtml += tokens[i];
                    }
                    
                    resultHtml += `]</div>`;
                    resultHtml += formatTokens(tokens);
                    
                    resultDiv.innerHTML = resultHtml;
                    
                    if (document.getElementById('stats').classList.contains('active')) {
                        refreshStats();
                    }
                } else {
                    resultDiv.innerHTML = `<span class="error">Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                hideLoading('encode-loading');
                resultDiv.innerHTML = `<span class="error">Ошибка: ${err.message}</span>`;
            });
        }
        
        function detokenize() {
            const tokensInput = document.getElementById('tokens').value;
            const resultDiv = document.getElementById('decode-result');
            
            if (!tokensInput.trim()) {
                resultDiv.innerHTML = '<span class="error">Введите токены для декодирования!</span>';
                return;
            }
            
            showLoading('decode-loading');
            resultDiv.innerHTML = 'Обработка...';
            
            const tokens = tokensInput.split(',').map(t => parseInt(t.trim())).filter(t => !isNaN(t));
            
            if (tokens.length === 0) {
                hideLoading('decode-loading');
                resultDiv.innerHTML = '<span class="error">Некорректный формат токенов!</span>';
                return;
            }
            
            fetch('/api/detokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tokens: tokens})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading('decode-loading');
                if (data.success) {
                    let displayText = data.text.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                    
                    resultDiv.innerHTML = `<strong>Декодированный текст:</strong><br>` +
                        `<pre style="margin:10px 0; padding:10px; background:#e8f5e8; border-radius:5px; white-space: pre-wrap;">${displayText}</pre>`;
                    
                    if (document.getElementById('stats').classList.contains('active')) {
                        refreshStats();
                    }
                } else {
                    resultDiv.innerHTML = `<span class="error">Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                hideLoading('decode-loading');
                resultDiv.innerHTML = `<span class="error">Ошибка: ${err.message}</span>`;
            });
        }
        
        function batchTokenize() {
            const codes = document.getElementById('batch-codes').value.split('\n').filter(c => c.trim());
            const resultDiv = document.getElementById('batch-result');
            
            if (codes.length === 0) {
                resultDiv.innerHTML = '<span class="error">Введите код для обработки!</span>';
                return;
            }
            
            showLoading('batch-loading');
            resultDiv.innerHTML = 'Обработка...';
            
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000);
            
            fetch('/api/batch/tokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({texts: codes}),
                signal: controller.signal
            })
            .then(response => response.json())
            .then(data => {
                clearTimeout(timeoutId);
                hideLoading('batch-loading');
                
                if (data.success) {
                    let resultHtml = `<strong>Результаты (${data.count} примеров):</strong><br>`;
                    
                    for (let i = 0; i < Math.min(data.results.length, 10); i++) {
                        const res = data.results[i];
                        resultHtml += `<div style="margin:10px 0; padding:10px; background:#f0f0f0; border-radius:5px;">`;
                        resultHtml += `<strong>Пример ${i+1}:</strong> ${res.token_count} токенов<br>`;
                        resultHtml += `<span style="font-size:12px;">[${res.tokens.slice(0, 20).join(', ')}${res.tokens.length > 20 ? '...' : ''}]</span>`;
                        resultHtml += `</div>`;
                    }
                    
                    if (data.results.length > 10) {
                        resultHtml += `<p>... и ещё ${data.results.length - 10} примеров</p>`;
                    }
                    
                    resultDiv.innerHTML = resultHtml;
                    
                    if (document.getElementById('stats').classList.contains('active')) {
                        refreshStats();
                    }
                } else {
                    resultDiv.innerHTML = `<span class="error">Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                clearTimeout(timeoutId);
                hideLoading('batch-loading');
                if (err.name === 'AbortError') {
                    resultDiv.innerHTML = '<span class="error">Таймаут запроса (более 30 сек)</span>';
                } else {
                    resultDiv.innerHTML = `<span class="error">Ошибка: ${err.message}</span>`;
                }
            });
        }
        
        function refreshStats() {
            const resultDiv = document.getElementById('stats-result');
            resultDiv.innerHTML = 'Загрузка статистики...';
            
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    let html = '<h3>Статистика токенизатора:</h3>';
                    html += '<table style="width:100%; border-collapse:collapse;">';
                    html += '<tr><th style="text-align:left; padding:8px; background:#f0f0f0;">Параметр</th><th style="text-align:right; padding:8px; background:#f0f0f0;">Значение</th></tr>';
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Размер словаря</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.vocab_size}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Правил слияния</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.merges_count}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Вызовов encode</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.encode_calls}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Вызовов decode</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.decode_calls}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Всего токенов</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.total_tokens}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Попаданий в кэш</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.cache_hits}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Промахов кэша</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${data.cache_misses}</strong></td></tr>`;
                    html += `<tr><td style="padding:8px; border-bottom:1px solid #ddd;">Эффективность кэша</td><td style="text-align:right; padding:8px; border-bottom:1px solid #ddd;"><strong>${(data.cache_hit_rate).toFixed(1)}%</strong></td></tr>`;
                    html += '</table>';
                    
                    resultDiv.innerHTML = html;
                })
                .catch(err => {
                    resultDiv.innerHTML = `<span class="error">Ошибка загрузки статистики: ${err.message}</span>`;
                });
        }
        
        function resetStats() {
            showConfirm('Вы уверены, что хотите сбросить всю статистику?', function() {
                const resetBtn = document.querySelector('.secondary[onclick="resetStats()"]');
                const originalText = resetBtn.textContent;
                resetBtn.textContent = 'Сброс...';
                resetBtn.disabled = true;
                
                fetch('/api/stats/reset', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'}
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showNotification('Статистика успешно сброшена!', 'success');
                        refreshStats();
                    } else {
                        showNotification('Ошибка при сбросе статистики', 'error');
                    }
                })
                .catch(err => {
                    showNotification('Ошибка соединения: ' + err.message, 'error');
                })
                .finally(() => {
                    resetBtn.textContent = originalText;
                    resetBtn.disabled = false;
                });
            });
        }
    </script>
</body>
</html>
)rawliteral";
}

// ============================================================================
// Основная функция
// ============================================================================

/**
 * @brief Точка входа в программу
 * 
 * @param argc Количество аргументов
 * @param argv Аргументы командной строки
 * @return int 0 при успешном запуске, 1 при ошибке
 */
int main(int argc, char* argv[]) {
    // ============================================================================
    // Парсинг аргументов командной строки
    // ============================================================================
    
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
            std::cout << "--port PORT    - Порт для сервера (по умолчанию: 8080)\n";
            std::cout << "--threads N    - Количество потоков (по умолчанию: все)\n";
            std::cout << "--model PATH   - Путь к файлу словаря\n";
            std::cout << "--merges PATH  - Путь к файлу слияний\n";
            std::cout << "--rate-limit N - Лимит запросов в минуту (по умолчанию: 100)\n";
            std::cout << "--cache-size N - Размер кэша (по умолчанию: 1000)\n";
            std::cout << "--help         - Показать справку\n";
            return 0;
        }
    }
    
    // Установка обработчика сигналов
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // ============================================================================
    // Инициализация токенизатора
    // ============================================================================
    
    std::cout << "============================================================\n";
    std::cout << "BPE TOKENIZER WEB SERVER v3.3.0\n";
    std::cout << "============================================================\n\n";
    
    std::cout << "Загрузка модели..." << std::endl;
    
    auto tokenizer = std::make_shared<FastBPETokenizer>(
        TokenizerConfig{10000, 10000, true}
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
        std::cout << "Проверка: " << paths.vocab << " и " << paths.merges << std::endl;
        
        if (std::ifstream(paths.vocab).good() && std::ifstream(paths.merges).good()) {
            std::cout << "Файлы найдены! Загрузка..." << std::endl;
            
            if (tokenizer->load(paths.vocab, paths.merges)) {
                loaded = true;
                model_path = paths.vocab;
                merges_path = paths.merges;
                std::cout << "Модель успешно загружена!" << std::endl;
                std::cout << "Словарь: " << paths.vocab << std::endl;
                std::cout << "Слияния: " << paths.merges << std::endl;
                break;
            } else {
                std::cout << "Не удалось загрузить модель из файлов!" << std::endl;
            }
        }
    }

    if (!loaded) {
        std::cerr << "\nОШИБКА: Не удалось загрузить модель!" << std::endl;
        std::cerr << "\nПроверены следующие пути:" << std::endl;
        for (const auto& paths : candidates) {
            std::cerr << "vocab:  " << paths.vocab << std::endl;
            std::cerr << "merges: " << paths.merges << std::endl;
        }
        std::cerr << "\nИспользуйте аргументы --model и --merges для указания путей" << std::endl;
        return 1;
    }
    
    std::cout << "\nИнформация о модели:" << std::endl;
    std::cout << "- Размер словаря: " << tokenizer->vocab_size() << std::endl;
    std::cout << "- Правил слияния: " << tokenizer->merges_count() << std::endl;
    std::cout << std::endl;
    
    // ============================================================================
    // Инициализация кэша
    // ============================================================================
    
    auto cache = std::make_shared<LRUCache<std::string, std::string>>(cache_size);
    std::cout << "Кэш инициализирован!" << std::endl;
    std::cout << "Размер: " << cache_size << std::endl;
    std::cout << std::endl;
    
    // ============================================================================
    // Настройка сервера Crow
    // ============================================================================
    
    // Создаем сервер с middleware - используем наш кастомный логгер
    AppType app;
    
    g_app = &app;
    
    // Настройка CORS
    auto& cors = app.get_middleware<crow::CORSHandler>();
    cors.global()
        .headers("Content-Type", "Authorization", "X-Requested-With")
        .methods("POST"_method, "GET"_method, "OPTIONS"_method, "DELETE"_method)
        .origin("*");

    // ============================================================================
    // Маршруты API
    // ============================================================================

    // Главная страница
    CROW_ROUTE(app, "/")
    ([]() {
        return crow::response(get_html_page());
    });

    // Swagger UI страница
    CROW_ROUTE(app, "/swagger")
    ([]() {
        std::string html = R"(
    <!DOCTYPE html>
    <html>
    <head>
        <title>BPE Tokenizer API</title>
        <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css">
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js"></script>
        <script>
            SwaggerUIBundle({
                url: '/swagger.json',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.presets.standalone
                ]
            });
        </script>
    </body>
    </html>
        )";
        return crow::response(html);
    });
    
    // Swagger JSON спецификация
    CROW_ROUTE(app, "/swagger.json")
    ([]() {
        crow::json::wvalue spec;
        
        spec["openapi"] = "3.0.0";
        spec["info"]["title"] = "BPE Tokenizer API";
        spec["info"]["description"] = "API для токенизации C++ кода с использованием BPE";
        spec["info"]["version"] = "3.2.0";
        
        spec["servers"][0]["url"] = "http://localhost:8080";
        spec["servers"][0]["description"] = "Local server";
        
        spec["paths"]["/api/tokenize"]["post"]["summary"] = "Токенизация текста";
        spec["paths"]["/api/tokenize"]["post"]["requestBody"]["required"] = true;
        spec["paths"]["/api/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["type"] = "object";
        spec["paths"]["/api/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["text"]["type"] = "string";
        spec["paths"]["/api/tokenize"]["post"]["responses"]["200"]["description"] = "Успешная токенизация";
        
        spec["paths"]["/api/detokenize"]["post"]["summary"] = "Декодирование токенов";
        spec["paths"]["/api/detokenize"]["post"]["requestBody"]["required"] = true;
        spec["paths"]["/api/detokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["type"] = "object";
        spec["paths"]["/api/detokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["tokens"]["type"] = "array";
        spec["paths"]["/api/detokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["tokens"]["items"]["type"] = "integer";
        spec["paths"]["/api/detokenize"]["post"]["responses"]["200"]["description"] = "Успешное декодирование";
        
        spec["paths"]["/api/batch/tokenize"]["post"]["summary"] = "Пакетная токенизация";
        spec["paths"]["/api/batch/tokenize"]["post"]["requestBody"]["required"] = true;
        spec["paths"]["/api/batch/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["type"] = "object";
        spec["paths"]["/api/batch/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["texts"]["type"] = "array";
        spec["paths"]["/api/batch/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]["texts"]["items"]["type"] = "string";
        spec["paths"]["/api/batch/tokenize"]["post"]["responses"]["200"]["description"] = "Успешная пакетная обработка";
        
        spec["paths"]["/api/stats"]["get"]["summary"] = "Статистика токенизатора";
        spec["paths"]["/api/stats"]["get"]["responses"]["200"]["description"] = "Статистика";
        
        spec["paths"]["/api/stats/reset"]["post"]["summary"] = "Сброс статистики";
        spec["paths"]["/api/stats/reset"]["post"]["responses"]["200"]["description"] = "Статистика сброшена";
        
        spec["paths"]["/api/health"]["get"]["summary"] = "Проверка работоспособности";
        spec["paths"]["/api/health"]["get"]["responses"]["200"]["description"] = "Сервер работает";
        
        return crow::response(spec.dump());
    });

    // Токенизация
    CROW_ROUTE(app, "/api/tokenize")
    .methods("POST"_method)
    ([&tokenizer, &cache](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json || !json.has("text")) {
            return crow::response(400, crow::json::wvalue{{"error", "Missing 'text' field"}, {"success", false}});
        }
        
        std::string text = json["text"].s();
        if (text.empty()) {
            return crow::response(400, crow::json::wvalue{{"error", "Empty text"}, {"success", false}});
        }
        
        auto tokens = tokenizer->encode(text);
        
        crow::json::wvalue result;
        result["tokens"] = std::vector<uint32_t>(tokens.begin(), tokens.end());
        result["count"] = tokens.size();
        result["success"] = true;
        
        return crow::response(result);
    });

    // Декодирование
    CROW_ROUTE(app, "/api/detokenize")
    .methods("POST"_method)
    ([&tokenizer](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json || !json.has("tokens")) {
            return crow::response(400, crow::json::wvalue{{"error", "Missing 'tokens' field"}, {"success", false}});
        }
        
        std::vector<uint32_t> tokens;
        for (const auto& t : json["tokens"]) {
            tokens.push_back(static_cast<uint32_t>(t.i()));
        }
        
        if (tokens.empty()) {
            return crow::response(400, crow::json::wvalue{{"error", "Empty tokens array"}, {"success", false}});
        }
        
        std::string text = tokenizer->decode(tokens);
        
        crow::json::wvalue result;
        result["text"] = text;
        result["count"] = tokens.size();
        result["success"] = true;
        
        return crow::response(result);
    });

    // Пакетная токенизация
    CROW_ROUTE(app, "/api/batch/tokenize")
    .methods("POST"_method)
    ([&tokenizer](const crow::request& req) {
        auto json = crow::json::load(req.body);
        if (!json || !json.has("texts")) {
            return crow::response(400, crow::json::wvalue{{"error", "Missing 'texts' field"}, {"success", false}});
        }
        
        std::vector<std::string> texts;
        for (const auto& t : json["texts"]) {
            texts.push_back(t.s());
        }
        
        if (texts.empty()) {
            return crow::response(400, crow::json::wvalue{{"error", "Empty texts array"}, {"success", false}});
        }
        
        crow::json::wvalue::list results;
        
        for (const auto& text : texts) {
            auto tokens = tokenizer->encode(text);
            
            crow::json::wvalue item;
            item["text"] = text;
            item["token_count"] = tokens.size();
            
            crow::json::wvalue::list token_list;
            for (auto t : tokens) {
                token_list.push_back(t);
            }
            item["tokens"] = std::move(token_list);
            
            results.push_back(std::move(item));
        }
        
        crow::json::wvalue result;
        result["results"] = std::move(results);
        result["count"] = texts.size();
        result["success"] = true;
        
        return crow::response(result);
    });

    // Статистика
    CROW_ROUTE(app, "/api/stats")
    ([&tokenizer]() {
        auto stats = tokenizer->stats();
        
        crow::json::wvalue result;
        result["vocab_size"] = tokenizer->vocab_size();
        result["merges_count"] = tokenizer->merges_count();
        result["cache_hits"] = stats.cache_hits;
        result["cache_misses"] = stats.cache_misses;
        result["cache_hit_rate"] = stats.cache_hit_rate() * 100;
        result["encode_calls"] = stats.encode_calls;
        result["decode_calls"] = stats.decode_calls;
        result["total_tokens"] = stats.total_tokens_processed;
        
        return crow::response(result);
    });

    // Сброс статистики
    CROW_ROUTE(app, "/api/stats/reset")
    .methods("POST"_method)
    ([&tokenizer]() {
        tokenizer->reset_stats();
        crow::json::wvalue result;
        result["success"] = true;
        return crow::response(result);
    });

    // Проверка здоровья
    CROW_ROUTE(app, "/api/health")
    ([]() {
        crow::json::wvalue result;
        result["status"] = "healthy";
        result["timestamp"] = std::time(nullptr);
        result["version"] = "3.2.0";
        
        return crow::response(result);
    });    

    // Статика для Swagger UI (опционально)
    CROW_ROUTE(app, "/swagger-ui.css")
    ([]() {
        crow::response res(302);
        res.add_header("Location", "https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css");
        return res;
    });

    CROW_ROUTE(app, "/swagger-ui-bundle.js")
    ([]() {
        crow::response res(302);
        res.add_header("Location", "https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js");
        return res;
    });

    // ============================================================================
    // Запуск сервера
    // ============================================================================
    
    std::cout << "Сервер запускается на http://localhost:" << port << std::endl;
    std::cout << "Документация:   http://localhost:" << port << "/" << std::endl;
    std::cout << "Swagger UI:     http://localhost:" << port << "/swagger" << std::endl;
    std::cout << "Потоков:        " << threads << std::endl;
    std::cout << "Лимит запросов: " << rate_limit << " в минуту" << std::endl;
    std::cout << "Размер кэша:    " << cache_size << std::endl;
    std::cout << "Нажмите Ctrl+C для остановки!" << std::endl;
    std::cout << std::endl;
    
    app.signal_clear();
    app.port(port).multithreaded().run();
    
    std::cout << "\nСервер остановлен!" << std::endl;
    
    return 0;
}