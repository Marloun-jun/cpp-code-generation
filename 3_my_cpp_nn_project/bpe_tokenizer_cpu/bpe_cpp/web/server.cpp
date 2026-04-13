/**
 * @file server.cpp
 * @brief Простой HTTP сервер для BPE токенизатора на чистых сокетах
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Легковесный веб-сервер, реализованный на чистых сокетах
 *          без использования внешних библиотек. Предоставляет REST API
 *          для токенизации и декодирования C++ кода.
 * 
 *          **Архитектура:**
 *          ┌───────────────────────────────────────────────┐
 *          │                Клиент (браузер)               │
 *          │                       │                       │
 *          │    ┌───────────────────────────────────┐      │
 *          │    │      epoll (асинхронный I/O)      │      │
 *          │    └───────────────────────────────────┘      │
 *          │                       │                       │
 *          │   ┌───────────────────────────────────────┐   │
 *          │   │         ThreadPool (пул потоков)      │   │
 *          │   │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │   │
 *          │   │  │worker│ │worker│ │worker│ │worker│  │   │
 *          │   │  │  1   │ │  2   │ │  3   │ │  N   │  │   │
 *          │   │  └──────┘ └──────┘ └──────┘ └──────┘  │   │
 *          │   └───────────────────────────────────────┘   │
 *          │                       │                       │
 *          │             FastBPETokenizer (общий)          │
 *          └───────────────────────────────────────────────┘
 * 
 *          **Особенности:**
 *          - Полностью самодостаточный - только стандартная библиотека C++
 *          - Многопоточная обработка запросов (пул потоков)
 *          - Асинхронный I/O через epoll (Linux)
 *          - Поддержка CORS для кросс-доменных запросов
 *          - Красивый веб-интерфейс для тестирования
 *          - Статистика в реальном времени
 *          - Graceful shutdown
 *          - Лимиты на размер запросов (защита от DoS)
 * 
 *          **API эндпоинты:**
 *          ┌─────────┬─────────────────────┬───────────────────────┐
 *          │ Метод   │ Путь                │ Описание              │
 *          ├─────────┼─────────────────────┼───────────────────────┤
 *          │ GET     │ /                   │ HTML интерфейс        │
 *          │ POST    │ /api/tokenize       │ Токенизация текста    │
 *          │ POST    │ /api/detokenize     │ Декодирование токенов │
 *          │ POST    │ /api/batch/tokenize │ Пакетная токенизация  │
 *          │ GET     │ /api/stats          │ Статистика            │
 *          │ POST    │ /api/stats/reset    │ Сброс статистики      │
 *          │ GET     │ /api/health         │ Проверка здоровья     │
 *          │ OPTIONS │ *                   │ CORS preflight        │
 *          └─────────┴─────────────────────┴───────────────────────┘
 * 
 * @compile g++ -std=c++17 -Iinclude -O3 -pthread server.cpp -o server
 * @run ./server [--port PORT] [--threads N] [--vocab PATH] [--merges PATH] [--cache-size N]
 * 
 * @note Требует наличия файлов модели в одной из директорий:
 *       - ../models/bpe_10000/cpp_vocab.json
 *       - models/bpe_10000/cpp_vocab.json
 *       - ../../bpe_cpp/models/vocab_trained.json
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "fast_tokenizer.hpp"
#include "utils.hpp"
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <regex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace bpe;
using json = nlohmann::json;

// ============================================================================
// Константы и глобальные переменные
// ============================================================================

const size_t MAX_REQUEST_SIZE = 10 * 1024 * 1024;    ///< Максимальный размер запроса (10 МБ)
const int MAX_HEADERS_SIZE = 64 * 1024;              ///< Максимальный размер заголовков (64 КБ)
const int MAX_CONNECTIONS = 1000;                    ///< Максимальное количество одновременных соединений
const int READ_TIMEOUT_SEC = 30;                     ///< Таймаут чтения (секунды)
const int WRITE_TIMEOUT_SEC = 30;                    ///< Таймаут записи (секунды)

std::atomic<bool> g_running{true};    ///< Флаг работы сервера
int g_server_fd = -1;                 ///< Дескриптор серверного сокета
int g_epoll_fd = -1;                  ///< Дескриптор epoll

/**
 * @brief Обработчик сигналов (SIGINT, SIGTERM)
 */
void signal_handler(int) {
    std::cout << "\nПолучен сигнал остановки. Завершение работы..." << std::endl;
    g_running = false;
    if (g_server_fd != -1) {
        close(g_server_fd);
    }
    if (g_epoll_fd != -1) {
        close(g_epoll_fd);
    }
}

// ============================================================================
// Пул потоков
// ============================================================================

/**
 * @brief Пул потоков для параллельной обработки запросов
 * 
 * Реализует паттерн "производитель-потребитель" с очередью задач.
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;           ///< Рабочие потоки
    std::queue<std::function<void()>> tasks_;    ///< Очередь задач
    mutable std::mutex queue_mutex_;             ///< Мьютекс для очереди
    std::condition_variable condition_;          ///< Условная переменная
    std::atomic<bool> stop_{false};              ///< Флаг остановки
    
public:
    /**
     * @brief Конструктор
     * @param threads Количество потоков
     */
    explicit ThreadPool(size_t threads) 
        : workers_()
        , tasks_()
        , queue_mutex_()
        , condition_()
        , stop_(false)
    {
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this] { worker_thread(); });
        }
    }
    
    /**
     * @brief Деструктор
     */
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
    /**
     * @brief Добавить задачу в очередь
     * @param task Задача
     */
    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.push(std::move(task));
        }
        condition_.notify_one();
    }
    
    /**
     * @brief Получить размер очереди
     */
    size_t queue_size() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }
    
private:
    /**
     * @brief Функция рабочего потока
     */
    void worker_thread() {
        while (true) {
            std::function<void()> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                
                // Ждем, пока не появится задача или не будет сигнала остановки
                condition_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return stop_ || !tasks_.empty();
                });
                
                if (stop_ && tasks_.empty()) {
                    return;
                }
                
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
            }
            
            if (task) {
                try {
                    task();
                } catch (const std::exception& e) {
                    std::cerr << "Ошибка в задаче: " << e.what() << std::endl;
                }
            }
        }
    }
};

// ============================================================================
// Класс для формирования HTTP ответов
// ============================================================================

/**
 * @brief Вспомогательный класс для создания HTTP ответов
 */
class HttpResponse {
public:
    /**
     * @brief Экранирование для JSON
     */
    static std::string json_escape(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '"':  result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\b': result += "\\b"; break;
                case '\f': result += "\\f"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", c);
                        result += buf;
                    } else {
                        result += c;
                    }
            }
        }
        return result;
    }
    
    /**
     * @brief HTML ответ
     */
    static std::string html(const std::string& body, int status = 200) {
        return response(status, "text/html; charset=utf-8", body);
    }
    
    /**
     * @brief JSON ответ
     */
    static std::string json(const std::string& body, int status = 200) {
        return response(status, "application/json; charset=utf-8", body);
    }
    
    /**
     * @brief Простой текстовый ответ
     */
    static std::string plain(const std::string& body, int status = 200) {
        return response(status, "text/plain; charset=utf-8", body);
    }
    
    /**
     * @brief JSON ответ с ошибкой
     */
    static std::string error(const std::string& message, int status = 400) {
        std::ostringstream oss;
        oss << "{\"error\":\"" << json_escape(message) 
            << "\",\"success\":false,\"status\":" << status << "}";
        return json(oss.str(), status);
    }
    
    /**
     * @brief 404 Not Found
     */
    static std::string not_found(const std::string& path) {
        return error("Resource not found: " + path, 404);
    }
    
    /**
     * @brief 405 Method Not Allowed
     */
    static std::string method_not_allowed(const std::string& method) {
        return error("Method not allowed: " + method, 405);
    }
    
    /**
     * @brief 413 Payload Too Large
     */
    static std::string payload_too_large() {
        return error("Request entity too large", 413);
    }
    
    /**
     * @brief 429 Too Many Requests
     */
    static std::string too_many_requests() {
        return error("Too many requests", 429);
    }
    
    /**
     * @brief 500 Internal Server Error
     */
    static std::string internal_error(const std::string& details = "") {
        std::string msg = "Internal server error";
        if (!details.empty()) {
            msg += ": " + details;
        }
        return error(msg, 500);
    }
    
private:
    /**
     * @brief Сформировать полный HTTP ответ
     */
    static std::string response(int status, const std::string& content_type, const std::string& body) {
        std::ostringstream oss;
        oss << "HTTP/1.1 " << status << " " << get_status_message(status) << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.length() << "\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            << "Access-Control-Allow-Headers: Content-Type\r\n"
            << "Access-Control-Max-Age: 86400\r\n"
            << "Connection: close\r\n"
            << "Server: BPE-Tokenizer/3.4.0\r\n"
            << "X-Content-Type-Options: nosniff\r\n"
            << "\r\n"
            << body;
        return oss.str();
    }
    
    /**
     * @brief Получить текстовое описание HTTP статуса
     */
    static std::string get_status_message(int status) {
        switch(status) {
            case 200: return "OK";
            case 400: return "Bad Request";
            case 403: return "Forbidden";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 413: return "Payload Too Large";
            case 429: return "Too Many Requests";
            case 500: return "Internal Server Error";
            default: return "Unknown";
        }
    }
};

// ============================================================================
// Парсинг HTTP запросов
// ============================================================================

/**
 * @brief Структура, представляющая HTTP запрос
 */
struct HttpRequest {
    std::string method;                                 ///< HTTP метод (GET, POST, etc)
    std::string path;                                   ///< Путь запроса
    std::string version;                                ///< Версия HTTP
    std::map<std::string, std::string> headers;         ///< Заголовки
    std::string body;                                   ///< Тело запроса
    std::map<std::string, std::string> query_params;    ///< Query параметры
    
    bool is_valid() const {
        return !method.empty() && !path.empty();
    }
    
    std::string get_header(const std::string& name) const {
        auto it = headers.find(name);
        return it != headers.end() ? it->second : "";
    }
    
    std::string get_query_param(const std::string& name) const {
        auto it = query_params.find(name);
        return it != query_params.end() ? it->second : "";
    }
};

/**
 * @brief Разобрать query параметры из URL
 */
std::map<std::string, std::string> parse_query_params(const std::string& query) {
    std::map<std::string, std::string> params;
    std::istringstream iss(query);
    std::string pair;
    
    while (std::getline(iss, pair, '&')) {
        size_t eq = pair.find('=');
        if (eq != std::string::npos) {
            std::string key = pair.substr(0, eq);
            std::string value = pair.substr(eq + 1);
            params[key] = value;
        }
    }
    
    return params;
}

/**
 * @brief Разобрать сырой HTTP запрос
 */
HttpRequest parse_request(const std::string& raw) {
    HttpRequest req;
    
    size_t header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return req;
    }
    
    std::string headers_str = raw.substr(0, header_end);
    std::istringstream iss(headers_str);
    std::string line;
    
    // Парсим первую строку: "GET /path HTTP/1.1"
    if (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream line_ss(line);
        line_ss >> req.method >> req.path >> req.version;
        
        // Разбираем query параметры
        size_t qpos = req.path.find('?');
        if (qpos != std::string::npos) {
            std::string query = req.path.substr(qpos + 1);
            req.path = req.path.substr(0, qpos);
            req.query_params = parse_query_params(query);
        }
    }
    
    // Парсим заголовки
    while (std::getline(iss, line) && line != "\r") {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            
            // Удаляем пробелы в начале значения
            size_t start = value.find_first_not_of(" \t");
            if (start != std::string::npos) {
                value = value.substr(start);
            }
            
            req.headers[key] = value;
        }
    }
    
    // Парсим тело запроса
    if (header_end + 4 < raw.length()) {
        req.body = raw.substr(header_end + 4);
    }
    
    return req;
}

// ============================================================================
// HTML страница интерфейса (с кастомными уведомлениями)
// ============================================================================

/**
 * @brief Возвращает HTML страницу с пользовательским интерфейсом
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
        
        .result p, .result ul, .result li {
            margin: 0;
            padding: 0;
        }

        .result p {
            margin-top: 8px;
        }

        .result ul {
            margin-top: 4px;
            margin-bottom: 8px;
            padding-left: 20px;
        }

        .result li {
            margin: 2px 0;
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
                <p><strong>BPE Tokenizer Web Service</strong></p><p>Версия: 3.4.0</p><p>API эндпоинты:</p><ul><li><code>POST /api/tokenize</code> - токенизация текста</li><li><code>POST /api/detokenize</code> - декодирование токенов</li><li><code>POST /api/batch/tokenize</code> - пакетная токенизация</li><li><code>GET /api/stats</code> - статистика токенизатора</li><li><code>POST /api/stats/reset</code> - сброс статистики</li><li><code>GET /api/health</code> - проверка работоспособности</li></ul><p><strong>Технические детали:</strong></p><ul><li>Многопоточная обработка запросов</li><li>Защита от DoS-атак</li><li>Поддержка CORS</li>
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
// Обработка клиентских соединений
// ============================================================================

/**
 * @brief Установить таймауты на сокет
 */
bool set_socket_timeout(int fd, int sec) {
    struct timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = 0;
    
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
        return false;
    }
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
        return false;
    }
    return true;
}

/**
 * @brief Прочитать запрос от клиента с учетом лимитов
 */
std::string read_request(int client_fd) {
    std::string request;
    char buffer[4096];
    size_t total_read = 0;
    bool headers_complete = false;
    
    while (g_running) {
        ssize_t bytes = read(client_fd, buffer, sizeof(buffer));
        
        if (bytes < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            return "";
        }
        
        if (bytes == 0) {
            break;
        }
        
        request.append(buffer, bytes);
        total_read += bytes;
        
        if (total_read > MAX_REQUEST_SIZE) {
            return "TOO_LARGE";
        }
        
        if (!headers_complete && request.find("\r\n\r\n") != std::string::npos) {
            headers_complete = true;
            
            size_t header_end = request.find("\r\n\r\n");
            if (header_end > static_cast<size_t>(MAX_HEADERS_SIZE)) {
                return "HEADERS_TOO_LARGE";
            }
            
            size_t content_length_pos = request.find("Content-Length:");
            if (content_length_pos != std::string::npos) {
                size_t start = request.find(':', content_length_pos) + 1;
                size_t end = request.find("\r\n", start);
                if (end != std::string::npos) {
                    std::string len_str = request.substr(start, end - start);
                    try {
                        size_t content_length = std::stoul(len_str);
                        if (content_length > MAX_REQUEST_SIZE) {
                            return "TOO_LARGE";
                        }
                    } catch (...) {}
                }
            }
        }
        
        if (headers_complete) {
            size_t header_end = request.find("\r\n\r\n");
            size_t content_length = request.length() - header_end - 4;
            
            size_t cl_pos = request.find("Content-Length:");
            if (cl_pos != std::string::npos && cl_pos < header_end) {
                size_t start = request.find(':', cl_pos) + 1;
                size_t end = request.find("\r\n", start);
                if (end != std::string::npos) {
                    std::string len_str = request.substr(start, end - start);
                    try {
                        size_t expected = std::stoul(len_str);
                        if (content_length >= expected) {
                            break;
                        }
                    } catch (...) {}
                }
            } else {
                break;
            }
        }
    }
    
    return request;
}

/**
 * @brief Обработать запрос клиента
 */
void handle_client(int client_fd, FastBPETokenizer& tokenizer) {
    
    set_socket_timeout(client_fd, READ_TIMEOUT_SEC);
    
    std::string request_str = read_request(client_fd);
    
    if (request_str.empty()) {
        close(client_fd);
        return;
    }
    
    if (request_str == "TOO_LARGE") {
        std::string error = HttpResponse::payload_too_large();
        send(client_fd, error.c_str(), error.length(), 0);
        close(client_fd);
        return;
    }
    
    if (request_str == "HEADERS_TOO_LARGE") {
        std::string error = HttpResponse::error("Headers too large", 431);
        send(client_fd, error.c_str(), error.length(), 0);
        close(client_fd);
        return;
    }
    
    HttpRequest req = parse_request(request_str);
    
    if (!req.is_valid()) {
        std::string error = HttpResponse::error("Invalid request", 400);
        send(client_fd, error.c_str(), error.length(), 0);
        close(client_fd);
        return;
    }
    
    std::cout << "[" << std::this_thread::get_id() << "] " 
              << req.method << " " << req.path << std::endl;
    
    std::string response;
    
    // CORS preflight
    if (req.method == "OPTIONS") {
        response = HttpResponse::plain("", 200);
    }
    // Токенизация
    else if (req.path == "/api/tokenize" && req.method == "POST") {
        try {
            auto j = json::parse(req.body);
            
            if (j.contains("text") && j["text"].is_string()) {
                std::string text = j["text"];
                
                if (text.empty()) {
                    response = HttpResponse::error("Empty text field");
                } else {
                    auto tokens = tokenizer.encode(text);
                    
                    json result;
                    result["tokens"] = tokens;
                    result["count"] = tokens.size();
                    result["success"] = true;
                    
                    response = HttpResponse::json(result.dump());
                }
            } else {
                response = HttpResponse::error("Missing or invalid 'text' field");
            }
        } catch (const std::exception& e) {
            response = HttpResponse::internal_error(e.what());
        }
    }
    // Декодирование
    else if (req.path == "/api/detokenize" && req.method == "POST") {
        try {
            auto j = json::parse(req.body);
            
            std::vector<uint32_t> tokens;
            
            if (j.contains("tokens") && j["tokens"].is_array()) {
                tokens = j["tokens"].get<std::vector<uint32_t>>();
            }
            
            if (tokens.empty()) {
                response = HttpResponse::error("Missing or empty 'tokens' field");
            } else {
                std::string text = tokenizer.decode(tokens);
                
                json result;
                result["text"] = text;
                result["count"] = tokens.size();
                result["success"] = true;
                
                response = HttpResponse::json(result.dump());
            }
        } catch (const std::exception& e) {
            response = HttpResponse::internal_error(e.what());
        }
    }
    // Пакетная токенизация
    else if (req.path == "/api/batch/tokenize" && req.method == "POST") {
        try {
            auto j = json::parse(req.body);
            
            if (j.contains("texts") && j["texts"].is_array()) {
                std::vector<std::string> texts = j["texts"].get<std::vector<std::string>>();
                
                if (texts.empty()) {
                    response = HttpResponse::error("Empty texts array");
                } else {
                    if (texts.size() > 100) {
                        response = HttpResponse::error("Too many texts (max 100)");
                    } else {
                        json results = json::array();
                        bool error = false;
                        std::string error_msg;
                        
                        for (const auto& text : texts) {
                            try {
                                auto tokens = tokenizer.encode(text);
                                
                                json item;
                                item["text"] = text;
                                item["token_count"] = tokens.size();
                                item["tokens"] = tokens;
                                
                                results.push_back(item);
                            } catch (const std::exception& e) {
                                error = true;
                                error_msg = e.what();
                                break;
                            }
                        }
                        
                        if (!error) {
                            json result;
                            result["results"] = results;
                            result["count"] = texts.size();
                            result["success"] = true;
                            
                            response = HttpResponse::json(result.dump());
                        } else {
                            response = HttpResponse::internal_error(error_msg);
                        }
                    }
                }
            } else {
                response = HttpResponse::error("Missing or invalid 'texts' field");
            }
        } catch (const json::parse_error& e) {
            response = HttpResponse::error("Invalid JSON: " + std::string(e.what()));
        } catch (const std::exception& e) {
            response = HttpResponse::internal_error(e.what());
        }
    }
    // Статистика
    else if (req.path == "/api/stats" && req.method == "GET") {
        auto stats = tokenizer.stats();
        
        json result;
        result["vocab_size"] = tokenizer.vocab_size();
        result["merges_count"] = tokenizer.merges_count();
        result["cache_hits"] = stats.cache_hits;
        result["cache_misses"] = stats.cache_misses;
        result["cache_hit_rate"] = stats.cache_hit_rate() * 100;
        result["encode_calls"] = stats.encode_calls;
        result["decode_calls"] = stats.decode_calls;
        result["total_tokens"] = stats.total_tokens_processed;
        
        response = HttpResponse::json(result.dump());
    }
    // Сброс статистики
    else if (req.path == "/api/stats/reset" && req.method == "POST") {
        tokenizer.reset_stats();
        json result;
        result["success"] = true;
        response = HttpResponse::json(result.dump());
    }
    // Проверка здоровья
    else if (req.path == "/api/health" && req.method == "GET") {
        json result;
        result["status"] = "healthy";
        result["timestamp"] = std::time(nullptr);
        result["version"] = "3.4.0";
        
        response = HttpResponse::json(result.dump());
    }
    // Главная страница
    else if (req.path == "/" || req.path.empty()) {
        response = HttpResponse::html(get_html_page());
    }
    // 404
    else {
        response = HttpResponse::not_found(req.path);
    }
    
    set_socket_timeout(client_fd, WRITE_TIMEOUT_SEC);
    send(client_fd, response.c_str(), response.length(), 0);
    
    close(client_fd);
}

// ============================================================================
// Поиск файлов модели
// ============================================================================

/**
 * @brief Найти файлы модели в стандартных расположениях
 */
bool find_model_files(FastBPETokenizer& tokenizer, std::string& vocab_path, 
                      std::string& merges_path, int argc, char* argv[]) {
    
    // Проверяем аргументы командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        }
    }
    
    // Проверяем переменные окружения
    const char* env_vocab = std::getenv("BPE_VOCAB");
    const char* env_merges = std::getenv("BPE_MERGES");
    
    if (env_vocab && env_merges) {
        vocab_path = env_vocab;
        merges_path = env_merges;
    }
    
    // Если указаны явно - пробуем загрузить
    if (!vocab_path.empty() && !merges_path.empty()) {
        std::ifstream vfile(vocab_path);
        std::ifstream mfile(merges_path);
        
        if (vfile.good() && mfile.good()) {
            if (tokenizer.load(vocab_path, merges_path)) {
                std::cout << "Модель загружена: " << vocab_path << std::endl;
                return true;
            }
        }
    }
    
    // Список возможных путей
    std::vector<int> model_sizes = {8000, 10000, 12000};
    std::vector<std::pair<std::string, std::string>> candidates;
    
    for (int size : model_sizes) {
        candidates.emplace_back(
            "../models/bpe_" + std::to_string(size) + "/cpp_vocab.json",
            "../models/bpe_" + std::to_string(size) + "/cpp_merges.txt"
        );
        candidates.emplace_back(
            "models/bpe_" + std::to_string(size) + "/cpp_vocab.json",
            "models/bpe_" + std::to_string(size) + "/cpp_merges.txt"
        );
        candidates.emplace_back(
            "../../models/bpe_" + std::to_string(size) + "/cpp_vocab.json",
            "../../models/bpe_" + std::to_string(size) + "/cpp_merges.txt"
        );
    }
    
    candidates.emplace_back("../models/cpp_vocab.json", "../models/cpp_merges.txt");
    candidates.emplace_back("models/cpp_vocab.json", "models/cpp_merges.txt");
    candidates.emplace_back("vocab.json", "merges.txt");
    
    // Пробуем все пути
    for (const auto& [vpath, mpath] : candidates) {
        std::ifstream vfile(vpath);
        std::ifstream mfile(mpath);
        
        if (vfile.good() && mfile.good()) {
            vocab_path = vpath;
            merges_path = mpath;
            
            if (tokenizer.load(vpath, mpath)) {
                std::cout << "Модель загружена: " << vpath << std::endl;
                return true;
            }
        }
    }
    
    std::cerr << "\nМодель не найдена!\n" << std::endl;
    std::cerr << "Проверьте расположение файлов:" << std::endl;
    for (const auto& [vpath, _] : candidates) {
        std::cerr << "  - " << vpath << std::endl;
    }
    
    return false;
}

// ============================================================================
// Основная функция
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "============================================================\n";
    std::cout << "BPE TOKENIZER WEB SERVICE v3.4.0\n";
    std::cout << "============================================================\n\n";
    
    int port = 8080;
    int num_threads = std::thread::hardware_concurrency();
    
    // Парсинг аргументов командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "--port PORT   - Порт для сервера (по умолчанию: 8080)\n";
            std::cout << "--threads N   - Количество потоков (по умолчанию: " 
                      << num_threads << ")\n";
            std::cout << "--vocab PATH  - Путь к файлу словаря\n";
            std::cout << "--merges PATH - Путь к файлу слияний\n";
            std::cout << "--help        - Показать справку\n";
            return 0;
        }
    }
    
    // Установка обработчика сигналов
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Загрузка модели
    FastBPETokenizer tokenizer;
    std::cout << "Загрузка модели..." << std::endl;
    
    std::string vocab_path, merges_path;
    if (!find_model_files(tokenizer, vocab_path, merges_path, argc, argv)) {
        return 1;
    }
    
    std::cout << "Модель загружена!" << std::endl;
    std::cout << "Словарь:        " << vocab_path << std::endl;
    std::cout << "Слияния:        " << merges_path << std::endl;
    std::cout << "Размер словаря: " << tokenizer.vocab_size() << std::endl;
    std::cout << "Правил слияния: " << tokenizer.merges_count() << std::endl;
    std::cout << std::endl;
    
    // Создание пула потоков
    ThreadPool pool(num_threads);
    
    std::cout << "Запуск сервера..." << std::endl;
    std::cout << "Потоков: " << num_threads << std::endl;
    std::cout << std::endl;
    
    // Создание сокета
    g_server_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (g_server_fd < 0) {
        std::cerr << "Ошибка создания сокета!" << std::endl;
        return 1;
    }
    
    int opt = 1;
    if (setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Ошибка настройки сокета!" << std::endl;
        return 1;
    }
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(g_server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Ошибка привязки к порту " << port << "!" << std::endl;
        return 1;
    }
    
    if (listen(g_server_fd, MAX_CONNECTIONS) < 0) {
        std::cerr << "Ошибка прослушивания!" << std::endl;
        return 1;
    }
    
    // Создание epoll
    g_epoll_fd = epoll_create1(0);
    if (g_epoll_fd < 0) {
        std::cerr << "Ошибка создания epoll!" << std::endl;
        return 1;
    }
    
    struct epoll_event event;
    event.events = EPOLLIN | EPOLLET;
    event.data.fd = g_server_fd;
    
    if (epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, g_server_fd, &event) < 0) {
        std::cerr << "Ошибка добавления сокета в epoll!" << std::endl;
        return 1;
    }
    
    std::cout << "Сервер запущен на http://localhost:" << port << std::endl;
    std::cout << "API эндпоинты:" << std::endl;
    std::cout << "    POST /api/tokenize       - токенизация текста" << std::endl;
    std::cout << "    POST /api/detokenize     - декодирование токенов" << std::endl;
    std::cout << "    POST /api/batch/tokenize - пакетная токенизация" << std::endl;
    std::cout << "    GET  /api/stats          - статистика токенизатора" << std::endl;
    std::cout << "    POST /api/stats/reset    - сброс статистики" << std::endl;
    std::cout << "    GET  /api/health         - проверка работоспособности" << std::endl;
    std::cout << "    GET  /                   - веб-интерфейс" << std::endl;
    std::cout << "Нажмите Ctrl+C для остановки!" << std::endl;
    std::cout << std::endl;
    
    std::atomic<int> client_count{0};
    
    // Главный цикл обработки соединений
    while (g_running) {
        struct epoll_event events[32];
        int nfds = epoll_wait(g_epoll_fd, events, 32, 1000);
        
        if (!g_running) break;
        
        for (int i = 0; i < nfds; ++i) {
            // Новое соединение
            if (events[i].data.fd == g_server_fd) {
                while (g_running) {
                    struct sockaddr_in client_addr;
                    socklen_t client_len = sizeof(client_addr);
                    int client_fd = accept4(g_server_fd, (struct sockaddr*)&client_addr,
                                            &client_len, SOCK_NONBLOCK);
                    
                    if (client_fd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;
                        }
                        continue;
                    }
                    
                    client_count++;
                    
                    struct epoll_event client_event;
                    client_event.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
                    client_event.data.fd = client_fd;
                    
                    if (epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, client_fd, &client_event) < 0) {
                        close(client_fd);
                        continue;
                    }
                }
            } 
            // Данные от клиента готовы для чтения
            else {
                int client_fd = events[i].data.fd;
                
                epoll_ctl(g_epoll_fd, EPOLL_CTL_DEL, client_fd, nullptr);
                
                pool.enqueue([client_fd, &tokenizer]() {
                    handle_client(client_fd, tokenizer);
                });
            }
        }
    }
    
    std::cout << "\nОстановка сервера..." << std::endl;
    
    close(g_epoll_fd);
    close(g_server_fd);
    
    std::cout << "Сервер остановлен!" << std::endl;
    std::cout << "Всего обработано запросов: " << client_count << std::endl;
    
    return 0;
}