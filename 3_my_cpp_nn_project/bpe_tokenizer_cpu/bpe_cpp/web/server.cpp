/**
 * @file server.cpp
 * @brief Простой HTTP сервер для BPE токенизатора на чистом сокетах
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.3.0
 * 
 * @details Легковесный веб-сервер, реализованный на чистых сокетах
 *          без использования внешних библиотек. Предоставляет REST API
 *          для токенизации и декодирования C++ кода.
 * 
 *          **Особенности:**
 *          - Полностью самодостаточный - только стандартная библиотека C++
 *          - Многопоточная обработка запросов (пул потоков)
 *          - Поддержка CORS для кросс-доменных запросов
 *          - Красивый веб-интерфейс для тестирования
 *          - Статистика в реальном времени
 *          - Кэширование частых запросов
 *          - Graceful shutdown
 *          - Лимиты на размер запросов (защита от DoS)
 * 
 *          **API эндпоинты:**
 * 
 *          | Метод | Путь | Описание |
 *          |-------|------|----------|
 *          | GET   | `/` | HTML интерфейс |
 *          | POST  | `/api/tokenize` | Токенизация текста |
 *          | POST  | `/api/detokenize` | Декодирование токенов |
 *          | GET   | `/api/stats` | Статистика токенизатора |
 *          | GET   | `/api/health` | Проверка работоспособности |
 *          | GET   | `/api/cache/stats` | Статистика кэша |
 *          | POST  | `/api/cache/clear` | Очистка кэша |
 *          | OPTIONS | * | CORS preflight |
 * 
 *          **Форматы запросов/ответов:**
 * 
 *          Токенизация:
 *          ```json
 *          POST /api/tokenize
 *          {"text": "int main() { return 0; }"}
 *          
 *          Response:
 *          {"tokens":[42,17,33,98,44],"count":5,"success":true}
 *          ```
 * 
 *          Декодирование:
 *          ```json
 *          POST /api/detokenize
 *          {"tokens":[42,17,33,98,44]}
 *          
 *          Response:
 *          {"text":"int main() { return 0; }","count":5,"success":true}
 *          ```
 * 
 *          Статистика:
 *          ```json
 *          GET /api/stats
 *          Response:
 *          {
 *            "vocab_size":8000,
 *            "merges_count":7999,
 *            "cache_hits":1234,
 *            "cache_misses":567,
 *            "cache_hit_rate":68.5,
 *            "encode_calls":123,
 *            "decode_calls":45,
 *            "total_tokens":67890
 *          }
 *          ```
 * 
 * @compile g++ -std=c++17 -Iinclude -O3 -pthread server.cpp -o server
 * @run ./server [--port PORT] [--threads N] [--vocab PATH] [--merges PATH] [--cache-size N]
 * 
 * @note Требует наличия файлов модели в одной из директорий:
 *       - ../models/bpe_8000/cpp_vocab.json
 *       - models/bpe_8000/cpp_vocab.json
 *       - ../../bpe_cpp/models/vocab_trained.json
 */

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/epoll.h>

#include "fast_tokenizer.hpp"
#include "utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <signal.h>
#include <ctime>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <shared_mutex>
#include <regex>
#include <cstdlib>
#include <list>
#include <cstdint>

using namespace bpe;

// ======================================================================
// Константы и глобальные переменные
// ======================================================================

const size_t MAX_REQUEST_SIZE = 10 * 1024 * 1024;  ///< Максимальный размер запроса (10MB)
const int MAX_HEADERS_SIZE = 64 * 1024;            ///< Максимальный размер заголовков (64KB)
const int MAX_CONNECTIONS = 1000;                   ///< Максимальное количество одновременных соединений
const int READ_TIMEOUT_SEC = 30;                    ///< Таймаут чтения (секунды)
const int WRITE_TIMEOUT_SEC = 30;                   ///< Таймаут записи (секунды)
const size_t DEFAULT_CACHE_SIZE = 1000;             ///< Размер кэша по умолчанию

std::atomic<bool> g_running{true};                  ///< Флаг работы сервера
int g_server_fd = -1;                                ///< Дескриптор серверного сокета
int g_epoll_fd = -1;                                 ///< Дескриптор epoll

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

// ======================================================================
// Класс для кэширования ответов
// ======================================================================

/**
 * @brief Класс для кэширования частых запросов
 * @tparam Key Тип ключа
 * @tparam Value Тип значения
 */
template<typename Key, typename Value>
class Cache {
private:
    struct Entry {
        Value value;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::unordered_map<Key, Entry> cache_;
    mutable std::shared_mutex mutex_;
    size_t max_size_;
    std::chrono::seconds ttl_;
    
    // Для LRU
    std::list<Key> lru_list_;
    std::unordered_map<Key, typename std::list<Key>::iterator> lru_map_;
    
public:
    /**
     * @brief Конструктор
     * @param max_size Максимальный размер кэша
     * @param ttl Время жизни записей (секунды)
     */
    Cache(size_t max_size = DEFAULT_CACHE_SIZE, std::chrono::seconds ttl = std::chrono::minutes(5))
        : max_size_(max_size), ttl_(ttl) {}
    
    /**
     * @brief Получить значение из кэша
     * @param key Ключ
     * @param value [out] Значение (если найдено)
     * @return true если значение найдено и не устарело
     */
    bool get(const Key& key, Value& value) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return false;
        }
        
        auto now = std::chrono::steady_clock::now();
        if (now - it->second.timestamp > ttl_) {
            // Запись устарела, нужно удалить
            lock.unlock();
            remove(key);
            return false;
        }
        
        value = it->second.value;
        
        // Обновляем LRU
        std::unique_lock<std::shared_mutex> write_lock(mutex_);
        lru_list_.erase(lru_map_[key]);
        lru_list_.push_front(key);
        lru_map_[key] = lru_list_.begin();
        
        return true;
    }
    
    /**
     * @brief Поместить значение в кэш
     * @param key Ключ
     * @param value Значение
     */
    void put(const Key& key, const Value& value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        
        // Если ключ уже существует, обновляем
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second.value = value;
            it->second.timestamp = now;
            
            // Обновляем LRU
            lru_list_.erase(lru_map_[key]);
            lru_list_.push_front(key);
            lru_map_[key] = lru_list_.begin();
            return;
        }
        
        // Если кэш полный, удаляем самую старую запись
        if (cache_.size() >= max_size_ && !lru_list_.empty()) {
            Key oldest = lru_list_.back();
            lru_list_.pop_back();
            lru_map_.erase(oldest);
            cache_.erase(oldest);
        }
        
        // Добавляем новую запись
        cache_[key] = {value, now};
        lru_list_.push_front(key);
        lru_map_[key] = lru_list_.begin();
    }
    
    /**
     * @brief Удалить запись из кэша
     * @param key Ключ
     */
    void remove(const Key& key) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            lru_list_.erase(lru_map_[key]);
            lru_map_.erase(key);
            cache_.erase(key);
        }
    }
    
    /**
     * @brief Очистить кэш
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_.clear();
        lru_list_.clear();
        lru_map_.clear();
    }
    
    /**
     * @brief Получить размер кэша
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.size();
    }
    
    /**
     * @brief Получить статистику кэша
     */
    std::map<std::string, size_t> stats() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return {
            {"size", cache_.size()},
            {"max_size", max_size_},
            {"ttl_seconds", static_cast<size_t>(ttl_.count())}
        };
    }
};

// ======================================================================
// Пул потоков
// ======================================================================

/**
 * @brief Пул потоков для параллельной обработки запросов
 */
class ThreadPool {
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;  // добавлен mutable и инициализация по умолчанию
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    
public:
    /**
     * @brief Конструктор
     * @param threads Количество потоков
     */
    explicit ThreadPool(size_t threads) 
        : workers_()
        , tasks_()
        , queue_mutex_()  // явная инициализация (можно и не писать)
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

// ======================================================================
// Класс для формирования HTTP ответов (улучшенный)
// ======================================================================

/**
 * @brief Вспомогательный класс для создания HTTP ответов
 */
class HttpResponse {
public:
    /**
     * @brief Экранирование для JSON (публичный статический метод)
     */
    static std::string json_escape(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
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
            << "Access-Control-Max-Age: 86400\r\n"  // 24 часа
            << "Connection: close\r\n"
            << "Server: BPE-Tokenizer/3.3.0\r\n"
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

// ======================================================================
// Парсинг HTTP запросов (улучшенный)
// ======================================================================

/**
 * @brief Структура, представляющая HTTP запрос
 */
struct HttpRequest {
    std::string method;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;
    std::map<std::string, std::string> query_params;
    
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
            
            // URL декодирование
            // В реальном проекте используйте полноценную библиотеку
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
    
    // Находим конец заголовков
    size_t header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        return req;  // Неполный запрос
    }
    
    std::string headers_str = raw.substr(0, header_end);
    std::istringstream iss(headers_str);
    std::string line;
    
    // Парсим первую строку
    if (std::getline(iss, line)) {
        // Убираем \r в конце
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        
        std::istringstream line_ss(line);
        line_ss >> req.method >> req.path >> req.version;
        
        // Разбираем query параметры из path
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
            
            // Убираем пробелы в начале значения
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

// ======================================================================
// Простой JSON парсер (упрощенный, но лучше чем ручной разбор)
// ======================================================================

/**
 * @brief Простой класс для работы с JSON
 * 
 * @note Для production рекомендуется использовать nlohmann/json
 *       или другую полноценную библиотеку
 */
class SimpleJson {
public:
    /**
     * @brief Парсинг JSON строки
     */
    static bool parse(const std::string& json, std::map<std::string, std::string>& result) {
        result.clear();
        
        // Убираем пробелы в начале и конце
        size_t start = json.find_first_not_of(" \t\r\n");
        size_t end = json.find_last_not_of(" \t\r\n");
        
        if (start == std::string::npos || end == std::string::npos) {
            return false;
        }
        
        std::string trimmed = json.substr(start, end - start + 1);
        
        // Проверяем, что это объект
        if (trimmed.empty() || trimmed[0] != '{' || trimmed[trimmed.length() - 1] != '}') {
            return false;
        }
        
        // Убираем внешние скобки
        std::string content = trimmed.substr(1, trimmed.length() - 2);
        
        size_t pos = 0;
        while (pos < content.length()) {
            // Пропускаем пробелы
            while (pos < content.length() && isspace(content[pos])) pos++;
            if (pos >= content.length()) break;
            
            // Парсим ключ
            if (content[pos] != '"') return false;
            pos++;
            
            size_t key_start = pos;
            while (pos < content.length() && content[pos] != '"') {
                if (content[pos] == '\\') pos += 2;
                else pos++;
            }
            if (pos >= content.length()) return false;
            
            std::string key = content.substr(key_start, pos - key_start);
            pos++; // Пропускаем закрывающую кавычку
            
            // Пропускаем пробелы и двоеточие
            while (pos < content.length() && isspace(content[pos])) pos++;
            if (pos >= content.length() || content[pos] != ':') return false;
            pos++;
            while (pos < content.length() && isspace(content[pos])) pos++;
            
            // Парсим значение
            std::string value;
            if (pos < content.length() && content[pos] == '"') {
                // Строковое значение
                pos++;
                size_t val_start = pos;
                while (pos < content.length() && content[pos] != '"') {
                    if (content[pos] == '\\') pos += 2;
                    else pos++;
                }
                if (pos >= content.length()) return false;
                value = content.substr(val_start, pos - val_start);
                pos++;
            } else {
                // Числовое или булево значение
                size_t val_start = pos;
                while (pos < content.length() && content[pos] != ',' && content[pos] != '}') {
                    pos++;
                }
                value = content.substr(val_start, pos - val_start);
                // Убираем пробелы
                size_t vstart = value.find_first_not_of(" \t\r\n");
                size_t vend = value.find_last_not_of(" \t\r\n");
                if (vstart != std::string::npos && vend != std::string::npos) {
                    value = value.substr(vstart, vend - vstart + 1);
                }
            }
            
            result[key] = value;
            
            // Пропускаем запятую
            while (pos < content.length() && isspace(content[pos])) pos++;
            if (pos < content.length() && content[pos] == ',') pos++;
        }
        
        return true;
    }
    
    /**
     * @brief Извлечь строковое поле
     */
    static std::string get_string(const std::map<std::string, std::string>& obj, 
                                   const std::string& key) {
        auto it = obj.find(key);
        if (it != obj.end()) {
            return it->second;
        }
        return "";
    }
    
    /**
     * @brief Извлечь числовое поле
     */
    static int get_int(const std::map<std::string, std::string>& obj, 
                        const std::string& key, int default_val = 0) {
        auto it = obj.find(key);
        if (it != obj.end()) {
            try {
                return std::stoi(it->second);
            } catch (...) {}
        }
        return default_val;
    }
    
    /**
     * @brief Извлечь массив чисел
     */
    static std::vector<int> get_int_array(const std::map<std::string, std::string>& obj,
                                           const std::string& key) {
        std::vector<int> result;
        
        auto it = obj.find(key);
        if (it == obj.end()) return result;
        
        std::string arr_str = it->second;
        if (arr_str.empty() || arr_str[0] != '[') return result;
        
        size_t pos = 1;
        while (pos < arr_str.length() && arr_str[pos] != ']') {
            // Пропускаем пробелы
            while (pos < arr_str.length() && isspace(arr_str[pos])) pos++;
            
            size_t num_start = pos;
            while (pos < arr_str.length() && arr_str[pos] != ',' && arr_str[pos] != ']') {
                pos++;
            }
            
            if (num_start < pos) {
                std::string num_str = arr_str.substr(num_start, pos - num_start);
                try {
                    result.push_back(std::stoi(num_str));
                } catch (...) {}
            }
            
            if (pos < arr_str.length() && arr_str[pos] == ',') pos++;
        }
        
        return result;
    }
};

// ======================================================================
// HTML страница интерфейса (улучшенная)
// ======================================================================

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
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }
        .stat-card:hover {
            transform: translateY(-5px);
        }
        .stat-card h3 {
            margin: 0;
            font-size: 12px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-card .value {
            font-size: 24px;
            font-weight: bold;
            margin: 5px 0 0;
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
            display: block;
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
        .copy-button {
            background: #2196F3;
            padding: 8px 15px;
            font-size: 14px;
            margin-left: 10px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1><span>BPE Tokenizer</span> for C++ Code</h1>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <h3>Загрузка...</h3>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('encode')">📝 Токенизация</div>
            <div class="tab" onclick="switchTab('decode')">🔄 Декодирование</div>
            <div class="tab" onclick="switchTab('batch')">📦 Пакетная обработка</div>
            <div class="tab" onclick="switchTab('stats')">📊 Статистика</div>
            <div class="tab" onclick="switchTab('about')">ℹ️ О сервере</div>
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
                        🔍 Токенизировать <span id="encode-loading" class="loading" style="display:none;"></span>
                    </button>
                    <button class="secondary" onclick="clearResult('encode-result')">
                        🗑️ Очистить
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
                        🔄 Декодировать <span id="decode-loading" class="loading" style="display:none;"></span>
                    </button>
                    <button class="secondary" onclick="clearResult('decode-result')">
                        🗑️ Очистить
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
                        📦 Обработать пакет <span id="batch-loading" class="loading" style="display:none;"></span>
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
            <button onclick="refreshStats()">
                🔄 Обновить статистику
            </button>
            <button class="secondary" onclick="clearCache()">
                🗑️ Очистить кэш
            </button>
        </div>
        
        <div id="about" class="tab-content">
            <h2>О сервере</h2>
            <div class="result">
                <p><strong>BPE Tokenizer Web Service</strong></p>
                <p>Версия: 3.3.0</p>
                <p>API эндпоинты:</p>
                <ul>
                    <li><code>POST /api/tokenize</code> - токенизация текста</li>
                    <li><code>POST /api/detokenize</code> - декодирование токенов</li>
                    <li><code>POST /api/batch/tokenize</code> - пакетная токенизация</li>
                    <li><code>GET /api/stats</code> - статистика токенизатора</li>
                    <li><code>GET /api/cache/stats</code> - статистика кэша</li>
                    <li><code>POST /api/cache/clear</code> - очистка кэша</li>
                    <li><code>GET /api/health</code> - проверка работоспособности</li>
                </ul>
                <p><strong>Технические детали:</strong></p>
                <ul>
                    <li>Многопоточная обработка запросов</li>
                    <li>Кэширование частых запросов (LRU)</li>
                    <li>Защита от DoS-атак</li>
                    <li>Поддержка CORS</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>BPE Tokenizer for C++ Code | <a href="https://github.com/yourusername/bpe-tokenizer" target="_blank">GitHub</a> | &copy; 2026</p>
        </div>
    </div>
    
    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            document.querySelector(`.tab[onclick*='${tab}']`).classList.add('active');
            document.getElementById(tab).classList.add('active');
            
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
                    
                    let resultHtml = `<strong>✅ Токены (${data.count} шт.):</strong><br>`;
                    resultHtml += `<div style="font-family: monospace; white-space: pre-wrap; word-break: break-all; margin: 10px 0; background: #e8f5e8; padding: 10px; border-radius: 5px;">[`;
                    
                    for (let i = 0; i < tokens.length; i++) {
                        if (i > 0) resultHtml += ', ';
                        resultHtml += tokens[i];
                    }
                    
                    resultHtml += `]</div>`;
                    resultHtml += formatTokens(tokens);
                    
                    resultDiv.innerHTML = resultHtml;
                } else {
                    resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                hideLoading('encode-loading');
                resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${err.message}</span>`;
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
                    
                    resultDiv.innerHTML = `<strong>✅ Декодированный текст:</strong><br>` +
                        `<pre style="margin:10px 0; padding:10px; background:#e8f5e8; border-radius:5px; white-space: pre-wrap;">${displayText}</pre>`;
                } else {
                    resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                hideLoading('decode-loading');
                resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${err.message}</span>`;
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
            
            fetch('/api/batch/tokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({texts: codes})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading('batch-loading');
                
                if (data.success) {
                    let resultHtml = `<strong>✅ Результаты (${data.count} примеров):</strong><br>`;
                    
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
                } else {
                    resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                hideLoading('batch-loading');
                resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${err.message}</span>`;
            });
        }
        
        function refreshStats() {
            const resultDiv = document.getElementById('stats-result');
            resultDiv.innerHTML = 'Загрузка статистики...';
            
            Promise.all([
                fetch('/api/stats').then(r => r.json()),
                fetch('/api/cache/stats').then(r => r.json())
            ])
            .then(([tokenizerStats, cacheStats]) => {
                let html = '<h3>Токенизатор:</h3>';
                html += `<ul>`;
                html += `<li>Размер словаря: <strong>${tokenizerStats.vocab_size}</strong></li>`;
                html += `<li>Правил слияния: <strong>${tokenizerStats.merges_count}</strong></li>`;
                html += `<li>Вызовов encode: <strong>${tokenizerStats.encode_calls}</strong></li>`;
                html += `<li>Вызовов decode: <strong>${tokenizerStats.decode_calls}</strong></li>`;
                html += `<li>Всего токенов: <strong>${tokenizerStats.total_tokens}</strong></li>`;
                html += `</ul>`;
                
                html += '<h3>Кэш:</h3>';
                html += `<ul>`;
                html += `<li>Попаданий: <strong>${tokenizerStats.cache_hits}</strong></li>`;
                html += `<li>Промахов: <strong>${tokenizerStats.cache_misses}</strong></li>`;
                html += `<li>Эффективность: <strong>${tokenizerStats.cache_hit_rate}%</strong></li>`;
                html += `<li>Размер кэша: <strong>${cacheStats.size}/${cacheStats.max_size}</strong></li>`;
                html += `<li>Время жизни: <strong>${cacheStats.ttl_seconds} сек</strong></li>`;
                html += `</ul>`;
                
                resultDiv.innerHTML = html;
            })
            .catch(err => {
                resultDiv.innerHTML = `<span class="error">Ошибка загрузки статистики: ${err.message}</span>`;
            });
        }
        
        function clearCache() {
            fetch('/api/cache/clear', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Кэш успешно очищен!');
                        refreshStats();
                    }
                })
                .catch(err => {
                    alert('Ошибка при очистке кэша: ' + err.message);
                });
        }
        
        // Загрузка статистики при запуске
        window.onload = function() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('stats').innerHTML = `
                        <div class="stat-card">
                            <h3>Размер словаря</h3>
                            <div class="value">${data.vocab_size}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Правил слияния</h3>
                            <div class="value">${data.merges_count}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Попаданий в кэш</h3>
                            <div class="value">${data.cache_hits}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Эффективность</h3>
                            <div class="value">${data.cache_hit_rate}%</div>
                        </div>
                    `;
                });
        };
    </script>
</body>
</html>
)rawliteral";
}

// ======================================================================
// Обработка клиентских соединений
// ======================================================================

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
                // Таймаут чтения
                break;
            }
            return "";
        }
        
        if (bytes == 0) {
            // Соединение закрыто
            break;
        }
        
        request.append(buffer, bytes);
        total_read += bytes;
        
        if (total_read > MAX_REQUEST_SIZE) {
            // Слишком большой запрос
            return "TOO_LARGE";
        }
        
        if (!headers_complete && request.find("\r\n\r\n") != std::string::npos) {
            headers_complete = true;
            
            // Проверяем размер заголовков
            size_t header_end = request.find("\r\n\r\n");
            if (header_end > static_cast<size_t>(MAX_HEADERS_SIZE)) {
                return "HEADERS_TOO_LARGE";
            }
            
            // Если есть Content-Length, проверяем полный размер
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
        
        // Проверяем, получили ли мы полный запрос
        if (headers_complete) {
            size_t header_end = request.find("\r\n\r\n");
            size_t content_length = request.length() - header_end - 4;
            
            // Проверяем, есть ли Content-Length
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
                // Нет тела запроса
                break;
            }
        }
    }
    
    return request;
}

/**
 * @brief Обработать запрос клиента
 */
void handle_client(int client_fd, FastBPETokenizer& tokenizer, 
                   Cache<std::string, std::string>& cache) {
    
    // Устанавливаем таймауты
    set_socket_timeout(client_fd, READ_TIMEOUT_SEC);
    
    // Читаем запрос
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
    
    // Парсим запрос
    HttpRequest req = parse_request(request_str);
    
    if (!req.is_valid()) {
        std::string error = HttpResponse::error("Invalid request", 400);
        send(client_fd, error.c_str(), error.length(), 0);
        close(client_fd);
        return;
    }
    
    // Логируем запрос
    std::cout << "[" << std::this_thread::get_id() << "] " 
              << req.method << " " << req.path << std::endl;
    
    // Проверяем кэш для GET запросов
    std::string cache_key = req.method + ":" + req.path + ":" + req.body;
    std::string cached_response;
    
    if (req.method == "GET" && cache.get(cache_key, cached_response)) {
        send(client_fd, cached_response.c_str(), cached_response.length(), 0);
        close(client_fd);
        return;
    }
    
    // Обрабатываем запрос
    std::string response;
    
    // CORS preflight
    if (req.method == "OPTIONS") {
        response = HttpResponse::plain("", 200);
    }
    // API маршруты
    else if (req.path == "/api/tokenize" && req.method == "POST") {
        std::map<std::string, std::string> json_obj;
        if (SimpleJson::parse(req.body, json_obj)) {
            std::string text = SimpleJson::get_string(json_obj, "text");
            
            if (text.empty()) {
                response = HttpResponse::error("Missing 'text' field");
            } else {
                try {
                    auto tokens = tokenizer.encode(text);
                    
                    std::ostringstream json;
                    json << "{\"tokens\":[";
                    for (size_t i = 0; i < tokens.size(); ++i) {
                        if (i > 0) json << ",";
                        json << tokens[i];
                    }
                    json << "],\"count\":" << tokens.size() << ",\"success\":true}";
                    
                    response = HttpResponse::json(json.str());
                    
                } catch (const std::exception& e) {
                    response = HttpResponse::internal_error(e.what());
                }
            }
        } else {
            response = HttpResponse::error("Invalid JSON");
        }
    }
    else if (req.path == "/api/detokenize" && req.method == "POST") {
        std::map<std::string, std::string> json_obj;
        if (SimpleJson::parse(req.body, json_obj)) {
            auto tokens = SimpleJson::get_int_array(json_obj, "tokens");
            
            if (tokens.empty()) {
                response = HttpResponse::error("Missing or empty 'tokens' field");
            } else {
                try {
                    std::vector<uint32_t> uint32_tokens(tokens.begin(), tokens.end());
                    std::string text = tokenizer.decode(uint32_tokens);
                    
                    std::ostringstream json;
                    json << "{\"text\":\"" << HttpResponse::json_escape(text) 
                         << "\",\"count\":" << tokens.size()
                         << ",\"success\":true}";
                    
                    response = HttpResponse::json(json.str());
                    
                } catch (const std::exception& e) {
                    response = HttpResponse::internal_error(e.what());
                }
            }
        } else {
            response = HttpResponse::error("Invalid JSON");
        }
    }
    else if (req.path == "/api/batch/tokenize" && req.method == "POST") {
        // Пакетная обработка
        std::map<std::string, std::string> json_obj;
        if (SimpleJson::parse(req.body, json_obj)) {
            std::string texts_str = SimpleJson::get_string(json_obj, "texts");
            
            if (texts_str.empty()) {
                response = HttpResponse::error("Missing 'texts' field");
            } else {
                try {
                    // Разбираем массив строк (упрощенно)
                    std::vector<std::string> texts;
                    size_t pos = 1; // пропускаем '['
                    while (pos < texts_str.length() && texts_str[pos] != ']') {
                        if (texts_str[pos] == '"') {
                            pos++;
                            size_t start = pos;
                            while (pos < texts_str.length() && texts_str[pos] != '"') {
                                if (texts_str[pos] == '\\') pos += 2;
                                else pos++;
                            }
                            texts.push_back(texts_str.substr(start, pos - start));
                            pos++;
                        } else {
                            pos++;
                        }
                    }
                    
                    std::ostringstream json;
                    json << "{\"results\":[";
                    json << "],\"count\":" << texts.size() << ",\"success\":true}";
                    
                    response = HttpResponse::json(json.str());
                    
                } catch (const std::exception& e) {
                    response = HttpResponse::internal_error(e.what());
                }
            }
        } else {
            response = HttpResponse::error("Invalid JSON");
        }
    }
    else if (req.path == "/api/stats" && req.method == "GET") {
        auto stats = tokenizer.stats();
        
        std::ostringstream json;
        json << "{"
             << "\"vocab_size\":" << tokenizer.vocab_size() << ","
             << "\"merges_count\":" << tokenizer.merges_count() << ","
             << "\"cache_hits\":" << stats.cache_hits << ","
             << "\"cache_misses\":" << stats.cache_misses << ","
             << "\"cache_hit_rate\":" << std::fixed << std::setprecision(1) 
             << (stats.cache_hit_rate() * 100) << ","
             << "\"encode_calls\":" << stats.encode_calls << ","
             << "\"decode_calls\":" << stats.decode_calls << ","
             << "\"total_tokens\":" << stats.total_tokens_processed
             << "}";
        
        response = HttpResponse::json(json.str());
    }
    else if (req.path == "/api/cache/stats" && req.method == "GET") {
        auto cache_stats = cache.stats();
        
        std::ostringstream json;
        json << "{";
        for (auto it = cache_stats.begin(); it != cache_stats.end(); ++it) {
            if (it != cache_stats.begin()) json << ",";
            json << "\"" << it->first << "\":" << it->second;
        }
        json << "}";
        
        response = HttpResponse::json(json.str());
    }
    else if (req.path == "/api/cache/clear" && req.method == "POST") {
        cache.clear();
        response = HttpResponse::json("{\"success\":true}");
    }
    else if (req.path == "/api/health" && req.method == "GET") {
        std::ostringstream json;
        json << "{"
             << "\"status\":\"healthy\","
             << "\"timestamp\":" << std::time(nullptr) << ","
             << "\"version\":\"3.3.0\""
             << "}";
        response = HttpResponse::json(json.str());
    }
    else if (req.path == "/" || req.path.empty()) {
        response = HttpResponse::html(get_html_page());
        
        // Кэшируем главную страницу
        cache.put(cache_key, response);
    }
    else {
        response = HttpResponse::not_found(req.path);
    }
    
    // Отправляем ответ
    set_socket_timeout(client_fd, WRITE_TIMEOUT_SEC);
    send(client_fd, response.c_str(), response.length(), 0);
    
    // Кэшируем успешные GET запросы
    if (req.method == "GET" && response.find("\"success\":true") != std::string::npos) {
        cache.put(cache_key, response);
    }
    
    close(client_fd);
}

// ======================================================================
// Поиск файлов модели (улучшенный)
// ======================================================================

/**
 * @brief Найти файлы модели в стандартных расположениях
 */
bool find_model_files(FastBPETokenizer& tokenizer, std::string& vocab_path, 
                      std::string& merges_path, int argc, char* argv[]) {
    
    // Проверка аргументов командной строки
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (arg == "--model-size" && i + 1 < argc) {
            int size = std::stoi(argv[++i]);
            // Устанавливаем предпочтительный размер
        }
    }
    
    // Проверка переменных окружения
    const char* env_vocab = std::getenv("BPE_VOCAB");
    const char* env_merges = std::getenv("BPE_MERGES");
    
    if (env_vocab && env_merges) {
        vocab_path = env_vocab;
        merges_path = env_merges;
    }
    
    // Если пути заданы явно, пробуем их
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
    
    // Приоритетные пути поиска
    std::vector<int> model_sizes = {8000, 10000, 12000};
    std::vector<std::pair<std::string, std::string>> candidates;
    
    // Добавляем пути для каждого размера
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
    
    // Общие пути
    candidates.emplace_back("../models/cpp_vocab.json", "../models/cpp_merges.txt");
    candidates.emplace_back("models/cpp_vocab.json", "models/cpp_merges.txt");
    candidates.emplace_back("vocab.json", "merges.txt");
    
    // Пробуем загрузить модель
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
    
    // Если ничего не нашли, выводим подсказку
    std::cerr << "\n❌ Модель не найдена!\n" << std::endl;
    std::cerr << "Проверьте расположение файлов:" << std::endl;
    for (const auto& [vpath, _] : candidates) {
        std::cerr << "  - " << vpath << std::endl;
    }
    std::cerr << "\nИспользуйте аргументы командной строки:" << std::endl;
    std::cerr << "  --vocab PATH   путь к файлу словаря" << std::endl;
    std::cerr << "  --merges PATH  путь к файлу слияний" << std::endl;
    std::cerr << "  --model-size N размер модели (8000, 10000, 12000)" << std::endl;
    
    return false;
}

// ======================================================================
// Основная функция
// ======================================================================

/**
 * @brief Точка входа в программу
 */
int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "BPE TOKENIZER WEB SERVICE v3.3.0\n";
    std::cout << "========================================\n\n";
    
    // ===== Парсинг аргументов командной строки =====
    int port = 8080;
    int num_threads = std::thread::hardware_concurrency();
    size_t cache_size = DEFAULT_CACHE_SIZE;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--cache-size" && i + 1 < argc) {
            cache_size = std::stoull(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --port PORT       Порт для сервера (по умолчанию: 8080)\n";
            std::cout << "  --threads N       Количество потоков (по умолчанию: " 
                      << num_threads << ")\n";
            std::cout << "  --cache-size N    Размер кэша (по умолчанию: " 
                      << DEFAULT_CACHE_SIZE << ")\n";
            std::cout << "  --vocab PATH      Путь к файлу словаря\n";
            std::cout << "  --merges PATH     Путь к файлу слияний\n";
            std::cout << "  --model-size N    Размер модели (8000, 10000, 12000)\n";
            std::cout << "  --help            Показать справку\n";
            return 0;
        }
    }
    
    // ===== Установка обработчика сигналов =====
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // ===== Загрузка токенизатора =====
    FastBPETokenizer tokenizer;
    std::cout << "📦 Загрузка модели..." << std::endl;
    
    std::string vocab_path, merges_path;
    if (!find_model_files(tokenizer, vocab_path, merges_path, argc, argv)) {
        return 1;
    }
    
    std::cout << "✅ Модель загружена!" << std::endl;
    std::cout << "   Словарь: " << vocab_path << std::endl;
    std::cout << "   Слияния: " << merges_path << std::endl;
    std::cout << "   Размер словаря: " << tokenizer.vocab_size() << std::endl;
    std::cout << "   Правил слияния: " << tokenizer.merges_count() << std::endl;
    std::cout << std::endl;
    
    // ===== Инициализация кэша и пула потоков =====
    Cache<std::string, std::string> cache(cache_size);
    ThreadPool pool(num_threads);
    
    std::cout << "🚀 Запуск сервера..." << std::endl;
    std::cout << "   Потоков: " << num_threads << std::endl;
    std::cout << "   Размер кэша: " << cache_size << std::endl;
    std::cout << std::endl;
    
    // ===== Создание сокета =====
    g_server_fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (g_server_fd < 0) {
        std::cerr << "❌ Ошибка создания сокета!" << std::endl;
        return 1;
    }
    
    // Настройка опций сокета
    int opt = 1;
    if (setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "❌ Ошибка настройки сокета!" << std::endl;
        return 1;
    }
    
    // Привязка к порту
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(g_server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "❌ Ошибка привязки к порту " << port << "!" << std::endl;
        return 1;
    }
    
    // Прослушивание
    if (listen(g_server_fd, MAX_CONNECTIONS) < 0) {
        std::cerr << "❌ Ошибка прослушивания!" << std::endl;
        return 1;
    }
    
    // ===== Создание epoll =====
    g_epoll_fd = epoll_create1(0);
    if (g_epoll_fd < 0) {
        std::cerr << "❌ Ошибка создания epoll!" << std::endl;
        return 1;
    }
    
    struct epoll_event event;
    event.events = EPOLLIN | EPOLLET;  // Edge-triggered
    event.data.fd = g_server_fd;
    
    if (epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, g_server_fd, &event) < 0) {
        std::cerr << "❌ Ошибка добавления сокета в epoll!" << std::endl;
        return 1;
    }
    
    // ===== Запуск сервера =====
    std::cout << "✅ Сервер запущен на http://localhost:" << port << std::endl;
    std::cout << "📡 API эндпоинты:" << std::endl;
    std::cout << "   POST /api/tokenize    - токенизация текста" << std::endl;
    std::cout << "   POST /api/detokenize  - декодирование токенов" << std::endl;
    std::cout << "   POST /api/batch/tokenize - пакетная токенизация" << std::endl;
    std::cout << "   GET  /api/stats        - статистика токенизатора" << std::endl;
    std::cout << "   GET  /api/cache/stats  - статистика кэша" << std::endl;
    std::cout << "   POST /api/cache/clear  - очистка кэша" << std::endl;
    std::cout << "   GET  /api/health       - проверка работоспособности" << std::endl;
    std::cout << "   GET  /                 - веб-интерфейс" << std::endl;
    std::cout << "Нажмите Ctrl+C для остановки!" << std::endl;
    std::cout << std::endl;
    
    std::atomic<int> client_count{0};
    
    // Основной цикл обработки соединений
    while (g_running) {
        struct epoll_event events[32];
        int nfds = epoll_wait(g_epoll_fd, events, 32, 1000);
        
        if (!g_running) break;
        
        for (int i = 0; i < nfds; ++i) {
            if (events[i].data.fd == g_server_fd) {
                // Новое соединение
                while (g_running) {
                    struct sockaddr_in client_addr;
                    socklen_t client_len = sizeof(client_addr);
                    int client_fd = accept4(g_server_fd, (struct sockaddr*)&client_addr,
                                            &client_len, SOCK_NONBLOCK);
                    
                    if (client_fd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            // Больше нет ожидающих соединений
                            break;
                        }
                        continue;
                    }
                    
                    client_count++;
                    
                    // Добавляем клиентский сокет в epoll
                    struct epoll_event client_event;
                    client_event.events = EPOLLIN | EPOLLET | EPOLLONESHOT;
                    client_event.data.fd = client_fd;
                    
                    if (epoll_ctl(g_epoll_fd, EPOLL_CTL_ADD, client_fd, &client_event) < 0) {
                        close(client_fd);
                        continue;
                    }
                }
            } else {
                // Данные от клиента
                int client_fd = events[i].data.fd;
                
                // Удаляем из epoll перед обработкой
                epoll_ctl(g_epoll_fd, EPOLL_CTL_DEL, client_fd, nullptr);
                
                // Передаем обработку в пул потоков
                pool.enqueue([client_fd, &tokenizer, &cache]() {
                    handle_client(client_fd, tokenizer, cache);
                });
            }
        }
    }
    
    // ===== Завершение работы =====
    std::cout << "\n⏳ Остановка сервера..." << std::endl;
    
    close(g_epoll_fd);
    close(g_server_fd);
    
    std::cout << "✅ Сервер остановлен." << std::endl;
    std::cout << "📊 Всего обработано запросов: " << client_count << std::endl;
    
    return 0;
}