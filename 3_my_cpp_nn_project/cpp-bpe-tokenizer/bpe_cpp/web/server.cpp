/**
 * @file server.cpp
 * @brief Простой HTTP сервер для BPE токенизатора на чистом сокетах
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Легковесный веб-сервер без внешних зависимостей:
 *          - / - HTML интерфейс
 *          - /api/tokenize - POST эндпоинт для токенизации
 *          - /api/detokenize - POST эндпоинт для декодирования
 *          - /api/stats - GET статистика
 *          - /api/health - GET проверка работоспособности
 * 
 * @compile g++ -std=c++17 -Iinclude server.cpp -o server -lpthread
 * @run ./server [--port PORT]
 */

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

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

using namespace bpe;

// ======================================================================
// Глобальные переменные
// ======================================================================

std::atomic<bool> g_running{true};
int g_server_fd = -1;

void signal_handler(int) {
    std::cout << "\n🛑 Получен сигнал остановки. Завершение работы..." << std::endl;
    g_running = false;
    if (g_server_fd != -1) {
        close(g_server_fd);
    }
}

// ======================================================================
// Класс для HTTP ответов
// ======================================================================

class HttpResponse {
public:
    static std::string html(const std::string& body, int status = 200) {
        return response(status, "text/html; charset=utf-8", body);
    }
    
    static std::string json(const std::string& body, int status = 200) {
        return response(status, "application/json; charset=utf-8", body);
    }
    
    static std::string plain(const std::string& body, int status = 200) {
        return response(status, "text/plain; charset=utf-8", body);
    }
    
    static std::string error(const std::string& message, int status = 400) {
        return json("{\"error\":\"" + message + "\",\"success\":false}", status);
    }
    
private:
    static std::string response(int status, const std::string& content_type, const std::string& body) {
        std::ostringstream oss;
        oss << "HTTP/1.1 " << status << " " << get_status_message(status) << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.length() << "\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            << "Access-Control-Allow-Headers: Content-Type\r\n"
            << "Connection: close\r\n"
            << "\r\n"
            << body;
        return oss.str();
    }
    
    static std::string get_status_message(int status) {
        switch(status) {
            case 200: return "OK";
            case 400: return "Bad Request";
            case 404: return "Not Found";
            case 405: return "Method Not Allowed";
            case 500: return "Internal Server Error";
            default: return "Unknown";
        }
    }
};

// ======================================================================
// Парсинг HTTP запросов
// ======================================================================

struct HttpRequest {
    std::string method;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;
    
    bool is_valid() const {
        return !method.empty() && !path.empty();
    }
    
    std::string get_header(const std::string& name) const {
        auto it = headers.find(name);
        return it != headers.end() ? it->second : "";
    }
};

HttpRequest parse_request(const std::string& raw) {
    HttpRequest req;
    
    std::istringstream iss(raw);
    std::string line;
    
    // Парсим первую строку
    if (std::getline(iss, line)) {
        std::istringstream line_ss(line);
        line_ss >> req.method >> req.path >> req.version;
    }
    
    // Парсим заголовки
    while (std::getline(iss, line) && line != "\r") {
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 2);
            // Убираем \r в конце
            if (!value.empty() && value.back() == '\r') {
                value.pop_back();
            }
            req.headers[key] = value;
        }
    }
    
    // Парсим тело
    size_t body_start = raw.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        req.body = raw.substr(body_start + 4);
    }
    
    return req;
}

// ======================================================================
// Парсинг JSON (упрощенный)
// ======================================================================

std::string extract_json_field(const std::string& json, const std::string& field) {
    std::string search = "\"" + field + "\":\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        // Пробуем без кавычек для чисел
        search = "\"" + field + "\":";
        pos = json.find(search);
        if (pos != std::string::npos) {
            pos += search.length();
            size_t end = json.find_first_of(",}", pos);
            if (end != std::string::npos) {
                return json.substr(pos, end - pos);
            }
        }
        return "";
    }
    
    pos += search.length();
    std::string result;
    while (pos < json.length() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.length()) {
            pos++; // пропускаем экранирование
        }
        result += json[pos];
        pos++;
    }
    return result;
}

std::vector<uint32_t> extract_json_array(const std::string& json, const std::string& field) {
    std::vector<uint32_t> result;
    
    std::string search = "\"" + field + "\":[";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;
    
    pos += search.length();
    size_t end = json.find("]", pos);
    if (end == std::string::npos) return result;
    
    std::string array_str = json.substr(pos, end - pos);
    std::istringstream iss(array_str);
    std::string token;
    
    while (std::getline(iss, token, ',')) {
        try {
            result.push_back(std::stoul(token));
        } catch (...) {
            // Игнорируем ошибки
        }
    }
    
    return result;
}

// ======================================================================
// HTML страница
// ======================================================================

std::string get_html_page() {
    return R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BPE Tokenizer</title>
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
            max-width: 1200px; 
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .stat-card h3 {
            margin: 0;
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stat-card .value {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0 0;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            transition: all 0.3s;
        }
        .tab:hover {
            background: #f0f0f0;
        }
        .tab.active {
            background: #4CAF50;
            color: white;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        textarea, input[type="text"] { 
            width: 100%; 
            padding: 15px; 
            margin: 10px 0; 
            font-family: 'Consolas', 'Monaco', monospace; 
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border 0.3s;
        }
        textarea:focus, input:focus {
            outline: none;
            border-color: #4CAF50;
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
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        button:active {
            transform: translateY(0);
        }
        .result { 
            background: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            margin-top: 20px;
            border-left: 4px solid #4CAF50;
            font-family: 'Consolas', 'Monaco', monospace;
            white-space: pre-wrap;
            word-wrap: break-word;
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
    </style>
</head>
<body>
    <div class="container">
        <h1><span>🚀 BPE Tokenizer</span></h1>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <h3>Загрузка...</h3>
            </div>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('encode')">🔤 Токенизация</div>
            <div class="tab" onclick="switchTab('decode')">🔄 Декодирование</div>
            <div class="tab" onclick="switchTab('about')">ℹ️ О сервере</div>
        </div>
        
        <div id="encode" class="tab-content active">
            <h2>Токенизация C++ кода</h2>
            <textarea id="code" rows="10" placeholder="Введите C++ код...">int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}</textarea>
            <button onclick="tokenize()">
                Токенизировать <span id="encode-loading" class="loading" style="display:none;"></span>
            </button>
            <div class="result" id="encode-result"></div>
        </div>
        
        <div id="decode" class="tab-content">
            <h2>Декодирование токенов</h2>
            <input type="text" id="tokens" placeholder="Введите токены через запятую (например: 42, 43, 44)">
            <button onclick="detokenize()">
                Декодировать <span id="decode-loading" class="loading" style="display:none;"></span>
            </button>
            <div class="result" id="decode-result"></div>
        </div>
        
        <div id="about" class="tab-content">
            <h2>О сервере</h2>
            <div class="result">
                <p><strong>BPE Tokenizer Web Service</strong></p>
                <p>Версия: 1.0.0</p>
                <p>API эндпоинты:</p>
                <ul>
                    <li><code>POST /api/tokenize</code> - токенизация текста</li>
                    <li><code>POST /api/detokenize</code> - декодирование токенов</li>
                    <li><code>GET /api/stats</code> - статистика токенизатора</li>
                    <li><code>GET /api/health</code> - проверка работоспособности</li>
                </ul>
            </div>
        </div>
    </div>
    
    <script>
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            document.querySelector(`.tab[onclick*='${tab}']`).classList.add('active');
            document.getElementById(tab).classList.add('active');
        }
        
        function showLoading(elementId) {
            document.getElementById(elementId).style.display = 'inline-block';
        }
        
        function hideLoading(elementId) {
            document.getElementById(elementId).style.display = 'none';
        }
        
        function tokenize() {
            const code = document.getElementById('code').value;
            const resultDiv = document.getElementById('encode-result');
            
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
                    resultDiv.innerHTML = `
                        <strong>✅ Токены (${data.count} шт.):</strong><br>
                        [${data.tokens.join(', ')}]
                    `;
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
            
            showLoading('decode-loading');
            resultDiv.innerHTML = 'Обработка...';
            
            const tokens = tokensInput.split(',').map(t => parseInt(t.trim()));
            
            fetch('/api/detokenize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({tokens: tokens})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading('decode-loading');
                if (data.success) {
                    resultDiv.innerHTML = `
                        <strong>✅ Декодированный текст:</strong><br>
                        <pre>${data.text}</pre>
                    `;
                } else {
                    resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${data.error}</span>`;
                }
            })
            .catch(err => {
                hideLoading('decode-loading');
                resultDiv.innerHTML = `<span class="error">❌ Ошибка: ${err.message}</span>`;
            });
        }
        
        // Загрузка статистики
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
    </script>
</body>
</html>
)rawliteral";
}

// ======================================================================
// Обработка запросов
// ======================================================================

std::string handle_request(const HttpRequest& req, FastBPETokenizer& tokenizer) {
    // CORS preflight
    if (req.method == "OPTIONS") {
        return HttpResponse::plain("", 200);
    }
    
    // API маршруты
    if (req.path == "/api/tokenize" && req.method == "POST") {
        std::string text = extract_json_field(req.body, "text");
        
        if (text.empty()) {
            return HttpResponse::error("Missing 'text' field");
        }
        
        try {
            auto tokens = tokenizer.encode(text);
            
            std::ostringstream json;
            json << "{\"tokens\":[";
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) json << ",";
                json << tokens[i];
            }
            json << "],\"count\":" << tokens.size() << ",\"success\":true}";
            
            std::cout << "✅ Токенизировано " << tokens.size() << " токенов" << std::endl;
            return HttpResponse::json(json.str());
            
        } catch (const std::exception& e) {
            return HttpResponse::error(e.what(), 500);
        }
    }
    
    else if (req.path == "/api/detokenize" && req.method == "POST") {
        auto tokens = extract_json_array(req.body, "tokens");
        
        if (tokens.empty()) {
            return HttpResponse::error("Missing or empty 'tokens' field");
        }
        
        try {
            std::string text = tokenizer.decode(tokens);
            
            // Экранируем для JSON
            std::string escaped;
            for (char c : text) {
                if (c == '"' || c == '\\') {
                    escaped += '\\';
                }
                escaped += c;
            }
            
            std::ostringstream json;
            json << "{\"text\":\"" << escaped << "\",\"count\":" << tokens.size()
                 << ",\"success\":true}";
            
            std::cout << "✅ Детокенизировано " << tokens.size() << " токенов" << std::endl;
            return HttpResponse::json(json.str());
            
        } catch (const std::exception& e) {
            return HttpResponse::error(e.what(), 500);
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
             << "\"cache_hit_rate\":" << (stats.cache_hit_rate() * 100) << ","
             << "\"encode_calls\":" << stats.encode_calls << ","
             << "\"decode_calls\":" << stats.decode_calls << ","
             << "\"total_tokens\":" << stats.total_tokens_processed
             << "}";
        
        return HttpResponse::json(json.str());
    }
    
    else if (req.path == "/api/health" && req.method == "GET") {
        std::ostringstream json;
        json << "{"
             << "\"status\":\"healthy\","
             << "\"timestamp\":" << std::time(nullptr) << ","
             << "\"version\":\"1.0.0\""
             << "}";
        return HttpResponse::json(json.str());
    }
    
    else if (req.path == "/" || req.path.empty()) {
        return HttpResponse::html(get_html_page());
    }
    
    else {
        return HttpResponse::error("Not found", 404);
    }
}

// ======================================================================
// Поиск файлов модели
// ======================================================================

bool find_model_files(FastBPETokenizer& tokenizer, std::string& vocab_path, std::string& merges_path) {
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"../../bpe/vocab_trained.json", "../../bpe/merges_trained.txt"},
        {"../models/cpp_vocab.json", "../models/cpp_merges.txt"},
        {"models/cpp_vocab.json", "models/cpp_merges.txt"},
        {"../../bpe/vocab.json", "../../bpe/merges.txt"},
        {"vocab.json", "merges.txt"}
    };
    
    for (const auto& [vpath, mpath] : candidates) {
        std::ifstream vfile(vpath);
        std::ifstream mfile(mpath);
        
        if (vfile.good() && mfile.good()) {
            vocab_path = vpath;
            merges_path = mpath;
            
            if (tokenizer.load(vpath, mpath)) {
                return true;
            }
        }
    }
    
    return false;
}

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "🌐 BPE TOKENIZER WEB SERVICE\n";
    std::cout << "========================================\n\n";
    
    // Парсинг аргументов
    int port = 8080;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --port PORT    Порт для сервера (по умолчанию: 8080)\n";
            std::cout << "  --help         Показать справку\n";
            return 0;
        }
    }
    
    // Установка обработчика сигналов
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Загружаем токенизатор
    FastBPETokenizer tokenizer;
    std::cout << "📚 Загрузка модели..." << std::endl;
    
    std::string vocab_path, merges_path;
    if (!find_model_files(tokenizer, vocab_path, merges_path)) {
        std::cerr << "❌ Не удалось загрузить модель!" << std::endl;
        std::cerr << "   Убедитесь, что файлы существуют:" << std::endl;
        std::cerr << "   - ../../bpe/vocab_trained.json" << std::endl;
        std::cerr << "   - ../../bpe/merges_trained.txt" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Модель загружена!" << std::endl;
    std::cout << "   📚 Словарь: " << vocab_path << std::endl;
    std::cout << "   📚 Слияния: " << merges_path << std::endl;
    std::cout << "   📊 Размер словаря: " << tokenizer.vocab_size() << std::endl;
    std::cout << "   🔗 Правил слияния: " << tokenizer.merges_count() << std::endl;
    std::cout << std::endl;
    
    // Создаем сокет
    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        std::cerr << "❌ Ошибка создания сокета" << std::endl;
        return 1;
    }
    
    int opt = 1;
    if (setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "❌ Ошибка настройки сокета" << std::endl;
        return 1;
    }
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(g_server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "❌ Ошибка привязки к порту " << port << std::endl;
        return 1;
    }
    
    if (listen(g_server_fd, 5) < 0) {
        std::cerr << "❌ Ошибка прослушивания" << std::endl;
        return 1;
    }
    
    std::cout << "🌐 Сервер запущен на http://localhost:" << port << std::endl;
    std::cout << "   API эндпоинты:" << std::endl;
    std::cout << "   - POST /api/tokenize" << std::endl;
    std::cout << "   - POST /api/detokenize" << std::endl;
    std::cout << "   - GET /api/stats" << std::endl;
    std::cout << "   - GET /api/health" << std::endl;
    std::cout << "   Нажмите Ctrl+C для остановки" << std::endl;
    std::cout << std::endl;
    
    int client_count = 0;
    
    while (g_running) {
        int addrlen = sizeof(address);
        int client_fd = accept(g_server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
        
        if (!g_running) break;
        
        if (client_fd < 0) {
            if (g_running) {
                std::cerr << "❌ Ошибка принятия соединения" << std::endl;
            }
            continue;
        }
        
        client_count++;
        
        char buffer[16384] = {0};
        int bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read > 0) {
            std::string request_str(buffer);
            HttpRequest req = parse_request(request_str);
            
            if (req.is_valid()) {
                std::cout << "📝 [" << client_count << "] " 
                          << req.method << " " << req.path << std::endl;
                
                std::string response = handle_request(req, tokenizer);
                send(client_fd, response.c_str(), response.length(), 0);
            } else {
                std::string error = HttpResponse::error("Invalid request", 400);
                send(client_fd, error.c_str(), error.length(), 0);
            }
        }
        
        close(client_fd);
    }
    
    close(g_server_fd);
    std::cout << "\n👋 Сервер остановлен. Всего обработано запросов: " << client_count << std::endl;
    
    return 0;
}