/**
 * @file server_crow.cpp
 * @brief Веб-сервер для BPE токенизатора на базе CrowCpp
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details REST API сервер для токенизации C++ кода:
 *          - /tokenize - POST: кодирование текста в токены
 *          - /detokenize - POST: декодирование токенов в текст
 *          - /stats - GET: статистика токенизатора
 *          - /batch - POST: пакетная обработка
 *          - /health - GET: проверка работоспособности
 *          - /metrics - GET: метрики Prometheus
 *          - Swagger UI документация
 * 
 * @compile g++ -std=c++17 -pthread server_crow.cpp -o server_crow -lcrow -lfast_tokenizer
 * @run ./server_crow [--port PORT] [--threads N]
 */

#include "crow/app.h"
#include "crow/middlewares/cors.h"

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

using namespace bpe;

// ======================================================================
// Глобальные переменные для graceful shutdown
// ======================================================================
crow::SimpleApp* g_app = nullptr;
bool g_running = true;

void signal_handler(int) {
    std::cout << "\nПолучен сигнал остановки. Завершение работы..." << std::endl;
    g_running = false;
    if (g_app) {
        g_app->stop();
    }
}

// ======================================================================
// Middleware для логирования
// ======================================================================
struct LoggingMiddleware {
    struct context {
        std::chrono::time_point<std::chrono::steady_clock> start_time;
    };

    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        ctx.start_time = std::chrono::steady_clock::now();
    }

    void after_handle(crow::request& req, crow::response& res, context& ctx) {
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - ctx.start_time);
        
        // Преобразуем метод в строку для вывода
        std::string method_str;
        switch (req.method) {
            case crow::HTTPMethod::GET: method_str = "GET"; break;
            case crow::HTTPMethod::POST: method_str = "POST"; break;
            case crow::HTTPMethod::PUT: method_str = "PUT"; break;
            case crow::HTTPMethod::DELETE: method_str = "DELETE"; break;
            default: method_str = "OTHER";
        }
        
        std::cout << std::setw(6) << duration.count() << "μs | "
                  << std::setw(4) << res.code << " | "
                  << std::setw(10) << method_str << " | "
                  << req.url << std::endl;
    }
};

// ======================================================================
// Основная функция
// ======================================================================

int main(int argc, char* argv[]) {
    // ======================================================================
    // Парсинг аргументов
    // ======================================================================
    
    int port = 8080;
    int threads = std::thread::hardware_concurrency();
    std::string model_path = "../../bpe/vocab_trained.json";
    std::string merges_path = "../../bpe/merges_trained.txt";
    
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
        } else if (arg == "--help") {
            std::cout << "Использование: " << argv[0] << " [options]\n";
            std::cout << "  --port PORT      Порт для сервера (по умолчанию: 8080)\n";
            std::cout << "  --threads N      Количество потоков (по умолчанию: все)\n";
            std::cout << "  --model PATH     Путь к файлу словаря\n";
            std::cout << "  --merges PATH    Путь к файлу слияний\n";
            std::cout << "  --help           Показать справку\n";
            return 0;
        }
    }
    
    // ======================================================================
    // Инициализация
    // ======================================================================
    
    std::cout << "========================================\n";
    std::cout << "BPE TOKENIZER WEB SERVER\n";
    std::cout << "========================================\n\n";
    
    // Загружаем токенизатор
    std::cout << "Загрузка модели..." << std::endl;
    
    auto tokenizer = std::make_shared<FastBPETokenizer>(
        TokenizerConfig{32000, 10000, true, true}
    );
    
    // Пробуем разные пути к модели
    std::vector<std::string> possible_paths = {
        model_path,
        "../../bpe/vocab_trained.json",
        "../models/cpp_vocab.json",
        "models/cpp_vocab.json",
        "vocab.json"
    };
    
    bool loaded = false;
    for (const auto& path : possible_paths) {
        if (tokenizer->load(path, merges_path)) {
            loaded = true;
            model_path = path;
            break;
        }
    }
    
    if (!loaded) {
        std::cerr << "Ошибка загрузки модели!" << std::endl;
        std::cerr << "   Путь: " << model_path << std::endl;
        std::cerr << "   Слияния: " << merges_path << std::endl;
        return 1;
    }
    
    std::cout << "Модель загружена!" << std::endl;
    std::cout << "   Размер словаря: " << tokenizer->vocab_size() << std::endl;
    std::cout << "   Правил слияния: " << tokenizer->merges_count() << std::endl;
    std::cout << std::endl;
    
    // ======================================================================
    // Настройка сервера
    // ======================================================================
    
    // Создаем сервер с middleware
    crow::App<LoggingMiddleware, crow::CORSHandler> app;
    
    // Настройка CORS
    auto& cors = app.get_middleware<crow::CORSHandler>();
    cors.global()
        .headers("Content-Type", "Authorization")
        .methods("POST"_method, "GET"_method, "OPTIONS"_method);
    
    // ======================================================================
    // Маршруты
    // ======================================================================
    
    // Главная страница
    CROW_ROUTE(app, "/")([](){
        return R"(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BPE Tokenizer API</title>
    <style>
        body { font-family: Arial; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        h1 { color: #333; text-align: center; }
        .endpoint { background: #f8f9fa; padding: 15px; margin: 15px 0; border-left: 4px solid #4CAF50; border-radius: 5px; }
        .endpoint:hover { box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; font-family: monospace; }
        .method { display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-weight: bold; margin-right: 10px; }
        .get { background: #61affe; }
        .post { background: #49cc90; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 30px; }
        .stat-card { background: #4CAF50; color: white; padding: 15px; border-radius: 10px; text-align: center; }
        .stat-card h3 { margin: 0; font-size: 14px; opacity: 0.9; }
        .stat-card .value { font-size: 24px; font-weight: bold; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BPE Tokenizer API</h1>
        <p style="text-align: center;">Fast C++ BPE Tokenizer for C++ code</p>
        
        <div class="stats" id="stats">
            <div class="stat-card">
                <h3>Загрузка...</h3>
            </div>
        </div>
        
        <h2>Доступные эндпоинты:</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/stats</strong> - статистика токенизатора<br>
            <code>curl http://localhost:8080/stats</code>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/tokenize</strong> - токенизация текста<br>
            <code>curl -X POST http://localhost:8080/tokenize -H "Content-Type: application/json" -d '{"text": "int main() {}"}'</code>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/detokenize</strong> - детокенизация токенов<br>
            <code>curl -X POST http://localhost:8080/detokenize -H "Content-Type: application/json" -d '{"tokens": [42, 43, 44]}'</code>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/batch</strong> - пакетная обработка<br>
            <code>curl -X POST http://localhost:8080/batch -H "Content-Type: application/json" -d '{"texts": ["int a;", "float b;"]}'</code>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/health</strong> - проверка работоспособности<br>
            <code>curl http://localhost:8080/health</code>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/metrics</strong> - метрики Prometheus<br>
            <code>curl http://localhost:8080/metrics</code>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/swagger</strong> - Swagger документация<br>
            <code>curl http://localhost:8080/swagger</code>
        </div>
    </div>
    
    <script>
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
                const stats = document.getElementById('stats');
                stats.innerHTML = `
                    <div class="stat-card"><h3>Размер словаря</h3><div class="value">${data.vocab_size}</div></div>
                    <div class="stat-card"><h3>Правил слияния</h3><div class="value">${data.merges_count}</div></div>
                    <div class="stat-card"><h3>Попаданий в кэш</h3><div class="value">${data.cache_hits}</div></div>
                    <div class="stat-card"><h3>Промахов кэша</h3><div class="value">${data.cache_misses}</div></div>
                `;
            });
    </script>
</body>
</html>
)";
    });
    
    // API эндпоинты
    CROW_ROUTE(app, "/tokenize").methods("POST"_method)([tokenizer](const crow::request& req){
        auto json = crow::json::load(req.body);
        if (!json || !json.has("text")) {
            return crow::response(400, "Missing 'text' field");
        }
        
        try {
            std::string text = json["text"].s();
            auto tokens = tokenizer->encode(text);
            
            crow::json::wvalue result;
            result["tokens"] = tokens;
            result["count"] = tokens.size();
            result["success"] = true;
            
            return crow::response{result};
        } catch (const std::exception& e) {
            crow::json::wvalue error;
            error["error"] = e.what();
            error["success"] = false;
            return crow::response(500, error);
        }
    });
    
    CROW_ROUTE(app, "/detokenize").methods("POST"_method)([tokenizer](const crow::request& req){
        auto json = crow::json::load(req.body);
        if (!json || !json.has("tokens")) {
            return crow::response(400, "Missing 'tokens' field");
        }
        
        try {
            std::vector<uint32_t> tokens;
            for (const auto& t : json["tokens"]) {
                tokens.push_back(t.i());
            }
            
            std::string text = tokenizer->decode(tokens);
            
            crow::json::wvalue result;
            result["text"] = text;
            result["count"] = tokens.size();
            result["success"] = true;
            
            return crow::response{result};
        } catch (const std::exception& e) {
            crow::json::wvalue error;
            error["error"] = e.what();
            error["success"] = false;
            return crow::response(500, error);
        }
    });
    
    CROW_ROUTE(app, "/batch").methods("POST"_method)([tokenizer](const crow::request& req){
        auto json = crow::json::load(req.body);
        if (!json || !json.has("texts")) {
            return crow::response(400, "Missing 'texts' field");
        }
        
        try {
            std::vector<std::string> texts;
            for (const auto& t : json["texts"]) {
                texts.push_back(t.s());
            }
            
            std::vector<std::string_view> views;
            for (const auto& t : texts) {
                views.push_back(t);
            }
            
            auto batch_result = tokenizer->encode_batch(views);
            
            crow::json::wvalue result;
            std::vector<crow::json::wvalue> results;
            for (const auto& tokens : batch_result) {
                crow::json::wvalue item;
                item["tokens"] = tokens;
                item["count"] = tokens.size();
                results.push_back(std::move(item));
            }
            result["results"] = std::move(results);
            result["success"] = true;
            
            return crow::response{result};
        } catch (const std::exception& e) {
            crow::json::wvalue error;
            error["error"] = e.what();
            error["success"] = false;
            return crow::response(500, error);
        }
    });
    
    CROW_ROUTE(app, "/stats")([tokenizer](){
        crow::json::wvalue result;
        
        auto stats = tokenizer->stats();
        
        result["vocab_size"] = tokenizer->vocab_size();
        result["merges_count"] = tokenizer->merges_count();
        result["cache_hits"] = stats.cache_hits;
        result["cache_misses"] = stats.cache_misses;
        result["encode_calls"] = stats.encode_calls;
        result["decode_calls"] = stats.decode_calls;
        result["total_tokens"] = stats.total_tokens_processed;
        result["cache_hit_rate"] = stats.cache_hit_rate() * 100;
        result["avg_encode_time_ms"] = stats.avg_encode_time_ms();
        result["avg_decode_time_ms"] = stats.avg_decode_time_ms();
        
        return crow::response{result};
    });
    
    CROW_ROUTE(app, "/health")([](){
        crow::json::wvalue result;
        result["status"] = "healthy";
        result["timestamp"] = std::time(nullptr);
        result["uptime"] = "OK";
        return crow::response{result};
    });
    
    CROW_ROUTE(app, "/metrics")([tokenizer](){
        std::stringstream metrics;
        auto stats = tokenizer->stats();
        
        metrics << "# HELP bpe_vocab_size Vocabulary size\n";
        metrics << "# TYPE bpe_vocab_size gauge\n";
        metrics << "bpe_vocab_size " << tokenizer->vocab_size() << "\n\n";
        
        metrics << "# HELP bpe_merges_count Number of merge rules\n";
        metrics << "# TYPE bpe_merges_count gauge\n";
        metrics << "bpe_merges_count " << tokenizer->merges_count() << "\n\n";
        
        metrics << "# HELP bpe_cache_hits_total Total cache hits\n";
        metrics << "# TYPE bpe_cache_hits_total counter\n";
        metrics << "bpe_cache_hits_total " << stats.cache_hits << "\n\n";
        
        metrics << "# HELP bpe_cache_misses_total Total cache misses\n";
        metrics << "# TYPE bpe_cache_misses_total counter\n";
        metrics << "bpe_cache_misses_total " << stats.cache_misses << "\n\n";
        
        metrics << "# HELP bpe_encode_calls_total Total encode calls\n";
        metrics << "# TYPE bpe_encode_calls_total counter\n";
        metrics << "bpe_encode_calls_total " << stats.encode_calls << "\n\n";
        
        metrics << "# HELP bpe_tokens_processed_total Total tokens processed\n";
        metrics << "# TYPE bpe_tokens_processed_total counter\n";
        metrics << "bpe_tokens_processed_total " << stats.total_tokens_processed << "\n";
        
        return crow::response(metrics.str());
    });
    
    // Swagger UI
    CROW_ROUTE(app, "/swagger")([](){
        return R"(
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Swagger UI - BPE Tokenizer API</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css">
    <style>
        body { margin: 0; padding: 0; }
        #swagger-ui { max-width: 1200px; margin: 0 auto; }
    </style>
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
            ],
            layout: "BaseLayout"
        });
    </script>
</body>
</html>
)";
    });
    
    // OpenAPI spec (упрощенный)
    CROW_ROUTE(app, "/swagger.json")([](){
        return R"({
  "openapi": "3.0.0",
  "info": {
    "title": "BPE Tokenizer API",
    "description": "Fast C++ BPE Tokenizer for C++ code",
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "http://localhost:8080",
      "description": "Local development server"
    }
  ],
  "paths": {
    "/tokenize": {
      "post": {
        "summary": "Tokenize C++ code",
        "description": "Convert C++ code to token IDs",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "text": {
                    "type": "string",
                    "example": "int main() { return 0; }"
                  }
                },
                "required": ["text"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful response"
          }
        }
      }
    },
    "/detokenize": {
      "post": {
        "summary": "Detokenize tokens",
        "description": "Convert token IDs back to C++ code",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "tokens": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "example": [42, 43, 44, 45]
                  }
                },
                "required": ["tokens"]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Successful response"
          }
        }
      }
    },
    "/stats": {
      "get": {
        "summary": "Get tokenizer statistics",
        "description": "Returns current tokenizer statistics",
        "responses": {
          "200": {
            "description": "Successful response"
          }
        }
      }
    }
  }
})";
    });
    
    // ======================================================================
    // Запуск сервера (без глобального указателя для сигналов)
    // ======================================================================
    
    std::cout << "Сервер запускается на http://localhost:" << port << std::endl;
    std::cout << "Документация: http://localhost:" << port << "/" << std::endl;
    std::cout << "Swagger UI: http://localhost:" << port << "/swagger" << std::endl;
    std::cout << "Потоков: " << threads << std::endl;
    std::cout << "Нажмите Ctrl+C для остановки" << std::endl;
    std::cout << std::endl;
    
    // Запускаем сервер (блокирующий вызов)
    app.port(port).multithreaded().run();
    
    std::cout << "\nСервер остановлен" << std::endl;
    
    return 0;
}