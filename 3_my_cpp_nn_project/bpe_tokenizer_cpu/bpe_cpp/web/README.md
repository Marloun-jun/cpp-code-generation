# 🌐 Веб-серверы BPE токенизатора

Этот каталог содержит две реализации веб-сервера для BPE токенизатора:
- **Простой сервер на сокетах** (без внешних зависимостей)
- **REST API сервер на CrowCpp** (с Swagger UI и Prometheus метриками)

## 📋 Содержание

- [🌐 Веб-серверы BPE токенизатора](#-веб-серверы-bpe-токенизатора)
  - [📋 Содержание](#-содержание)
  - [🔧 Сборка серверов](#-сборка-серверов)
    - [**Сборка обоих серверов**](#сборка-обоих-серверов)
    - [**Сборка только простого сервера**](#сборка-только-простого-сервера)
    - [**Сборка только Crow сервера**](#сборка-только-crow-сервера)
  - [🚀 Запуск серверов](#-запуск-серверов)
    - [**Запуск простого сервера**](#запуск-простого-сервера)
    - [**Запуск Crow сервера**](#запуск-crow-сервера)
  - [🔄 Сравнение серверов](#-сравнение-серверов)
  - [📡 API эндпоинты](#-api-эндпоинты)
    - [**Простой сервер**](#простой-сервер)
    - [**Crow сервер**](#crow-сервер)
  - [📊 Метрики Prometheus (только Crow)](#-метрики-prometheus-только-crow)
  - [📚 Swagger UI (только Crow)](#-swagger-ui-только-crow)
  - [🔧 Требования к зависимостям](#-требования-к-зависимостям)
    - [**Для простого сервера**](#для-простого-сервера)
    - [**Для Crow сервера**](#для-crow-сервера)
  - [📁 Структура каталога](#-структура-каталога)


## 🔧 Сборка серверов

### **Сборка обоих серверов**
```bash
# Из корневой директории bpe_cpp/
cd bpe_cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### **Сборка только простого сервера**
```bash
make server
```

### **Сборка только Crow сервера**
```bash
make tokenizer_server_crow
```

## 🚀 Запуск серверов

### **Запуск простого сервера**
```bash
# Из директории build
./web/server [--port PORT]

# Или через цель make
make run_simple_server
```

**Параметры:**

- --port PORT - порт для сервера (по умолчанию 8080)

**Пример:**
```bash
./web/server --port 8080
```

### **Запуск Crow сервера**
```bash
# Из директории build
./web/tokenizer_server_crow [--port PORT] [--threads N]

# Или через цель make
make run_web_server
```

**Параметры:**

- --port PORT - порт для сервера (по умолчанию 8080)
- --threads N - количество потоков (по умолчанию все доступные)
- --model PATH - путь к файлу словаря
- --merges PATH - путь к файлу слияний

**Пример:**
```bash
./web/tokenizer_server_crow --port 8080 --threads 4
```

## 🔄 Сравнение серверов

| Характеристика | Простой сервер (server.cpp) | Crow сервер (server_crow.cpp) |
|----------------|------------------------------|-------------------------------|
| **Зависимости** | Нет (только стандартная библиотека) | CrowCpp, Zlib |
| **Размер** | ~100 КБ | ~1-2 МБ (с CrowCpp) |
| **Производительность** | Средняя | Высокая (асинхронный) |
| **API** | Базовый | Полноценный REST |
| **Swagger UI** | Нет | ✅ Да |
| **Prometheus метрики** | Нет | ✅ Да |
| **CORS поддержка** | Ручная | ✅ Автоматическая |
| **Многопоточность** | Последовательная обработка | ✅ Пул потоков |
| **Сжатие ответов** | Нет | ✅ Да (zlib) |
| **Graceful shutdown** | ✅ Да | ✅ Да |

## 📡 API эндпоинты

### Простой сервер

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | HTML интерфейс |
| POST | `/api/tokenize` | Токенизация текста |
| POST | `/api/detokenize` | Декодирование токенов |
| GET | `/api/stats` | Статистика токенизатора |
| GET | `/api/health` | Проверка работоспособности |
| OPTIONS | `*` | CORS preflight |

**Пример запроса:**
```bash
curl -X POST http://localhost:8080/api/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "int main() { return 0; }"}'
```

**Пример ответа:**
```json
{
  "tokens": [42, 17, 35, 98, 3, 105, 32, 12],
  "count": 8,
  "time_ms": 0.15
}
```

### Crow сервер

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/` | HTML документация |
| POST | `/tokenize` | Токенизация текста |
| POST | `/detokenize` | Декодирование токенов |
| POST | `/batch` | Пакетная обработка |
| GET | `/stats` | Статистика токенизатора |
| GET | `/health` | Проверка работоспособности |
| GET | `/metrics` | Метрики Prometheus |
| GET | `/swagger` | Swagger UI |
| GET | `/swagger.json` | OpenAPI спецификация |

**Пример запроса:**
```bash
curl -X POST http://localhost:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "int main() { return 0; }"}'
```

## 📊 Метрики Prometheus (только Crow)

Эндпоинт /metrics возвращает метрики в формате Prometheus:
```text
# HELP bpe_vocab_size Vocabulary size
# TYPE bpe_vocab_size gauge
bpe_vocab_size 10000

# HELP bpe_cache_hits_total Total cache hits
# TYPE bpe_cache_hits_total counter
bpe_cache_hits_total 1234
```

Можно настроить сбор метрик в Prometheus:
```yaml
scrape_configs:
  - job_name: 'bpe_tokenizer'
    static_configs:
      - targets: ['localhost:8080']
```

## 📚 Swagger UI (только Crow)

После запуска сервера откройте в браузере:
```text
http://localhost:8080/swagger
```

Swagger UI предоставляет интерактивную документацию API с возможностью тестирования эндпоинтов прямо из браузера.

## 🔧 Требования к зависимостям

### Для простого сервера

- Компилятор с поддержкой C++17
- Стандартная библиотека

### Для Crow сервера

**Установка CrowCpp:**
```bash
# Клонирование в third_party
git clone https://github.com/CrowCpp/Crow.git ../../third_party/crow
```

**Установка zlib:**
```bash
# Ubuntu/Debian
sudo apt install zlib1g-dev

# macOS
brew install zlib

# Windows (vcpkg)
vcpkg install zlib
```

## 📁 Структура каталога
```text
web/
├── CMakeLists.txt     # Конфигурация сборки
├── README.md          # Этот файл
├── server_crow.cpp    # REST API сервер на CrowCpp
├── server.cpp         # Простой сервер на сокетах
└── swagger.html       # Swagger UI (опционально)
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026