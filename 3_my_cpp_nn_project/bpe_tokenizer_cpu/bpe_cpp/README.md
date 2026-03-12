# 🚀 BPE Tokenizer - Высокопроизводительная C++ реализация

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.16+-brightgreen.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Fast BPE Tokenizer** - это высокопроизводительная реализация алгоритма Byte Pair Encoding (BPE) для токенизации C++ кода. Проект включает как базовую, так и оптимизированную версии токенизатора, а также полный набор инструментов для тестирования, бенчмаркинга и веб-интеграции.

## 📋 Содержание

- [🚀 BPE Tokenizer - Высокопроизводительная C++ реализация](#-bpe-tokenizer---высокопроизводительная-c-реализация)
  - [📋 Содержание](#-содержание)
  - [✨ Особенности](#-особенности)
  - [🔧 Требования](#-требования)
  - [🚀 Быстрый старт](#-быстрый-старт)
  - [📚 Компоненты](#-компоненты)
    - [**Основные классы**](#основные-классы)
    - [**Оптимизации**](#оптимизации)
    - [**Утилиты и скрипты**](#утилиты-и-скрипты)
  - [⚙️ Конфигурация сборки](#️-конфигурация-сборки)
    - [**Основные опции**](#основные-опции)
    - [**Оптимизации производительности**](#оптимизации-производительности)
    - [**Инструменты разработчика**](#инструменты-разработчика)
  - [📊 Производительность](#-производительность)
    - [**Сравнение с Python**](#сравнение-с-python)
    - [**Влияние размера словаря**](#влияние-размера-словаря)
    - [**Масштабирование с числом потоков**](#масштабирование-с-числом-потоков)
  - [🌐 Веб-сервер](#-веб-сервер)
  - [📈 Бенчмарки](#-бенчмарки)
  - [🧪 Тестирование](#-тестирование)
  - [🐍 Python интеграция](#-python-интеграция)
  - [📦 Модели](#-модели)
  - [📁 Структура каталога](#-структура-каталога)
  - [🔧 Устранение неполадок](#-устранение-неполадок)
  - [📄 Лицензия](#-лицензия)

## ✨ Особенности

- 🔥 **Две реализации**: базовая (`BPETokenizer`) и оптимизированная (`FastBPETokenizer`)
- ⚡ **SIMD оптимизации**: AVX2, SSE4.2 для массовой обработки
- 🧵 **Многопоточность**: OpenMP для параллельного обучения и обработки
- 💾 **Кэширование**: потокобезопасный LRU-кэш для частых запросов
- 📦 **Пул памяти**: уменьшение количества аллокаций
- 🔬 **Профилирование**: встроенный профайлер для поиска узких мест
- 🌐 **Веб-сервер**: REST API на сокетах и CrowCpp
- 🐍 **Python биндинги**: использование из Python через pybind11
- 📊 **Бенчмарки**: Google Benchmark для измерения производительности
- 🧪 **Тесты**: Google Test для проверки корректности
- 📚 **Примеры**: полный набор примеров использования

## 🔧 Требования

- **Компилятор**: GCC 9+ / Clang 10+ / MSVC 2019+ с поддержкой C++17
- **CMake**: версия 3.16 или выше
- **Python**: версия 3.8 или выше (для биндингов и скриптов)

**Опциональные зависимости:**
- **OpenMP**: для параллельного обучения
- **pybind11**: для Python биндингов
- **CrowCpp**: для веб-сервера
- **zlib**: для сжатия ответов в Crow сервере

## 🚀 Быстрый старт

```bash
# 1. Клонирование репозитория
git clone https://github.com/yourusername/bpe_tokenizer_cpu.git
cd bpe_tokenizer_cpu/bpe_cpp

# 2. Конфигурация и сборка
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 3. Запуск простого примера
./examples/simple_example

# 4. Запуск тестов
make test

# 5. Запуск бенчмарков
make benchmark-all
```

## 📚 Компоненты

### Основные классы

| Класс | Файл | Описание |
|-------|------|----------|
| `BPETokenizer` | `bpe_tokenizer.hpp` | Базовая реализация BPE алгоритма. Подходит для обучения и отладки |
| `FastBPETokenizer` | `fast_tokenizer.hpp` | Оптимизированная версия с SIMD, кэшем и пулом памяти. Рекомендуется для продакшена |
| `Vocabulary` | `vocabulary.hpp` | Двустороннее отображение токен ↔ ID с сериализацией |
| `ParallelTrainer` | `parallel_trainer.hpp` | Многопоточное обучение на больших корпусах |

### Оптимизации

| Компонент | Описание | Где используется |
|-----------|----------|------------------|
| `simd_utils.hpp` | AVX2/SSE4.2 для массовой обработки | `FastBPETokenizer::byte_level_encode()` |
| `thread_safe_cache.hpp` | LRU-кэш для частых слов | `FastBPETokenizer` |
| `memory_pool.hpp` | Пул памяти для уменьшения аллокаций | `FastBPETokenizer` |
| `optimized_types.hpp` | 64-битные ключи вместо строк | `merges_` в `FastBPETokenizer` |
| `parallel_trainer.hpp` | OpenMP для параллельного обучения | `ParallelTrainer` |

### Утилиты и скрипты

| Файл | Назначение |
|------|------------|
| `tools/convert_vocab.py` | Конвертация словаря из Python в C++ формат |
| `scripts/build.sh` | Универсальный скрипт сборки |
| `scripts/run_tests.sh` | Запуск тестов с анализом покрытия |
| `scripts/run_benchmarks.sh` | Запуск бенчмарков с графиками |
| `scripts/compare_performance.py` | Сравнение производительности Python и C++ |

## ⚙️ Конфигурация сборки

### Основные опции
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \        # Release, Debug, RelWithDebInfo
         -DBUILD_PYTHON_BINDINGS=ON \        # Собрать Python биндинги
         -DBUILD_TESTING=ON \                 # Собрать тесты
         -DBUILD_BENCHMARKS=ON \              # Собрать бенчмарки
         -DBUILD_EXAMPLES=ON \                 # Собрать примеры
         -DBUILD_WEB_SERVER=ON                 # Собрать веб-сервер
```

### Оптимизации производительности
```bash
cmake .. -DUSE_OPENMP=ON \        # OpenMP для параллельного обучения
         -DUSE_TBB=OFF \           # Intel TBB (альтернатива OpenMP)
         -DUSE_SIMD=ON \           # SIMD оптимизации (AVX2, SSE4.2)
         -DUSE_CACHE=ON             # Кэширование результатов
```

### Инструменты разработчика
```bash
cmake .. -DBUILD_WITH_PROFILING=ON \   # Поддержка профилирования (-pg)
         -DBUILD_WITH_ASAN=ON \         # AddressSanitizer
         -DBUILD_WITH_UBSAN=ON           # UndefinedBehaviorSanitizer
```

## 📊 Производительность

### Сравнение с Python
| Реализация | Время (100KB) | Относительная скорость |
|------------|---------------|------------------------|
| Python (чистый) | 15.2 ms | 1x |
| C++ базовая | 2.8 ms | 5.4x |
| C++ оптимизированная | 1.2 ms | 12.7x |

### Влияние размера словаря
| Размер словаря | Время encode | Память | Сжатие |
|----------------|--------------|--------|--------|
| 8000 | 1.2 ms | 2.5 MB | 24% |
| 10000 | 1.5 ms | 3.2 MB | 29% |
| 12000 | 1.8 ms | 3.8 MB | 33% |

### Масштабирование с числом потоков
| Потоков | Время обучения | Ускорение |
|---------|---------------|-----------|
| 1 | 100 с | 1x |
| 2 | 52 с | 1.92x |
| 4 | 28 с | 3.57x |
| 8 | 15 с | 6.67x |

## 🌐 Веб-сервер

Проект включает два варианта веб-сервера:
```bash
# Простой сервер на сокетах (без зависимостей)
./web/server --port 8080

# REST API сервер на CrowCpp (со Swagger UI)
./web/tokenizer_server_crow --port 8080 --threads 4
```

**Документация API:** http://localhost:8080/swagger

## 📈 Бенчмарки
```bash
# Запуск всех бенчмарков
make benchmark-all

# Или через скрипт
./scripts/run_benchmarks.sh --type all --format plot
```

## 🧪 Тестирование
```bash
# Запуск всех тестов
make test
# Или через скрипт с анализом покрытия
./scripts/run_tests.sh --coverage --html
```

## 🐍 Python интеграция
```bash
# Сборка Python модуля
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make

# Тестирование
python3 -c "import bpe_tokenizer_cpp; print(dir(bpe_tokenizer_cpp))"
```

## 📦 Модели

В папке models/ находятся предобученные модели:

- bpe_8000/ - основная модель для повседневного использования
- bpe_10000/ - для бенчмарков и сравнения
- bpe_12000/ - для анализа влияния размера словаря

Для конвертации своих моделей из Python:
```bash
python3 tools/convert_vocab.py --input ../bpe_python/models/bpe_8000 --output models/bpe_8000
```

## 🔧 Устранение неполадок

### Проблемы со сборкой

```bash
# Очистка и пересборка
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Python биндинги не собираются

```bash
# Установка pybind11
pip install pybind11

# Или через систему
sudo apt install pybind11-dev
```

### Crow сервер не компилируется

```bash
# Установка CrowCpp
git clone https://github.com/CrowCpp/Crow.git ../third_party/crow

# Установка zlib
sudo apt install zlib1g-dev  # Ubuntu/Debian
brew install zlib             # macOS
```

## 📄 Лицензия

Проект распространяется под лицензией MIT. См. файл LICENSE для подробностей.


## 📁 Структура каталога
```text
bpe_cpp/
├── benchmarks/                     # Бенчмарки
│   ├── bench_tokenizer.cpp         # Бенчмарк базовой версии
│   ├── bench_fast_tokenizer.cpp    # Бенчмарк оптимизированной версии
│   └── bench_comparison.cpp        # Сравнение версий
├── build/                          # директория сборки (создается автоматически)
│   └── ...                         # сгенерированные файлы сборки
├── examples/                       # Примеры использования
├── include/                        # Заголовочные файлы
│   ├── bpe_tokenizer.hpp           # Базовая версия
│   ├── fast_tokenizer.hpp          # Оптимизированная версия
│   ├── vocabulary.hpp              # Управление словарём
│   └── ...                         # Остальные заголовки
├── models/                         # Обученные модели
│   ├── bpe_8000/                   # Модель на 8000 токенов
│   ├── bpe_10000/                  # Модель на 10000 токенов
│   └── bpe_12000/                  # Модель на 12000 токенов
├── scripts/                        # Вспомогательные скрипты
│   ├── build.sh                    # Сборка проекта
│   ├── run_tests.sh                # Запуск тестов
│   └── compare_performance.py      # Сравнение с Python
├── src/                            # Исходные файлы
│   ├── bpe_tokenizer.cpp           # Реализация базовой версии
│   ├── fast_tokenizer.cpp          # Реализация оптимизированной версии
│   └── ...                         # Остальные исходники
├── tests/                          # Модульные тесты
├── tools/                          # Инструменты
│   └── convert_vocab.py            # Конвертация словаря из Python
├── web/                            # Веб-сервер
│   ├── server.cpp                  # Простой сервер на сокетах
│   └── server_crow.cpp             # REST API сервер на CrowCpp
├── CMakeLists.txt                  # Основной файл сборки
└── README.md                       # Этот файл
```

