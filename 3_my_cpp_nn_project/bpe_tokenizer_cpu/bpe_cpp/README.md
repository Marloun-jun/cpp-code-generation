# ⚡ BPE Tokenizer — C++ реализация

Высокопроизводительная реализация Byte Pair Encoding (BPE) токенизатора на C++ с поддержкой SIMD-оптимизаций, параллельного обучения и интеграции с Python.

[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.16+-green.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Содержание

- [⚡ BPE Tokenizer — C++ реализация](#-bpe-tokenizer--c-реализация)
  - [📋 Содержание](#-содержание)
  - [🎯 О проекте](#-о-проекте)
  - [📊 Ключевые характеристики](#-ключевые-характеристики)
  - [🚀 Быстрый старт](#-быстрый-старт)
    - [1. Клонирование и сборка](#1-клонирование-и-сборка)
    - [2. Запуск примеров](#2-запуск-примеров)
    - [3. Запуск тестов](#3-запуск-тестов)
    - [4. Запуск бенчмарков](#4-запуск-бенчмарков)
    - [5. Запуск веб-сервера](#5-запуск-веб-сервера)
  - [📈 Производительность](#-производительность)
  - [🔧 Компоненты](#-компоненты)
  - [📁 Структура проекта](#-структура-проекта)
  - [📚 Документация](#-документация)
  - [🤝 Интеграция с Python](#-интеграция-с-python)
  - [⚙️ Опции сборки](#️-опции-сборки)
  - [📊 Поддерживаемые модели](#-поддерживаемые-модели)
  - [⭐ Заключение](#-заключение)

## 🎯 О проекте

**Цель:** Создание максимально быстрого BPE токенизатора для обработки C++ кода.

**Ключевые особенности:**
- ⚡ **Высокая производительность** — ускорение до 175x по сравнению с базовой версией
- 🚀 **SIMD оптимизации** — поддержка AVX2, AVX, SSE4.2
- 🔄 **Параллельное обучение** — использование всех ядер CPU через OpenMP
- 💾 **Экономия памяти** — пул памяти, кэширование, компактные структуры
- 🐍 **Python биндинги** — вызов C++ кода из Python через pybind11
- 🌐 **Веб-сервер** — REST API с Swagger UI и Prometheus метриками
- ✅ **Полное покрытие тестами** — модульные тесты, бенчмарки, валидация с Python

## 📊 Ключевые характеристики

| Показатель | Базовая версия | Оптимизированная | Ускорение |
|------------|---------------|------------------|-----------|
| Encode (128 KB) | ~2.6 мс | ~0.015 мс | **~175x** |
| Пакетная обработка (x64) | ~2.76 мс | ~0.047 мс | **~59x** |
| Декодирование | 8.28 мкс | 2.75 мкс | **~3x** |
| Пропускная способность | 48.7 МБ/с | **10 ГБ/с** | **~205x** |
| Пиковая память | ~12.6 МБ | ~10.5 МБ | -17% |

*Данные из бенчмарков на Intel i5-1240P (16 ядер, 4.4 GHz)*

## 🚀 Быстрый старт

### 1. Клонирование и сборка
```bash
# Клонирование репозитория
git clone <repo_url>
cd bpe_tokenizer_cpu/bpe_cpp

# Сборка (Release)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Или через скрипт
cd ../scripts && ./build.sh Release
```

### 2. Запуск примеров
```bash
# Простой пример
./examples/simple_example    # Из директории build/

# Пакетная обработка
./examples/batch_example

# Демо оптимизаций
./examples/fast_tokenizer_demo

# Обучение модели
./examples/train_example --corpus ../data/corpus/train_code.txt
```

### 3. Запуск тестов
```bash
# Все тесты
cd build && ctest -V

# Или через скрипт
cd ../scripts && ./run_tests.sh --verbose

# Конкретный тест
./tests/test_fast_tokenizer    # Из директории build/

# С фильтром
./tests/test_fast_tokenizer --gtest_filter="*Encode*"
```

### 4. Запуск бенчмарков
```bash
# Все бенчмарки
cd ../scripts && ./run_benchmarks.sh --type all

# Только сравнение версий
./run_benchmarks.sh --type comparison --format plot

# С сохранением в JSON
./run_benchmarks.sh --type fast --output results.json
```

### 5. Запуск веб-сервера
```bash
# Crow сервер (REST API + Swagger)
cd build && make run_crow_server
# Открыть http://localhost:8080/swagger

# Простой сервер (без зависимостей)
make run_simple_server
# Открыть http://localhost:8080
```

## 📈 Производительность

| Операция | Время | Примечание |
|----------|-------|------------|
| Encode 1 KB | 0.15 мкс | Короткий текст |
| Encode 128 KB | 14.7 мкс | Средний файл |
| Encode 1 MB | 150 мкс | Большой файл |
| Decode 1000 токенов | 2.75 мкс | ~1.4 КБ текста |
| Batch encode (x64) | 47 мкс | 64 текста одновременно |
| Загрузка модели | 5.8 мс | Модель 10000 |

## 🔧 Компоненты

| Компонент | Назначение | Документация |
|-----------|------------|--------------|
| `BPETokenizer` | Базовая реализация BPE | [include/bpe_tokenizer.hpp](include/bpe_tokenizer.hpp) |
| `FastBPETokenizer` | Оптимизированная версия | [include/fast_tokenizer.hpp](include/fast_tokenizer.hpp) |
| `Vocabulary` | Управление словарём | [include/vocabulary.hpp](include/vocabulary.hpp) |
| `ParallelTrainer` | Параллельное обучение | [include/parallel_trainer.hpp](include/parallel_trainer.hpp) |
| `MemoryPool` | Пул памяти | [include/memory_pool.hpp](include/memory_pool.hpp) |
| `StringViewCache` | Кэширование результатов | [include/thread_safe_cache.hpp](include/thread_safe_cache.hpp) |
| `SIMDUtils` | SIMD оптимизации | [include/simd_utils.hpp](include/simd_utils.hpp) |

## 📁 Структура проекта

```text
bpe_cpp/
├── benchmarks/       # Бенчмарки производительности
│   └── ...
├── examples/         # Примеры использования
│   └── ...
├── include/          # Заголовочные файлы
│   └── ...
├── models/           # Обученные модели
│   └── ...
├── scripts/          # Вспомогательные скрипты
│   └── ...
├── src/              # Исходные файлы
│   └── ...
├── tests/            # Модульные тесты
│   └── ...
├── tools/            # Утилиты
│   └── ...
├── web/              # Веб-сервер
│   └── ...
├── CMakeLists.txt    # Основной файл сборки
└── README.md         # Этот файл
```
**Примечание:** В данной структуре указаны только корневые папки. Содержимое папок приведено в файлах README.md (смотри раздел "Документация")

## 📚 Документация

Подробная документация находится в README.md файлах каждой подпапки:

| Папка | Содержание |
|-------|------------|
| [`benchmarks/`](benchmarks/README.md) | Бенчмарки и сравнение производительности |
| [`examples/`](examples/README.md) | Примеры использования |
| [`include/`](include/README.md) | Заголовочные файлы и API |
| [`models/`](models/README.md) | Обученные модели |
| [`scripts/`](scripts/README.md) | Вспомогательные скрипты |
| [`src/`](src/README.md) | Исходный код |
| [`tests/`](tests/README.md) | Модульные тесты |
| [`web/`](web/README.md) | Веб-сервер |

## 🤝 Интеграция с Python

```python
# Импорт C++ токенизатора
from bpe_tokenizer_cpp import FastBPETokenizer

# Создание и загрузка модели
tokenizer = FastBPETokenizer(vocab_size=10000, byte_level=True)
tokenizer.load("models/bpe_10000/cpp_vocab.json", 
               "models/bpe_10000/cpp_merges.txt")

# Токенизация
tokens = tokenizer.encode("int main() { return 0; }")
print(f"Токенов: {len(tokens)}")

# Декодирование
text = tokenizer.decode(tokens)
print(f"Текст: {text}")

# Статистика
stats = tokenizer.stats()
print(f"Cache hit rate: {stats.cache_hit_rate():.1f}%")
```

## ⚙️ Опции сборки

| Опция | По умолчанию | Описание |
|-------|--------------|----------|
| `BUILD_PYTHON_BINDINGS` | ON | Сборка Python модуля |
| `BUILD_TESTING` | ON | Сборка тестов |
| `BUILD_BENCHMARKS` | ON | Сборка бенчмарков |
| `BUILD_EXAMPLES` | ON | Сборка примеров |
| `BUILD_WEB_SERVER` | ON | Сборка веб-сервера |
| `USE_OPENMP` | ON | Параллельное обучение |
| `USE_SIMD` | ON | SIMD оптимизации |
| `USE_CACHE` | ON | Кэширование результатов |
| `USE_LOCAL_JSON` | OFF | Локальный json.hpp |

**Пример конфигурации:**
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=OFF -DUSE_OPENMP=ON -DUSE_SIMD=ON
```

## 📊 Поддерживаемые модели

| Модель | Размер | Назначение | Статус |
|--------|--------|------------|--------|
| **bpe_10000** | 10000 | Основная рабочая модель | ✅ Рекомендуется |
| bpe_8000 | 8000 | Для бенчмарков | 🔬 Экспериментальная |
| bpe_12000 | 12000 | Для бенчмарков | 🔬 Экспериментальная |

## ⭐ Заключение

C++ реализация BPE токенизатора обеспечивает максимальную производительность благодаря SIMD оптимизациям, параллельной обработке и эффективному использованию памяти. Подходит как для встраивания в C++ приложения, так и для использования из Python.

---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026