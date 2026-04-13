# Бенчмарки BPE токенизатора

Этот каталог содержит набор тестов производительности для BPE токенизатора, реализованных с использованием библиотеки [Google Benchmark](https://github.com/google/benchmark).

## 📋 Содержание

- [Бенчмарки BPE токенизатора](#бенчмарки-bpe-токенизатора)
  - [📋 Содержание](#-содержание)
  - [🔧 Требования](#-требования)
  - [🚀 Запуск бенчмарков](#-запуск-бенчмарков)
    - [Быстрый старт](#быстрый-старт)
    - [Запуск всех бенчмарков](#запуск-всех-бенчмарков)
    - [Запуск отдельных бенчмарков](#запуск-отдельных-бенчмарков)
  - [💾 Сохранение результатов](#-сохранение-результатов)
    - [Пути сохранения](#пути-сохранения)
  - [📊 Описание бенчмарков](#-описание-бенчмарков)
    - [1. bench\_tokenizer (базовая версия)](#1-bench_tokenizer-базовая-версия)
    - [2. bench\_fast\_tokenizer (оптимизированная версия)](#2-bench_fast_tokenizer-оптимизированная-версия)
    - [3. bench\_comparison (сравнение)](#3-bench_comparison-сравнение)
  - [📈 Интерпретация результатов](#-интерпретация-результатов)
    - [Ключевые метрики](#ключевые-метрики)
    - [Пример вывода](#пример-вывода)
  - [🔄 Сравнение с Python реализацией](#-сравнение-с-python-реализацией)
  - [🔍 Профилирование](#-профилирование)
  - [📁 Структура каталога](#-структура-каталога)

## 🔧 Требования

Перед запуском бенчмарков убедитесь, что у вас есть:

1. **Собранный проект** (см. [основной README](../README.md#-быстрый-старт))
2. **Модели токенизатора** в директории `bpe_cpp/models/`:
   - `bpe_8000/cpp_vocab.json` и `bpe_8000/cpp_merges.txt` (модель 8000)
   - `bpe_10000/cpp_vocab.json` и `bpe_10000/cpp_merges.txt` (модель 10000)
   - `bpe_12000/cpp_vocab.json` и `bpe_12000/cpp_merges.txt` (модель 12000)
3. **Тестовые данные** (опционально) — файл `bench_data/sample_code.txt` с C++ кодом

> **Примечание:** Если модели отсутствуют, их можно сконвертировать из Python версии:
> ```bash
> cd ../tools/
> python convert_vocab.py --model-size 8000
> python convert_vocab.py --model-size 10000
> python convert_vocab.py --model-size 12000
> ```

## 🚀 Запуск бенчмарков

### Быстрый старт

```bash
# Переход в директорию проекта
cd ~/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/

# Сборка с бенчмарками
mkdir -p build && cd build
cmake .. -DBUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
make -j4

# Проверка наличия моделей
make check_models

# Запуск всех бенчмарков
make run_benchmarks
```
### Запуск всех бенчмарков

```bash
make run_benchmarks
```

**Эта команда последовательно запустит:**

1. bench_tokenizer — базовая версия
2. bench_fast_tokenizer — оптимизированная версия
3. bench_comparison — прямое сравнение

### Запуск отдельных бенчмарков

```bash
# Только базовая версия
make run_bench_tokenizer

# Только оптимизированная версия
make run_bench_fast_tokenizer

# Только сравнение
make run_bench_comparison
```

## 💾 Сохранение результатов

### Пути сохранения

Результаты сохраняются в директорию `bpe_cpp/reports/benchmarks/`:

| Команда | Формат | Путь сохранения |
|---------|--------|------------------|
| `make run_benchmarks` | JSON | `../reports/benchmarks/bench_*_YYYYMMDD_HHMMSS.json` |
| Ручной запуск | JSON/CSV | Указанный путь или текущая директория |

**Примеры команд для сохранения:**

```bash
# Автоматическое сохранение (тихий режим)
make run_benchmarks

# Ручное сохранение в JSON
./benchmarks/bench_comparison \
    --benchmark_out=../reports/benchmarks/comparison.json \
    --benchmark_out_format=json

# Ручное сохранение в CSV
./benchmarks/bench_comparison \
    --benchmark_out=../reports/benchmarks/comparison.csv \
    --benchmark_out_format=csv
```

## 📊 Описание бенчмарков

### 1. **bench_tokenizer** (базовая версия)

Тестирует производительность BPETokenizer — базовой реализации без оптимизаций.
| Тест | Описание | Что измеряет |
|------|----------|--------------|
| `BM_EncodeShort` | Кодирование короткого текста (1 строка) | Накладные расходы на вызов |
| `BM_EncodeLong` | Кодирование полного тестового корпуса | Базовая производительность |
| `BM_Decode` | Декодирование обратно в текст | Скорость обратной операции |
| `BM_BatchEncode` | Пакетная обработка 10 текстов | Эффективность при батчах |
| `BM_EncodeVariableLength/64` | Текст 64 байта | Масштабирование с размером |
| `BM_EncodeVariableLength/65536` | Текст 64 КБ | Масштабирование с размером |
| `BM_EncodeDifferentVocab/8000` | Словарь 8000 | Влияние размера словаря |
| `BM_EncodeDifferentVocab/12000` | Словарь 12000 | Влияние размера словаря |
| `BM_CompareModes/0` | Обычный режим | Сравнение режимов |
| `BM_CompareModes/1` | Byte-level режим | Сравнение режимов |
| `BM_Repeatability` | Повторяемость результатов | Стабильность измерений |
| `BM_MemoryUsage` | Использование памяти | Потребление RAM |

### 2. **bench_fast_tokenizer** (оптимизированная версия)

Тестирует производительность FastBPETokenizer с оптимизациями:
- SIMD инструкции для ускорения поиска
- Пул памяти для уменьшения аллокаций
- Кэширование частых результатов
- Компактные структуры данных

| Тест | Описание |
|------|----------|
| `BM_FastTokenizer_Encode` | Кодирование текстов разного размера |
| `BM_FastTokenizer_Decode` | Декодирование разного количества токенов |
| `BM_FastTokenizer_EncodeBatch` | Пакетная обработка (1-64 текста) |
| `BM_FastTokenizer_CacheEfficiency` | Эффективность кэша (100-10000 элементов) |
| `BM_FastTokenizer_Multithreaded` | Масштабирование (1-8 потоков) |
| `BM_FastTokenizer_Multithreaded_Advanced` | Разные нагрузки на потоки |
| `BM_FastTokenizer_CompareSizes` | Сравнение размеров словарей |
| `BM_FastTokenizer_MemoryLeak` | Проверка утечек памяти |

### 3. **bench_comparison** (сравнение)

Прямое сравнение двух версий на одинаковых данных

| Тест | Описание |
|------|----------|
| `BM_Basic_EncodeShort` / `BM_Fast_EncodeShort` | Короткий текст |
| `BM_Basic_EncodeLong` / `BM_Fast_EncodeLong` | Длинный текст (1KB-256KB) |
| `BM_Basic_Decode` / `BM_Fast_Decode` | Декодирование |
| `BM_Basic_BatchEncode` / `BM_Fast_BatchEncode` | Пакетная обработка |
| `BM_Basic_MemoryUsage` / `BM_Fast_MemoryUsage` | Использование памяти |
| `BM_Basic_LoadTime` / `BM_Fast_LoadTime` | Время загрузки моделей |
| `BM_Compare_VocabSizes` | Влияние размера словаря |

Ключевой тест: Показывает итоговое ускорение оптимизированной версии.

## 📈 Интерпретация результатов

### Ключевые метрики

| Метрика | Что означает | Хорошее значение |
|---------|--------------|------------------|
| **Время (us/ms)** | Абсолютное время выполнения | Чем меньше, тем лучше |
| **BytesProcessed** | Пропускная способность | > 100 MB/s |
| **ItemsProcessed** | Для пакетной обработки | > 1000 items/s |
| **CacheHitRate** | Эффективность кэша (fast) | > 90% |
| **Speedup_x** | Ускорение относительно базовой версии | > 2.0x |
| **CV_percent** | Коэффициент вариации (стабильность) | < 5% |

### Пример вывода
```text
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_EncodeShort                   1.23 us         1.23 us       568182
BM_EncodeLong                     245 us          245 us         2857
BM_FastTokenizer_Encode/1024     3.45 us         3.45 us       202899
BM_CompareWithOriginal           15.6 ms         15.6 ms           45
  Original BPETokenizer          45.2 ms         45.2 ms           15
  FastBPETokenizer               15.6 ms         15.6 ms           45
```
*Примечание: реальные значения зависят от процессора и модели.*

**Интерпретация:**
- Короткий текст кодируется за 1.23 микросекунды
- Длинный текст (весь корпус) — за 245 микросекунд
- FastTokenizer обрабатывает 1 KB за 3.45 мкс
- Ускорение оптимизированной версии: 2.9x (45.2 / 15.6)

## 🔄 Сравнение с Python реализацией
Для сравнения производительности C++ и Python версий используйте скрипты из корневой директории:
```bash
# Переход в корневую директорию
cd ~/Projects/NS/

# Сравнение всех трех версий (HF, Python, C++)
python scripts/benchmark_all.py

# Визуализация результатов
python scripts/plot_results.py

# Валидация корректности C++ токенизатора
python scripts/validate_cpp_tokenizer.py
```
Результаты сохраняются в:
- `../reports/benchmark_results.json`
- `../reports/figures/comparison.png`
- `../reports/figures/comparison.pdf`



## 🔍 Профилирование
Для детального анализа узких мест используйте профилировщик:
```bash
# Переход в директорию сборки
cd ~/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/

# Профилирование с perf (Linux)
perf record ./benchmarks/bench_comparison --benchmark_min_time=0.1
perf report

# Профилирование с gprof
./benchmarks/bench_comparison
gprof ./benchmarks/bench_comparison gmon.out > analysis.txt

# Использование встроенного профайлера (если включен)
./benchmarks/bench_fast_tokenizer --benchmark_enable_profiling=true

# Сохранение профиля в файл
perf record -o perf_comparison.data ./benchmarks/bench_comparison
perf report -i perf_comparison.data
```

## 📁 Структура каталога
```text
benchmarks/
├── bench_data/                 # Тестовые данные
│   └── sample_code.txt         # Примеры C++ кода
├── bench_comparison.cpp        # Прямое сравнение версий
├── bench_fast_tokenizer.cpp    # Бенчмарки оптимизированной версии
├── bench_tokenizer.cpp         # Бенчмарки базовой версии
├── CMakeLists.txt              # Конфигурация сборки
└── README.md                   # Этот файл
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026