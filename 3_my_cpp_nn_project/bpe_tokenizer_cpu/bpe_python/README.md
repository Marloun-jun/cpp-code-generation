# 🐍 BPE Tokenizer - Python реализация

Эта директория содержит Python реализацию BPE токенизатора, которая служит эталоном для сравнения с C++ версиями и используется для обучения моделей.

## 📋 Содержание

- [🐍 BPE Tokenizer - Python реализация](#-bpe-tokenizer---python-реализация)
  - [📋 Содержание](#-содержание)
  - [📦 Основные модули](#-основные-модули)
    - [**`tokenizer.py`** — ядро токенизатора](#tokenizerpy--ядро-токенизатора)
    - [**`trainer.py`** — обучение моделей](#trainerpy--обучение-моделей)
    - [**`pytorch_integration.py`** — интеграция с PyTorch](#pytorch_integrationpy--интеграция-с-pytorch)
  - [🧪 Тестирование](#-тестирование)
    - [**`test_bpe_tokenizer.py`** — общие тесты](#test_bpe_tokenizerpy--общие-тесты)
    - [**`test_compare_speed.py`** — тесты производительности](#test_compare_speedpy--тесты-производительности)
    - [**`test_compare_models.py`** — сравнение моделей](#test_compare_modelspy--сравнение-моделей)
  - [📊 Результаты сравнения моделей](#-результаты-сравнения-моделей)
  - [🚀 Быстрый старт](#-быстрый-старт)
    - [**Установка зависимостей**](#установка-зависимостей)
    - [**Обучение модели**](#обучение-модели)
    - [**Использование токенизатора**](#использование-токенизатора)
  - [🔧 Расширенное использование](#-расширенное-использование)
    - [**Пакетная обработка**](#пакетная-обработка)
    - [**Работа с кэшем**](#работа-с-кэшем)
    - [**Сериализация**](#сериализация)
  - [🤝 Интеграция с PyTorch](#-интеграция-с-pytorch)
  - [📊 Сводная таблица](#-сводная-таблица)
  - [📁 Структура каталога](#-структура-каталога)


## 📦 Основные модули

### **`tokenizer.py`** — ядро токенизатора

**Назначение:** Основная реализация BPE алгоритма с поддержкой byte-level режима и кэширования.

**Ключевые особенности:**
- **Byte-level кодирование** — полная поддержка UTF-8 (русский, китайский, эмодзи)
- **LRU-кэш** — ускорение повторяющихся вызовов encode
- **Специальные токены** — `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<CPP>`, `<CODE>`
- **Сериализация** — JSON и бинарный форматы

**Пример использования:**
```python
from tokenizer import BPETokenizer

# создание токенизатора
tokenizer = BPETokenizer(vocab_size=8000, byte_level=True, cache_size=10000)

# обучение
tokenizer.train(["пример текста", "другой пример"])

# сохранение
tokenizer.save("vocab.json", "merges.txt")
tokenizer.save_binary("model.bin")

# кодирование
tokens = tokenizer.encode("int main() { return 0; }")
print(tokens)  # [45, 67, 89, 123, 45]

# декодирование
text = tokenizer.decode(tokens)
print(text)  # "int main() { return 0; }"

# статистика кэша
print(tokenizer.cache_stats())
```

### **`trainer.py`** — обучение моделей

**Назначение:** Высокоуровневый интерфейс для обучения токенизатора на корпусе текстов.

**Возможности:**

 - Загрузка корпуса из текстового файла
 - Валидация входных данных
 - Прогресс-бары для больших файлов
 - Автоматическое сохранение в трёх форматах
 - Тестирование на примерах C++ кода

**Пример использования:**
```bash
# Обучение с параметрами по умолчанию
python trainer.py

# Указание корпуса и размера словаря
python trainer.py --corpus ../data/corpus.txt --vocab-size 8000

# Сохранение в свою директорию
python trainer.py --output-dir ./my_models

# Отключение byte-level режима
python trainer.py --no-byte-level
```
### **`pytorch_integration.py`** — интеграция с PyTorch

**Назначение:** Адаптация токенизатора для использования в пайплайнах машинного обучения.

**Компоненты:**

 - BPETokenizerWrapper — обертка для PyTorch
 - CodeDataset — Dataset для C++ кода
 - Функции для создания DataLoader

**Пример использования:**
```python
from pytorch_integration import BPETokenizerWrapper, CodeDataset, create_dataloader
from tokenizer import BPETokenizer

# Загрузка обученной модели
tokenizer = BPETokenizer.load('vocab.json', 'merges.txt')

# Создание обертки
wrapper = BPETokenizerWrapper(tokenizer, max_length=128)

# Подготовка данных
texts = ["int main()", "std::cout << \"Hello\";"]
dataset = CodeDataset(texts, wrapper)
dataloader = create_dataloader(texts, wrapper, batch_size=32)

# Использование в обучении
for batch in dataloader:
    output = model(batch['input_ids'], batch['attention_mask'])
```

## 🧪 Тестирование

### **`test_bpe_tokenizer.py`** — общие тесты

Комплексное тестирование всех аспектов работы токенизатора:

 - ✅ Byte-level кодирование/декодирование
 - ✅ Обучение на маленьком корпусе
 - ✅ Сохранение и загрузка (JSON + бинарный)
 - ✅ Бенчмарк производительности
 - ✅ Граничные случаи (пустой текст, спецсимволы, 4-байтовый UTF-8)
```bash
python tests/test_bpe_tokenizer.py
python tests/test_bpe_tokenizer.py --verbose
```

### **`test_compare_speed.py`** — тесты производительности

Измерение скорости операций на большом тексте:

 - ⚡ Скорость encode (операций/сек, MB/сек)
 - ⚡ Скорость decode (операций/сек, MB/сек)
 - 📊 Статистика по длинам последовательностей
 - 🔄 Точность roundtrip
```bash
# С кэшированием (реальное использование)
python tests/test_compare_speed.py --model-size 8000 --iterations 50

# Без кэширования (чистая производительность)
python tests/test_compare_speed.py --model-size 8000 --iterations 50 --no-cache
```

### **`test_compare_models.py`** — сравнение моделей

Детальное сравнение моделей 8000, 10000 и 12000:

 - 🎯 Точность по категориям
 - 🗜️ Степень сжатия
 - ⚡ Скорость работы (без кэша)
 - 📚 Анализ словарей
 - 🔍 OOV анализ (Out of Vocabulary)
 - 📏 Глубина сжатия
```bash
# Полный анализ (рекомендуется)
python tests/test_compare_models.py --full-analysis

# Быстрый режим (только основные метрики)
python tests/test_compare_models.py --quick

# Только графики
python tests/test_compare_models.py --plot-only
```

## 📊 Результаты сравнения моделей

В директории `reports/` находятся результаты сравнения трёх моделей:

| Файл | Описание |
|------|----------|
| `three_model_report.txt` | Полный текстовый отчет |
| `three_model_comparison.json` | Сырые данные в JSON |
| `three_model_comparison.png` | Графики (PNG) |
| `three_model_comparison.pdf` | Графики (PDF) |
| `test_categories.json` | Категории тестов |

### Ключевые результаты

| Модель | Точность | Сжатие | Скорость encode | Скорость decode | Размер |
|--------|----------|--------|-----------------|-----------------|--------|
| bpe_8000 | **95.73%** | 1.76x | 75.9 млн/с | 6.4 млн/с | 7 998 |
| bpe_10000 | **95.73%** | 1.77x | **69.9 млн/с** | 5.5 млн/с | 9 998 |
| bpe_12000 | **95.73%** | **1.77x** | 80.8 млн/с | 5.8 млн/с | 11 997 |

### Анализ по категориям

| Категория | Точность | Проблемные области |
|-----------|----------|---------------------|
| Базовые конструкции | **100%** | Препроцессор, функции, классы |
| STL контейнеры | **80%** | `std::set`, `std::deque`, `std::shared_ptr` |
| Многопоточность | **80%** | `std::atomic`, `std::promise` |
| C++20 Concepts | **50%** | Сложные концепты |
| Реальные файлы | **98%** | Только `#include <string>` |

### Статистика словарей

| Показатель | bpe_8000 | bpe_10000 | bpe_12000 |
|------------|----------|-----------|-----------|
| Размер словаря | 7 998 | 9 998 | 11 997 |
| ASCII токенов | 6 506 | 8 197 | 9 889 |
| Unicode токенов | 1 476 | 1 784 | 2 090 |
| Средняя длина токена | 7.29 | 7.74 | 8.16 |
| Макс. длина токена | 54 | 61 | 61 |

### Глубина сжатия

| Показатель | bpe_8000 | bpe_10000 | bpe_12000 |
|------------|----------|-----------|-----------|
| Средняя длина токена | 2.15 | 2.16 | 2.16 |
| Токенов длины 1-2 | **74.4%** | **74.2%** | **74.1%** |
| Самый частый токен | `" "` (893) | `" "` (893) | `" "` (893) |

### OOV анализ

| Показатель | Значение |
|------------|----------|
| Уникальных символов | 179 |
| Покрытие словаря | **51.4%** |
| Неизвестных символов | 87 |
| Кириллица | 39 |
| Китайский | 23 |
| Эмодзи | 5 |

🏆 **Рекомендуемая модель:** `bpe_10000` — лучший баланс точности, скорости и размера словаря.

## 🚀 Быстрый старт

### Установка зависимостей
```bash
# Основные зависимости
pip install numpy matplotlib

# Для PyTorch интеграции
pip install torch

# Для бенчмарков
pip install pytest
```

### Обучение модели
```bash
# Обучить модель 8000 на корпусе по умолчанию
python trainer.py --vocab-size 8000

# Обучить модель 10000 (рекомендуется)
python trainer.py --vocab-size 10000

# Обучить модель 12000 на своем корпусе
python trainer.py --corpus ../data/corpus.txt --vocab-size 12000 --output-dir ./models
```

### Использование токенизатора
```python
from tokenizer import BPETokenizer

# Загрузка обученной модели (рекомендуется bpe_10000)
tokenizer = BPETokenizer.load('models/bpe_10000/vocab.json', 'models/bpe_10000/merges.txt')

# Токенизация
code = "int main() { std::cout << \"Hello, мир!\" << std::endl; }"
tokens = tokenizer.encode(code)
print(f"Токенов: {len(tokens)}")

# Декодирование
decoded = tokenizer.decode(tokens)
print(f"Декодировано: {decoded}")

# Статистика кэша
print(tokenizer.cache_stats())
```

## 🔧 Расширенное использование

### Пакетная обработка
```python
texts = ["int a;", "float b;", "char c;"]
batch_tokens = [tokenizer.encode(text) for text in texts]
```

### Работа с кэшем
```python
# Первый вызов - промах кэша
tokens1 = tokenizer.encode("int main()")

# Второй вызов - попадание в кэш (мгновенно)
tokens2 = tokenizer.encode("int main()")

# Статистика
stats = tokenizer.cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Сериализация
```python
# JSON формат (читаемый)
tokenizer.save("vocab.json", "merges.txt")

# Бинарный формат (компактный)
tokenizer.save_binary("model.bin")

# Загрузка
tokenizer = BPETokenizer.load("vocab.json", "merges.txt")
tokenizer = BPETokenizer.load_binary("model.bin")
```

## 🤝 Интеграция с PyTorch
```python
from pytorch_integration import BPETokenizerWrapper, create_dataloader
from tokenizer import BPETokenizer

# Загрузка модели
tokenizer = BPETokenizer.load('models/bpe_10000/vocab.json', 'models/bpe_10000/merges.txt')
wrapper = BPETokenizerWrapper(tokenizer, max_length=128)

# Подготовка данных
texts = ["int main()", "std::cout << \"Hello\";"]
dataloader = create_dataloader(texts, wrapper, batch_size=32)

# Обучение модели
for epoch in range(10):
    for batch in dataloader:
        output = model(batch['input_ids'], batch['attention_mask'])
        loss = output.loss
        loss.backward()
        optimizer.step()
```

## 📊 Сводная таблица

| Модуль | Назначение | Ключевые классы/функции |
|--------|------------|------------------------|
| `tokenizer.py` | Ядро токенизатора | `BPETokenizer`, `LRUCache` |
| `trainer.py` | Обучение моделей | `train_from_corpus()` |
| `pytorch_integration.py` | PyTorch интеграция | `BPETokenizerWrapper`, `CodeDataset` |
| `test_bpe_tokenizer.py` | Общие тесты | `BPETokenizerTest` |
| `test_compare_speed.py` | Тесты скорости | `SpeedTest` |
| `test_compare_models.py` | Сравнение моделей | `ThreeModelComparison` |

## 📁 Структура каталога
```text
bpe_python/
├── models/                       # Обученные модели
│   ├── bpe_8000/                 # Модель на 8000 токенов
│   │   ├── vocab.json
│   │   ├── model.bin
│   │   └── merges.txt
│   ├── bpe_10000/                # Модель на 10000 токенов
│   │   ├── vocab.json
│   │   ├── model.bin
│   │   └── merges.txt
│   └── bpe_12000/                # Модель на 12000 токенов
│       ├── vocab.json
│       ├── model.bin
│       └── merges.txt
├── reports/                      # Результаты сравнения
├── tests/                        # Тесты
│   ├── test_bpe_tokenizer.py     # Общие тесты
│   ├── test_compare_speed.py     # Тесты производительности
│   └── test_compare_models.py    # Сравнение моделей
├── pytorch_integration.py        # Интеграция с PyTorch
├── README.md                     # Этот файл
├── tokenizer.py                  # Основной класс BPETokenizer
└── trainer.py                    # Обучение моделей на корпусе
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026