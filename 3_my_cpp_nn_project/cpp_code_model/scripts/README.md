# 🛠️ Скрипты для обучения и запуска модели

Эта папка содержит исполняемые скрипты для обучения моделей, подготовки данных и запуска веб-сервера. Все скрипты используют общую структуру проекта и конфигурации.

## 📋 Содержание

- [🛠️ Скрипты для обучения и запуска модели](#️-скрипты-для-обучения-и-запуска-модели)
  - [📋 Содержание](#-содержание)
  - [🎯 Обучение моделей](#-обучение-моделей)
    - [**`train_tiny.py`** — обучение Tiny (5.2M)](#train_tinypy--обучение-tiny-52m)
    - [**`train_small.py`** — обучение Small (8.3M)](#train_smallpy--обучение-small-83m)
    - [**`train_medium.py`** — обучение Medium (18.3M)](#train_mediumpy--обучение-medium-183m)
    - [**`train_finetune_lora.py`** — LoRA дообучение](#train_finetune_lorapy--lora-дообучение)
  - [📦 Подготовка данных](#-подготовка-данных)
    - [**`prepare_train_dataset.py`** — токенизация корпуса](#prepare_train_datasetpy--токенизация-корпуса)
    - [**`prepare_finetune_dataset.py`** — датасет для LoRA](#prepare_finetune_datasetpy--датасет-для-lora)
  - [🌐 Веб-сервер](#-веб-сервер)
    - [**`final_server.py`** — веб-интерфейс](#final_serverpy--веб-интерфейс)
  - [🔤 Токенизатор русских инструкций](#-токенизатор-русских-инструкций)
    - [**`description_tokenizator/extract_descriptions.py`** — извлечение описаний](#description_tokenizatorextract_descriptionspy--извлечение-описаний)
    - [**`description_tokenizator/train_rus_tokenizer.py`** — обучение русского BPE](#description_tokenizatortrain_rus_tokenizerpy--обучение-русского-bpe)
    - [**`description_tokenizator/convert_rus_tokenizer_to_cpp.py`** — конвертация в C++ формат](#description_tokenizatorconvert_rus_tokenizer_to_cpppy--конвертация-в-c-формат)
  - [📊 Сводная таблица скриптов](#-сводная-таблица-скриптов)
  - [🚀 Типичные сценарии использования](#-типичные-сценарии-использования)
    - [**Полный цикл обучения с нуля**](#полный-цикл-обучения-с-нуля)
    - [**LoRA дообучение**](#lora-дообучение)
    - [**Запуск веб-сервера**](#запуск-веб-сервера)
  - [📁 Структура каталога](#-структура-каталога)

## 🎯 Обучение моделей

### **`train_tiny.py`** — обучение Tiny (5.2M)

**Назначение:** Обучение компактной модели Tiny с 5.2M параметров.

**Параметры модели:**
| Параметр | Значение |
|----------|----------|
| `d_model` | 192 |
| `nhead` | 3 |
| `num_layers` | 3 |
| `max_len` | 768 |
| `batch_size` | 12 |
| `learning_rate` | 3.5e-4 |
| `epochs` | 12 |

**Запуск:**
```bash
python scripts/train_tiny.py
```

### **`train_small.py`** — обучение Small (8.3M)

**Назначение:** Обучение модели Small с 8.3M параметров — баланс между качеством и скоростью.

**Параметры модели:**
| Параметр | Значение |
|----------|----------|
| `d_model` | 256 |
| `nhead` | 4 |
| `num_layers` | 4 |
| `max_len` | 512 |
| `batch_size` | 12 |
| `learning_rate` | 2.5e-4 |
| `epochs` | 20 |

**Запуск:**
```bash
python scripts/train_small.py
```

### **`train_medium.py`** — обучение Medium (18.3M)

**Назначение:** Обучение основной модели Medium с 18.3M параметров — лучшее качество.

**Параметры модели:**
| Параметр | Значение |
|----------|----------|
| `d_model` | 384 |
| `nhead` | 6 |
| `num_layers` | 6 |
| `max_len` | 512 |
| `batch_size` | 4 |
| `learning_rate` | 3e-4 |
| `epochs` | 20 |

**Запуск:**
```bash
python scripts/train_medium.py
```

**Результат:** Лучшая модель — cpp-code-epoch=19-val_loss=0.87.ckpt

### **`train_finetune_lora.py`** — LoRA дообучение

**Назначение:** Дообучение модели Medium на русских инструкциях с помощью LoRA (Low-Rank Adaptation).

**Особенности:**
- Использует два токенизатора (русский + C++)
- Расширяет словарь до 14000 токенов
- Заменяет 37 Linear слоёв на LoRA
- Добавляет 1.79M обучаемых параметров (7.7%)

**Параметры LoRA:**
| Параметр | Значение |
|----------|----------|
| `r` | 32 |
| `alpha` | 64 |
| `dropout` | 0.1 |
| `batch_size` | 4 |
| `learning_rate` | 2e-4 |
| `epochs` | 5 |

**Запуск:**
```bash
# Обучение с нуля
python scripts/train_finetune_lora.py

# Продолжить с эпохи 2
python scripts/train_finetune_lora.py --resume 2

# Автоматическое продолжение с последней эпохи
python scripts/train_finetune_lora.py --auto-resume
```

**Результат:** Лучшая LoRA — lora_weights_epoch_5.pt (val_loss=0.7763)

## 📦 Подготовка данных

### **`prepare_train_dataset.py`** — токенизация корпуса

**Назначение:** Токенизация текстового корпуса C++ кода с помощью C++ BPE токенизатора и сохранение в формате PyTorch.

**Входные данные:**
- data/corpus/train_code.txt (8000 примеров)
- data/corpus/val_code.txt (800 примеров)
- data/corpus/test_code.txt (500 примеров)

**Выходные данные:**
- data/tokenized/train_tokens.pt
- data/tokenized/val_tokens.pt
- data/tokenized/test_tokens.pt

**Запуск:**
```bash
python scripts/prepare_train_dataset.py
```

**Примечание:** Данные НЕ обрезаются — паддинг выполняется динамически в DataLoader.

### **`prepare_finetune_dataset.py`** — датасет для LoRA

**Назначение:** Подготовка датасета для LoRA дообучения из исходного CSV файла.

**Особенности:**
- Парсинг CSV с ручным поиском кавычек
- Очистка экранированных символов (`\\"` -> `"`, `\\n` -> `\n`)
- Аугментация описаний синонимами (70% примеров, 2 вариации)
- Сохранение в формате JSONL

**Входные данные:** `data/raw/2_cpp_code_generation_dataset.csv`

**Выходные данные:** `data/instruction_train.jsonl` (~20540 примеров)

**Запуск:**
```bash
python scripts/prepare_finetune_dataset.py
```

## 🌐 Веб-сервер

### **`final_server.py`** — веб-интерфейс

**Назначение:** Демонстрация генерации C++ кода.

**Режимы работы:**
| Вкладка | Модель | Назначение |
|---------|--------|------------|
| Продолжение кода | Medium | Дополнение незаконченного C++ кода |
| Генерация по инструкции | Medium + LoRA | Генерация кода по русскому описанию |

**Запуск:**
```bash
python scripts/final_server.py
# Открыть в браузере: http://localhost:8080
```

**Управление:**
- Ctrl+Enter — быстрая генерация
- Выбор температуры (0.5-1.2)
- Настройка максимального количества токенов

## 🔤 Токенизатор русских инструкций

### **`description_tokenizator/extract_descriptions.py`** — извлечение описаний

**Назначение:** Обучение BPE токенизатора специально для русских инструкций.

**Параметры:**
| Параметр | Значение |
|----------|----------|
| VOCAB_SIZE | 4000 |
| Входной файл | `data/rus_descriptions.txt` |
| Выходная папка | `tokenizers/rus_bpe_4000/` |

**Результат:**
- `tokenizers/rus_bpe_4000/vocab.json` — словарь (4000 токенов)
- `tokenizers/rus_bpe_4000/merges.txt` — правила слияния

**Запуск:**
```bash
python scripts/description_tokenizator/extract_descriptions.py
```

### **`description_tokenizator/convert_rus_tokenizer_to_cpp.py`** — конвертация в C++ формат

**Назначение:** Конвертация русского токенизатора из Python формата в формат, совместимый с C++ FastBPETokenizer.

**Входные данные:**
- `tokenizers/rus_bpe_4000/vocab.json`
- `tokenizers/rus_bpe_4000/merges.txt`

**Выходные данные:**
- `tokenizers/rus_bpe_4000_cpp/cpp_vocab.json`
- `tokenizers/rus_bpe_4000_cpp/cpp_merges.txt`

**Запуск:**
```bash
python scripts/description_tokenizator/convert_rus_tokenizer_to_cpp.py
```

## 📊 Сводная таблица скриптов

| Скрипт | Назначение | Вход | Выход | Время |
|--------|------------|------|-------|-------|
| `train_tiny.py` | Обучение Tiny (5.2M) | `data/tokenized/*.pt` | `checkpoints/tiny/` | 2-4 ч |
| `train_small.py` | Обучение Small (8.3M) | `data/tokenized/*.pt` | `checkpoints/small/` | 3-6 ч |
| `train_medium.py` | Обучение Medium (18.3M) | `data/tokenized/*.pt` | `checkpoints/medium/` | 8-10 ч |
| `train_finetune_lora.py` | LoRA дообучение | `instruction_train.jsonl` | `checkpoints/lora_medium/` | 3-6 ч |
| `prepare_train_dataset.py` | Токенизация корпуса | `corpus/*.txt` | `tokenized/*.pt` | - |
| `prepare_finetune_dataset.py` | Датасет для LoRA | `raw/*.csv` | `instruction_train.jsonl` | - |
| `final_server.py` | Веб-сервер | Модели | HTTP :8080 | - |
| `extract_descriptions.py` | Извлечение описаний | `raw/*.csv` | `rus_descriptions.txt` | - |
| `train_rus_tokenizer.py` | Обучение русского BPE | `rus_descriptions.txt` | `tokenizers/rus_bpe_4000/` | - |
| `convert_rus_tokenizer_to_cpp.py` | Конвертация в C++ | `rus_bpe_4000/` | `rus_bpe_4000_cpp/` | - |

## 🚀 Типичные сценарии использования

### Полный цикл обучения с нуля

```bash
# 1. Подготовка данных
python scripts/prepare_train_dataset.py
python scripts/description_tokenizator/extract_descriptions.py
python scripts/description_tokenizator/train_rus_tokenizer.py
python scripts/prepare_finetune_dataset.py

# 2. Обучение базовой модели
python scripts/train_medium.py

# 3. LoRA дообучение
python scripts/train_finetune_lora.py
```

### LoRA дообучение

```bash
# Подготовка датасета (если ещё не сделано)
python scripts/prepare_finetune_dataset.py

# Обучение
python scripts/train_finetune_lora.py

# Продолжение с последней эпохи
python scripts/train_finetune_lora.py --auto-resume
```

### Запуск веб-сервера

```bash
python scripts/final_server.py
# Открыть http://localhost:8080
```

## 📁 Структура каталога

```text
scripts/
├── description_tokenizator/               # Токенизатор поля dataset - description (русские инструкции)
│   ├── convert_rus_tokenizer_to_cpp.py    # Конвертация в C++
│   ├── extract_descriptions.py            # Извлечение описаний из dataset
│   └── train_rus_tokenizer.py             # Обучение русского BPE
├── final_server.py                        # Веб-сервер для демонстрации работы моделей
├── prepare_finetune_dataset.py            # Подготовка dataset для обучения finetune LoRA
├── prepare_train_dataset.py               # Подготовка dataset для обучения базовой модели
├── README.md                              # Этот файл
├── train_finetune_lora.py                 # Дообучение модели LoRA
├── train_medium.py                        # Обучение модели medium размером 18,3M
├── train_small.py                         # Обучение модели small размером 8,3M
└── train_tiny.py                          # Обучение модели tiny размером 5,2M
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026