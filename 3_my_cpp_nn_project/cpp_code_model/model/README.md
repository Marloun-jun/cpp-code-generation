# 🏗️ Архитектура модели Transformer для генерации C++ кода

Эта папка содержит полную реализацию модели Transformer для генерации C++ кода. Модель построена на PyTorch Lightning и включает все необходимые компоненты: внимание, позиционное кодирование, блоки трансформера и модуль загрузки данных.

## 📋 Содержание

- [🏗️ Архитектура модели Transformer для генерации C++ кода](#️-архитектура-модели-transformer-для-генерации-c-кода)
  - [📋 Содержание](#-содержание)
  - [🏛️ Архитектурные компоненты](#️-архитектурные-компоненты)
    - [**`attention.py`** — многоголовое внимание](#attentionpy--многоголовое-внимание)
    - [**`feedforward.py`** — сеть прямого распространения](#feedforwardpy--сеть-прямого-распространения)
    - [**`positional.py`** — позиционное кодирование](#positionalpy--позиционное-кодирование)
    - [**`transformer.py`** — блок трансформера](#transformerpy--блок-трансформера)
    - [**`model.py`** — полная модель](#modelpy--полная-модель)
  - [📦 Загрузка данных](#-загрузка-данных)
    - [**`data_module.py`** — PyTorch Lightning DataModule](#data_modulepy--pytorch-lightning-datamodule)
  - [📊 Параметры моделей](#-параметры-моделей)
  - [📁 Структура каталога](#-структура-каталога)

## 🏛️ Архитектурные компоненты

### **`attention.py`** — многоголовое внимание

**Назначение:** Реализация Multi-Head Attention с causal masking для авторегрессионной генерации.

**Особенности:**
- Scaled dot-product attention
- Causal mask (запрет "подглядывания" в будущее)
- Поддержка внешней маски для паддинга
- Кэширование causal mask для ускорения

**Класс:** `MultiHeadAttention`

**Параметры:**
| Параметр | Описание | Пример |
|----------|----------|--------|
| `d_model` | Размерность эмбеддингов | 384 |
| `nhead` | Количество голов | 6 |
| `dropout` | Вероятность dropout | 0.1 |

**Пример:**
```python
from model.architecture import MultiHeadAttention

attention = MultiHeadAttention(d_model=384, nhead=6)
output = attention(x, x, x, mask=attention_mask)
```

### **`feedforward.py`** — сеть прямого распространения

**Назначение:** Двухслойная полносвязная сеть с GELU активацией.

**Особенности:**
- Расширение размерности (d_model → 4×d_model)
- GELU активация (лучше ReLU для Transformer)
- Dropout после каждого слоя

**Класс:** `FeedForward`

**Параметры:**
| Параметр | Описание | Пример |
|----------|----------|--------|
| `d_model` | Входная/выходная размерность | - |
| `d_ff` | Размерность скрытого слоя | d_model × 4 |
| `dropout` | Вероятность dropout | 0.1 |

**Пример:**
```python
from model.architecture import FeedForward

ffn = FeedForward(d_model=384, d_ff=1536)
output = ffn(x)    # x.shape = (batch, seq_len, 384)
```

### **`positional.py`** — позиционное кодирование

**Назначение:** Синусоидальное позиционное кодирование (Positional Encoding) из статьи "Attention Is All You Need".

**Особенности:**
- Детерминистическое (не обучаемое)
- Динамическое расширение для seq_len > max_len
- Кэширование вычисленных значений

**Класс:** `PositionalEncoding`

**Параметры:**
| Параметр | Описание | Пример |
|----------|----------|--------|
| `d_model` | Размерность эмбеддингов | - |
| `max_len` | Максимальная длина | 5000 |
| `dropout` | Вероятность dropout | 0.1 |

**Формулы:**
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

**Пример:**
```python
from model.architecture import PositionalEncoding

pos_enc = PositionalEncoding(d_model=384, max_len=512)
x = pos_enc(token_embeddings)
```

### **`transformer.py`** — блок трансформера

**Назначение:** Один слой (блок) трансформера с Pre-LN архитектурой.

**Особенности:**
- Multi-Head Self-Attention
- Feed-Forward Network
- Residual connections
- Pre-LN (LayerNorm перед сложением)

**Класс:** `TransformerBlock`

**Pre-LN архитектура (современная):**
1. x = x + Dropout(Attention(LayerNorm(x)))
2. x = x + Dropout(FFN(LayerNorm(x)))

**Отличие от Post-LN (оригинальный Transformer):**
- Post-LN: x = LayerNorm(x + Dropout(Attention(x)))
- Pre-LN:  x = x + Dropout(Attention(LayerNorm(x)))

Pre-LN обеспечивает лучшую стабильность градиентов для глубоких сетей.

**Параметры:**
| Параметр | Описание | Пример |
|----------|----------|--------|
| `d_model` | Размерность эмбеддингов | 384 |
| `nhead` | Количество голов | 6 |
| `dropout` | Вероятность dropout | 0.1 |

**Пример:**
```python
from model.architecture import TransformerBlock

block = TransformerBlock(d_model=384, nhead=6)
output = block(x, mask=attention_mask)
```

### **`model.py`** — полная модель

**Назначение:** Полная модель Transformer для генерации C++ кода на PyTorch Lightning.

**Особенности:**
- Token Embedding + Positional Encoding
- N блоков Transformer
- LM Head для предсказания токенов
- Встроенная генерация с temperature и top-k/p
- Автоматический scheduler с warmup

**Класс:** `CppCodeModel`

**Параметры:**
| Параметр | Описание | Пример |
|----------|----------|--------|
| `vocab_size` | Размер словаря | 10001 |
| `d_model` | Размерность эмбеддингов | 384 |
| `nhead` | Количество голов | 6 |
| `num_layers` | Количество слоев | 6 |
| `max_len` | Максимальная длина | 512 |
| `dropout` | Вероятность dropout | 0.1 |
| `learning_rate` | Скорость обучения | 3e-4 |
| `weight_decay` | L2 регуляризация | 0.01 |
| `warmup_steps` | Шагов warmup | 800 |

**Пример:**
```python
from model.architecture import CppCodeModel

model = CppCodeModel(
    vocab_size=10001,
    d_model=384,
    nhead=6,
    num_layers=6
)

# Обучение
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)

# Генерация
code = model.generate(tokenizer, "int main() {", temperature=0.8)
```

## 📦 Загрузка данных

### **`data_module.py`** — PyTorch Lightning DataModule

**Назначение:** Управление загрузкой токенизированных данных для обучения модели.

**Особенности:**
- Автоматическая загрузка train/val/test .pt файлов
- Динамический паддинг внутри батча
- Создание attention mask
- Статистика по длинам последовательностей

**Классы:** `CppCodeDataset` и `CppCodeDataModule`

**Параметры:**
| Параметр | Описание | Пример |
|----------|----------|--------|
| `data_dir` | Директория с .pt файлами | - |
| `batch_size` | Размер батча | 8 |
| `num_workers` | Количество воркеров | 4 |
| `max_len` | Максимальная длина | 1024 |

**Структура данных:**
```text
data/
└── corpus
    ├── train_tokens.pt    # Тренировочные данные
    ├── val_tokens.pt      # Валидационные данные
    └── test_tokens.pt     # Тестовые данные
```

**Пример:**
```python
from model.data_module import CppCodeDataModule

dm = CppCodeDataModule(
    data_dir='data/tokenized',
    batch_size=8,
    max_len=512
)
dm.setup()

# Обучение
trainer.fit(model, dm)
```

## 📊 Параметры моделей

| Модель | d_model | nhead | num_layers | Параметров | Размер | Batch | LR | Warmup |
|--------|---------|-------|------------|------------|--------|-------|----|--------|
| **Tiny** | 192 | 3 | 3 | 5.2M | 21 MB | 12 | 3.5e-4 | 800 |
| **Small** | 256 | 4 | 4 | 8.3M | 34 MB | 12 | 2.5e-4 | 1400 |
| **Medium** | 384 | 6 | 6 | 18.3M | 73 MB | 4 | 3e-4 | 4000 |

**Общие параметры:**
| Параметр | Значение |
|----------|----------|
| `vocab_size` | 10001 |
| `max_len` | 512 (Tiny: 768) |
| `dropout` | 0.15 |
| `weight_decay` | 0.01 |
| `gradient_clip` | 1.0 |
| `epochs` | 20 (Tiny: 12) |
| `pad_token_id` | 0 |
| `bos_token_id` | 2 |
| `eos_token_id` | 3 |

## 📁 Структура каталога

```text
model/
├── architecture/
│   ├── __init__.py       # Экспорт компонентов
│   ├── attention.py      # MultiHeadAttention
│   ├── feedforward.py    # FeedForward с GELU
│   ├── model.py          # CppCodeModel (Lightning)
│   ├── positional.py     # PositionalEncoding (sin/cos)
│   └── transformer.py    # TransformerBlock (Pre-LN)
├── data_module.py        # CppCodeDataModule
└── README.md             # Этот файл
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026