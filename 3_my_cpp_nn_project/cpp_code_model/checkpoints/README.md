# 💾 Сохраненные модели (Checkpoints)

Эта папка содержит обученные модели и LoRA веса для генерации C++ кода. Все чекпоинты сохранены в формате PyTorch Lightning (`.ckpt`) и PyTorch (`.pt`).

## 📋 Содержание

- [💾 Сохраненные модели (Checkpoints)](#-сохраненные-модели-checkpoints)
  - [📋 Содержание](#-содержание)
  - [📊 Обзор моделей](#-обзор-моделей)
  - [🎯 Medium — основная модель](#-medium--основная-модель)
    - [**Базовая модель**](#базовая-модель)
    - [**LoRA веса**](#lora-веса)
  - [🔬 Small — экспериментальная](#-small--экспериментальная)
  - [🔬 Tiny — экспериментальная](#-tiny--экспериментальная)
  - [📁 Структура каталога](#-структура-каталога)

## 📊 Обзор моделей

| Модель | Параметры | Val Loss | Эпох | Размер | Статус |
|--------|-----------|----------|------|--------|--------|
| **Medium** | 18.3M | **0.87** | 19 | ~222.5 МБ | ✅ Основная |
| Medium + LoRA | 18.3M + 1.8M | **0.78** | 5 | ~222.5 МБ + 7.2 МБ | ✅ Инструкции |
| Small | 8.3M | 1.47 | 19 | ~101.1 МБ | 🔬 Эксперимент |
| Tiny | 5.2M | 1.76 | 11 | ~64.7 МБ | 🔬 Эксперимент |

## 🎯 Medium — основная модель

### **Базовая модель**

**Путь:** `medium/`

| Файл | Описание | Val Loss |
|------|----------|----------|
| `cpp-code-epoch=19-val_loss=0.87.ckpt` | Лучшая эпоха (19) | 0.87 |
| `last.ckpt` | Последнее состояние | - |

**Параметры:**
- `vocab_size`: 10001
- `d_model`: 384
- `nhead`: 6
- `num_layers`: 6
- `max_len`: 512

**Загрузка:**
```python
from model.architecture import CppCodeModel

model = CppCodeModel.load_from_checkpoint(
    'checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt'
)
```

### **LoRA веса**

**Путь:** `lora_medium/`

| Файл | Описание | Val Loss |
|------|----------|----------|
| `lora_weights_best.pt` | Симлинк на лучшую | 0.7763 |
| `lora_weights_epoch_5.pt` | Лучшая эпоха - Эпоха 5 | 0.7763 |

**Параметры LoRA:**
- `r`: 32
- `alpha`: 64
- `dropout`: 0.1
- `Обучаемых параметров:`: 1.79M (7.7%)

**Загрузка:**
```python
import torch
from scripts.train_finetune_lora import replace_with_lora, load_lora_weights

# Загружаем базовую модель
model = CppCodeModel.load_from_checkpoint('checkpoints/medium/...ckpt')

# Добавляем LoRA
replace_with_lora(model)
load_lora_weights(model, 'checkpoints/lora_medium/lora_weights_epoch_5.pt')
```

### **🔬 Small — экспериментальная**

**Путь:** `small/`

| Файл | Описание | Val Loss |
|------|----------|----------|
| `cpp-code-small-epoch=19-val_loss=1.47.ckpt` | Лучшая эпоха (19) | 1.47 |
| `last.ckpt` | Последнее состояние | - |

**Параметры:**
- `vocab_size`: 10001
- `d_model`: 256
- `nhead`: 4
- `num_layers`: 4
- `max_len`: 512

### **🔬 Tiny — экспериментальная**

**Путь:** `tiny/`

| Файл | Описание | Val Loss |
|------|----------|----------|
| `cpp-code-tiny-epoch=11-val_loss=1.76.ckpt` | Лучшая эпоха (11) | 1.76 |
| `last.ckpt` | Последнее состояние | - |

**Параметры:**
- `vocab_size`: 10001
- `d_model`: 192
- `nhead`: 3
- `num_layers`: 3
- `max_len`: 768

### **📁 Структура каталога**
```text
checkpoints/        # Cохраненные модели
├── lora_medium/    # LoRA модели для medium
    ├── lora_weights_best.pt
│   ├── ...
│   └── lora_weights_epoch_5.pt
├── medium/         # Базовая модель размером 18,3M
│   ├── ...
│   ├── cpp-code-epoch=19-val_loss=0.87.ckpt
│   └── last.ckpt
├── small/          # Базовая модель размером 8,3M
│   ├── ...
│   ├── cpp-code-small-epoch=19-val_loss=1.47.ckpt
│   └── last.ckpt
├── tiny/           # Базовая модель размером 5,2M
│   ├── ...
│   ├── cpp-code-tiny-epoch=11-val_loss=1.76.ckpt
│   └── last.ckpt
└── README.md       # Этот файл
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026