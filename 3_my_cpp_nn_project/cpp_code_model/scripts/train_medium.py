#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# train_medium.py - Обучение CppCodeModel-Medium (23M параметров)
# ======================================================================
#
# @file train_medium.py
# @brief Скрипт для обучения модели C++ кода среднего размера с 23M параметров
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот скрипт обучает модель CppCodeModel-Medium с 23M параметров,
#          оптимизированную для GPU с 4GB VRAM (например, GTX 1650), 
#          для датасета из 9.3K примеров.
#          Модель обеспечивает наилучшее качество.
#
#          **Основные возможности:**
#
#          1. **Улучшенная конфигурация Medium**
#             - d_model=384
#             - nhead=6 (384/64 = 6 голов)
#             - num_layers=6
#             - max_len=512
#
#          2. **Регуляризация и стабилизация**
#             - dropout=0.15
#             - weight_decay=0.01 (L2 регуляризация)
#             - gradient_clip=1.0
#             - warmup_steps=1000
#
#          3. **Оптимизация памяти**
#             - mixed precision (fp16) для GPU
#             - Еще уменьшенна длина последовательности (512)
#             - batch_size (4)
#
#          4. **Мониторинг и сохранение**
#             - Сохранение лучших чекпоинтов (top-3)
#             - Логирование loss и perplexity
#             - Прогноз времени обучения
#
# @usage
#     python scripts/train_medium.py
#
# @example
#     # Стандартный запуск (автоматически продолжит при наличии чекпоинта)
#     python train_medium.py
#
#     # Принудительное новое обучение (удалите папку checkpoints/medium)
#     rm -rf checkpoints/medium && python train_medium.py
#
# ======================================================================

import sys
import json
import torch
import pytorch_lightning as pl

from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Добавляем путь к модели
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture.model import CppCodeModel
from model.data_module import CppCodeDataModule

# ======================================================================
# КОНФИГУРАЦИЯ МОДЕЛИ MEDIUM (23M)
# ======================================================================

config = {
    "name": "CppCodeModel-Medium",
    "description": "18.3M параметров",
    "vocab_size": 10001,
    
    # Архитектура
    "d_model": 384,
    "nhead": 6,
    "num_layers": 6,
    
    # Длина и память (оптимизировано для GTX 1650)
    "max_len": 512,
    "batch_size": 4,
    
    # Регуляризация (борьба с переобучением на 9.3K примерах)
    "dropout": 0.15,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    
    # Обучение
    "epochs": 20,
    "learning_rate": 3e-4,
    "warmup_steps": 4000,
    
    # Специальные токены (согласовано с C++ токенизатором)
    "pad_token_id": 0,
    "bos_token_id": 2,
    "eos_token_id": 3,
}
# Оптимальные параметры для 9.3K примеров

# ======================================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "tokenized"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "medium"
CONFIG_DIR = PROJECT_ROOT / "configs"

# Создаем папки
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# ПРОВЕРКА НАЛИЧИЯ ДАННЫХ
# ======================================================================

if not DATA_DIR.exists():
    print(f"Ошибка: Директория с данными не найдена: {DATA_DIR}!")
    print("Запустите сначала скрипт подготовки данных (prepare_train_dataset.py)")
    sys.exit(1)

# Проверяем наличие файлов
required_files = ['train_tokens.pt', 'val_tokens.pt', 'test_tokens.pt']
missing_files = [f for f in required_files if not (DATA_DIR / f).exists()]
if missing_files:
    print(f"Ошибка: Отсутствуют файлы: {', '.join(missing_files)}!")
    print(f"Полный путь: {DATA_DIR}")
    sys.exit(1)

# ======================================================================
# СОХРАНЕНИЕ КОНФИГУРАЦИИ
# ======================================================================

# Создаем папки
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Сохраняем файл конфигураций
with open(CONFIG_DIR / "medium.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# ======================================================================
# ВЫВОД ИНФОРМАЦИИ О КОНФИГУРАЦИИ
# ======================================================================

print("=" * 60)
print("ОБУЧЕНИЕ CppCodeModel-Medium (23M)")
print("=" * 60)

# Проверка GPU
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU найден: {device_name}")
    print(f"VRAM:       {vram:.1f} ГБ")
    
    if vram < 4:
        print(f"ВНИМАНИЕ: VRAM ({vram:.1f}ГБ) меньше рекомендуемого 4ГБ!")
        print("Может потребоваться уменьшить batch_size или max_len")
else:
    print("GPU не найден, обучение на CPU!")

print("\nПАРАМЕТРЫ МОДЕЛИ:")
print(f"- d_model:       {config['d_model']}")
print(f"- nhead:         {config['nhead']}")
print(f"- num_layers:    {config['num_layers']}")
print(f"- Параметров:    ~23M")
print(f"- Batch size:    {config['batch_size']}")
print(f"- Max length:    {config['max_len']}")
print(f"- Dropout:       {config['dropout']}")
print(f"- Learning rate: {config['learning_rate']}")
print(f"- Weight decay:  {config['weight_decay']}")
print(f"- Warmup steps:  {config['warmup_steps']}")
print(f"- Эпох:          {config['epochs']}")
print(f"- PAD ID:        {config['pad_token_id']}")
print(f"- BOS ID:        {config['bos_token_id']}")
print(f"- EOS ID:        {config['eos_token_id']}")

print("\nПРОГНОЗ ВРЕМЕНИ:")
if torch.cuda.is_available():
    print("GPU (GTX 1650): 8-10 часов")
else:
    print("CPU: 30-40 часов")

print("\nПУТИ:")
print(f"- Данные:    {DATA_DIR}")
print(f"- Чекпоинты: {CHECKPOINT_DIR}")
print(f"- Конфиг:    {CONFIG_DIR / 'medium.json'}")
print("=" * 60)

# ======================================================================
# ЗАГРУЗКА ДАННЫХ
# ======================================================================

print("\nЗагрузка данных...")
data_module = CppCodeDataModule(
    data_dir=DATA_DIR,
    batch_size=config['batch_size'],
    num_workers=4,
    max_len=config['max_len']
)
data_module.setup()
print(f"Train: {len(data_module.train_dataset)} | Val: {len(data_module.val_dataset)} | Test: {len(data_module.test_dataset)}")

# ======================================================================
# СОЗДАНИЕ МОДЕЛИ
# ======================================================================

print("\n🔧 Создание модели...")
model = CppCodeModel(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    max_len=config['max_len'],
    dropout=config['dropout'],
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    warmup_steps=config['warmup_steps']
)

# Подсчет параметров
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nМодель создана!")
print(f"Всего параметров: {total_params:,} ({total_params/1e6:.2f}M)")

# ======================================================================
# НАСТРОЙКА CALLBACKS
# ======================================================================

callbacks = [
    pl.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='cpp-code-medium-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    ),
    pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,               
        mode='min'
    )
]

# ======================================================================
# НАСТРОЙКА TRAINER
# ======================================================================

trainer = pl.Trainer(
    max_epochs=config['epochs'],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    precision='16-mixed' if torch.cuda.is_available() else '32-true',
    gradient_clip_val=config['gradient_clip'],
    log_every_n_steps=10,
    enable_progress_bar=True,
    callbacks=callbacks
)

print("\nНАЧАЛО ОБУЧЕНИЯ")
print(f"- Эпох:          {config['epochs']}")
print(f"- Learning rate: {config['learning_rate']}")
print(f"- Warmup steps:  {config['warmup_steps']}")
print(f"- Batch size:    {config['batch_size']}")
print(f"- Max length:    {config['max_len']}")
print("-" * 60)

# ======================================================================
# ОБУЧЕНИЕ С ОБРАБОТКОЙ ОШИБОК
# ======================================================================

try:
    trainer.fit(model, data_module)
except KeyboardInterrupt:
    print("\nОбучение прервано пользователем!")
    print(f"Чекпоинты сохранены в: {CHECKPOINT_DIR}")
    sys.exit(0)
except torch.cuda.OutOfMemoryError:
    print("\nCUDA OUT OF MEMORY!")
    print("Попробуйте уменьшить batch_size или max_len")
    sys.exit(1)

# ======================================================================
# ЗАВЕРШЕНИЕ ОБУЧЕНИЯ
# ======================================================================

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print(f"Лучшие чекпоинты сохранены в: {CHECKPOINT_DIR}")
print("=" * 60)