#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# train_finetune_lora.py - LoRA Fine-Tuning (эксперимент)
# ======================================================================
#
# @file train_finetune_lora.py
# @brief LoRA дообучение модели Medium на инструкциях с двумя токенизаторами (эксперимент)
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль выполняет LoRA (Low-Rank Adaptation) дообучение
#          предварительно обученной модели Medium на датасете инструкций.
#          Использует два токенизатора:
#          - Русский BPE (Python) для кодирования инструкций
#          - C++ BPE для кодирования кода
#
# @usage
#     python scripts/train_finetune_lora.py
#     python scripts/train_finetune_lora.py --resume 2  # продолжить с эпохи 2
#
# ======================================================================

import torch
import json
import sys
import random
import math
import argparse

from pathlib import Path
from tqdm import tqdm

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture.model import CppCodeModel

# Путь к Python BPE токенизатору (русский)
bpe_python_dir = Path(__file__).parent.parent.parent / "bpe_tokenizer_cpu" / "bpe_python"
sys.path.insert(0, str(bpe_python_dir))
from tokenizer import BPETokenizer

# Путь к C++ токенизатору (для кода)
bpe_cpp_dir = Path(__file__).parent.parent.parent / "bpe_tokenizer_cpu" / "bpe_cpp"
bpe_build_dir = bpe_cpp_dir / "build"
sys.path.insert(0, str(bpe_build_dir))
import bpe_tokenizer_cpp as bpe

# ======================================================================
# ПУТИ
# ======================================================================

PROJECT_ROOT = Path(__file__).parent.parent

# Используем лучшую модель Medium
MODEL_PATH = PROJECT_ROOT / "checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt"
DATASET_PATH = PROJECT_ROOT / "data" / "instruction_train.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "lora_medium"

# Русский токенизатор (Python)
RUS_TOKENIZER_DIR = PROJECT_ROOT / "tokenizers" / "rus_bpe_4000"
RUS_VOCAB = RUS_TOKENIZER_DIR / "vocab.json"
RUS_MERGES = RUS_TOKENIZER_DIR / "merges.txt"

# C++ токенизатор
CPP_VOCAB = PROJECT_ROOT.parent / "bpe_tokenizer_cpu" / "bpe_cpp" / "models" / "bpe_10000" / "cpp_vocab.json"
CPP_MERGES = PROJECT_ROOT.parent / "bpe_tokenizer_cpu" / "bpe_cpp" / "models" / "bpe_10000" / "cpp_merges.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================================
# ПАРАМЕТРЫ LoRA (ОПТИМИЗИРОВАНЫ НА ОСНОВЕ ЭКСПЕРИМЕНТОВ)
# ======================================================================

# LoRA параметры (проверено: r=32 даёт лучшее качество)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# Целевые модули — используем ВСЕ Linear слои (проверено: лучше понимает инструкции)
TARGET_MODULES = None

# Обучение (проверено: LR=2e-4, эпохи=6 дают лучший результат)
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS
EPOCHS = 5
MAX_LEN = 768
WARMUP_STEPS = 800
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
NUM_WORKERS = 4
VAL_SPLIT = 0.05

# Количество сохраняемых чекпоинтов (сохраняем все)
KEEP_LAST_N = 6

# Специальные токены
BOS_TOKEN = 2
EOS_TOKEN = 3
SEP_TOKEN = 10000

# Глобальная переменная для продолжения
RESUME_EPOCH = 0
RESUME_LOADED = False

# ======================================================================
# КЛАСС LoRALinear
# ======================================================================

class LoRALinear(torch.nn.Module):
    """
    LoRA слой для линейной проекции.
    
    Args:
        original_linear (nn.Linear): Оригинальный линейный слой
        r (int):                     Ранг LoRA
        alpha (int):                 Коэффициент масштабирования
        dropout (float):             Вероятность dropout
    """
    
    def __init__(self, original_linear, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        self.r = r
        self.alpha = alpha
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        self.lora_A = torch.nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, r))
        
        # Инициализация
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / r
    
    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        if self.dropout:
            lora_out = self.dropout(lora_out)
        return result + self.scaling * lora_out


# ======================================================================
# ФУНКЦИЯ ЗАМЕНЫ СЛОЁВ НА LoRA
# ======================================================================

def replace_with_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT, target_modules=None):
    """
    Рекурсивная замена Linear слоёв на LoRALinear.
    
    Args:
        model (nn.Module):     Модель
        r (int):               Ранг LoRA
        alpha (int):           Коэффициент масштабирования
        dropout (float):       Вероятность dropout
        target_modules (list): Список имён модулей для замены (None = все Linear)
        
    Returns:
        int: Количество заменённых слоёв
    """
    replaced_count = 0
    
    for name, module in model.named_children():
        # Проверяем, нужно ли заменить этот модуль
        should_replace = False
        
        if target_modules is None:
            # Заменяем все Linear слои
            should_replace = isinstance(module, torch.nn.Linear)
        else:
            module_name = name.lower()
            for target in target_modules:
                if target.lower() in module_name:
                    should_replace = True
                    break
        
        if should_replace and isinstance(module, torch.nn.Linear):
            print(f"Замена: {name} (Linear) -> LoRALinear")
            lora_layer = LoRALinear(module, r, alpha, dropout)
            setattr(model, name, lora_layer)
            replaced_count += 1
        else:
            # Рекурсивно обрабатываем дочерние модули
            replaced_count += replace_with_lora(module, r, alpha, dropout, target_modules)
    
    return replaced_count


# ======================================================================
# ФУНКЦИИ ДЛЯ УПРАВЛЕНИЯ ЧЕКПОИНТАМИ
# ======================================================================

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=KEEP_LAST_N):
    """
    Удаляет старые чекпоинты, оставляя только последние n.
    
    Args:
        checkpoint_dir (Path): Директория с чекпоинтами
        keep_last_n (int):     Количество последних чекпоинтов для сохранения
    """
    checkpoint_files = list(checkpoint_dir.glob("lora_weights_epoch_*.pt"))
    
    if len(checkpoint_files) <= keep_last_n:
        return
    
    def get_epoch_num(filename):
        import re
        match = re.search(r'epoch_(\d+)', str(filename))
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=get_epoch_num)
    to_delete = checkpoint_files[:-keep_last_n]
    for f in to_delete:
        f.unlink()
        print(f"Удалён старый чекпоинт: {f.name}")


def find_latest_checkpoint(checkpoint_dir):
    """Находит последний чекпоинт LoRA по номеру эпохи."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None, 0
    
    checkpoint_files = list(checkpoint_dir.glob("lora_weights_epoch_*.pt"))
    if not checkpoint_files:
        return None, 0
    
    def get_epoch_num(filename):
        import re
        match = re.search(r'epoch_(\d+)', str(filename))
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=get_epoch_num)
    latest_file = checkpoint_files[-1]
    latest_epoch = get_epoch_num(latest_file)
    
    return latest_file, latest_epoch


def load_lora_weights(model, checkpoint_path):
    """Загружает LoRA веса из чекпоинта."""
    if not checkpoint_path.exists():
        print(f"Чекпоинт не найден: {checkpoint_path}!")
        return False
    
    lora_state = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].copy_(value)
            loaded += 1
    
    print(f"Загружено {loaded} LoRA весов из {checkpoint_path.name}")
    return loaded > 0

# ======================================================================
# ФУНКЦИЯ COLLATE
# ======================================================================

def collate_fn(batch):
    """
    Функция коллации с динамическим паддингом.
    
    Args:
        batch: Список кортежей (input, target)
        
    Returns:
        dict: input_ids, labels, attention_mask
    """
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = []
    padded_targets = []
    attention_masks = []
    
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        if pad_len > 0:
            padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
            padded_targets.append(torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)]))
            mask = torch.cat([torch.ones(len(inp)), torch.zeros(pad_len)])
        else:
            padded_inputs.append(inp)
            padded_targets.append(tgt)
            mask = torch.ones(len(inp))
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.stack(padded_inputs),
        'labels': torch.stack(padded_targets),
        'attention_mask': torch.stack(attention_masks)
    }

# ======================================================================
# ФУНКЦИЯ КОДИРОВАНИЯ С ДВУМЯ ТОКЕНИЗАТОРАМИ
# ======================================================================

def encode_with_two_tokenizers(instruction, code, rus_tokenizer, cpp_tokenizer, max_len):
    """
    Кодирование инструкции и кода с двумя токенизаторами.
    
    Формат: [BOS] + токены_инструкции_СО_СМЕЩЕНИЕМ + [SEP] + токены_кода + [EOS]
    """
    # Получаем русские токены (0-3999)
    rus_tokens = rus_tokenizer.encode(instruction)
    
    # Смещаем русские токены в диапазон 10000-13999
    RUS_OFFSET = 10000
    shifted_rus_tokens = [t + RUS_OFFSET for t in rus_tokens]
    
    # C++ токены остаются как есть (0-9999)
    code_tokens = cpp_tokenizer.encode(code)
    
    full_tokens = [BOS_TOKEN] + shifted_rus_tokens + [SEP_TOKEN] + code_tokens + [EOS_TOKEN]
    
    if len(full_tokens) > max_len:
        full_tokens = full_tokens[:max_len]
    
    return full_tokens

# ======================================================================
# ФУНКЦИЯ ЗАГРУЗКИ ДАТАСЕТА
# ======================================================================

def load_dataset(dataset_path, rus_tokenizer, cpp_tokenizer, max_len):
    """Загрузка и токенизация датасета инструкций."""
    print(f"Загрузка датасета: {dataset_path}")
    
    texts = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    
    print(f"Загружено {len(texts)} примеров")
    
    input_ids_list = []
    skipped = 0
    
    for text in tqdm(texts, desc="Токенизация"):
        parts = text.split('\n\n', 1)
        if len(parts) == 2:
            instruction = parts[0]
            code = parts[1]
        else:
            instruction = text
            code = ""
        
        try:
            tokens = encode_with_two_tokenizers(instruction, code, rus_tokenizer, cpp_tokenizer, max_len)
            if len(tokens) < 5:
                skipped += 1
                continue
            input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
        except Exception as e:
            skipped += 1
            continue
    
    print(f"Токенизировано: {len(input_ids_list)} примеров")
    if skipped > 0:
        print(f"Пропущено: {skipped}!")
    
    return input_ids_list

# ======================================================================
# КЛАСС InstructionDataset
# ======================================================================

class InstructionDataset(torch.utils.data.Dataset):
    """Dataset для инструкций с C++ кодом."""
    
    def __init__(self, input_ids_list):
        self.input_ids = input_ids_list
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        return tokens[:-1], tokens[1:]

# ======================================================================
# ФУНКЦИЯ ВАЛИДАЦИИ
# ======================================================================

def validate(model, val_loader, device):
    """
    Валидация модели на отложенной выборке.
    
    Returns:
        float: Средняя loss на валидации
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Валидация", leave=False):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += (labels != 0).sum().item()
    
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция обучения LoRA."""
    global RESUME_EPOCH
    
    parser = argparse.ArgumentParser(description='LoRA Fine-Tuning')
    parser.add_argument('--resume', type=int, default=0, help='Номер эпохи для продолжения (0 - начать заново)')
    parser.add_argument('--auto-resume', action='store_true', help='Автоматически продолжить с последней эпохи')
    args = parser.parse_args()
    
    # Определяем эпоху для продолжения
    start_epoch = args.resume
    
    if args.auto_resume and start_epoch == 0:
        latest_checkpoint, latest_epoch = find_latest_checkpoint(OUTPUT_DIR)
        if latest_epoch > 0:
            start_epoch = latest_epoch
            print(f"\nАвтоматически определена последняя эпоха: {latest_epoch}")
        else:
            print("\nЧекпоинтов не найдено, начинаем с нуля!")
    
    print("=" * 60)
    print("LoRA FINE-TUNING (ДВА ТОКЕНИЗАТОРА СО СМЕЩЕНИЕМ)")
    print("Русский (Python BPE) + C++ (FastBPE)")
    print("Базовая модель: Medium (18.3M параметров)")
    print("=" * 60)
    print(f"- LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"- Target modules:       {'ВСЕ Linear слои' if TARGET_MODULES is None else TARGET_MODULES}")
    print(f"- Learning rate:        {LEARNING_RATE}")
    print(f"- Batch size:           {BATCH_SIZE} (accumulation: {ACCUMULATION_STEPS})")
    print(f"- Effective batch size: {EFFECTIVE_BATCH_SIZE}")
    print(f"- Epochs:               {EPOCHS}")
    print(f"- Max length:           {MAX_LEN}")
    print(f"- Сохранять последние:  {KEEP_LAST_N} чекпоинтов")
    if start_epoch > 0:
        print(f"Продолжение с эпохи {start_epoch + 1}")
    print("=" * 60)
    
    # 1. Загрузка токенизаторов
    print("\nЗагрузка русского токенизатора (Python)...")
    if not RUS_VOCAB.exists() or not RUS_MERGES.exists():
        print(f"Токенизатор не найден: {RUS_TOKENIZER_DIR}!")
        return
    
    rus_tokenizer = BPETokenizer.load(str(RUS_VOCAB), str(RUS_MERGES))
    rus_vocab_size = len(rus_tokenizer.vocab)
    print(f"Русский токенизатор загружен (vocab_size={rus_vocab_size})")
    
    print("\nЗагрузка C++ токенизатора...")
    cpp_tokenizer = bpe.FastBPETokenizer()
    cpp_tokenizer.load(str(CPP_VOCAB), str(CPP_MERGES))
    cpp_vocab_size = cpp_tokenizer.vocab_size
    print(f"C++ токенизатор загружен (vocab_size={cpp_vocab_size})")
    
    # 2. Датасет
    print("\nПодготовка датасета...")
    input_ids_list = load_dataset(DATASET_PATH, rus_tokenizer, cpp_tokenizer, MAX_LEN)
    
    if len(input_ids_list) == 0:
        print("Нет данных для обучения!")
        return
    
    random.seed(42)
    indices = list(range(len(input_ids_list)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - VAL_SPLIT))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = InstructionDataset([input_ids_list[i] for i in train_indices])
    val_dataset = InstructionDataset([input_ids_list[i] for i in val_indices])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )
    
    print(f"Датасет готов: train={len(train_dataset)}, val={len(val_dataset)}")
    
    # 3. Базовая модель
    print(f"\nЗагрузка базовой модели из {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"Модель не найдена: {MODEL_PATH}!")
        return
    
    base_model = CppCodeModel.load_from_checkpoint(
        MODEL_PATH,
        map_location='cpu',
        strict=False
    )
    
    # Расширение словаря для поддержки русских токенов со смещением
    OLD_VOCAB_SIZE = base_model.token_embedding.num_embeddings    # 10001
    RUS_VOCAB_SIZE = len(rus_tokenizer.vocab)                     # 4000
    NEW_VOCAB_SIZE = OLD_VOCAB_SIZE + RUS_VOCAB_SIZE              # 14001
    RUS_OFFSET = OLD_VOCAB_SIZE                                   # 10001
    
    print(f"\nРасширение словаря модели:")
    print(f"- Старый размер:   {OLD_VOCAB_SIZE}")
    print(f"- Русских токенов: {RUS_VOCAB_SIZE}")
    print(f"- Новый размер:    {NEW_VOCAB_SIZE}")
    print(f"- Смещение для русских токенов: {RUS_OFFSET}")
    
    # Расширяем token_embedding
    old_embedding = base_model.token_embedding
    d_model = old_embedding.embedding_dim
    new_embedding = torch.nn.Embedding(NEW_VOCAB_SIZE, d_model)
    
    # Копируем старые веса
    with torch.no_grad():
        new_embedding.weight.data[:OLD_VOCAB_SIZE] = old_embedding.weight.data
    base_model.token_embedding = new_embedding
    
    # Расширяем lm_head
    old_head = base_model.lm_head
    new_head = torch.nn.Linear(old_head.in_features, NEW_VOCAB_SIZE)
    
    with torch.no_grad():
        new_head.weight.data[:OLD_VOCAB_SIZE] = old_head.weight.data
        if old_head.bias is not None:
            new_head.bias.data[:OLD_VOCAB_SIZE] = old_head.bias.data
    base_model.lm_head = new_head
    
    # Обновляем hparams
    base_model.hparams.vocab_size = NEW_VOCAB_SIZE
    
    print(f"Словарь успешно расширен!")
    
    # Заморозка всех параметров
    print("\nЗаморозка параметров базовой модели...")
    for param in base_model.parameters():
        param.requires_grad = False
    
    # 4. Добавление LoRA
    print(f"Добавление LoRA слоёв...")
    replaced_count = replace_with_lora(
        base_model, 
        r=LORA_R, 
        alpha=LORA_ALPHA, 
        dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    print(f"Заменено {replaced_count} Linear слоёв на LoRALinear")
    
    model = base_model
    
    # 5. Загрузка весов для продолжения
    if start_epoch > 0:
        checkpoint_path = OUTPUT_DIR / f"lora_weights_epoch_{start_epoch}.pt"
        if not checkpoint_path.exists():
            checkpoint_path = OUTPUT_DIR / "lora_weights_best.pt"
            print(f"Чекпоинт для эпохи {start_epoch} не найден, используем best.pt!")
        
        if checkpoint_path.exists():
            load_lora_weights(model, checkpoint_path)
            RESUME_LOADED = True
        else:
            print(f"Чекпоинт не найден: {checkpoint_path}!")
            start_epoch = 0
    
    # 6. Статистика
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nСТАТИСТИКА ПАРАМЕТРОВ:")
    print(f"- Всего:            {total_params:,}")
    print(f"- Обучаемых (LoRA): {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")
    
    if trainable_params == 0:
        print("\nОШИБКА: Нет обучаемых параметров!")
        return
    
    # 7. Оптимизатор
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 8. Scheduler с cosine decay
    total_steps = len(train_loader) * EPOCHS
    
    def lambda_lr(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    
    # 9. Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nУстройство: {device}")
    
    # 10. Обучение с gradient accumulation
    print("\nНАЧАЛО LoRA FINE-TUNING")
    print(f"Gradient accumulation: {ACCUMULATION_STEPS} шагов")
    print("-" * 60)
    
    best_val_loss = float('inf')
    optimizer.zero_grad()
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
            
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            
            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        val_loss = validate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}: Train loss = {avg_loss:.4f}, Val loss = {val_loss:.4f}")
        
        # Сохраняем чекпоинт для каждой эпохи
        lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k}
        
        epoch_path = OUTPUT_DIR / f"lora_weights_epoch_{epoch+1}.pt"
        torch.save(lora_state, epoch_path)
        print(f"Сохранена эпоха {epoch+1}: {epoch_path.name}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(lora_state, OUTPUT_DIR / "lora_weights_best.pt")
            print(f"Лучшие веса обновлены (val_loss={val_loss:.4f})")
        
        cleanup_old_checkpoints(OUTPUT_DIR, KEEP_LAST_N)
    
    print("\n" + "=" * 60)
    print("LoRA FINE-TUNING ЗАВЕРШЁН!")
    print("=" * 60)
    print(f"Лучшая val_loss: {best_val_loss:.4f}")
    print(f"LoRA веса:       {OUTPUT_DIR / 'lora_weights_best.pt'}")
    print(f"Последние {KEEP_LAST_N} эпох сохранены в {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()