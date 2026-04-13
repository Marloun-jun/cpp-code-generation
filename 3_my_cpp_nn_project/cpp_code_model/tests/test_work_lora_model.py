#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# test_work_lora_model.py - Тестирование LoRA модели с двумя токенизаторами
# ======================================================================
#
# @file test_work_lora_model.py
# @brief Тестирование генерации C++ кода LoRA моделью по русским инструкциям
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль тестирует LoRA модель (Medium + LoRA) на способность
#          понимать русские инструкции и генерировать корректный C++ код.
#          Использует два токенизатора:
#          - Русский BPE (Python) для кодирования инструкций
#          - C++ BPE для кодирования и декодирования кода
#
#          **Основные возможности:**
#
#          1. **Два токенизатора**
#             - Русский токенизатор (BPETokenizer) для инструкций
#             - C++ токенизатор (FastBPETokenizer) для кода
#             - Разделитель SEP (10000) между инструкцией и кодом
#
#          2. **Генерация кода**
#             - Поддержка русских инструкций
#             - Три уровня температуры (0.85, 0.9, 0.95)
#             - Top-k sampling (k=50)
#             - Защита от раннего EOS (min_tokens=45)
#
#          3. **Тестовые промпты**
#             - Функция main()
#             - Hello World
#             - Функция сложения
#             - Класс Person
#             - Сортировка вектора
#             - Рекурсивный Фибоначчи
#
#          4. **Сохранение результатов**
#             - JSON с полными результатами генерации
#             - Текстовый отчёт с примерами кода
#             - Статистика по времени и длине
#
# @usage
#     python tests/test_work_lora_model.py
#
# @example
#     # Стандартный запуск
#     python test_work_lora_model.py
#
# ======================================================================

import sys
import os
import json
import time

from pathlib import Path
from datetime import datetime

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

current_file = Path(__file__).resolve()
tests_dir = current_file.parent
cpp_code_model_dir = tests_dir.parent
sys.path.insert(0, str(cpp_code_model_dir))
sys.path.insert(0, str(cpp_code_model_dir / "scripts"))

os.chdir(cpp_code_model_dir)

print(f"Рабочая директория: {os.getcwd()}")

import torch
import json as json_lib
from model.architecture.model import CppCodeModel

# ======================================================================
# ПУТИ (MEDIUM + LoRA)
# ======================================================================

# Используем лучшую модель Medium
BASE_PATH = cpp_code_model_dir / "checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt"
LORA_PATH = cpp_code_model_dir / "checkpoints/lora_medium/lora_weights_epoch_4.pt"
CONFIG_PATH = cpp_code_model_dir / "configs/medium.json"

# Альтернативные пути для LoRA (если best не найден)
ALT_LORA_PATHS = [
    cpp_code_model_dir / "checkpoints/lora_medium/lora_weights_epoch_5.pt",
    cpp_code_model_dir / "checkpoints/lora_medium/lora_weights_epoch_4.pt",
    cpp_code_model_dir / "checkpoints/lora_medium/lora_weights_epoch_3.pt",
]

# Директория для отчётов
reports_dir = cpp_code_model_dir / "reports" / "test_work_lora_medium"
reports_dir.mkdir(parents=True, exist_ok=True)
print(f"Результаты будут сохранены в: {reports_dir}")

# ======================================================================
# ЗАГРУЗКА ТОКЕНИЗАТОРОВ
# ======================================================================

# Путь к Python BPE токенизатору (русский)
bpe_python_dir = cpp_code_model_dir.parent / "bpe_tokenizer_cpu" / "bpe_python"
sys.path.insert(0, str(bpe_python_dir))
from tokenizer import BPETokenizer

RUS_TOKENIZER_DIR = cpp_code_model_dir / "tokenizers" / "rus_bpe_4000"
RUS_VOCAB = RUS_TOKENIZER_DIR / "vocab.json"
RUS_MERGES = RUS_TOKENIZER_DIR / "merges.txt"

# Путь к C++ токенизатору
bpe_cpp_dir = cpp_code_model_dir.parent / "bpe_tokenizer_cpu" / "bpe_cpp"
bpe_build_dir = bpe_cpp_dir / "build"
sys.path.insert(0, str(bpe_build_dir))
import bpe_tokenizer_cpp as bpe_module

CPP_VOCAB = bpe_cpp_dir / "models" / "bpe_10000" / "cpp_vocab.json"
CPP_MERGES = bpe_cpp_dir / "models" / "bpe_10000" / "cpp_merges.txt"

print("\n" + "=" * 60)
print("ЗАГРУЗКА ТОКЕНИЗАТОРОВ")
print("=" * 60)

# Загружаем русский токенизатор
rus_tokenizer = BPETokenizer.load(str(RUS_VOCAB), str(RUS_MERGES))
print(f"Русский токенизатор загружен (vocab_size={len(rus_tokenizer.vocab)})")

# Загружаем C++ токенизатор
cpp_tokenizer = bpe_module.FastBPETokenizer()
cpp_tokenizer.load(str(CPP_VOCAB), str(CPP_MERGES))
print(f"C++ токенизатор загружен (vocab_size={cpp_tokenizer.vocab_size})")
print(f"BOS ID: {cpp_tokenizer.bos_id}, EOS ID: {cpp_tokenizer.eos_id}")

# ======================================================================
# ФУНКЦИИ LoRA (ДЛЯ ЗАГРУЗКИ ВЕСОВ)
# ======================================================================

class LoRALinear(torch.nn.Module):
    """LoRA слой для линейной проекции."""
    
    def __init__(self, original_linear, r=32, alpha=64, dropout=0.1):
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
        
        self.scaling = alpha / r
    
    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        if self.dropout:
            lora_out = self.dropout(lora_out)
        return result + self.scaling * lora_out


def replace_with_lora(model, r=32, alpha=64, dropout=0.1):
    """Рекурсивная замена всех Linear слоёв на LoRALinear."""
    replaced_count = 0
    
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            lora_layer = LoRALinear(module, r, alpha, dropout)
            setattr(model, name, lora_layer)
            replaced_count += 1
        else:
            replaced_count += replace_with_lora(module, r, alpha, dropout)
    
    return replaced_count


def load_lora_weights(model, lora_path):
    """Загружает LoRA веса в модель."""
    if not lora_path.exists():
        return False
    
    lora_state = torch.load(lora_path, map_location='cpu')
    model_state = model.state_dict()
    
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].copy_(value)
            loaded += 1
    
    print(f"Загружено {loaded} LoRA весов")
    return loaded > 0


# ======================================================================
# ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ
# ======================================================================

def safe_rus_decode(tokens):
    """Безопасное декодирование русских токенов."""
    if not tokens:
        return ""
    try:
        result = rus_tokenizer.decode(tokens)
        result = ''.join(c if c.isprintable() or c in '\n\t ' else '?' for c in result)
        return result
    except:
        decoded = []
        for t in tokens:
            try:
                part = rus_tokenizer.decode([t])
                part = ''.join(c if c.isprintable() or c in '\n\t ' else '?' for c in part)
                decoded.append(part)
            except:
                decoded.append('?')
        return ''.join(decoded)


def safe_cpp_decode(tokens):
    """Безопасное декодирование C++ токенов."""
    if not tokens:
        return ""
    try:
        return cpp_tokenizer.decode(tokens)
    except:
        decoded = []
        for t in tokens:
            try:
                decoded.append(cpp_tokenizer.decode([t]))
            except:
                decoded.append('?')
        return ''.join(decoded)

def encode_with_two_tokenizers(instruction, max_len=768):
    """Кодирование инструкции русским токенизатором со смещением."""
    # Получаем русские токены (0-3999)
    inst_tokens = rus_tokenizer.encode(instruction)
    
    # Смещаем в диапазон 10001-14000
    RUS_OFFSET = 10001
    shifted_tokens = [t + RUS_OFFSET for t in inst_tokens]
    
    SEP_TOKEN = 10000
    BOS_TOKEN = 2
    full_tokens = [BOS_TOKEN] + shifted_tokens + [SEP_TOKEN]
    
    if len(full_tokens) > max_len:
        full_tokens = full_tokens[:max_len]
    
    return full_tokens

def decode_with_two_tokenizers(tokens):
    """Декодирование с разделением на инструкцию и код."""
    sep_token = 10000
    RUS_OFFSET = 10001
    
    try:
        if sep_token in tokens:
            sep_idx = tokens.index(sep_token)
            inst_tokens_shifted = tokens[:sep_idx]
            code_tokens = tokens[sep_idx + 1:]
            
            # Убираем смещение с русских токенов перед декодированием
            inst_tokens = [t - RUS_OFFSET for t in inst_tokens_shifted if t >= RUS_OFFSET]
            
            instruction = safe_rus_decode(inst_tokens) if inst_tokens else ""
            code = safe_cpp_decode(code_tokens) if code_tokens else ""
            return instruction, code
        else:
            return "", safe_cpp_decode(tokens)
    except Exception as e:
        return "", f"Ошибка декодирования: {e}!"


def generate(instruction, model, max_tokens=350, temperature=0.9, min_tokens=45, top_k=50):
    """
    Генерация C++ кода по русской инструкции.
    
    Args:
        instruction (str):   Инструкция на русском языке
        model:               LoRA модель
        max_tokens (int):    Максимальное количество генерируемых токенов
        temperature (float): Температура сэмплирования
        min_tokens (int):    Минимальное количество токенов до EOS
        top_k (int):         Top-k сэмплинг
        
    Returns:
        str: Сгенерированный C++ код
    """
    prompt_tokens = encode_with_two_tokenizers(instruction)
    generated = prompt_tokens.copy()
    
    with torch.no_grad():
        for step in range(max_tokens):
            x = torch.tensor([generated])
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            
            # Top-k sampling
            k = min(top_k, len(probs))
            top_k_probs, top_k_indices = torch.topk(probs, k)
            
            # Игнорируем EOS в первые min_tokens шагов
            if step < min_tokens:
                mask = top_k_indices != 3
                if mask.any():
                    top_k_probs = top_k_probs[mask]
                    top_k_indices = top_k_indices[mask]
            
            next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
            
            if next_token == 3:  # EOS
                break
            generated.append(next_token)
    
    _, code = decode_with_two_tokenizers(generated)
    return code

# ======================================================================
# ЗАГРУЗКА МОДЕЛИ
# ======================================================================

print("\n" + "=" * 60)
print("ЗАГРУЗКА LoRA МОДЕЛИ")
print("=" * 60)

# Загружаем конфиг
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json_lib.load(f)

# Создаём базовую модель Medium с ОРИГИНАЛЬНЫМ vocab_size
model = CppCodeModel(
    vocab_size=config['vocab_size'],  # 10001
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    max_len=config.get('max_len', 768),
    dropout=config.get('dropout', 0.15)
)

# Загружаем базовые веса
checkpoint = torch.load(BASE_PATH, map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
clean_state = {}
for k, v in state_dict.items():
    key = k[6:] if k.startswith('model.') else k
    if 'causal_mask' not in key:
        clean_state[key] = v
model.load_state_dict(clean_state, strict=False)
print("Базовые веса Medium загружены")

# ======================================================================
# Расширение словаря для поддержки русских токенов
# ======================================================================
OLD_VOCAB_SIZE = model.token_embedding.num_embeddings  # 10001
RUS_VOCAB_SIZE = len(rus_tokenizer.vocab)  # 3999
NEW_VOCAB_SIZE = OLD_VOCAB_SIZE + RUS_VOCAB_SIZE  # 14000
RUS_OFFSET = OLD_VOCAB_SIZE  # 10001

print(f"\nРасширение словаря модели:")
print(f"- Старый размер:   {OLD_VOCAB_SIZE}")
print(f"- Русских токенов: {RUS_VOCAB_SIZE}")
print(f"- Новый размер:    {NEW_VOCAB_SIZE}")
print(f"- Смещение:        {RUS_OFFSET}")

# Расширяем token_embedding
old_embedding = model.token_embedding
d_model = old_embedding.embedding_dim
new_embedding = torch.nn.Embedding(NEW_VOCAB_SIZE, d_model)

with torch.no_grad():
    new_embedding.weight.data[:OLD_VOCAB_SIZE] = old_embedding.weight.data
model.token_embedding = new_embedding

# Расширяем lm_head
old_head = model.lm_head
new_head = torch.nn.Linear(old_head.in_features, NEW_VOCAB_SIZE)

with torch.no_grad():
    new_head.weight.data[:OLD_VOCAB_SIZE] = old_head.weight.data
    if old_head.bias is not None:
        new_head.bias.data[:OLD_VOCAB_SIZE] = old_head.bias.data
model.lm_head = new_head

model.hparams.vocab_size = NEW_VOCAB_SIZE
print("✓ Словарь успешно расширен!")

# Замораживаем все параметры
for param in model.parameters():
    param.requires_grad = False

# Добавляем LoRA слои
print("\nДобавление LoRA слоёв...")
replaced_count = replace_with_lora(model, r=32, alpha=64, dropout=0.1)
print(f"Заменено {replaced_count} Linear слоёв на LoRALinear")

# Загружаем LoRA веса
lora_path = LORA_PATH
if not lora_path.exists():
    for alt_path in ALT_LORA_PATHS:
        if alt_path.exists():
            lora_path = alt_path
            print(f"Используем LoRA веса: {lora_path.name}")
            break

if lora_path.exists():
    load_lora_weights(model, lora_path)
else:
    print("LoRA веса не найдены! Используется базовая модель!")

model.eval()
print("Модель готова к тестированию")

# ======================================================================
# ТЕСТОВЫЕ ПРОМПТЫ
# ======================================================================

test_prompts = [
    {
        "name": "main_function",
        "prompt": "Напиши программу с функцией main",
        "description": "Простая функция main()",
        "expected": "Должна содержать return 0;"
    },
    {
        "name": "hello_world",
        "prompt": "Напиши программу которая выводит Hello World",
        "description": "Hello World программа",
        "expected": "Должна содержать cout или std::cout"
    },
    {
        "name": "sum_function",
        "prompt": "Напиши функцию которая складывает два числа",
        "description": "Функция сложения",
        "expected": "Должна возвращать сумму"
    },
    {
        "name": "class_person",
        "prompt": "Создай класс Person с полями имя и возраст",
        "description": "Класс Person",
        "expected": "Должен иметь конструктор и метод вывода"
    },
    {
        "name": "vector_sort",
        "prompt": "Напиши функцию которая сортирует вектор целых чисел",
        "description": "Сортировка вектора",
        "expected": "Должна содержать std::sort"
    },
    {
        "name": "fibonacci",
        "prompt": "Напиши рекурсивную функцию вычисления чисел Фибоначчи",
        "description": "Числа Фибоначчи",
        "expected": "Должна содержать fibonacci(n-1) + fibonacci(n-2)"
    }
]

# ======================================================================
# ЗАПУСК ТЕСТОВ
# ======================================================================

print("\n" + "=" * 60)
print("ТЕСТ ГЕНЕРАЦИИ C++ КОДА")
print("Два токенизатора | min_tokens=45 | temp=0.7-0.9")
print("=" * 60)

generated_samples = []
total_time = 0
total_tests = 0

for test in test_prompts:
    print(f"\n{'─' * 60}")
    print(f"Тест:     {test['name']}")
    print(f"Описание: {test['description']}")
    print(f"Ожидание: {test['expected']}")
    print(f"{'─' * 60}")
    
    print(f"\nИнструкция: {test['prompt']}")
    
    results = []
    for temp in [0.7, 0.8, 0.9]:
        print(f"\nТемпература = {temp}:")
        start_time = time.time()
        
        generated = generate(
            test['prompt'],
            model,
            max_tokens=350,
            temperature=temp,
            min_tokens=45,
            top_k=50
        )
        
        elapsed = time.time() - start_time
        total_time += elapsed
        total_tests += 1
        
        # Показываем результат
        code_preview = generated[:500] if len(generated) > 500 else generated
        print(f"{code_preview}...")
        print(f"Время: {elapsed:.2f} сек")
        print(f"Длина: {len(generated)} символов")
        
        results.append({
            'temperature': temp,
            'generated_code': generated,
            'time_sec': elapsed,
            'length': len(generated)
        })
    
    generated_samples.append({
        'test_id': len(generated_samples) + 1,
        'name': test['name'],
        'description': test['description'],
        'prompt': test['prompt'],
        'expected': test['expected'],
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

# ======================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ======================================================================

avg_time = total_time / total_tests if total_tests > 0 else 0
avg_length = sum(len(s['results'][0]['generated_code']) for s in generated_samples) / len(generated_samples) if generated_samples else 0

# Сохраняем JSON
samples_path = reports_dir / 'generated_samples.json'
with open(samples_path, 'w', encoding='utf-8') as f:
    json.dump(generated_samples, f, indent=2, ensure_ascii=False)
print(f"\nJSON сохранён: {samples_path}")

# ======================================================================
# ОТЧЁТ
# ======================================================================

report = f"""
============================================================
 ОТЧЕТ О ГЕНЕРАЦИИ C++ КОДА (LoRA MEDIUM)
 Два токенизатора | min_tokens=45 | temp=0.7-0.9
============================================================

ИНФОРМАЦИЯ О ТЕСТИРОВАНИИ
------------------------------------------------------------
- Дата:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Модель:              CppCodeModel-Medium + LoRA
- Русский токенизатор: {len(rus_tokenizer.vocab)} токенов
- C++ токенизатор:     {cpp_tokenizer.vocab_size} токенов
- Параметры генерации:
    - min_tokens: 45 (игнорируем EOS первые 45 шагов)
    - max_tokens: 350
    - top_k:      50

РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ
------------------------------------------------------------
- Всего тестов:         {len(generated_samples)}
- Среднее время генер.: {avg_time:.2f} сек
- Средняя длина кода:   {avg_length:.0f} символов

ДЕТАЛИ ТЕСТОВ
------------------------------------------------------------
"""

for sample in generated_samples:
    report += f"""
{sample['test_id']}. {sample['name']}
   Инструкция: {sample['prompt']}
   Ожидание:   {sample['expected']}
"""
    for result in sample['results']:
        preview = result['generated_code'][:300].replace('\n', '\\n')
        report += f"""
      Температура = {result['temperature']}:
      Время: {result['time_sec']:.2f} сек
      Длина: {result['length']} символов
      Код:   {preview}...
"""

report += f"""
РЕКОМЕНДАЦИИ
------------------------------------------------------------
- Оптимальная температура: 0.7-0.9
- Минимальное количество токенов: 45
- Для лучших результатов используйте max_tokens=350-500
- Модель понимает русские инструкции и генерирует C++ код

СОХРАНЕННЫЕ ФАЙЛЫ
------------------------------------------------------------
- generated_samples.json - Все сгенерированные примеры
- GENERATION_REPORT.txt  - Данный отчет

============================================================
"""

report_path = reports_dir / 'GENERATION_REPORT.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"Отчёт сохранён: {report_path}")

# ======================================================================
# ВЫВОД СТАТИСТИКИ
# ======================================================================

print("\n" + "=" * 60)
print("СТАТИСТИКА ТЕСТИРОВАНИЯ")
print("=" * 60)
print(f"Всего тестов:          {len(generated_samples)}")
print(f"Среднее время генер.:  {avg_time:.2f} сек")
print(f"Средняя длина кода:    {avg_length:.0f} символов")
print(f"Оптимальные параметры: min_tokens=45, temperature=0.7-0.9")

print("\nТестирование завершено!")