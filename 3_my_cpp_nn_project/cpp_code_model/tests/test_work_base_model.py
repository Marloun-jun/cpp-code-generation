#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# test_work_base_model.py - Тестирование генерации C++ кода "из затравки"
# ======================================================================
#
# @file test_work_base_model.py
# @brief Тестирование генерации C++ кода "из затравки" (продолжение кода)
#        базовых моделей tiny, small или medium (меняется в коде в ручную)
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль тестирует генерацию C++ кода базовой моделью Tiny
#          с использованием C++ BPE токенизатора. Модель обучена только
#          на C++ коде, поэтому тестирование выполняется только с затравками
#          кода (code prompts).
#
#          **Основные возможности:**
#
#          1. **C++ токенизатор**
#             - FastBPETokenizer для генерации кода
#             - Безопасное декодирование с обработкой ошибок UTF-8
#
#          2. **Гибкая генерация**
#             - Три уровня температуры (0.7, 0.8, 0.9)
#             - Top-k sampling (k=50)
#             - Защита от раннего EOS (min_tokens=15)
#
#          3. **Тестирование на затравках кода**
#             - main() функция
#             - Hello World программа
#             - Функции, классы, шаблоны
#
#          4. **Сохранение результатов**
#             - JSON с полными результатами генерации
#             - Текстовый отчёт с примерами кода
#             - Статистика по времени и длине
#
# @usage
#     python tests/test_work_base_model.py
#
# @example
#     # Стандартный запуск
#     python test_work_base_model.py
#
# ======================================================================

import sys
import os
import json
import time
import re

from pathlib import Path
from datetime import datetime

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Определяем пути
current_file = Path(__file__).resolve()
tests_dir = current_file.parent
cpp_code_model_dir = tests_dir.parent
sys.path.insert(0, str(cpp_code_model_dir))

# Путь к C++ токенизатору
bpe_cpp_dir = cpp_code_model_dir.parent / "bpe_tokenizer_cpu" / "bpe_cpp"
bpe_build_dir = bpe_cpp_dir / "build"

# Добавляем путь к скомпилированному модулю
sys.path.insert(0, str(bpe_build_dir))

os.chdir(cpp_code_model_dir)

print(f"Рабочая директория:  {os.getcwd()}")
print(f"C++ BPE токенизатор: {bpe_cpp_dir}")
# ======================================================================

import torch
import torch.nn.functional as F
import json as json_lib

from model.architecture.model import CppCodeModel

# ======================================================================
# НАСТРОЙКА ДИРЕКТОРИЙ
# ======================================================================

# Создаем директорию для отчетов
reports_dir = cpp_code_model_dir / "reports" / "test_work_medium"
reports_dir.mkdir(parents=True, exist_ok=True)
print(f"Результаты будут сохранены в: {reports_dir}")

# ======================================================================
# КОНСТАНТЫ
# ======================================================================

VOCAB_SIZE = 10000
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0

# Температуры для тестирования
TEMPERATURES = [0.7, 0.8, 0.9]

# Параметры генерации
MAX_NEW_TOKENS = 200
MIN_TOKENS = 15
TOP_K = 50

# ======================================================================
# ЗАГРУЗКА ТОКЕНИЗАТОРА
# ======================================================================

print("\n" + "=" * 60)
print("ЗАГРУЗКА ТОКЕНИЗАТОРА")
print("=" * 60)

cpp_tokenizer = None
cpp_tokenizer_available = False

print("\nЗагрузка C++ BPE токенизатора...")
try:
    import bpe_tokenizer_cpp as bpe_module
    print(f"Модуль bpe_tokenizer_cpp загружен")
    
    cpp_tokenizer = bpe_module.FastBPETokenizer()
    
    model_dir = bpe_cpp_dir / "models" / f"bpe_{VOCAB_SIZE}"
    vocab_file = model_dir / "cpp_vocab.json"
    merges_file = model_dir / "cpp_merges.txt"
    
    if vocab_file.exists() and merges_file.exists():
        cpp_tokenizer.load(str(vocab_file), str(merges_file))
        print(f"C++ токенизатор загружен из {model_dir}")
        print(f"- Размер словаря: {cpp_tokenizer.vocab_size}")
        print(f"- BOS ID:         {cpp_tokenizer.bos_id}")
        print(f"- EOS ID:         {cpp_tokenizer.eos_id}")
        print(f"- PAD ID:         {cpp_tokenizer.pad_id}")
        print(f"- UNK ID:         {cpp_tokenizer.unknown_id}")
        cpp_tokenizer_available = True
    else:
        print(f"Модель не найдена: {model_dir}!")
        
except Exception as e:
    print(f"Ошибка загрузки C++ токенизатора: {e}!")

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def safe_cpp_decode(tokens):
    """Безопасное декодирование C++ токенов с обработкой ошибок UTF-8"""
    if not tokens or not cpp_tokenizer_available:
        return ""
    try:
        return cpp_tokenizer.decode(tokens)
    except UnicodeDecodeError:
        # Декодируем по одному токену
        result = []
        for token in tokens:
            try:
                result.append(cpp_tokenizer.decode([token]))
            except:
                result.append('�')
        return ''.join(result)
    except Exception:
        return ""


def generate_from_prompt(model, prompt_tokens, max_new_tokens=MAX_NEW_TOKENS, 
                         temperature=0.8, min_tokens=MIN_TOKENS, top_k=TOP_K):
    """
    Генерация кода из токенов промпта.
    
    Args:
        model:                Модель CppCodeModel
        prompt_tokens (list): Начальные токены
        max_new_tokens (int): Максимальное количество новых токенов
        temperature (float):  Температура сэмплирования
        min_tokens (int):     Минимальное количество токенов до EOS
        top_k (int):          Top-k сэмплинг
        
    Returns:
        list: Сгенерированные токены
    """
    generated = prompt_tokens.copy()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            x = torch.tensor([generated])
            logits = model(x)
            next_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            vocab_size = next_logits.size(0)
            k = min(top_k, vocab_size)
            top_k_logits, top_k_indices = torch.topk(next_logits, k)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            # Игнорируем EOS в первые min_tokens шагов
            if step < min_tokens:
                mask = top_k_indices != EOS_TOKEN_ID
                if mask.any():
                    top_k_probs = probs[mask]
                    top_k_indices_filtered = top_k_indices[mask]
                    if len(top_k_probs) > 0:
                        next_token = top_k_indices_filtered[torch.multinomial(top_k_probs, 1)].item()
                    else:
                        next_token = top_k_indices[torch.multinomial(probs, 1)].item()
                else:
                    next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            else:
                next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            
            if next_token == EOS_TOKEN_ID and step >= min_tokens:
                break
            
            generated.append(next_token)
    
    return generated


def generate_code(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, 
                  temperature=0.8, min_tokens=MIN_TOKENS, top_k=TOP_K):
    """
    Генерация C++ кода из затравки.
    
    Args:
        model:                Модель CppCodeModel
        tokenizer:            C++ токенизатор
        prompt (str):         Начальный код (затравка)
        max_new_tokens (int): Максимальное количество новых токенов
        temperature (float):  Температура сэмплирования
        min_tokens (int):     Минимальное количество токенов до EOS
        top_k (int):          Top-k сэмплинг
        
    Returns:
        str: Сгенерированный код
    """
    if not cpp_tokenizer_available or tokenizer is None:
        return f"{prompt}\n    // C++ BPE токенизатор не доступен!\n    return 0;\n}}"
    
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    
    # Токенизация
    tokens = tokenizer.encode(prompt)
    tokens = [bos_id] + tokens
    
    generated_tokens = generate_from_prompt(
        model, tokens, max_new_tokens, temperature, min_tokens, top_k
    )
    
    # Декодируем только сгенерированную часть
    prompt_length = len(tokenizer.encode(prompt))
    generated_code = safe_cpp_decode(generated_tokens[1 + prompt_length:])
    
    return generated_code

# ======================================================================
# ЗАГРУЗКА МОДЕЛИ
# ======================================================================

print("\n" + "=" * 60)
print("ЗАГРУЗКА МОДЕЛИ")
print("=" * 60)

# Пути к модели
config_path = cpp_code_model_dir / "configs" / "medium.json"
model_path = cpp_code_model_dir / "checkpoints" / "medium" / "cpp-code-epoch=19-val_loss=0.87.ckpt"

# Если указанный файл не существует, ищем любой чекпоинт
if not model_path.exists():
    checkpoints_dir = cpp_code_model_dir / "checkpoints" / "medium"
    ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
    if ckpt_files:
        # Исключаем last.ckpt, выбираем с наибольшей эпохой
        epoch_files = [f for f in ckpt_files if 'epoch' in f.name]
        if epoch_files:
            def get_epoch(filename):
                match = re.search(r'epoch=(\d+)', str(filename))
                return int(match.group(1)) if match else 0
            epoch_files.sort(key=get_epoch)
            model_path = epoch_files[-1]
        else:
            model_path = ckpt_files[0]
        print(f"Найден чекпоинт: {model_path.name}")

print(f"Конфиг: {config_path}")
print(f"Модель: {model_path}")

with open(config_path, 'r', encoding='utf-8') as f:
    config = json_lib.load(f)

model = CppCodeModel(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    max_len=config.get('max_len', 1024),
    dropout=config.get('dropout', 0.1)
)

checkpoint = torch.load(model_path, map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('model.'):
        new_state_dict[k[6:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Модель загружена: {total_params:,} параметров")

# ======================================================================
# ТЕСТОВЫЕ ПРОМПТЫ ("ЗАТРАВКИ КОДА")
# ======================================================================

test_prompts = [
    {
        "name": "main_function",
        "prompt": "int main() {",
        "description": "Простая функция main()",
        "expected": "Должна содержать return 0;"
    },
    {
        "name": "hello_world",
        "prompt": '#include <iostream>\n\nint main() {\n    std::cout << "Hello, World!" << std::endl;',
        "description": "Hello World программа",
        "expected": "Должна содержать корректный вывод"
    },
    {
        "name": "function_definition",
        "prompt": "int add(int a, int b) {\n    return a + b;",
        "description": "Простая функция сложения",
        "expected": "Должна корректно возвращать сумму"
    },
    {
        "name": "class_definition",
        "prompt": "class MyClass {\npublic:\n    MyClass() {}\n    void print() {",
        "description": "Определение класса",
        "expected": "Должна содержать метод print"
    },
    {
        "name": "template_function",
        "prompt": "template<typename T>\nT max(T a, T b) {\n    return (a > b) ? a : b;",
        "description": "Шаблонная функция",
        "expected": "Должна корректно работать с разными типами"
    },
    {
        "name": "vector_usage",
        "prompt": '#include <vector>\n#include <algorithm>\n\nvoid sortVector(std::vector<int>& vec) {\n    std::sort(vec.begin(), vec.end());',
        "description": "Использование STL вектора",
        "expected": "Должна содержать сортировку"
    },
    {
        "name": "recursive_fibonacci",
        "prompt": "int fibonacci(int n) {\n    if (n <= 1) return n;\n    return fibonacci(n-1) + fibonacci(n-2);",
        "description": "Рекурсивная функция Фибоначчи",
        "expected": "Должна корректно вычислять числа Фибоначчи"
    }
]

# ======================================================================
# ЗАПУСК ТЕСТОВ
# ======================================================================

print("\n" + "=" * 60)
print("ТЕСТ ГЕНЕРАЦИИ C++ КОДА (БАЗОВАЯ МОДЕЛЬ MEDIUM)")
print("=" * 60)
print(f"C++ токенизатор:    {'Доступен' if cpp_tokenizer_available else 'Не доступен'}")
print(f"Температуры:        {TEMPERATURES}")
print(f"Max новых токенов:  {MAX_NEW_TOKENS}")
print(f"Min токенов до EOS: {MIN_TOKENS}")
print(f"Top-k:              {TOP_K}")
print("=" * 60)

# Список для сохранения результатов
all_results = []
total_time = 0
total_tests = 0

for test in test_prompts:
    print(f"\n{'─'*60}")
    print(f"Тест: {test['name']}")
    print(f"Описание: {test['description']}")
    print(f"Ожидание: {test['expected']}")
    print(f"{'─'*60}")
    
    print(f"\nПромпт (затравка):")
    print(test['prompt'][:100] + "..." if len(test['prompt']) > 100 else test['prompt'])
    
    test_results = []
    
    for temp in TEMPERATURES:
        print(f"\nТемпература = {temp}:")
        start_time = time.time()
        
        generated = generate_code(
            model, cpp_tokenizer, test['prompt'], 
            max_new_tokens=MAX_NEW_TOKENS, 
            temperature=temp, 
            min_tokens=MIN_TOKENS, 
            top_k=TOP_K
        )
        
        elapsed = time.time() - start_time
        total_time += elapsed
        total_tests += 1
        
        # Показываем результат
        code_preview = generated[:400] if len(generated) > 400 else generated
        print(f"{code_preview}...")
        print(f"Время: {elapsed:.2f} сек")
        print(f"Длина: {len(generated)} символов")
        
        test_results.append({
            'temperature': temp,
            'generated_code': generated,
            'time_sec': elapsed,
            'length': len(generated)
        })
    
    all_results.append({
        'test_id': len(all_results) + 1,
        'name': test['name'],
        'description': test['description'],
        'prompt': test['prompt'],
        'expected': test['expected'],
        'results': test_results,
        'timestamp': datetime.now().isoformat()
    })

# ======================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ======================================================================

# Статистика
avg_time = total_time / total_tests if total_tests > 0 else 0
avg_length = sum(len(r['results'][0]['generated_code']) for r in all_results) / len(all_results) if all_results else 0

# Сохраняем JSON
samples_path = reports_dir / 'generated_samples.json'
with open(samples_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Создаем текстовый отчет
report = f"""
============================================================
 ОТЧЕТ О ГЕНЕРАЦИИ C++ КОДА (БАЗОВАЯ МОДЕЛЬ)
 ГЕНЕРАЦИЯ "ИЗ ЗАТРАВКИ"
============================================================

ИНФОРМАЦИЯ О ТЕСТИРОВАНИИ
------------------------------------------------------------
- Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Модель:            {config.get('name', 'CppCodeModel-Tiny')}
- Параметры:         {total_params:,}
- Устройство:        CPU
- Тип генерации:     "Из затравки" (code prompt)

СТАТУС ТОКЕНИЗАТОРА
------------------------------------------------------------
- C++ токенизатор: {'Доступен' if cpp_tokenizer_available else 'Не доступен'}
- Размер словаря:  {cpp_tokenizer.vocab_size if cpp_tokenizer_available else 'N/A'}

ПАРАМЕТРЫ ГЕНЕРАЦИИ
------------------------------------------------------------
- Температуры:        {TEMPERATURES}
- Max новых токенов:  {MAX_NEW_TOKENS}
- Min токенов до EOS: {MIN_TOKENS}
- Top-k:              {TOP_K}

РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ
------------------------------------------------------------
- Всего тестов:            {len(all_results)}
- Среднее время генерации: {avg_time:.2f} сек
- Средняя длина кода:      {avg_length:.0f} символов

ДЕТАЛИ ТЕСТОВ
------------------------------------------------------------
"""

for result in all_results:
    report += f"""
{result['test_id']}. {result['name']}
   Описание: {result['description']}
   Промпт:   {result['prompt'][:80]}...
   Ожидание: {result['expected']}
"""
    for res in result['results']:
        preview = res['generated_code'][:200].replace('\n', '\\n')
        report += f"""
      Температура = {res['temperature']}:
      Время: {res['time_sec']:.2f} сек
      Длина: {res['length']} символов
      Код:   {preview}...
"""

report += f"""
РЕКОМЕНДАЦИИ ПО ТЕМПЕРАТУРАМ
------------------------------------------------------------
- 0.7 - Низкая: детерминированная генерация, предсказуемый код
- 0.8 - Средняя: баланс между качеством и разнообразием
- 0.9 - Высокая: креативная генерация, больше вариативности

ВАЖНО
------------------------------------------------------------
- Базовая модель обучена ТОЛЬКО на C++ коде
- Русские инструкции НЕ ПОДДЕРЖИВАЮТСЯ
- Для русских инструкций используйте LoRA модель

СОХРАНЕННЫЕ ФАЙЛЫ
------------------------------------------------------------
- generated_samples.json - Все сгенерированные примеры
- GENERATION_REPORT.txt  - Данный отчет

============================================================
"""

report_path = reports_dir / 'GENERATION_REPORT.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print("\n" + "=" * 60)
print("СТАТИСТИКА ТЕСТИРОВАНИЯ")
print("=" * 60)
print(f"Всего тестов:            {len(all_results)}")
print(f"Среднее время генерации: {avg_time:.2f} сек")
print(f"Средняя длина кода:      {avg_length:.0f} символов")
print(f"Температуры:             {TEMPERATURES}")

print("\n" + "=" * 60)
print("СОХРАНЕННЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"JSON:  {samples_path}")
print(f"Отчет: {report_path}")

print("\nТестирование генерации завершено!")