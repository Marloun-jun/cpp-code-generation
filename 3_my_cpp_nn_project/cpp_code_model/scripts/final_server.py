#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# final_server_fixed.py - Веб-сервер для генерации C++ кода
# ======================================================================
#
# @file final_server_fixed.py
# @brief Веб-сервер для генерации C++ кода (Medium + LoRA)
#
# @author Евгений П.
# @date 2026
# @version 3.3.1
#
# @details Этот модуль запускает веб-сервер для генерации C++ кода.
#          Использует модель Medium для продолжения кода и Medium + LoRA
#          для генерации по русским инструкциям.
#
#          **Основные возможности:**
#
#          1. **Продолжение C++ кода**
#             - Модель Medium (10001 токенов)
#             - Дополнение незаконченного кода
#
#          2. **Генерация по русским инструкциям**
#             - Модель Medium + LoRA (14000 токенов)
#             - Понимание русских инструкций
#             - Генерация C++ кода с нуля
#
#          3. **Веб-интерфейс**
#             - Две вкладки для разных режимов
#             - Настройка температуры и длины генерации
#             - Примеры промптов
#
# @usage
#     python scripts/final_server_fixed.py
#
# @example
#     python final_server_fixed.py
#     # Открыть в браузере: http://localhost:8080
#
# ======================================================================

import os
import sys
import json
import torch
import time
import math
from pathlib import Path
from aiohttp import web

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.architecture.model import CppCodeModel

PROJECT_ROOT = Path(__file__).parent.parent

# ======================================================================
# ПУТИ (MEDIUM для продолжения кода + LoRA для инструкций)
# ======================================================================

import os
from pathlib import Path

# Автоопределение окружения
IN_DOCKER = os.path.exists('/app/cpp_code_model')

if IN_DOCKER:
    PROJECT_ROOT = Path('/app/cpp_code_model')
    BPE_ROOT = Path('/app/bpe_tokenizer_cpu')
    print("Запущено в Docker-контейнере")
else:
    PROJECT_ROOT = Path(__file__).parent.parent
    BPE_ROOT = PROJECT_ROOT.parent / 'bpe_tokenizer_cpu'
    print("Локальный запуск")

# Базовая модель Medium (для продолжения кода)
MODEL_PATH_MEDIUM = PROJECT_ROOT / 'checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt'
CONFIG_PATH = PROJECT_ROOT / 'configs/medium.json'

# LoRA веса (эпоха 5 - лучшая для инструкций)
LORA_PATH = PROJECT_ROOT / 'checkpoints/lora_medium/lora_weights_epoch_5.pt'

# Если файл не найден, пробуем best.pt
if not LORA_PATH.exists():
    LORA_PATH = PROJECT_ROOT / 'checkpoints/lora_medium/lora_weights_best.pt'

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CONFIG_PATH: {CONFIG_PATH}")

# Параметры LoRA
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# Параметры модели
FIXED_SEQ_LEN = 768
BOS_TOKEN = 2
EOS_TOKEN = 3
SEP_TOKEN = 10000
PAD_TOKEN = 0
RUS_OFFSET = 10001
VOCAB_SIZE = 10001
RUS_VOCAB_SIZE = 3999
TOTAL_VOCAB_SIZE = VOCAB_SIZE + RUS_VOCAB_SIZE

print("=" * 60)
print("C++ CODE GENERATOR")
print("=" * 60)

# ======================================================================
# ЗАГРУЗКА КОНФИГУРАЦИИ
# ======================================================================

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)

print(f"\nКонфигурация модели:")
print(f"- d_model:    {config['d_model']}")
print(f"- nhead:      {config['nhead']}")
print(f"- num_layers: {config['num_layers']}")
print(f"- max_len:    {config.get('max_len', 768)} -> {FIXED_SEQ_LEN}")

# ======================================================================
# ЗАГРУЗКА МОДЕЛИ 1: MEDIUM (для продолжения кода)
# ======================================================================

print("1. Загрузка модели Medium...")

model_medium = CppCodeModel(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    max_len=config.get('max_len', 768),
    dropout=0.0
)

checkpoint = torch.load(MODEL_PATH_MEDIUM, map_location='cpu')
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

new_state_dict = {}
for k, v in state_dict.items():
    key = k[6:] if k.startswith('model.') else k
    if 'causal_mask' not in key:
        new_state_dict[key] = v

model_medium.load_state_dict(new_state_dict, strict=False)
model_medium.eval()
print("Модель Medium загружена (для продолжения кода)")

# ======================================================================
# КЛАССЫ LoRA ДЛЯ ЗАГРУЗКИ ВЕСОВ
# ======================================================================

class LoRALinear(torch.nn.Module):
    """LoRA слой для линейной проекции."""
    
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
        
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / r
    
    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        if self.dropout:
            lora_out = self.dropout(lora_out)
        return result + self.scaling * lora_out


def replace_with_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
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
        print(f"LoRA файл не найден: {lora_path}")
        return False
    
    lora_state = torch.load(lora_path, map_location='cpu')
    model_state = model.state_dict()
    
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                model_state[key].copy_(value)
                loaded += 1
        elif key.startswith('model.'):
            new_key = key[6:]
            if new_key in model_state:
                if model_state[new_key].shape == value.shape:
                    model_state[new_key].copy_(value)
                    loaded += 1
        else:
            for model_key in model_state:
                if key.endswith('lora_A') and model_key.endswith('lora_A'):
                    if model_state[model_key].shape == value.shape:
                        model_state[model_key].copy_(value)
                        loaded += 1
                        break
                elif key.endswith('lora_B') and model_key.endswith('lora_B'):
                    if model_state[model_key].shape == value.shape:
                        model_state[model_key].copy_(value)
                        loaded += 1
                        break
    
    print(f"Загружено {loaded} LoRA весов")
    return loaded > 0


# ======================================================================
# ЗАГРУЗКА МОДЕЛИ 2: MEDIUM + LoRA (для инструкций)
# ======================================================================

print("\n2. Загрузка модели Medium + LoRA...")

# Используем оригинальный max_len из конфига (512)
ORIGINAL_MAX_LEN = config.get('max_len', 768)

# Создаем модель с расширенным словарем и оригинальной длиной
model_lora = CppCodeModel(
    vocab_size=TOTAL_VOCAB_SIZE,
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    max_len=ORIGINAL_MAX_LEN,
    dropout=0.0
)

# Расширяем веса из чекпоинта
print("   Расширение словаря...")
old_embedding = model_lora.token_embedding.weight.data
old_head_weight = model_lora.lm_head.weight.data
old_head_bias = model_lora.lm_head.bias.data if model_lora.lm_head.bias is not None else None

# Копируем веса из базовой модели
checkpoint_weights = {}
for k, v in new_state_dict.items():
    if k == 'token_embedding.weight':
        old_embedding[:VOCAB_SIZE] = v
    elif k == 'lm_head.weight':
        old_head_weight[:VOCAB_SIZE] = v
    elif k == 'lm_head.bias' and old_head_bias is not None:
        old_head_bias[:VOCAB_SIZE] = v
    else:
        checkpoint_weights[k] = v

model_lora.load_state_dict(checkpoint_weights, strict=False)
print(f"Token embedding:     {VOCAB_SIZE} -> {TOTAL_VOCAB_SIZE} токенов")
print(f"LM head:             {VOCAB_SIZE} -> {TOTAL_VOCAB_SIZE} токенов")
print(f"Positional encoding: сохранен оригинальный размер")

# Заморозка параметров
for param in model_lora.parameters():
    param.requires_grad = False

# Добавление LoRA слоёв
print("Добавление LoRA слоёв...")
replaced_count = replace_with_lora(model_lora)
print(f"Заменено {replaced_count} Linear слоёв")

# Загрузка LoRA весов
if LORA_PATH.exists():
    if load_lora_weights(model_lora, LORA_PATH):
        print("LoRA веса загружены")
    else:
        print("Не удалось загрузить LoRA веса!")
else:
    print(f"LoRA веса не найдены: {LORA_PATH}!")

model_lora.eval()
print("Модель Medium + LoRA загружена (для генерации по инструкции)")

# ======================================================================
# ЗАГРУЗКА ТОКЕНИЗАТОРОВ
# ======================================================================

print("\n3. Загрузка токенизаторов...")

# C++ токенизатор
bpe_cpp_dir = BPE_ROOT / "bpe_cpp"
bpe_build_dir = bpe_cpp_dir / "build"
sys.path.insert(0, str(bpe_build_dir))
import bpe_tokenizer_cpp as bpe_module

cpp_tokenizer = bpe_module.FastBPETokenizer()
cpp_vocab = bpe_cpp_dir / "models" / "bpe_10000" / "cpp_vocab.json"
cpp_merges = bpe_cpp_dir / "models" / "bpe_10000" / "cpp_merges.txt"

if cpp_vocab.exists() and cpp_merges.exists():
    cpp_tokenizer.load(str(cpp_vocab), str(cpp_merges))
    print(f"C++ токенизатор загружен (vocab_size={cpp_tokenizer.vocab_size})")
else:
    print(f"C++ модель не найдена: {cpp_vocab}!")

# Русский токенизатор
bpe_python_dir = BPE_ROOT / "bpe_python"
sys.path.insert(0, str(bpe_python_dir))
from tokenizer import BPETokenizer

rus_tokenizer_dir = PROJECT_ROOT / "tokenizers" / "rus_bpe_4000"
rus_tokenizer = BPETokenizer.load(
    str(rus_tokenizer_dir / "vocab.json"),
    str(rus_tokenizer_dir / "merges.txt")
)
print(f"Русский токенизатор загружен (vocab_size={len(rus_tokenizer.vocab)})")

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def safe_decode(tokens, tokenizer):
    """Безопасное декодирование токенов с обработкой ошибок."""
    if not tokens:
        return ""
    try:
        result = tokenizer.decode(tokens)
        # Заменяем экранированные последовательности на реальные символы
        result = result.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        return result
    except:
        result = []
        for t in tokens:
            try:
                part = tokenizer.decode([t])
                result.append(part)
            except:
                result.append('?')
        result = ''.join(result)
        result = result.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
        return result

def generate_code_completion(code_prefix, temperature=0.8, max_tokens=200):
    """Продолжение кода (только Medium) - БЕЗ СПЕЦТОКЕНОВ"""
    EOS_TOKEN = 3
    
    # Только код, без BOS и SEP!
    code_tokens = cpp_tokenizer.encode(code_prefix)
    generated = code_tokens.copy()
    min_tokens = 30
    
    with torch.no_grad():
        for step in range(max_tokens):
            x = torch.tensor([generated])
            logits = model_medium(x)
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            
            top_k = min(50, len(probs))
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            
            if step < min_tokens:
                mask = top_k_indices != EOS_TOKEN
                if mask.any():
                    top_k_probs = top_k_probs[mask]
                    top_k_indices = top_k_indices[mask]
            
            if len(top_k_probs) == 0:
                break
            
            next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
            
            if next_token == EOS_TOKEN and step >= min_tokens:
                break
            
            generated.append(next_token)
    
    # Декодируем только сгенерированную часть
    generated_tokens = generated[len(code_tokens):]
    result = safe_decode(generated_tokens, cpp_tokenizer)
    return result

def generate_from_instruction(prompt, temperature=0.8, max_tokens=200):
    """
    Генерация C++ кода по русской инструкции (Medium + LoRA).
    """
    # Кодируем русскую инструкцию со смещением
    inst_tokens = rus_tokenizer.encode(prompt)
    shifted_tokens = [t + RUS_OFFSET for t in inst_tokens]
    
    prompt_tokens = [BOS_TOKEN] + shifted_tokens + [SEP_TOKEN]
    generated = prompt_tokens.copy()
    min_tokens = 30
    
    with torch.no_grad():
        for step in range(max_tokens):
            x = torch.tensor([generated])  # Без обрезки!
            logits = model_lora(x)
            next_logits = logits[0, -1, :] / temperature
            
            top_k = min(50, next_logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            
            if step < min_tokens:
                mask = top_k_indices != EOS_TOKEN
                if mask.any():
                    probs = probs[mask]
                    top_k_indices = top_k_indices[mask]
                    probs = probs / probs.sum()
            
            if probs.sum() == 0:
                break
            
            next_token = top_k_indices[torch.multinomial(probs, 1)].item()
            
            if next_token == EOS_TOKEN and step >= min_tokens:
                break
            
            generated.append(next_token)
    
    # Декодируем код после SEP_TOKEN
    try:
        sep_pos = prompt_tokens.index(SEP_TOKEN)
        code_tokens = generated[sep_pos + 1:]
    except ValueError:
        code_tokens = []
    
    if not code_tokens:
        return "// Не удалось сгенерировать код"
    
    result = safe_decode(code_tokens, cpp_tokenizer)
    return result

# ======================================================================
# HTML СТРАНИЦА
# ======================================================================

HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>C++ Code Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; background: white; border-radius: 20px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; margin-bottom: 10px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; border-bottom: 2px solid #e0e0e0; }
        .tab { padding: 10px 20px; cursor: pointer; border-radius: 8px 8px 0 0; transition: all 0.3s; }
        .tab:hover { background: #f5f5f5; }
        .tab.active { background: linear-gradient(135deg, #667eea, #764ba2); color: white; }
        .tab-content { display: none; animation: fadeIn 0.3s; }
        .tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        textarea { width: 100%; padding: 15px; font-family: 'Consolas', monospace; font-size: 14px; border: 2px solid #e0e0e0; border-radius: 10px; resize: vertical; background: #fafafa; }
        textarea:focus { outline: none; border-color: #667eea; background: white; }
        .controls { margin: 20px 0; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        .control-group { display: flex; align-items: center; gap: 10px; }
        button { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; padding: 12px 30px; border-radius: 25px; font-size: 16px; cursor: pointer; margin: 10px 5px; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        button.secondary { background: #f44336; }
        button.secondary:hover { background: #d32f2f; }
        pre { background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 10px; padding: 20px; overflow-x: auto; font-family: 'Consolas', monospace; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; min-height: 100px; max-height: 400px; overflow-y: auto; }
        .example-buttons { display: flex; gap: 10px; margin-bottom: 15px; flex-wrap: wrap; }
        .example-btn { background: #f0f0f0; color: #333; padding: 8px 15px; font-size: 12px; border-radius: 20px; cursor: pointer; border: none; transition: all 0.2s; }
        .example-btn:hover { background: #e0e0e0; transform: translateY(-1px); }
        .status { padding: 12px; border-radius: 10px; margin: 10px 0; }
        .info { background: #e3f2fd; color: #1976d2; }
        .success { background: #e8f5e9; color: #2e7d32; }
        .error { background: #ffebee; color: #c62828; }
        .button-group { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .footer { text-align: center; margin-top: 30px; color: #999; font-size: 12px; }
        input[type="range"] { width: 150px; }
        input[type="number"] { width: 80px; padding: 5px; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>C++ Code Generator</h1>
        <div class="subtitle">Продолжение C++ кода</div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('completion')">Продолжение кода</div>
            <div class="tab" onclick="switchTab('instruction')">Генерация по инструкции</div>
        </div>
        
        <!-- Вкладка 1: Продолжение кода -->
        <div id="completion" class="tab-content active">
            <div class="example-buttons">
                <button class="example-btn" onclick="document.getElementById('code').value = '#include <iostream>\\nusing namespace std;\\n\\nint main() {\\n    cout << &quot;Hello, World!&quot; << endl;\\n    '">Hello World</button>
                <button class="example-btn" onclick="document.getElementById('code').value = 'int add(int a, int b) {\\n    return a + b;\\n'">Сумма чисел</button>
                <button class="example-btn" onclick="document.getElementById('code').value = 'class Person {\\npublic:\\n    string name;\\n    int age;\\n    '">Класс Person</button>
                <button class="example-btn" onclick="document.getElementById('code').value = 'int fibonacci(int n) {\\n    if (n <= 1) return n;\\n    '">Фибоначчи</button>
                <button class="example-btn" onclick="document.getElementById('code').value = 'void sortArray(int arr[], int size) {\\n    for(int i = 0; i < size-1; i++) {\\n        '">Сортировка</button>
            </div>
            <textarea id="code" rows="8" placeholder="Введите начало C++ кода...">int main() {</textarea>
            
            <div class="controls">
                <div class="control-group">
                    Temperature: <input type="range" id="temp" min="0.5" max="1.2" step="0.05" value="0.8">
                    <span id="tempVal">0.80</span>
                </div>
                <div class="control-group">
                    Max tokens: <input type="number" id="maxTokens" value="200" min="50" max="400" step="25">
                </div>
            </div>
            
            <div class="button-group">
                <button onclick="generateCompletion()">Дополнить код</button>
                <button onclick="clearCode()" class="secondary">Очистить поле</button>
            </div>
            
            <div id="comp-status" class="status info">Готов к работе</div>
            <pre id="comp-result"></pre>
        </div>
                
        <!-- Вкладка 2: Генерация по инструкции -->
        <div id="instruction" class="tab-content">
            <div class="example-buttons">
                <button class="example-btn" onclick="document.getElementById('inst-prompt').value = 'напиши программу которая выводит Hello World'">Hello World</button>
                <button class="example-btn" onclick="document.getElementById('inst-prompt').value = 'напиши функцию которая складывает два числа'">Сумма чисел</button>
                <button class="example-btn" onclick="document.getElementById('inst-prompt').value = 'создай класс Person с полями имя и возраст'">Класс Person</button>
                <button class="example-btn" onclick="document.getElementById('inst-prompt').value = 'напиши рекурсивную функцию Фибоначчи'">Фибоначчи</button>
                <button class="example-btn" onclick="document.getElementById('inst-prompt').value = 'напиши функцию которая сортирует вектор целых чисел'">Сортировка</button>
            </div>
            
            <textarea id="inst-prompt" rows="4" placeholder="Например: напиши программу которая выводит Hello World">напиши программу которая выводит Hello World</textarea>
            
            <div class="controls">
                <div class="control-group">
                    Temperature: <input type="range" id="inst-temp" min="0.5" max="1.0" step="0.05" value="0.8">
                    <span id="inst-tempVal">0.8</span>
                </div>
                <div class="control-group">
                    Max tokens: <input type="number" id="inst-maxTokens" value="200" min="50" max="500" step="50">
                </div>
            </div>
            
            <div class="button-group">
                <button onclick="generateInstruction()">Сгенерировать код</button>
                <button onclick="clearInstruction()" class="secondary">Очистить поле</button>
            </div>
            
            <div id="inst-status" class="status info">Готов к работе</div>
            <pre id="inst-result"></pre>
        </div>
        
        <div class="footer">
            Модель: Medium (продолжение кода) + LoRA (инструкции)
        </div>
    </div>
    
    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            if (tabName === 'completion') {
                document.querySelector('.tab:first-child').classList.add('active');
                document.getElementById('completion').classList.add('active');
            } else {
                document.querySelector('.tab:last-child').classList.add('active');
                document.getElementById('instruction').classList.add('active');
            }
        }
        
        const tempSlider = document.getElementById('temp');
        const tempVal = document.getElementById('tempVal');
        if (tempSlider) tempSlider.oninput = () => tempVal.textContent = parseFloat(tempSlider.value).toFixed(2);
        
        const instTemp = document.getElementById('inst-temp');
        const instTempVal = document.getElementById('inst-tempVal');
        if (instTemp) instTemp.oninput = () => instTempVal.textContent = parseFloat(instTemp.value).toFixed(2);
        
        function clearCode() {
            document.getElementById('code').value = '';
            document.getElementById('comp-result').textContent = '';
            document.getElementById('comp-status').textContent = 'Поле очищено';
            document.getElementById('comp-status').className = 'status info';
        }
        
        function clearInstruction() {
            document.getElementById('inst-prompt').value = '';
            document.getElementById('inst-result').textContent = '';
            document.getElementById('inst-status').textContent = 'Поле очищено';
            document.getElementById('inst-status').className = 'status info';
        }
        
        async function generateCompletion() {
            const codePrefix = document.getElementById('code').value;
            const temperature = parseFloat(tempSlider.value);
            const maxTokens = parseInt(document.getElementById('maxTokens').value);
            const resultDiv = document.getElementById('comp-result');
            const statusDiv = document.getElementById('comp-status');
            const btn = event.target;
            
            if (!codePrefix.trim()) {
                statusDiv.textContent = 'Введите начало кода';
                statusDiv.className = 'status error';
                return;
            }
            
            resultDiv.textContent = '';
            statusDiv.textContent = 'Генерация продолжения...';
            statusDiv.className = 'status info';
            btn.disabled = true;
            
            try {
                const response = await fetch('/complete', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({code_prefix: codePrefix, temperature, max_tokens: maxTokens})
                });
                const data = await response.json();
                if (data.success) {
                    resultDiv.textContent = codePrefix + data.code;
                    statusDiv.textContent = `Время: ${data.time_sec} сек, длина: ${data.length} символов`;
                    statusDiv.className = 'status success';
                } else {
                    statusDiv.textContent = `${data.error}`;
                    statusDiv.className = 'status error';
                }
            } catch (error) {
                statusDiv.textContent = `Ошибка: ${error.message}`;
                statusDiv.className = 'status error';
            } finally {
                btn.disabled = false;
            }
        }
        
        async function generateInstruction() {
            const prompt = document.getElementById('inst-prompt').value;
            const temperature = parseFloat(instTemp.value);
            const maxTokens = parseInt(document.getElementById('inst-maxTokens').value);
            const resultDiv = document.getElementById('inst-result');
            const statusDiv = document.getElementById('inst-status');
            const btn = event.target;
            
            if (!prompt.trim()) {
                statusDiv.textContent = 'Введите описание';
                statusDiv.className = 'status error';
                return;
            }
            
            resultDiv.textContent = '';
            statusDiv.textContent = 'Генерация кода...';
            statusDiv.className = 'status info';
            btn.disabled = true;
            
            try {
                const response = await fetch('/generate_instruction', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, temperature, max_tokens: maxTokens})
                });
                const data = await response.json();
                if (data.success) {
                    resultDiv.textContent = data.code;
                    statusDiv.textContent = `Время: ${data.time_sec} сек, длина: ${data.length} символов`;
                    statusDiv.className = 'status success';
                } else {
                    statusDiv.textContent = `${data.error}`;
                    statusDiv.className = 'status error';
                }
            } catch (error) {
                statusDiv.textContent = `Ошибка: ${error.message}`;
                statusDiv.className = 'status error';
            } finally {
                btn.disabled = false;
            }
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                const activeTab = document.querySelector('.tab-content.active');
                if (activeTab && activeTab.id === 'completion') generateCompletion();
                else if (activeTab && activeTab.id === 'instruction') generateInstruction();
            }
        });
    </script>
</body>
</html>"""


# ======================================================================
# ОБРАБОТЧИКИ HTTP ЗАПРОСОВ
# ======================================================================

async def index(request):
    """Главная страница."""
    return web.Response(text=HTML, content_type='text/html')


async def complete(request):
    """Продолжение C++ кода."""
    try:
        data = await request.json()
        code_prefix = data.get('code_prefix', '')
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 200)
        
        print(f"[Completion] {code_prefix[:50]}...")
        
        start = time.time()
        code = generate_code_completion(code_prefix, temperature, max_tokens)
        elapsed = time.time() - start
        
        print(f"Сгенерировано за {elapsed:.2f} сек, {len(code)} символов")
        
        return web.json_response({
            'success': True,
            'code': code,
            'time_sec': round(elapsed, 2),
            'length': len(code)
        })
    except Exception as e:
        print(f"Ошибка: {e}!")
        return web.json_response({'success': False, 'error': str(e)})


async def generate_instruction(request):
    """Генерация кода по русской инструкции."""
    try:
        data = await request.json()
        prompt = data.get('prompt', '')
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 200)
        
        print(f"[Instruction] {prompt[:50]}...")
        
        start = time.time()
        code = generate_from_instruction(prompt, temperature, max_tokens)
        elapsed = time.time() - start
        
        print(f"Сгенерировано за {elapsed:.2f} сек, {len(code)} символов")
        
        return web.json_response({
            'success': True,
            'code': code,
            'time_sec': round(elapsed, 2),
            'length': len(code)
        })
    except Exception as e:
        print(f"Ошибка: {e}!")
        return web.json_response({'success': False, 'error': str(e)})


# ======================================================================
# ЗАПУСК СЕРВЕРА
# ======================================================================

app = web.Application()
app.router.add_get('/', index)
app.router.add_post('/complete', complete)
app.router.add_post('/generate_instruction', generate_instruction)

print(f"\nСервер запущен на http://localhost:8080")
print("Вкладка 1: Продолжение кода (Medium)")
print("Вкладка 2: Генерация по инструкции (Medium + LoRA)")
print("Нажмите Ctrl+C для остановки")
print("=" * 60)

web.run_app(app, host='0.0.0.0', port=8080)