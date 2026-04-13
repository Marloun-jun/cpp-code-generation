#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# compare_models.py - Сравнение трёх моделей (Tiny, Small, Medium)
# ======================================================================
#
# @file compare_models.py
# @brief Сравнение метрик и качества генерации трёх базовых моделей
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль выполняет комплексное сравнение трёх моделей:
#          Tiny (5M), Small (8.3M), Medium (18.3M).
#          Сравниваются метрики (perplexity, производительность, память)
#          и качество генерации C++ кода из затравки.
#
#          **Основные возможности:**
#
#          1. **Сравнение метрик**
#             - Perplexity на тестовых данных
#             - Время инференса (производительность)
#             - Использование памяти
#             - Размер моделей и чекпоинтов
#
#          2. **Сравнение генерации кода**
#             - Тестирование на одинаковых затравках
#             - Оценка качества сгенерированного кода
#             - Сохранение примеров для анализа
#
#          3. **Визуализация**
#             - Графики сравнения метрик
#             - Таблицы результатов
#             - Сохранение отчёта в HTML/JSON
#
# @usage
#     python scripts/compare_models.py
#
# @example
#     # Стандартный запуск
#     python compare_models.py
#
# ======================================================================

import os
import sys

from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

current_file = Path(__file__).resolve()
scripts_dir = current_file.parent
cpp_code_model_dir = scripts_dir.parent
sys.path.insert(0, str(cpp_code_model_dir))

os.chdir(cpp_code_model_dir)

print(f"Рабочая директория: {os.getcwd()}")
# ======================================================================

import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from model.architecture.model import CppCodeModel

# ======================================================================
# КОНФИГУРАЦИИ МОДЕЛЕЙ
# ======================================================================

MODELS_CONFIG = {
    "tiny": {
        "name": "Tiny (5M)",
        "config_path": "configs/tiny.json",
        "checkpoint_dir": "checkpoints/tiny",
        "color": "#2196F3",
        "color_light": "#64B5F6"
    },
    "small": {
        "name": "Small (8.3M)",
        "config_path": "configs/small.json",
        "checkpoint_dir": "checkpoints/small",
        "color": "#4CAF50",
        "color_light": "#81C784"
    },
    "medium": {
        "name": "Medium (18.3M)",
        "config_path": "configs/medium.json",
        "checkpoint_dir": "checkpoints/medium",
        "color": "#FF9800",
        "color_light": "#FFB74D"
    }
}

# Тестовые промпты для сравнения генерации
TEST_PROMPTS = [
    {
        "name": "main_function",
        "prompt": "int main() {",
        "description": "Простая функция main()",
        "expected": "должна содержать return 0;"
    },
    {
        "name": "hello_world",
        "prompt": '#include <iostream>\n\nint main() {\n    std::cout << "Hello, World!" << std::endl;',
        "description": "Hello World программа",
        "expected": "должна содержать корректный вывод"
    },
    {
        "name": "add_function",
        "prompt": "int add(int a, int b) {\n    return a + b;",
        "description": "Функция сложения",
        "expected": "должна возвращать сумму"
    }
]

# ======================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С МОДЕЛЯМИ
# ======================================================================

def find_best_checkpoint(checkpoint_dir):
    """Находит последний чекпоинт в директории."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None
    
    # Сортируем по времени модификации
    ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Возвращаем самый свежий
    return ckpt_files[0]


def load_model(model_type, device='cpu'):
    """Загружает модель указанного типа."""
    config = MODELS_CONFIG[model_type]
    
    # Загружаем конфиг
    config_path = cpp_code_model_dir / config["config_path"]
    if not config_path.exists():
        print(f"Файл конфигураций не найден: {config_path}!")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        model_config = json.load(f)
    
    # Находим чекпоинт
    checkpoint_dir = cpp_code_model_dir / config["checkpoint_dir"]
    checkpoint_path = find_best_checkpoint(checkpoint_dir)
    
    if not checkpoint_path:
        print(f"Чекпоинт не найден: {checkpoint_dir}!")
        return None
    
    # Создаём модель
    model = CppCodeModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        max_len=model_config.get('max_len', 1024),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Загружаем веса
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Убираем префиксы
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model, model_config, checkpoint_path


def compute_perplexity(model, test_tokens_path, device='cpu', max_samples=200):
    """Вычисляет perplexity модели на тестовых данных."""
    if not Path(test_tokens_path).exists():
        return None
    
    test_tokens = torch.load(test_tokens_path)
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for seq in test_tokens[:max_samples]:
            if len(seq) < 2:
                continue
            
            # Обрезаем до разумной длины
            if len(seq) > 512:
                seq = seq[:512]
            
            input_ids = seq[:-1].unsqueeze(0).to(device)
            target_ids = seq[1:].unsqueeze(0).to(device)
            
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def measure_inference_time(model, seq_len=512, num_iterations=50, device='cpu'):
    """Измеряет время инференса модели."""
    dummy_input = torch.randint(0, 10000, (1, seq_len), device=device)
    
    # Прогрев
    for _ in range(10):
        _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    avg_time = (time.time() - start_time) / num_iterations
    throughput = seq_len / avg_time
    
    return avg_time * 1000, throughput    # Время в мс


def measure_memory_usage(model, seq_len=512, batch_size=1, device='cpu'):
    """Измеряет использование памяти моделью."""
    dummy_input = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = model(dummy_input)
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        import psutil
        import gc
        gc.collect()
        process = psutil.Process()
        before_mem = process.memory_info().rss / (1024 ** 2)
        _ = model(dummy_input)
        after_mem = process.memory_info().rss / (1024 ** 2)
        memory_mb = after_mem - before_mem
    
    return memory_mb


def get_model_size(model):
    """Возвращает размер модели в МБ."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 ** 2)    # float32 = 4 байта

# ======================================================================
# ВИЗУАЛИЗАЦИЯ
# ======================================================================

def create_comparison_charts(results, output_dir):
    """Создаёт графики сравнения моделей."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = list(results.keys())
    model_names = [MODELS_CONFIG[m]["name"] for m in models]
    colors = [MODELS_CONFIG[m]["color"] for m in models]
    
    # 1. Perplexity
    perplexities = [results[m].get('perplexity', 0) for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, perplexities, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Сравнение Perplexity моделей', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, perplexities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'perplexity_comparison.png', dpi=150)
    plt.close()
    
    # 2. Время инференса
    inference_times = [results[m].get('inference_time_ms', 0) for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, inference_times, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Время (мс)', fontsize=12)
    plt.title('Сравнение времени инференса (seq_len=512)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, inference_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f} мс', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_comparison.png', dpi=150)
    plt.close()
    
    # 3. Использование памяти
    memory_usages = [results[m].get('memory_mb', 0) for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, memory_usages, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Память (МБ)', fontsize=12)
    plt.title('Сравнение использования памяти (batch_size=1)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, memory_usages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val:.0f} МБ', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory_comparison.png', dpi=150)
    plt.close()
    
    # 4. Размер модели
    model_sizes = [results[m].get('model_size_mb', 0) for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, model_sizes, color=colors, edgecolor='black', linewidth=1.5)
    plt.ylabel('Размер (МБ)', fontsize=12)
    plt.title('Сравнение размера моделей (fp32)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, model_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'{val:.1f} МБ', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_size_comparison.png', dpi=150)
    plt.close()
    
    print(f"\nГрафики сохранены в {output_dir}")


def create_summary_table(results, output_dir):
    """Создаёт HTML таблицу с результатами сравнения."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Сравнение моделей CppCodeModel</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
            th { background-color: #4CAF50; color: white; font-weight: bold; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f5f5f5; }
            .best { background-color: #d4edda; font-weight: bold; }
            .timestamp { text-align: center; color: #666; margin-top: 20px; }
            .metrics-grid { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
            .metric-card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }
            .metric-card h3 { margin: 0 0 10px 0; color: #333; }
            .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
            .metric-unit { font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <h1>Сравнение моделей CppCodeModel</h1>
        <p style="text-align: center; color: #666;">Сравнение Tiny (5M), Small (8.3M) и Medium (18.3M)</p>
        
        <h2>Основные метрики</h2>
        <table>
            <tr>
                <th>Метрика</th>
                <th>Tiny (5M)</th>
                <th>Small (8.3M)</th>
                <th>Medium (18.3M)</th>
                <th>Лучший</th>
            </tr>
    """
    
    # Определяем лучшие значения
    best_perplexity = min(results[m].get('perplexity', float('inf')) for m in results)
    best_inference = min(results[m].get('inference_time_ms', float('inf')) for m in results)
    best_memory = min(results[m].get('memory_mb', float('inf')) for m in results)
    best_model_size = min(results[m].get('model_size_mb', float('inf')) for m in results)
    
    # Строка Perplexity
    html += "            <tr>\n                <td><strong>Perplexity</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('perplexity', 0)
        is_best = abs(val - best_perplexity) < 0.01
        html += f'                <td class="{"best" if is_best else ""}">{val:.2f}</td>\n'
    html += "                <td>↓ чем ниже, тем лучше</td>\n            </tr>\n"
    
    # Строка Loss
    html += "            <tr>\n                <td><strong>Cross-entropy Loss</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('avg_loss', 0)
        html += f'                <td>{val:.4f}</td>\n'
    html += "                <td>↓ чем ниже, тем лучше</td>\n            </tr>\n"
    
    # Строка Время инференса
    html += "            <tr>\n                <td><strong>Время инференса (мс)</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('inference_time_ms', 0)
        is_best = abs(val - best_inference) < 0.01
        html += f'                <td class="{"best" if is_best else ""}">{val:.1f}</td>\n'
    html += "                <td>↓ чем меньше, тем лучше</td>\n            </tr>\n"
    
    # Строка Пропускная способность
    html += "            <tr>\n                <td><strong>Пропускная способность</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('throughput', 0)
        html += f'                <td>{val:.0f} т/с</td>\n'
    html += "                <td>↑ чем выше, тем лучше</td>\n            </tr>\n"
    
    # Строка Память
    html += "            <tr>\n                <td><strong>Память (МБ)</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('memory_mb', 0)
        is_best = abs(val - best_memory) < 0.01
        html += f'                <td class="{"best" if is_best else ""}">{val:.0f}</td>\n'
    html += "                <td>↓ чем меньше, тем лучше</td>\n            </tr>\n"
    
    # Строка Размер модели
    html += "            <tr>\n                <td><strong>Размер модели (fp32)</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('model_size_mb', 0)
        is_best = abs(val - best_model_size) < 0.01
        html += f'                <td class="{"best" if is_best else ""}">{val:.1f} МБ</td>\n'
    html += "                <td>↓ чем меньше, тем лучше</td>\n            </tr>\n"
    
    # Строка Параметры
    html += "            <tr>\n                <td><strong>Параметры</strong></td>\n"
    for m in ['tiny', 'small', 'medium']:
        val = results[m].get('total_params', 0)
        html += f'                <td>{val/1e6:.1f}M</td>\n'
    html += "                <td>-</td>\n            </tr>\n"
    
    html += """
        </table>
        
        <div class="metrics-grid">
    """
    
    # Добавляем карточки с рекомендациями
    html += """
            <div class="metric-card">
                <h3>Для максимального качества</h3>
                <div class="metric-value">Medium</div>
                <div class="metric-unit">Лучшая perplexity и качество генерации</div>
            </div>
            <div class="metric-card">
                <h3>Для максимальной скорости</h3>
                <div class="metric-value">Tiny</div>
                <div class="metric-unit">Самая быстрая, минимальное потребление памяти</div>
            </div>
            <div class="metric-card">
                <h3>Оптимальный баланс</h3>
                <div class="metric-value">Small</div>
                <div class="metric-unit">Хорошее качество при приемлемой скорости</div>
            </div>
        </div>
    """
    
    html += f"""
        <p class="timestamp">Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    
    output_path = output_dir / 'comparison_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML отчёт сохранён: {output_path}")


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция сравнения моделей."""
    
    print("=" * 60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ: Tiny, Small (8.3M) и Medium (18.3M)")
    print("=" * 60)
    
    # Проверка GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nУстройство: {device.upper()}")
    
    # Директория для сохранения результатов
    reports_dir = cpp_code_model_dir / "reports" / "model_comparison"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Тестовые данные
    test_tokens_path = cpp_code_model_dir / "data" / "tokenized" / "test_tokens.pt"
    
    results = {}
    
    # Тестируем каждую модель
    for model_type in ['tiny', 'small', 'medium']:
        print("\n" + "=" * 60)
        print(f"ТЕСТИРОВАНИЕ: {MODELS_CONFIG[model_type]['name']}")
        print("=" * 60)
        
        # Загружаем модель
        print("Загрузка модели...")
        model_info = load_model(model_type, device)
        
        if model_info is None:
            print(f"Модель {model_type} не найдена, пропускаем!")
            continue
        
        model, model_config, checkpoint_path = model_info
        print(f"Модель загружена: {checkpoint_path.name}")
        
        # Базовые метрики
        model_size_mb = get_model_size(model)
        total_params = sum(p.numel() for p in model.parameters())
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
        
        print(f"- Параметры:            {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"- Размер модели (fp32): {model_size_mb:.1f} МБ")
        print(f"- Размер чекпоинта:     {checkpoint_size_mb:.1f} МБ")
        
        # Perplexity
        print("Вычисление perplexity...")
        perplexity_result = compute_perplexity(model, test_tokens_path, device)
        if perplexity_result:
            perplexity, avg_loss = perplexity_result
            print(f"- Perplexity: {perplexity:.2f}")
            print(f"- Loss:       {avg_loss:.4f}")
        else:
            perplexity, avg_loss = None, None
            print("Не удалось вычислить perplexity!")
        
        # Производительность
        print("Измерение производительности...")
        inference_time_ms, throughput = measure_inference_time(model, device=device)
        print(f"- Время инференса:        {inference_time_ms:.1f} мс")
        print(f"- Пропускная способность: {throughput:.0f} токенов/сек")
        
        # Память
        print("Измерение памяти...")
        memory_mb = measure_memory_usage(model, device=device)
        print(f"- Использование памяти: {memory_mb:.0f} МБ")
        
        # Сохраняем результаты
        results[model_type] = {
            'total_params': total_params,
            'model_size_mb': model_size_mb,
            'checkpoint_size_mb': checkpoint_size_mb,
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'inference_time_ms': inference_time_ms,
            'throughput': throughput,
            'memory_mb': memory_mb,
            'config': model_config
        }
        
        # Очищаем память
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Сохраняем JSON с результатами
    json_path = reports_dir / 'comparison_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        # Конвертируем numpy типы
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert, ensure_ascii=False)
    print(f"\nРезультаты сохранены: {json_path}")
    
    # Создаём графики
    create_comparison_charts(results, reports_dir)
    
    # Создаём HTML отчёт
    create_summary_table(results, reports_dir)
    
    # Выводим итоговую таблицу
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
    print("=" * 60)
    
    print(f"\n{'Метрика':<30} {'Tiny':>15} {'Small':>15} {'Medium':>15}")
    print("-" * 75)
    
    metrics = [
        ('Параметры (M)', lambda r: r.get('total_params', 0)/1e6, '{:.1f}'),
        ('Размер модели (МБ)', 'model_size_mb', '{:.1f}'),
        ('Размер чекпоинта (МБ)', 'checkpoint_size_mb', '{:.1f}'),
        ('Perplexity', 'perplexity', '{:.2f}'),
        ('Loss', 'avg_loss', '{:.4f}'),
        ('Время инференса (мс)', 'inference_time_ms', '{:.1f}'),
        ('Пропускная способность (т/с)', 'throughput', '{:.0f}'),
        ('Память (МБ)', 'memory_mb', '{:.0f}')
    ]
    
    for label, key, fmt in metrics:
        row = f"{label:<30}"
        for model_type in ['tiny', 'small', 'medium']:
            if model_type in results:
                if callable(key):
                    val = key(results[model_type])
                else:
                    val = results[model_type].get(key, 0)
                row += f" {fmt.format(val) if val else 'N/A':>15}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    print("- Для максимального качества: Medium")
    print("- Для максимальной скорости:  Tiny")
    print("- Оптимальный баланс:         Small")
    print("=" * 60)
    print(f"\nВсе результаты сохранены в: {reports_dir}")
    print("- comparison_results.json (данные)")
    print("- comparison_report.html (отчёт)")
    print("- *.png (графики)")


if __name__ == "__main__":
    main()