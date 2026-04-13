#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# test_lora_model.py - Тестирование метрик LoRA модели (Medium + LoRA)
# ======================================================================
#
# @file test_lora_model.py
# @brief Комплексное тестирование метрик LoRA модели: Perplexity, Loss, производительность, память
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль выполняет всестороннее тестирование LoRA модели
#          (Medium + LoRA на инструкциях). Результаты сохраняются в reports/test_lora_medium/
#
#          **Основные тесты:**
#
#          1. **Базовые метрики модели**
#             - Количество параметров (обучаемых и всего)
#             - Размер модели в памяти
#             - Размер чекпоинта базовой модели и LoRA весов
#             - Архитектурные параметры
#
#          2. **Численная стабильность**
#             - Проверка на NaN значения
#             - Проверка на Inf значения
#             - Анализ максимальных logit значений
#
#          3. **Perplexity (качество модели)**
#             - Вычисление perplexity на тестовых данных
#             - Средняя потеря (cross-entropy loss)
#             - Статистика распределения loss
#
#          4. **Производительность**
#             - Время инференса для разных длин последовательностей
#             - Пропускная способность (токенов/сек)
#             - Визуализация графиков
#
#          5. **Использование памяти**
#             - Потребление памяти для разных batch size
#             - Поддержка CPU и GPU
#             - Визуализация результатов
#
# @usage
#     python tests/test_lora_model.py
#     python tests/test_lora_model.py --epoch 7
#
# @example
#     # Стандартный запуск (автоматический поиск)
#     python test_lora_model.py
#
#     # Тестирование конкретной эпохи
#     python test_lora_model.py --epoch 7
#
# ======================================================================

import os
import sys

from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Определяем пути автоматически
current_file = Path(__file__).resolve()
tests_dir = current_file.parent
cpp_code_model_dir = tests_dir.parent
sys.path.insert(0, str(cpp_code_model_dir))

# Устанавливаем рабочую директорию в корень проекта
os.chdir(cpp_code_model_dir)

print(f"Рабочая директория: {os.getcwd()}")
print(f"Директория модели:  {cpp_code_model_dir}")
# ======================================================================

import re
import argparse
import json
import time
import psutil
import gc
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from model.architecture.model import CppCodeModel

matplotlib.use('Agg')

# ======================================================================
# ПУТИ ДЛЯ LoRA МОДЕЛИ (MEDIUM + LoRA)
# ======================================================================

# Базовые пути
PROJECT_ROOT = Path(__file__).parent.parent

# Пути для Medium + LoRA
BASE_MODEL_PATH = PROJECT_ROOT / "checkpoints/medium/cpp-code-epoch=19-val_loss=0.87.ckpt"
CONFIG_PATH = PROJECT_ROOT / "configs/medium.json"
LORA_DIR = PROJECT_ROOT / "checkpoints/lora_medium"

# Параметры LoRA (должны совпадать с обучением)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1

# ======================================================================
# КЛАСС LoRALinear (ДЛЯ ЗАГРУЗКИ ВЕСОВ)
# ======================================================================

class LoRALinear(torch.nn.Module):
    """LoRA слой для линейной проекции (для загрузки весов)."""
    
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
        
        self.scaling = alpha / r
    
    def forward(self, x):
        result = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        if self.dropout:
            lora_out = self.dropout(lora_out)
        return result + self.scaling * lora_out


def replace_with_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT):
    """
    Рекурсивная замена целевых Linear слоёв на LoRALinear.
    Заменяем только attention слои (w_q, w_k, w_v, w_o)
    """
    replaced_count = 0
    target_names = ['w_q', 'w_k', 'w_v', 'w_o']
    
    for name, module in model.named_children():
        should_replace = False
        for target in target_names:
            if target in name:
                should_replace = True
                break
        
        if should_replace and isinstance(module, torch.nn.Linear):
            lora_layer = LoRALinear(module, r, alpha, dropout)
            setattr(model, name, lora_layer)
            replaced_count += 1
        else:
            replaced_count += replace_with_lora(module, r, alpha, dropout)
    
    return replaced_count


def load_lora_weights(model, lora_path):
    """Загружает LoRA веса в модель."""
    if not lora_path.exists():
        print(f"LoRA файл не найден: {lora_path}!")
        return False
    
    lora_state = torch.load(lora_path, map_location='cpu')
    model_state = model.state_dict()
    
    loaded = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].copy_(value)
            loaded += 1
    
    print(f"Загружено {loaded} LoRA весов из {lora_path.name}")
    return loaded > 0

# ======================================================================
# КЛАСС ModelMetricsTester
# ======================================================================

class ModelMetricsTester:
    """
    Класс для комплексного тестирования метрик LoRA модели.
    
    Args:
        base_model_path (str): Путь к базовой модели Medium
        lora_path (str):       Путь к LoRA весам
        config_path (str):     Путь к конфигу
        device (str):          Устройство для выполнения тестов ('cpu' или 'cuda')
    """
    
    def __init__(self, base_model_path, lora_path, config_path, device='cpu'):
        self.device = torch.device(device)
        self.base_model_path = Path(base_model_path)
        self.lora_path = Path(lora_path)
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.results = {}
        
        # Директория для сохранения результатов
        self.reports_dir = cpp_code_model_dir / "reports" / "test_lora_medium"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nРезультаты будут сохранены в: {self.reports_dir}")
    
    def _load_config(self, config_path):
        """Загрузка конфигурации из JSON файла."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model(self):
        """
        Загрузка базовой модели Medium с LoRA весами.
        
        Returns:
            CppCodeModel: Загруженная модель в режиме eval()
        """
        print(f"Загрузка базовой модели Medium из {self.base_model_path}")
        
        # Создаём базовую модель
        model = CppCodeModel(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            max_len=self.config.get('max_len', 512),
            dropout=self.config.get('dropout', 0.15)
        )
        
        # Загружаем базовые веса
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # Убираем префиксы
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        
        # Замораживаем все параметры
        for param in model.parameters():
            param.requires_grad = False
        
        # Добавляем LoRA слои (только attention)
        print("Добавление LoRA слоёв (только attention)...")
        replaced_count = replace_with_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
        print(f"Заменено {replaced_count} Linear слоёв на LoRALinear")
        
        # Загружаем LoRA веса
        if not load_lora_weights(model, self.lora_path):
            print("LoRA веса не загружены, используется базовая модель!")
        
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Модель загружена: {total_params:,} параметров")
        print(f"Обучаемых (LoRA): {trainable_params:,}")
        
        return model
    
    def test_basic_metrics(self):
        """Тест 1: Базовые метрики модели (размер, параметры, архитектура)."""
        print("\n" + "=" * 60)
        print("ТЕСТ 1: Базовые метрики LoRA модели")
        print("=" * 60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # 4 байта на параметр для float32
        param_size_mb = total_params * 4 / (1024 ** 2)
        
        # Размер LoRA весов
        lora_size_mb = self.lora_path.stat().st_size / (1024 ** 2) if self.lora_path.exists() else 0
        
        metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_memory_mb': param_size_mb,
            'base_checkpoint_size_mb': self.base_model_path.stat().st_size / (1024 ** 2),
            'lora_weights_size_mb': lora_size_mb,
            'vocab_size': self.config['vocab_size'],
            'd_model': self.config['d_model'],
            'nhead': self.config['nhead'],
            'num_layers': self.config['num_layers'],
            'max_len': self.config.get('max_len', 512),
            'dropout': self.config.get('dropout', 0.15)
        }
        
        self.results['basic_metrics'] = metrics
        
        print(f"\nПараметры модели:")
        print(f"- Всего параметров:       {total_params:,}")
        print(f"- Обучаемых (LoRA):       {trainable_params:,}")
        print(f"- Размер в памяти (fp32): {param_size_mb:.1f} МБ")
        print(f"- Размер базовой модели:  {metrics['base_checkpoint_size_mb']:.1f} МБ")
        print(f"- Размер LoRA весов:      {metrics['lora_weights_size_mb']:.2f} МБ")
        print(f"- Архитектура:            {self.config['num_layers']} слоев, "
              f"{self.config['d_model']} dim, {self.config['nhead']} heads")
        
        return metrics
    
    def test_numerical_stability(self):
        """Тест 2: Численная стабильность (NaN, Inf, max logit)."""
        print("\n" + "=" * 60)
        print("ТЕСТ 2: Численная стабильность")
        print("=" * 60)
        
        test_input = torch.randint(0, self.config['vocab_size'],
                                   (1, 100), device=self.device)
        
        with torch.no_grad():
            output = self.model(test_input)
        
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        max_logit = output.abs().max().item()
        
        stability = {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'max_logit': max_logit,
            'is_stable': not (has_nan or has_inf)
        }
        
        self.results['numerical_stability'] = stability
        
        print(f"\nРезультаты:")
        print(f"- NaN значения:       {'Есть' if has_nan else 'Нет'}")
        print(f"- Inf значения:       {'Есть' if has_inf else 'Нет'}")
        print(f"- Максимальный logit: {max_logit:.2f}")
        print(f"- Стабильность:       {'Стабильна' if stability['is_stable'] else 'Нестабильна'}")
        
        return stability
    
    def test_perplexity(self, test_tokens_path=None, max_samples=500):
        """
        Тест 3: Perplexity на тестовых данных.
        
        Args:
            test_tokens_path (str, optional): Путь к файлу test_tokens.pt
            max_samples (int):                Максимальное количество тестовых примеров
        """
        print("\n" + "=" * 60)
        print("ТЕСТ 3: Perplexity (LoRA модель)")
        print("=" * 60)
        
        if test_tokens_path is None:
            test_tokens_path = cpp_code_model_dir / "data" / "tokenized" / "test_tokens.pt"
        
        if not os.path.exists(test_tokens_path):
            print(f"Файл не найден: {test_tokens_path}!")
            print("Пропускаем тест perplexity")
            return None
        
        test_tokens = torch.load(test_tokens_path)
        print(f"Загружено {len(test_tokens)} примеров")
        
        total_loss = 0
        total_tokens = 0
        losses = []
        
        with torch.no_grad():
            for i, seq in enumerate(test_tokens[:max_samples]):
                if len(seq) < 2:
                    continue
                
                # Обрезаем до разумной длины для скорости
                if len(seq) > 512:
                    seq = seq[:512]
                
                input_ids = seq[:-1].unsqueeze(0).to(self.device)
                target_ids = seq[1:].unsqueeze(0).to(self.device)
                
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=0
                )
                
                total_loss += loss.item() * target_ids.numel()
                total_tokens += target_ids.numel()
                losses.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print(f"Обработано {i + 1} примеров...")
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        perplexity_results = {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'loss_std': np.std(losses),
            'loss_min': np.min(losses),
            'loss_max': np.max(losses),
            'total_tokens': total_tokens,
            'num_samples': len(losses)
        }
        
        self.results['perplexity'] = perplexity_results
        
        print(f"\nРезультаты perplexity:")
        print(f"- Perplexity:             {perplexity:.2f}")
        print(f"- Средняя потеря:         {avg_loss:.4f}")
        print(f"- Стандартное отклонение: {np.std(losses):.4f}")
        print(f"- Min loss:               {np.min(losses):.4f}")
        print(f"- Max loss:               {np.max(losses):.4f}")
        print(f"- Всего токенов:          {total_tokens:,}")
        
        return perplexity_results
    
    def test_performance(self):
        """Тест 4: Производительность инференса (время, пропускная способность)."""
        print("\n" + "=" * 60)
        print("ТЕСТ 4: Производительность (LoRA модель)")
        print("=" * 60)
        
        seq_lengths = [64, 128, 256, 512]
        performance_results = []
        
        for seq_len in seq_lengths:
            print(f"\nДлина последовательности: {seq_len}")
            
            dummy_input = torch.randint(0, self.config['vocab_size'],
                                        (1, seq_len), device=self.device)
            
            # Прогрев (компиляция CUDA ядер, если нужно)
            for _ in range(10):
                _ = self.model(dummy_input)
            
            # Измерение
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            num_iterations = 100
            
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            avg_time = (time.time() - start_time) / num_iterations
            throughput = seq_len / avg_time
            
            print(f"- Время:                  {avg_time * 1000:.2f} мс")
            print(f"- Пропускная способность: {throughput:.0f} токенов/сек")
            
            performance_results.append({
                'seq_len': seq_len,
                'time_ms': avg_time * 1000,
                'throughput': throughput
            })
        
        self.results['performance'] = performance_results
        
        # Визуализация
        self._plot_performance(performance_results)
        
        return performance_results
    
    def _plot_performance(self, performance_results):
        """Визуализация производительности (графики времени и пропускной способности)."""
        seq_lens = [p['seq_len'] for p in performance_results]
        times = [p['time_ms'] for p in performance_results]
        throughput = [p['throughput'] for p in performance_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(seq_lens, times, marker='o', linewidth=2)
        ax1.set_xlabel('Длина последовательности')
        ax1.set_ylabel('Время (мс)')
        ax1.set_title('Время инференса (LoRA)')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(seq_lens, throughput, marker='s', linewidth=2, color='orange')
        ax2.set_xlabel('Длина последовательности')
        ax2.set_ylabel('Токенов/сек')
        ax2.set_title('Пропускная способность (LoRA)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'performance_plot.png', dpi=150)
        plt.close()
        print(f"\nГрафик сохранен: {self.reports_dir / 'performance_plot.png'}")
    
    def test_memory_usage(self):
        """Тест 5: Использование памяти при инференсе для разных batch size."""
        print("\n" + "=" * 60)
        print("ТЕСТ 5: Использование памяти (LoRA модель)")
        print("=" * 60)
        
        batch_sizes = [1, 2, 4, 8]
        memory_results = []
        
        for bs in batch_sizes:
            try:
                seq_len = 512
                dummy_input = torch.randint(0, self.config['vocab_size'],
                                            (bs, seq_len), device=self.device)
                
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    _ = self.model(dummy_input)
                    memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    gc.collect()
                    process = psutil.Process()
                    before_mem = process.memory_info().rss / (1024 ** 2)
                    _ = self.model(dummy_input)
                    after_mem = process.memory_info().rss / (1024 ** 2)
                    memory = after_mem - before_mem
                
                memory_results.append({'batch_size': bs, 'memory_mb': memory})
                print(f"Batch size {bs}: {memory:.1f} МБ")
                
            except Exception as e:
                print(f"Batch size {bs}: ошибка - {e}")
                break
        
        self.results['memory_usage'] = memory_results
        
        # Визуализация
        self._plot_memory(memory_results)
        
        return memory_results
    
    def _plot_memory(self, memory_results):
        """Визуализация использования памяти (столбчатая диаграмма)."""
        batch_sizes = [m['batch_size'] for m in memory_results]
        memories = [m['memory_mb'] for m in memory_results]
        
        plt.figure(figsize=(8, 5))
        plt.bar(batch_sizes, memories, color='green', alpha=0.7)
        plt.xlabel('Batch size')
        plt.ylabel('Память (МБ)')
        plt.title('Использование памяти при инференсе (LoRA)')
        plt.grid(True, alpha=0.3)
        
        for i, (bs, mem) in enumerate(zip(batch_sizes, memories)):
            plt.text(bs, mem + 10, f'{mem:.0f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'memory_usage.png', dpi=150)
        plt.close()
        print(f"\nГрафик памяти сохранен: {self.reports_dir / 'memory_usage.png'}")
    
    def generate_report(self):
        """Генерация итогового отчёта в текстовом и JSON форматах."""
        print("\n" + "=" * 60)
        print("ГЕНЕРАЦИЯ ИТОГОВОГО ОТЧЕТА")
        print("=" * 60)
        
        # Формируем отчёт
        report = f"""
============================================================
 ОТЧЕТ О ТЕСТИРОВАНИИ LoRA МОДЕЛИ
 Medium + LoRA (инструкции)
============================================================

ОСНОВНЫЕ ХАРАКТЕРИСТИКИ МОДЕЛИ
------------------------------------------------------------
- Название:               {self.config.get('name', 'CppCodeModel-Medium')} + LoRA
- Всего параметров:       {self.results['basic_metrics']['total_parameters']:,}
- Обучаемых (LoRA):       {self.results['basic_metrics']['trainable_parameters']:,}
- Размер в памяти (fp32): {self.results['basic_metrics']['model_size_memory_mb']:.1f} МБ
- Размер базовой модели:  {self.results['basic_metrics']['base_checkpoint_size_mb']:.1f} МБ
- Размер LoRA весов:      {self.results['basic_metrics']['lora_weights_size_mb']:.2f} МБ
- Архитектура:            {self.config['num_layers']} слоев, {self.config['d_model']} dim, {self.config['nhead']} heads

КАЧЕСТВО МОДЕЛИ (на тестовых данных)
------------------------------------------------------------
- Perplexity:             {self.results.get('perplexity', {}).get('perplexity', 'N/A'):.2f}
- Средняя потеря:         {self.results.get('perplexity', {}).get('avg_loss', 'N/A'):.4f}
- Стандартное отклонение: {self.results.get('perplexity', {}).get('loss_std', 'N/A'):.4f}
- Min/Max loss:           {self.results.get('perplexity', {}).get('loss_min', 'N/A'):.4f} / {self.results.get('perplexity', {}).get('loss_max', 'N/A'):.4f}

ЧИСЛЕННАЯ СТАБИЛЬНОСТЬ
------------------------------------------------------------
• NaN значения:       {'Нет' if not self.results['numerical_stability']['has_nan'] else 'Есть'}
• Inf значения:       {'Нет' if not self.results['numerical_stability']['has_inf'] else 'Есть'}
• Максимальный logit: {self.results['numerical_stability']['max_logit']:.2f}
• Статус:             {'Стабильна' if self.results['numerical_stability']['is_stable'] else 'Нестабильна'}

ПРОИЗВОДИТЕЛЬНОСТЬ ({"GPU" if self.device.type == 'cuda' else "CPU"})
------------------------------------------------------------
"""
        
        # Добавляем данные о производительности
        if self.results.get('performance'):
            avg_throughput = np.mean([p['throughput'] for p in self.results['performance']])
            report += f"""
- Средняя пропускная способность: {avg_throughput:.0f} токенов/сек
"""
            for perf in self.results['performance']:
                report += f"  - seq_len={perf['seq_len']}: {perf['time_ms']:.2f} мс, {perf['throughput']:.0f} токенов/сек\n"
        
        # Добавляем данные о памяти
        if self.results.get('memory_usage'):
            report += f"""
ИСПОЛЬЗОВАНИЕ ПАМЯТИ
------------------------------------------------------------
"""
            for mem in self.results['memory_usage']:
                report += f"- Batch size {mem['batch_size']}: {mem['memory_mb']:.1f} МБ\n"
        
        report += f"""
СОХРАНЕННЫЕ ФАЙЛЫ
------------------------------------------------------------
- performance_plot.png - График производительности
- memory_usage.png     - График использования памяти
- metrics_report.json  - Полные метрики в JSON

============================================================
Дата тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Устройство: {self.device}
"""
        
        # Сохраняем отчёт
        report_path = self.reports_dir / 'METRICS_REPORT.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Сохраняем JSON с полными результатами
        json_path = self.reports_dir / 'metrics_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nОтчет сохранен: {report_path}")
        print(f"JSON сохранен:  {json_path}")
        print("\n" + report)


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция запуска тестирования."""
    parser = argparse.ArgumentParser(description='Тестирование LoRA модели')
    parser.add_argument('--epoch', type=int, default=None, 
                       help='Номер эпохи для тестирования (например, 7)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Устройство для тестов')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ LoRA МОДЕЛИ (Medium + инструкции)")
    print("=" * 60)
    
    # Проверка наличия файлов
    if not BASE_MODEL_PATH.exists():
        print(f"Базовая модель не найдена: {BASE_MODEL_PATH}!")
        print("Убедитесь, что модель Medium обучена")
        return
    
    if not CONFIG_PATH.exists():
        print(f"Конфиг не найден: {CONFIG_PATH}!")
        return
    
    # Поиск LoRA весов
    lora_path = None
    
    if args.epoch is not None:
        # Тестируем конкретную эпоху
        lora_path = LORA_DIR / f"lora_weights_epoch_{args.epoch}.pt"
        if lora_path.exists():
            print(f"Тестирование эпохи {args.epoch}: {lora_path}")
        else:
            print(f"Чекпоинт для эпохи {args.epoch} не найден!")
            return
    else:
        # Автоматический поиск лучшей эпохи
        checkpoint_files = list(LORA_DIR.glob("lora_weights_epoch_*.pt"))
        if checkpoint_files:
            def get_epoch_num(filename):
                match = re.search(r'epoch_(\d+)', str(filename))
                return int(match.group(1)) if match else 0
            checkpoint_files.sort(key=get_epoch_num)
            lora_path = checkpoint_files[-1]
            epoch_num = get_epoch_num(lora_path)
            print(f"Найдены LoRA веса: эпоха {epoch_num}")
        else:
            # Проверяем lora_weights_best.pt
            best_path = LORA_DIR / "lora_weights_best.pt"
            if best_path.exists():
                lora_path = best_path
                print(f"Найдены LoRA веса: {best_path.name}")
            else:
                print(f"LoRA веса не найдены в {LORA_DIR}!")
                print("Сначала запусти train_finetune_lora.py")
                return
    
    try:
        # Создание тестера
        tester = ModelMetricsTester(BASE_MODEL_PATH, lora_path, CONFIG_PATH, device=args.device)
        
        # Запуск всех тестов
        tester.test_basic_metrics()
        tester.test_numerical_stability()
        tester.test_perplexity(max_samples=500)
        tester.test_performance()
        tester.test_memory_usage()
        
        # Генерация отчёта
        tester.generate_report()
        
        print("\nТестирование LoRA модели завершено!")
        
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}!")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()