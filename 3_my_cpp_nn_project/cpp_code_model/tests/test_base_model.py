#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# test_base_model.py - Тестирование метрик базовой модели Tiny/Small/Medium
# ======================================================================
#
# @file test_base_model.py
# @brief Комплексное тестирование метрик базовых моделей: Perplexity, Loss, производительность, память
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль выполняет тестирование обученной базовой модели.
#          Результаты сохраняются в reports/test_.../
#          где ... - модели tiny, small или medium (меняется в коде в ручную)
#
#          **Основные тесты:**
#
#          1. **Базовые метрики модели**
#             - Количество параметров (обучаемых и всего)
#             - Размер модели в памяти
#             - Размер чекпоинта на диске
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
#     python tests/test_base_model.py
#
# @example
#     # Стандартный запуск (автоматический поиск модели)
#     python test_base_model.py
#
#     # С указанием конкретной модели
#     python test_base_model.py --model path/to/model.ckpt --config path/to/config.json
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
print(f"Директория модели: {cpp_code_model_dir}")
# ======================================================================

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
# КЛАСС ModelMetricsTester
# ======================================================================

class ModelMetricsTester:
    """
    Класс для комплексного тестирования метрик модели.
    
    Args:
        model_path (str, optional):  Путь к файлу чекпоинта.
                                     Если не указан, ищется автоматически.
        config_path (str, optional): Путь к файлу конфигурации.
                                     Если не указан, ищется автоматически.
        device (str):                Устройство для выполнения тестов ('cpu' или 'cuda').
    
    **Автоматический поиск:**
    - Чекпоинт ищется в checkpoints/medium/*.ckpt
    - Конфиг ищется в configs/medium.json
    - Выбирается последний чекпоинт (с наибольшим номером эпохи)
    """
    
    def __init__(self, model_path=None, config_path=None, device='cpu'):
        self.device = torch.device(device)
        
        # Автоматический поиск путей, если не указаны
        if model_path is None:
            model_path = self._find_checkpoint()
        if config_path is None:
            config_path = self._find_config()
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = self._load_model()
        self.results = {}
        
        # Директория для сохранения результатов
        self.reports_dir = cpp_code_model_dir / "reports" / "test_medium"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nРезультаты будут сохранены в: {self.reports_dir}")
    
    def _find_checkpoint(self):
        """
        Автоматический поиск чекпоинта модели.
        
        Returns:
            str: Путь к найденному чекпоинту
            
        Raises:
            FileNotFoundError: Если чекпоинт не найден
        
        **Алгоритм поиска:**
        1. Поиск в checkpoints/medium/*.ckpt
        2. Выбор чекпоинта с наибольшим номером эпохи
        3. Если не найден - проверка альтернативных путей
        """
        checkpoints_dir = cpp_code_model_dir / "checkpoints" / "medium"
        
        if checkpoints_dir.exists():
            ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
            # Исключаем last.ckpt, выбираем лучший по эпохе
            epoch_files = [f for f in ckpt_files if 'epoch' in f.name]
            if epoch_files:
                # Сортируем по номеру эпохи
                def get_epoch(filename):
                    import re
                    match = re.search(r'epoch=(\d+)', str(filename))
                    return int(match.group(1)) if match else 0
                
                epoch_files.sort(key=get_epoch)
                return str(epoch_files[-1])    # Последний = наибольшая эпоха
        
        raise FileNotFoundError(f"Чекпоинт не найден. Искал в: {checkpoints_dir}!")
    
    def _find_config(self):
        """
        Автоматический поиск файла конфигурации.
        
        Returns:
            str: Путь к файлу конфигурации
            
        Raises:
            FileNotFoundError: Если конфиг не найден
        """
        config_path = cpp_code_model_dir / "configs" / "medium.json"
        if config_path.exists():
            return str(config_path)
        
        raise FileNotFoundError(f"Конфиг не найден: {config_path}")
    
    def _load_config(self):
        """Загрузка конфигурации из JSON файла."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model(self):
        """
        Загрузка модели из чекпоинта.
        
        Returns:
            CppCodeModel: Загруженная модель в режиме eval()
        
        **Особенности:**
        - strict=False для совместимости версий
        - Автоматическое удаление префикса 'model.'
        - Перенос на указанное устройство
        """
        print(f"Загрузка модели из {self.model_path}")
        
        model = CppCodeModel(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            num_layers=self.config['num_layers'],
            max_len=self.config.get('max_len', 1024),
            dropout=self.config.get('dropout', 0.1)
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Убираем префиксы (например, 'model.' из сохранённого state_dict)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        # strict=False позволяет игнорировать несовпадающие ключи
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Пропущенные ключи: {len(missing)}!")
        if unexpected:
            print(f"Лишние ключи: {len(unexpected)}!")
        
        model = model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Модель загружена: {total_params:,} параметров")
        
        return model
    
    def test_basic_metrics(self):
        """Тест 1: Базовые метрики модели (размер, параметры, архитектура)."""
        print("\n" + "=" * 60)
        print("ТЕСТ 1: Базовые метрики модели")
        print("=" * 60)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # 4 байта на параметр для float32
        param_size_mb = total_params * 4 / (1024 ** 2)
        
        metrics = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_memory_mb': param_size_mb,
            'checkpoint_size_mb': self.model_path.stat().st_size / (1024 ** 2),
            'vocab_size': self.config['vocab_size'],
            'd_model': self.config['d_model'],
            'nhead': self.config['nhead'],
            'num_layers': self.config['num_layers'],
            'max_len': self.config.get('max_len', 1024),
            'dropout': self.config.get('dropout', 0.1)
        }
        
        self.results['basic_metrics'] = metrics
        
        print(f"\nПараметры модели:")
        print(f"- Всего параметров:       {total_params:,}")
        print(f"- Обучаемых:              {trainable_params:,}")
        print(f"- Размер в памяти (fp32): {param_size_mb:.1f} МБ")
        print(f"- Размер чекпоинта:       {metrics['checkpoint_size_mb']:.1f} МБ")
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
        print("ТЕСТ 3: Perplexity")
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
        print("ТЕСТ 4: Производительность")
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
        ax1.set_title('Время инференса')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(seq_lens, throughput, marker='s', linewidth=2, color='orange')
        ax2.set_xlabel('Длина последовательности')
        ax2.set_ylabel('Токенов/сек')
        ax2.set_title('Пропускная способность')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'performance_plot.png', dpi=150)
        plt.close()
        print(f"\nГрафик сохранен: {self.reports_dir / 'performance_plot.png'}")
    
    def test_memory_usage(self):
        """Тест 5: Использование памяти при инференсе для разных batch size."""
        print("\n" + "=" * 60)
        print("ТЕСТ 5: Использование памяти")
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
        plt.title('Использование памяти при инференсе')
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
 ОТЧЕТ О ТЕСТИРОВАНИИ МЕТРИК МОДЕЛИ
============================================================

ОСНОВНЫЕ ХАРАКТЕРИСТИКИ МОДЕЛИ
------------------------------------------------------------
- Название:               {self.config.get('name', 'CppCodeModel-Medium')}
- Параметры:              {self.results['basic_metrics']['total_parameters']:,}
- Размер в памяти (fp32): {self.results['basic_metrics']['model_size_memory_mb']:.1f} МБ
- Размер чекпоинта:       {self.results['basic_metrics']['checkpoint_size_mb']:.1f} МБ
- Архитектура:            {self.config['num_layers']} слоев, {self.config['d_model']} dim, {self.config['nhead']} heads

КАЧЕСТВО МОДЕЛИ
------------------------------------------------------------
- Perplexity:             {self.results.get('perplexity', {}).get('perplexity', 'N/A'):.2f}
- Средняя потеря:         {self.results.get('perplexity', {}).get('avg_loss', 'N/A'):.4f}
- Стандартное отклонение: {self.results.get('perplexity', {}).get('loss_std', 'N/A'):.4f}
- Min/Max loss:           {self.results.get('perplexity', {}).get('loss_min', 'N/A'):.4f} / {self.results.get('perplexity', {}).get('loss_max', 'N/A'):.4f}

ЧИСЛЕННАЯ СТАБИЛЬНОСТЬ
------------------------------------------------------------
- NaN значения:       {'Нет' if not self.results['numerical_stability']['has_nan'] else 'Есть'}
- Inf значения:       {'Нет' if not self.results['numerical_stability']['has_inf'] else 'Есть'}
- Максимальный logit: {self.results['numerical_stability']['max_logit']:.2f}
- Статус:             {'Стабильна' if self.results['numerical_stability']['is_stable'] else 'Нестабильна'}

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
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ МЕТРИК МОДЕЛИ")
    print("=" * 60)
    
    try:
        # Создание тестера (автоматически найдёт модель и конфиг)
        # Для GPU используйте device='cuda'
        tester = ModelMetricsTester(device='cpu')
        
        # Запуск всех тестов
        tester.test_basic_metrics()
        tester.test_numerical_stability()
        tester.test_perplexity(max_samples=500)
        tester.test_performance()
        tester.test_memory_usage()
        
        # Генерация отчёта
        tester.generate_report()
        
        print("\nТестирование метрик завершено!")
        
    except FileNotFoundError as e:
        print(f"\nОшибка: {e}!")
        print("\nУбедитесь, что:")
        print("1. Модель находится в checkpoints/")
        print("2. Файл конфигурации находится в configs/")
        print("3. Вы запускаете скрипт из директории cpp_code_model/scripts/")
        print(f"\nТекущая директория: {os.getcwd()}")
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}!")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()