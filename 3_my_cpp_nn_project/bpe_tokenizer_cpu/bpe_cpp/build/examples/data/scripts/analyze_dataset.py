#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# analyze_dataset.py - Анализ и визуализация датасета C++ кода
# ======================================================================
#
# @file analyze_dataset.py
# @brief Анализ и визуализация датасета C++ кода для BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Выполняет комплексный анализ датасета C++ кода:
#          - Статистика длины программ
#          - Частотность символов
#          - Распределение train/val/test
#          - Визуализация (гистограммы, boxplot, violin plot)
#          - Облако ключевых слов C++
#          - Генерация Markdown отчета
#
# @usage python analyze_dataset.py
#
# @requirements
#   pip install matplotlib seaborn numpy wordcloud scikit-learn psutil
#
# @example
#   python analyze_dataset.py
#   # После выполнения в data/reports/ появятся:
#   # - figures/*.png (графики)
#   # - dataset_report.md (отчет)
#
# ======================================================================

import json
import subprocess
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ======================================================================
# НАСТРОЙКА СТИЛЕЙ
# ======================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела.
    
    Args:
        title: Заголовок
        width: Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


def get_project_root() -> Path:
    """
    Получить корневую директорию проекта.
    
    Returns:
        Path: Путь к корню проекта (bpe_tokenizer_cpu/)
    """
    current_file = Path(__file__).resolve()    # .../bpe_tokenizer_cpu/data/scripts/analyze_dataset.py
    scripts_dir = current_file.parent          # .../bpe_tokenizer_cpu/data/scripts/
    data_dir = scripts_dir.parent              # .../bpe_tokenizer_cpu/data/
    project_root = data_dir.parent             # .../bpe_tokenizer_cpu/
    
    return project_root

# ======================================================================
# ОСНОВНОЙ КЛАСС
# ======================================================================

class CppDatasetAnalyzer:
    """
    Класс для анализа датасета C++ кода.
    
    Выполняет загрузку данных, расчет статистики, генерацию графиков
    и создание отчета в формате Markdown.
    """
    
    def __init__(self, project_root: str = "bpe_tokenizer_cpu", plot_max_length: int = 15000):
        """
        Инициализация анализатора датасета.
        
        Args:
            project_root:    Корневая директория проекта (bpe_tokenizer_cpu/)
            plot_max_length: Максимальная длина для отображения на графиках
        """
        self.root = Path(project_root)
        self.plot_max_length = plot_max_length

        # ======================================================================
        # Пути
        # ======================================================================
        
        # Директории
        self.data_dir = self.root / 'data'                 # data/
        self.corpus_dir = self.data_dir / 'corpus'         # data/corpus/
        self.metadata_dir = self.data_dir / 'metadata'     # data/metadata/
        self.reports_dir = self.data_dir / 'reports'       # data/reports/
        self.figures_dir = self.reports_dir / 'figures'    # data/reports/figures/
        
        # Создаем директории
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Файлы
        self.corpus_file = self.corpus_dir / 'corpus.txt'             # data/corpus/corpus.txt
        self.train_file = self.corpus_dir / 'train_code.txt'          # data/corpus/train_code.txt
        self.val_file = self.corpus_dir / 'val_code.txt'              # data/corpus/val_code.txt
        self.test_file = self.corpus_dir / 'test_code.txt'            # data/corpus/test_code.txt
        self.stats_file = self.metadata_dir / 'dataset_stats.json'    # data/metadata/dataset_stats.json
        self.report_file = self.reports_dir / 'dataset_report.md'     # data/reports/dataset_report.md
        
        # Данные
        self.codes: List[str] = []
        self.stats: Dict[str, Any] = {}
        
        print(f"Корневая директория: {self.root}")
        print(f"Директория данных:   {self.data_dir}")
        print(f"Директория корпуса:  {self.corpus_dir}")
        print(f"Директория отчетов:  {self.reports_dir}")
    
    # ======================================================================
    # ЗАГРУЗКА ДАННЫХ
    # ======================================================================
    
    def load_data(self) -> List[str]:
        """
        Загрузка данных из корпуса.
        
        Returns:
            List[str]: Список программ
        """
        print_header("ЗАГРУЗКА ДАННЫХ")
        
        if not self.corpus_file.exists():
            print(f"Файл корпуса не найден: {self.corpus_file}!")
            print(f"Убедитесь, что файл существует в {self.corpus_dir}")
            return []
        
        print(f"Загрузка {self.corpus_file}...")
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            self.codes = [line.strip() for line in f if line.strip()]
        
        print(f"Загружено {len(self.codes):,} программ")
        
        # Загружаем статистику если есть
        if self.stats_file.exists():
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            print(f"Загружена ранее сохраненная статистика")
        
        return self.codes
    
    # ======================================================================
    # РАСЧЕТ СТАТИСТИКИ
    # ======================================================================
    
    def calculate_basic_stats(self) -> Dict[str, Any]:
        """
        Расчет базовой статистики по длине кода.
        
        Returns:
            Dict[str, Any]: Статистика датасета
        """
        print_header("РАСЧЕТ СТАТИСТИКИ")
        
        lengths = [len(code) for code in self.codes]
        
        if not lengths:
            print("Нет данных для расчета статистики!")
            return {}
        
        stats = {
            'total_examples': len(self.codes),
            'code_length': {
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths)),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths)),
                'std': float(np.std(lengths)),
                'q1': float(np.percentile(lengths, 25)),
                'q3': float(np.percentile(lengths, 75)),
                # Добавляем percentile для более точного анализа
                'p80': float(np.percentile(lengths, 80)),
                'p90': float(np.percentile(lengths, 90)),
                'p95': float(np.percentile(lengths, 95)),
                'p99': float(np.percentile(lengths, 99)),
            },
            'total_chars': int(np.sum(lengths)),
            'unique_chars': len(set(''.join(self.codes))),
        }
        
        self.stats.update(stats)
        
        # Сохраняем статистику
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"- Всего примеров:      {stats['total_examples']:,}")
        print(f"- Всего символов:      {stats['total_chars']:,}")
        print(f"- Уникальных символов: {stats['unique_chars']:,}")
        print(f"- Средняя длина:       {stats['code_length']['mean']:.1f}")
        print(f"- Медианная длина:     {stats['code_length']['median']:.1f}")
        print(f"- Мин/Макс:            {stats['code_length']['min']} / {stats['code_length']['max']}")
        print(f"- 90% примеров ≤ {stats['code_length']['p90']:.0f} символов")
        print(f"- 95% примеров ≤ {stats['code_length']['p95']:.0f} символов")
        
        return stats
    
    # ======================================================================
    # ГРАФИКИ
    # ======================================================================

    def plot_length_distribution(self) -> None:
        """
        График 1: Распределение длины кода.
        
        Создает три подграфика:
        - Гистограмма с средней и медианой
        - box plot (ВЕРТИКАЛЬНЫЙ)
        - Кумулятивное распределение
        """
        print("\nГенерация: распределение длины кода...")
        
        lengths = [len(code) for code in self.codes]
        total_examples = len(self.codes)
        
        # ======================================================================
        # НАСТРОЙКА ПАРАМЕТРОВ ДЛЯ ГРАФИКА
        # ======================================================================

        x_limit = 6000
        n_bins = 50

        print(f"- Максимальная длина: {self.stats['code_length']['max']:.0f}")
        print(f"- Предел на графике:  {x_limit:.0f}")

        # Предел для оси Y (количество программ) - до общего числа примеров
        y_limit = len(self.codes)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # ======================================================================
        # 1. ГИСТОГРАММА
        # ======================================================================

        # Строим вертикальную гистограмму
        counts, bins, patches = axes[0].hist(
            lengths, 
            bins=n_bins, 
            edgecolor='black', 
            alpha=0.7, 
            color='#3498db', 
            orientation='vertical'
        )

        # Добавляем вертикальные линии для средней и медианы
        axes[0].axvline(
            self.stats['code_length']['mean'], 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f"Средняя: {self.stats['code_length']['mean']:.0f}"
        )
        axes[0].axvline(
            self.stats['code_length']['median'], 
            color='green', 
            linestyle='--', 
            linewidth=2,
            label=f"Медиана: {self.stats['code_length']['median']:.0f}"
        )

        # Подписи осей
        axes[0].set_xlabel('Длина кода (символы)', fontsize=12)
        axes[0].set_ylabel('Количество программ', fontsize=12)
        axes[0].set_title('Распределение длины C++ программ', fontsize=14, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3, axis='y')

        # Пределы
        axes[0].set_xlim(0, x_limit)
        axes[0].set_ylim(0, y_limit)

        # ======================================================================
        # 2. BOX PLOT
        # ======================================================================

        # Устанавливаем предел по вертикали
        box_y_limit = 2800

        # Строим вертикальный box plot
        bp = axes[1].boxplot(
            lengths, 
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor='#e74c3c', alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='blue', linewidth=1.5),
            capprops=dict(color='blue', linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5),
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='green', markersize=8)
        )

        # Настройка осей
        axes[1].set_ylabel('Длина кода (символы)', fontsize=12)
        axes[1].set_xlabel('Распределение', fontsize=12)
        axes[1].set_title('Box-plot распределения длины', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Устанавливаем предел
        axes[1].set_ylim(0, box_y_limit)

        # Добавляем статистику прямо на график
        stats_text = (f'Медиана: {self.stats["code_length"]["median"]:.0f}\n'
                    f'Q1 (25%): {self.stats["code_length"]["q1"]:.0f}\n'
                    f'Q3 (75%): {self.stats["code_length"]["q3"]:.0f}\n'
                    f'IQR: {self.stats["code_length"]["q3"] - self.stats["code_length"]["q1"]:.0f}\n'
                    f'Выбросы: от {self.stats["code_length"]["q3"] + 1.5 * (self.stats["code_length"]["q3"] - self.stats["code_length"]["q1"]):.0f}')

        # Размещаем статистику справа от графика
        axes[1].text(1.1, 1900, stats_text,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.9),
                    verticalalignment='top',
                    fontsize=10)

        # Добавляем горизонтальную линию на уровне 2500 (начало выбросов)
        axes[1].axhline(y=2500, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

        # Подсчитываем количество выбросов выше 2500
        outliers_above_2500 = sum(1 for l in lengths if l > 2500)
        outliers_percent = (outliers_above_2500 / len(lengths)) * 100

        # Добавляем информацию о выбросах
        axes[1].text(1.1, 2300,
                    f'Выбросов >2500: {outliers_above_2500}\n({outliers_percent:.1f}% всех программ)',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                    verticalalignment='center',
                    fontsize=9)

        # ======================================================================
        # 3. КУМУЛЯТИВНОЕ РАСПРЕДЕЛЕНИЕ
        # ======================================================================

        # Строим кумулятивное распределение
        n, bins, patches = axes[2].hist(
            lengths, 
            bins=100, 
            cumulative=True, 
            density=True,
            histtype='step', 
            color='#2ecc71', 
            linewidth=2.5
        )

        # Добавляем горизонтальные линии для перцентилей
        axes[2].axhline(0.5, color='red', linestyle='--', linewidth=2, label='50% (медиана)')
        axes[2].axhline(0.8, color='orange', linestyle='--', linewidth=2, label='80%')
        axes[2].axhline(0.9, color='purple', linestyle='--', linewidth=2, label='90%')
        axes[2].axhline(0.95, color='brown', linestyle='--', linewidth=2, label='95%')

        # Настройка осей
        axes[2].set_xlabel('Длина кода (символы)', fontsize=12)
        axes[2].set_ylabel('Кумулятивная вероятность', fontsize=12)
        axes[2].set_title('Кумулятивное распределение длины кода', fontsize=14, fontweight='bold')
        axes[2].legend(loc='lower right')
        axes[2].grid(True, alpha=0.3)

        # Устанавливаем пределы
        axes[2].set_xlim(0, 6000)
        axes[2].set_ylim(0, 1)

        # Добавляем вертикальные линии для перцентилей
        axes[2].axvline(
            self.stats['code_length']['p80'],
            color='orange', 
            linestyle=':', 
            linewidth=1.5, 
            alpha=0.7
        )
        axes[2].axvline(
            self.stats['code_length']['p90'], 
            color='purple', 
            linestyle=':', 
            linewidth=1.5, 
            alpha=0.7
        )
        axes[2].axvline(
            self.stats['code_length']['p95'], 
            color='brown', 
            linestyle=':', 
            linewidth=1.5, 
            alpha=0.7
        )

        # Добавляем подписи значений перцентилей
        axes[2].text(
            self.stats['code_length']['p80'] + 50, 0.75,
            f'P80: {self.stats["code_length"]["p80"]:.0f}',
            fontsize=10, color='orange', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )
        axes[2].text(
            self.stats['code_length']['p90'] + 50, 0.85,
            f'P90: {self.stats["code_length"]["p90"]:.0f}',
            fontsize=10, color='purple', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )
        axes[2].text(
            self.stats['code_length']['p95'] + 50, 0.92,
            f'P95: {self.stats["code_length"]["p95"]:.0f}',
            fontsize=10, color='brown', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

        # Добавляем заливку областей для наглядности
        axes[2].axvspan(0, self.stats['code_length']['p80'], alpha=0.1, color='green', label='80% данных')
        axes[2].axvspan(self.stats['code_length']['p80'], self.stats['code_length']['p95'], alpha=0.1, color='yellow', label='15% данных')
        axes[2].axvspan(self.stats['code_length']['p95'], 6000, alpha=0.1, color='red', label='5% данных (выбросы)')

        # Добавляем легенду с областями
        axes[2].legend(loc='lower right', fontsize=9)

        # ======================================================================
        # ОБЩИЕ НАСТРОЙКИ
        # ======================================================================
        
        plt.tight_layout()
        
        # Сохраняем график
        output_path = self.figures_dir / 'code_length_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Сохранено: {output_path.name}")
        print(f"Параметры: {n_bins} бинов, Y-предел: {y_limit:.0f}, X-предел: {x_limit}")

    def plot_char_frequency(self) -> None:
        """
        График 2: Частотность символов в C++ коде.
        
        Показывает топ-50 наиболее частотных символов.
        """
        print("\nГенерация: частотность символов...")
        
        # Собираем все символы
        all_chars = ''.join(self.codes)
        char_counts = Counter(all_chars)
        
        # Топ-50 символов
        top_chars = char_counts.most_common(50)
        chars, counts = zip(*top_chars)
        
        # Спецсимволы
        char_labels = []
        for c in chars:
            if c == '\n':
                char_labels.append('\\n')
            elif c == '\t':
                char_labels.append('\\t')
            elif c == ' ':
                char_labels.append('пробел')
            elif c == '\\':
                char_labels.append('\\\\')
            else:
                char_labels.append(c)
        
        # Увеличиваем размер фигуры
        fig, ax = plt.subplots(figsize=(18, 10))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(chars)))
        bars = ax.bar(range(len(chars)), counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Настройка осей
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(char_labels, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel('Символ', fontsize=14, fontweight='bold')
        ax.set_ylabel('Частота', fontsize=14, fontweight='bold')
        ax.set_title('Топ-50 наиболее частотных символов в C++ коде', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Добавляем значения на столбцы
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            
            # Определяем позицию для текста
            if height > max(counts) * 0.7:    # Очень высокие столбцы
                y_pos = height - max(counts) * 0.05
                color = 'black'
                fontweight = 'bold'
                bbox = None
            else:    # Средние и низкие столбцы
                y_pos = height + max(counts) * 0.01
                color = 'black'
                fontweight = 'normal'
                bbox = dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='none')
            
            # Текст (rotation=90)
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{count:,}', 
                ha='center', 
                va='bottom',
                fontsize=9, 
                rotation=90,
                color=color,
                fontweight=fontweight,
                bbox=bbox)
        
        # Добавляем сетку
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        
        # Добавляем горизонтальную линию среднего значения
        mean_count = np.mean(counts)
        ax.axhline(y=mean_count, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'Среднее: {mean_count:.0f}')
        
        # Легенда по центру сверху
        ax.legend(loc='upper center', fontsize=11, framealpha=0.9)

        # Форматируем ось Y с разделителями тысяч
        from matplotlib.ticker import FuncFormatter
        def format_y(value, _):
            return f'{int(value):,}'
        ax.yaxis.set_major_formatter(FuncFormatter(format_y))
        
        # Статистика в правом верхнем углу
        total_chars = sum(counts)
        unique_chars = len(chars)
        stats_text = f'Всего символов в топ-50: {total_chars:,}\nУникальных символов: {unique_chars}'
        
        # Добавляем статистику с отступом от края
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9))
        
        plt.tight_layout()
        
        # Сохраняем
        output_path = self.figures_dir / 'char_frequency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Сохранено: {output_path.name}")

    def plot_length_violin(self) -> None:
        """
        График 3: Violin plot распределения длины.
        
        Показывает плотность распределения длины кода.
        """
        print("\nГенерация:    violin plot...")
        
        lengths = [len(code) for code in self.codes]
        
        # Используем предел 6000 для согласованности с гистограммой
        x_limit = 6000
        
        # Создаем фигуру с красивыми пропорциями
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Строим горизонтальный violin plot с улучшенными настройками
        parts = ax.violinplot(
            lengths, 
            vert=False,
            showmedians=True,
            showextrema=True,
            positions=[1],
            widths=0.7,
            bw_method=0.3
        )
        
        # Настройка цветов - градиент от фиолетового к синему
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor('#9b59b6')
            pc.set_alpha(0.8)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.5)
        
        # Настройка линий
        parts['cmedians'].set_color('#e74c3c')
        parts['cmedians'].set_linewidth(3)
        parts['cmaxes'].set_color('#3498db')
        parts['cmaxes'].set_linewidth(2)
        parts['cmins'].set_color('#3498db')
        parts['cmins'].set_linewidth(2)
        parts['cbars'].set_color('#3498db')
        parts['cbars'].set_linewidth(2)
        
        # Добавляем box plot внутри violin для большей информативности
        bp = ax.boxplot(
            lengths,
            positions=[1],
            vert=False,
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor='white', color='black', linewidth=2, alpha=0.7),
            medianprops=dict(color='#e74c3c', linewidth=3),
            whiskerprops=dict(color='black', linewidth=1.5, linestyle='--'),
            capprops=dict(color='black', linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.3)
        )
        
        # Настройка осей
        ax.set_xlabel('Длина кода (символы)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Плотность распределения', fontsize=14, fontweight='bold')
        ax.set_title('Распределение длины C++ программ (Violin + Box plot)', 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Устанавливаем пределы (0 - 6000)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0.5, 1.5)
        
        # Убираем метки с оси Y
        ax.set_yticklabels([])
        ax.set_yticks([])
        
        # Добавляем сетку
        ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
        
        # Добавляем вертикальные линии для перцентилей
        ax.axvline(self.stats['code_length']['p90'], color='purple', 
                linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(self.stats['code_length']['p95'], color='brown', 
                linestyle=':', linewidth=2, alpha=0.7)
        
        # Добавляем маркеры для средней и медианы (крупные)
        ax.scatter(self.stats['code_length']['mean'], 1, 
                color='darkblue', s=200, zorder=10, marker='D', 
                edgecolor='white', linewidth=2)
        ax.scatter(self.stats['code_length']['median'], 1, 
                color='darkred', s=200, zorder=10, marker='s',
                edgecolor='white', linewidth=2)
        
        # Добавляем заливку областей для наглядности
        ax.axvspan(0, self.stats['code_length']['q1'], alpha=0.1, color='blue')
        ax.axvspan(self.stats['code_length']['q1'], self.stats['code_length']['median'], 
                alpha=0.1, color='green')
        ax.axvspan(self.stats['code_length']['median'], self.stats['code_length']['q3'], 
                alpha=0.1, color='yellow')
        ax.axvspan(self.stats['code_length']['q3'], self.stats['code_length']['p90'], 
                alpha=0.1, color='orange')
        ax.axvspan(self.stats['code_length']['p90'], x_limit, 
                alpha=0.1, color='red')
        
        # ======================================================================
        # ЕДИНАЯ ЛЕГЕНДА В ЦЕНТРЕ СНИЗУ
        # ======================================================================

        
        legend_elements = [
            Patch(facecolor='#9b59b6', alpha=0.8, label='Плотность распределения'),
            Patch(facecolor='white', edgecolor='black', label='Box plot (IQR)'),
            Line2D([0], [0], color='#e74c3c', linewidth=3, label='Медиана'),
            Line2D([0], [0], color='darkblue', marker='D', linestyle='None',
                markersize=8, label=f'Средняя: {self.stats["code_length"]["mean"]:.0f}'),
            Line2D([0], [0], color='purple', linestyle=':', linewidth=2, 
                label=f'P90: {self.stats["code_length"]["p90"]:.0f}'),
            Line2D([0], [0], color='brown', linestyle=':', linewidth=2, 
                label=f'P95: {self.stats["code_length"]["p95"]:.0f}'),
            Patch(facecolor='blue', alpha=0.1, label='0-25%'),
            Patch(facecolor='green', alpha=0.1, label='25-50%'),
            Patch(facecolor='yellow', alpha=0.1, label='50-75%'),
            Patch(facecolor='orange', alpha=0.1, label='75-90%'),
            Patch(facecolor='red', alpha=0.1, label='90-100%'),
        ]
        
        # Размещаем легенду в центре снизу
        ax.legend(handles=legend_elements, 
                loc='lower center',
                bbox_to_anchor=(0.5, -0.25),
                ncol=4,
                fontsize=10,
                framealpha=0.95,
                edgecolor='black')
        
        # Добавляем статистику в правом верхнем углу
        textstr = (f'Статистика распределения:\n'
                f'{"─"*25}\n'
                f'Средняя: {self.stats["code_length"]["mean"]:>8.0f}\n'
                f'Медиана: {self.stats["code_length"]["median"]:>8.0f}\n'
                f'Q1 (25%): {self.stats["code_length"]["q1"]:>8.0f}\n'
                f'Q3 (75%): {self.stats["code_length"]["q3"]:>8.0f}\n'
                f'IQR: {self.stats["code_length"]["q3"] - self.stats["code_length"]["q1"]:>8.0f}\n'
                f'{"─"*25}\n'
                f'P90: {self.stats["code_length"]["p90"]:>8.0f}\n'
                f'P95: {self.stats["code_length"]["p95"]:>8.0f}\n'
                f'Макс: {self.stats["code_length"]["max"]:>8.0f}')
        
        props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='black')
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, fontfamily='monospace')
        
        # Добавляем аннотацию о выбросах
        outliers_count = sum(1 for l in lengths if l > self.stats['code_length']['p95'])
        outliers_percent = (outliers_count / len(lengths)) * 100
        ax.text(0.98, 0.02, f'Выбросы (>P95): {outliers_count} ({outliers_percent:.1f}%)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Добавляем отступ снизу для легенды
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        
        # Сохраняем с высоким разрешением
        output_path = self.figures_dir / 'code_length_violin.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Сохранено:    {output_path.name}")
        print(f"Предел оси X: {x_limit}")
        print(f"Медиана:      {self.stats['code_length']['median']:.0f}")
        print(f"P90:          {self.stats['code_length']['p90']:.0f}")
        print(f"P95:          {self.stats['code_length']['p95']:.0f}")

    def plot_wordcloud(self) -> None:
        """
        График 4: Облако ключевых слов C++.
        
        Показывает наиболее частотные ключевые слова C++ в датасете.
        """
        print("\nГенерация: облако ключевых слов...")
        
        try:
            from wordcloud import WordCloud
            
            # Ключевые слова C++
            cpp_keywords = [
                'int', 'void', 'char', 'double', 'float', 'bool',
                'if', 'else', 'for', 'while', 'do', 'switch', 'case',
                'return', 'break', 'continue', 'default',
                'class', 'struct', 'public', 'private', 'protected',
                'virtual', 'static', 'const', 'template', 'typename',
                'namespace', 'using', 'include', 'define',
                'cout', 'cin', 'endl', 'std', 'vector', 'string',
                'new', 'delete', 'sizeof', 'true', 'false', 'nullptr',
                'main', 'std::cout', 'std::cin', 'std::endl',
                'ifndef', 'pragma', 'friend', 'explicit', 'mutable'
            ]
            
            # Считаем частоту
            text = ' '.join(self.codes)
            word_counts = {}
            for keyword in cpp_keywords:
                count = text.count(keyword)
                if count > 0:
                    word_counts[keyword] = count
            
            # Генерируем облако
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                max_words=100,
                colormap='viridis',
                contour_width=1,
                contour_color='steelblue',
                random_state=42
            ).generate_from_frequencies(word_counts)
            
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Облако ключевых слов C++ в датасете', 
                        fontsize=18, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            output_path = self.figures_dir / 'cpp_keywords_wordcloud.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Сохранено: {output_path.name}")
            
        except ImportError:
            print("!!! wordcloud не установлен. Пропускаем...")
            print("Установите: pip install wordcloud")

    def plot_train_val_test(self) -> None:
        """
        График 5: Распределение train, val и test выборок.
        
        Показывает соотношение размеров выборок.
        """
        print("\nГенерация: распределение train/val/test...")
        
        # Считаем размеры выборок из файлов
        train_size = 0
        val_size = 0
        test_size = 0
        
        if self.train_file.exists():
            with open(self.train_file, 'r', encoding='utf-8') as f:
                train_size = sum(1 for line in f if line.strip())
        
        if self.val_file.exists():
            with open(self.val_file, 'r', encoding='utf-8') as f:
                val_size = sum(1 for line in f if line.strip())
        
        if self.test_file.exists():
            with open(self.test_file, 'r', encoding='utf-8') as f:
                test_size = sum(1 for line in f if line.strip())
        
        # Если файлы не найдены, используем значения из stats
        if train_size == 0:
            train_size = self.stats.get('train_examples', 8000)
        if val_size == 0:
            val_size = self.stats.get('val_examples', 800)
        if test_size == 0:
            test_size = self.stats.get('test_examples', 500)
        
        sizes = [train_size, val_size, test_size]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Круговая диаграмма
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            labels=labels,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(sizes))})',
            colors=colors,
            startangle=90,
            explode=(0.05, 0.05, 0.05)
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax1.set_title('Распределение выборок', fontsize=14, fontweight='bold')
        ax1.axis('equal')
        
        # Столбчатая диаграмма
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=2)
        ax2.set_ylabel('Количество примеров', fontsize=12)
        ax2.set_title('Размеры выборок', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        total = sum(sizes)
        for bar, size, label in zip(bars, sizes, labels):
            height = bar.get_height()
            percentage = (size/total)*100
            
            # Процент внутри столбца (белым цветом)
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{percentage:.1f}%', 
                    ha='center', va='center',
                    color='white', fontweight='bold', fontsize=11)
        
            # Количество сверху столбца (черным цветом)
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:,}', 
                    ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
            
        plt.tight_layout()
        
        output_path = self.figures_dir / 'train_val_test_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Сохранено: {output_path.name}")
        
        # Обновляем stats
        self.stats['train_examples'] = train_size
        self.stats['val_examples'] = val_size
        self.stats['test_examples'] = test_size

    # ======================================================================
    # ОПРЕДЕЛЕНИЕ ХАРАКТЕРИСТИК ОБОРУДОВАНИЯ
    # ======================================================================
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """
        Определение характеристик оборудования для оптимизации параметров.
        
        Returns:
            Dict[str, Any]: Информация о GPU и RAM
        """
        hardware = {
            'gpu_available': False,
            'gpu_name':      'Not detected',
            'gpu_vram_gb':   0,
            'ram_gb':        0,
            'cpu_cores':     0
        }
        
        # Определяем RAM
        try:
            import psutil
            hardware['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            # Если psutil не установлен, пробуем через /proc/meminfo на Linux
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            mem_kb = int(line.split()[1])
                            hardware['ram_gb'] = round(mem_kb / (1024**2), 1)
                            break
            except:
                hardware['ram_gb'] = 16
        
        # Определяем CPU cores
        try:
            import multiprocessing
            hardware['cpu_cores'] = multiprocessing.cpu_count()
        except:
            hardware['cpu_cores'] = 8
        
        # Определяем GPU через nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    parts = lines[0].split(',')
                    hardware['gpu_available'] = True
                    hardware['gpu_name'] = parts[0].strip()
                    
                    # Парсим VRAM
                    vram_str = parts[1].strip()
                    vram_match = re.search(r'(\d+)', vram_str)
                    if vram_match:
                        vram_mib = int(vram_match.group(1))
                        hardware['gpu_vram_gb'] = round(vram_mib / 1024, 1)
        except:
            pass
        
        # Если nvidia-smi не сработал, пробуем через torch
        if not hardware['gpu_available']:
            try:
                import torch
                if torch.cuda.is_available():
                    hardware['gpu_available'] = True
                    hardware['gpu_name'] = torch.cuda.get_device_name(0)
                    hardware['gpu_vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            except:
                pass
        
        return hardware
    
    # ======================================================================
    # ГЕНЕРАЦИЯ ОТЧЕТА
    # ======================================================================
    
    def generate_report(self) -> None:
        """
        Генерация Markdown отчета.
        
        Создает файл dataset_report.md со всей статистикой и ссылками на графики.
        """
        print("\nГенерация отчета...")
        
        # Безопасное извлечение статистики
        total = self.stats.get('total_examples', 0)
        train = self.stats.get('train_examples', 0)
        val = self.stats.get('val_examples', 0)
        test = self.stats.get('test_examples', 0)
        total_chars = self.stats.get('total_chars', 0)
        unique_chars = self.stats.get('unique_chars', 0)
        
        code_length = self.stats.get('code_length', {})
        mean_len = code_length.get('mean', 0)
        median_len = code_length.get('median', 0)
        min_len = code_length.get('min', 0)
        max_len = code_length.get('max', 0)
        std_len = code_length.get('std', 0)
        q1 = code_length.get('q1', 0)
        q3 = code_length.get('q3', 0)
        p90 = code_length.get('p90', 0)
        p95 = code_length.get('p95', 0)
        
        # Определяем оборудование для отчета
        hardware = self._detect_hardware()
        
        # Пути к графикам относительно отчета
        report_dir = self.report_file.parent
        figures_rel_path = 'figures'
        
        report = f"""# Отчет по датасету C++ кода

## Общая статистика

| Метрика | Значение |
|---------|---------|
| **Всего примеров** | {total:,} |
| **Train** | {train:,} |
| **Validation** | {val:,} |
| **Test** | {test:,} |
| **Всего символов** | {total_chars:,} |
| **Уникальных символов** | {unique_chars:,} |

## Статистика длины кода

| Метрика | Символов |
|---------|---------|
| **Средняя длина** | {mean_len:.1f} |
| **Медианная длина** | {median_len:.1f} |
| **Минимальная длина** | {min_len} |
| **Максимальная длина** | {max_len} |
| **Стандартное отклонение** | {std_len:.1f} |
| **Q1 (25%)** | {q1:.0f} |
| **Q3 (75%)** | {q3:.0f} |
| **P80 (80%)** | {code_length.get('p80', 0):.0f} |
| **P90 (90%)** | {p90:.0f} |
| **P95 (95%)** | {p95:.0f} |

## Информация об оборудовании

| Компонент | Характеристика |
|-----------|----------------|
| **GPU** | {hardware.get('gpu_name', 'Not detected')} |
| **VRAM** | {hardware.get('gpu_vram_gb', 0):.1f} GB |
| **RAM** | {hardware.get('ram_gb', 0):.1f} GB |
| **CPU Cores** | {hardware.get('cpu_cores', 0)} |

## Графики и визуализации

### 1. Распределение длины кода
![Распределение длины кода]({figures_rel_path}/code_length_distribution.png)

### 2. Частотность символов
![Частотность символов]({figures_rel_path}/char_frequency.png)

### 3. Violin plot распределения
![Violin plot]({figures_rel_path}/code_length_violin.png)

### 4. Облако ключевых слов C++
![Облако ключевых слов]({figures_rel_path}/cpp_keywords_wordcloud.png)

### 5. Распределение выборок
![Train/Val/Test]({figures_rel_path}/train_val_test_distribution.png)

## Формат данных

Код сохранен **с экранированными `\\n`** — правильный формат для BPE токенизации:
```cpp
#include <iostream>\\nusing namespace std;\\n\\nint main() {{\\n cout << "Hello";\\n return 0;\\n}}
```

## Выводы для обучения:

1. Размер словаря BPE: рекомендуется 8000-12000 токенов (оптимально для C++ кода)

2. Максимальная длина последовательности: {min(512, int(p95 * 0.4))} токенов (на основе данных и GPU)

3. Объем данных: {total_chars:,} символов достаточно для обучения

4. Сбалансированность: {train:,}/{val:,}/{test:,}

Отчет сгенерирован автоматически analyze_dataset.py версии 3.2.0
"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
        print(f"Отчет сохранен: {self.report_file}")

    # ======================================================================
    # ЗАПУСК ВСЕХ ТЕСТОВ
    # ======================================================================

    def run(self) -> None:
        """
        Запуск полного анализа.
        
        Выполняет все этапы анализа в правильном порядке.
        """
        print_header("C++ DATASET ANALYZER")
        print(f"Путь к данным:  {self.data_dir}")
        print(f"Путь к корпусу: {self.corpus_file}")
        
        # Загружаем данные
        if not self.load_data():
            print("Нет данных для анализа!")
            return
        
        # Считаем статистику
        self.calculate_basic_stats()
        
        # Генерируем графики
        self.plot_length_distribution()
        self.plot_char_frequency()
        self.plot_length_violin()
        self.plot_wordcloud()
        self.plot_train_val_test()
        
        # Генерируем отчет
        self.generate_report()
        
        print_header("АНАЛИЗ ЗАВЕРШЕН")
        print(f"Графики: {self.figures_dir}")
        print(f"Отчет:   {self.report_file}")
        print("=" * 60)

# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """
    Основная функция.
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    try:
        # Определяем корень проекта
        project_root = get_project_root()
        
        # Создаем анализатор
        analyzer = CppDatasetAnalyzer(project_root=str(project_root))
        
        # Запускаем анализ
        analyzer.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nАнализ прерван пользователем!")
        return 1
    except Exception as e:
        print(f"\nОшибка: {e}!")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())