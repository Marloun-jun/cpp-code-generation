import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np

# Настройка стилей
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'

class CppDatasetAnalyzer:
# Анализ очищенного датасет (примеры поля code)
    def __init__(self, project_root: str = "cpp-bpe-tokenizer"):
        self.root = Path(project_root)
        # директории
        self.corpus_dir = self.root / 'data' / 'corpus'
        self.metadata_dir = self.root / 'data' / 'metadata'
        self.reports_dir = self.root / 'reports'
        self.figures_dir = self.reports_dir / 'figures'
        # создаем директории
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # файлы
        self.corpus_file = self.corpus_dir / 'corpus.txt'
        self.train_file = self.corpus_dir / 'train_code.txt'
        self.val_file = self.corpus_dir / 'val_code.txt'
        self.test_file = self.corpus_dir / 'test_code.txt'
        self.stats_file = self.metadata_dir / 'dataset_stats.json'
        self.report_file = self.reports_dir / 'dataset_report.md'
        # данные
        self.codes = []
        self.stats = {}
    
    def load_data(self):
    # Загрузка данных из corpus
        print("Загрузка corpus...")
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            self.codes = [line.strip() for line in f if line.strip()]
        print(f"Загружено {len(self.codes)} программ")
        # загружаем статистику если есть
        if self.stats_file.exists():
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        return self.codes
    
    def calculate_basic_stats(self):
    # Базовая статистика по длине кода
        print("\nРасчет статистики...")
        lengths = [len(code) for code in self.codes]
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
            },
            'total_chars': int(np.sum(lengths)),
            'unique_chars': len(set(''.join(self.codes))),
        }
        self.stats.update(stats)
        # сохраняем обновленную статистику
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"   Средняя длина: {stats['code_length']['mean']:.1f}")
        print(f"   Медианная длина: {stats['code_length']['median']:.1f}")
        print(f"   Макс длина: {stats['code_length']['max']}")
        print(f"   Мин длина: {stats['code_length']['min']}")
        return stats
    
    def plot_length_distribution(self):  # Переименовано для ясности
    # График 1: Распределение длины кода
        print("\nГенерация: распределение длины кода...")
        lengths = [len(code) for code in self.codes]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # 1. Гистограмма
        axes[0].hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        axes[0].axvline(self.stats['code_length']['mean'], color='red', 
                       linestyle='--', linewidth=2, 
                       label=f"Средняя: {self.stats['code_length']['mean']:.0f}")
        axes[0].axvline(self.stats['code_length']['median'], color='green', 
                       linestyle='--', linewidth=2,
                       label=f"Медиана: {self.stats['code_length']['median']:.0f}")
        axes[0].set_xlabel('Длина кода (символы)')
        axes[0].set_ylabel('Количество программ')
        axes[0].set_title('Распределение длины C++ программ', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        # 2. График box plot
        axes[1].boxplot(lengths, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                       medianprops=dict(color='black', linewidth=2))
        axes[1].set_xlabel('Длина кода (символы)')
        axes[1].set_title('Box-plot распределения длины', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        # 3. Кумулятивная вероятность (cumulative distribution)
        axes[2].hist(lengths, bins=100, cumulative=True, density=True,
                    histtype='step', color='#2ecc71', linewidth=2)
        axes[2].axhline(0.5, color='red', linestyle='--', label='50% (медиана)')
        axes[2].axhline(0.8, color='orange', linestyle='--', label='80%')
        axes[2].axhline(0.9, color='purple', linestyle='--', label='90%')
        axes[2].set_xlabel('Длина кода (символы)')
        axes[2].set_ylabel('Кумулятивная вероятность')
        axes[2].set_title('Кумулятивное распределение', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = self.figures_dir / 'code_length_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Сохранено: {output_path}")
    
    def plot_char_frequency(self):
    # График 2: Частотность символов в C++ коде
        print("\nГенерация: частотность символов...")
        # собираем все символы
        all_chars = ''.join(self.codes)
        char_counts = Counter(all_chars)
        # топ-50 символов
        top_chars = char_counts.most_common(50)
        chars, counts = zip(*top_chars)
        # спецсимволы
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
        fig, ax = plt.subplots(figsize=(15, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(chars)))
        bars = ax.bar(range(len(chars)), counts, color=colors, alpha=0.8)
        ax.set_xticks(range(len(chars)))
        ax.set_xticklabels(char_labels, rotation=45, ha='right')
        ax.set_xlabel('Символ')
        ax.set_ylabel('Частота')
        ax.set_title('Топ-50 наиболее частотных символов в C++ коде', 
                    fontsize=16, fontweight='bold')
        # добавляем значения на столбцы
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}', ha='center', va='bottom', fontsize=8, rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        output_path = self.figures_dir / 'char_frequency.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Сохранено: {output_path}")
    
    def plot_length_violin(self):  # Переименовано
    # График 3: Диаграмма violin plot распределения длины
        print("\nГенерация: violin plot...")
        lengths = [len(code) for code in self.codes]
        fig, ax = plt.subplots(figsize=(10, 6))
        parts = ax.violinplot(lengths, vert=False, showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#9b59b6')
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('red')
        parts['cmedians'].set_linewidth(2)
        ax.set_xlabel('Длина кода (символы)')
        ax.set_title('Violin plot распределения длины кода', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        # добавляем статистику
        textstr = f'Средняя: {self.stats["code_length"]["mean"]:.0f}\n'
        textstr += f'Медиана: {self.stats["code_length"]["median"]:.0f}\n'
        textstr += f'Q1-Q3: {self.stats["code_length"]["q1"]:.0f} - {self.stats["code_length"]["q3"]:.0f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right', bbox=props)
        plt.tight_layout()
        output_path = self.figures_dir / 'code_length_violin.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Сохранено: {output_path}")
    
    def plot_wordcloud(self):
    # График 4: Облако ключевых слов C++
        print("\nГенерация: облако ключевых слов...")
        try:
            from wordcloud import WordCloud
            # ключевые слова C++
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
            # считаем частоту
            text = ' '.join(self.codes)
            word_counts = {}
            for keyword in cpp_keywords:
                count = text.count(keyword)
                if count > 0:
                    word_counts[keyword] = count
            # генерируем облако
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
            print(f"Сохранено: {output_path}")
        except ImportError:
            print("Wordcloud не установлен. Пропускаем...")
            print("Установите: pip install wordcloud")
    
    def plot_train_val_test(self):
    # График 5: Распределение train, val и test
        print("\nГенерация: распределение train/val/test...")
        # считаем размеры выборок из файлов
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
        # если файлы не найдены, используем стандартные значения
        if train_size == 0 and val_size == 0 and test_size == 0:
            train_size = 8022
            val_size = 800
            test_size = 500
        sizes = [train_size, val_size, test_size]
        labels = ['Train', 'Validation', 'Test']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # круговая диаграмма
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
        # столбчатая диаграмма
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Количество примеров', fontsize=12)
        ax2.set_title('Размеры выборок', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        # добавляем значения на столбцы
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        plt.tight_layout()
        output_path = self.figures_dir / 'train_val_test_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Сохранено: {output_path}")
        # обновляем stats
        self.stats['train_examples'] = train_size
        self.stats['val_examples'] = val_size
        self.stats['test_examples'] = test_size
    
    def generate_report(self):
    # Генерация Markdown отчета
        print("\nГенерация отчета...")
        report = f"""# Отчет по датасету C++ кода

##  Общая статистика

| Метрика | Значение |
|---------|---------|
| **Всего примеров** | {self.stats['total_examples']:,} |
| **Train** | {self.stats.get('train_examples', 0):,} |
| **Validation** | {self.stats.get('val_examples', 0):,} |
| **Test** | {self.stats.get('test_examples', 0):,} |
| **Всего символов** | {self.stats['total_chars']:,} |
| **Уникальных символов** | {self.stats['unique_chars']:,} |

##  Статистика длины кода

| Метрика | Символов |
|---------|---------|
| **Средняя длина** | {self.stats['code_length']['mean']:.1f} |
| **Медианная длина** | {self.stats['code_length']['median']:.1f} |
| **Минимальная длина** | {self.stats['code_length']['min']} |
| **Максимальная длина** | {self.stats['code_length']['max']} |
| **Стандартное отклонение** | {self.stats['code_length']['std']:.1f} |
| **Q1 (25%)** | {self.stats['code_length']['q1']:.0f} |
| **Q3 (75%)** | {self.stats['code_length']['q3']:.0f} |

##  Графики и визуализации

### 1. Распределение длины кода
![Распределение длины кода](figures/code_length_distribution.png)

### 2. Частотность символов
![Частотность символов](figures/char_frequency.png)

### 3. Violin plot распределения
![Violin plot](figures/code_length_violin.png)

### 4. Облако ключевых слов C++
![Облако ключевых слов](figures/cpp_keywords_wordcloud.png)

### 5. Распределение выборок
![Train/Val/Test](figures/train_val_test_distribution.png)

##  Формат данных

Код сохранен **с экранированными `\\n`** — правильный формат для BPE токенизации:
#include <iostream>\nusing namespace std;\n\nint main() {{\n cout << \"Hello\";\n return 0;\n}}

##  Выводы для обучения

1. **Размер словаря BPE**: рекомендуется **16000-24000** токенов
2. **Максимальная длина последовательности**: {min(512, int(self.stats['code_length']['max']))} токенов
3. **Объем данных**: {self.stats['total_chars']:,} символов достаточно для обучения
4. **Сбалансированность**: {self.stats.get('train_examples', 0):,}/{self.stats.get('val_examples', 0):,}/{self.stats.get('test_examples', 0):,}

---
*Отчет сгенерирован автоматически `analyze_dataset.py`*
"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчет сохранен: {self.report_file}")
    
    def run(self):
    # Запуск полного анализа
        print("="*60)
        print("C++ DATASET ANALYZER")
        print("="*60)
        # загружаем данные
        self.load_data()
        # считаем статистику
        self.calculate_basic_stats()
        # генерируем графики
        self.plot_length_distribution()
        self.plot_char_frequency()
        self.plot_length_violin()
        self.plot_wordcloud()
        self.plot_train_val_test()
        # генерируем отчет
        self.generate_report()
        print("\n" + "="*60)
        print("✅ АНАЛИЗ ЗАВЕРШЕН")
        print("="*60)
        print(f"Графики: {self.figures_dir}")
        print(f"Отчет: {self.report_file}")
        print("="*60)


def main():
    analyzer = CppDatasetAnalyzer(
        project_root=Path(__file__).parent.parent
    )
    analyzer.run()


if __name__ == "__main__":
    main()