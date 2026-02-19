#!/usr/bin/env python3
# ======================================================================
# train_tokenizer.py - Обучение BPE токенизатора на корпусе C++ кода
# ======================================================================
#
# @file train_tokenizer.py
# @brief Скрипт для обучения BPE токенизатора на датасете C++ кода
#
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @usage ./train_tokenizer.py [options]
#   --vocab-size N    Размер словаря (по умолчанию: 8000)
#   --corpus PATH     Путь к корпусу для обучения
#   --output-dir PATH Директория для сохранения модели
#   --byte-level      Использовать byte-level режим
#   --special FILE    Файл со специальными токенами
#   --validate        Валидация на тестовой выборке
#   --plot            Построить график частот
#   --verbose         Подробный вывод
#
# @example
#   python train_tokenizer.py --vocab-size 16000 --corpus data/train.txt
#   python train_tokenizer.py --byte-level --validate --plot
#
# ======================================================================

import os
import sys
import time
import json
import argparse
from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Получаем абсолютные пути
SCRIPT_DIR = Path(__file__).parent.absolute()        # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent.absolute()          # корень проекта
BPE_DIR = PROJECT_ROOT / "bpe"                        # папка с Python токенизатором

# Добавляем пути в sys.path для импорта
sys.path.insert(0, str(PROJECT_ROOT))  # чтобы видеть папку bpe как модуль
sys.path.insert(0, str(BPE_DIR))       # чтобы видеть сами файлы напрямую

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    # Прямой импорт из файла tokenizer.py
    from tokenizer import BPETokenizer
    print("✅ Импорт через tokenizer успешен")
except ImportError as e:
    print(f"⚠️ Прямой импорт не удался: {e}")
    
    try:
        # Импорт как модуля bpe
        from bpe.tokenizer import BPETokenizer
        print("✅ Импорт через bpe.tokenizer успешен")
    except ImportError as e2:
        print(f"❌ Не удалось импортировать BPETokenizer")
        print(f"   Ошибка: {e2}")
        print(f"\n🔍 Проверьте структуру папок:")
        print(f"   PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"   BPE_DIR = {BPE_DIR}")
        print(f"   Файл tokenizer.py существует: {(BPE_DIR / 'tokenizer.py').exists()}")
        print(f"   Файл __init__.py существует: {(BPE_DIR / '__init__.py').exists()}")
        
        print("\n📋 Содержимое BPE_DIR:")
        if BPE_DIR.exists():
            for f in BPE_DIR.iterdir():
                print(f"   {f.name}")
        else:
            print(f"   Папка {BPE_DIR} не существует!")
        
        sys.exit(1)

# ======================================================================
# КЛАСС ДЛЯ ОБУЧЕНИЯ
# ======================================================================

class BPETrainer:
    """Расширенный класс для обучения BPE токенизатора"""
    
    def __init__(self, vocab_size=8000, byte_level=True, special_tokens=None):
        """
        Инициализация тренера
        
        Args:
            vocab_size: Целевой размер словаря
            byte_level: Использовать byte-level режим
            special_tokens: Список специальных токенов
        """
        self.vocab_size = vocab_size
        self.byte_level = byte_level
        self.special_tokens = special_tokens or ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
        self.tokenizer = None
        self.stats = {
            'corpus_size': 0,
            'unique_chars': 0,
            'training_time': 0,
            'final_vocab_size': 0,
            'num_merges': 0
        }
    
    def load_corpus(self, corpus_path):
        """
        Загрузка корпуса из файла
        
        Args:
            corpus_path: Путь к файлу с корпусом
            
        Returns:
            list: Список строк для обучения
        """
        print(f"\n📖 Загрузка корпуса из {corpus_path}")
        
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Файл не найден: {corpus_path}")
        
        corpus = []
        total_size = 0
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith('//'):  # Пропускаем пустые строки и комментарии
                    corpus.append(line)
                    total_size += len(line)
                
                if i % 100000 == 0 and i > 0:
                    print(f"  Загружено {i} строк...")
        
        self.stats['corpus_size'] = total_size
        print(f"  ✅ Загружено {len(corpus)} строк")
        print(f"  📊 Общий размер: {total_size / 1024 / 1024:.2f} MB")
        
        # Анализ уникальных символов
        chars = set()
        for line in corpus:
            chars.update(line)
        self.stats['unique_chars'] = len(chars)
        print(f"  🔤 Уникальных символов: {len(chars)}")
        
        return corpus
    
    def train(self, corpus_path, output_dir="bpe/", validate=True):
        """
        Обучение токенизатора
        
        Args:
            corpus_path: Путь к корпусу
            output_dir: Директория для сохранения
            validate: Валидировать на тестовой выборке
        """
        print("=" * 60)
        print(f"🚀 ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРА")
        print("=" * 60)
        print(f"📊 Параметры:")
        print(f"   • Размер словаря: {self.vocab_size}")
        print(f"   • Byte-level: {self.byte_level}")
        print(f"   • Спецтокены: {', '.join(self.special_tokens)}")
        print(f"   • Выходная директория: {output_dir}")
        
        # Загрузка корпуса
        corpus = self.load_corpus(corpus_path)
        
        # Создание выходной директории
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Создание токенизатора
        self.tokenizer = BPETokenizer(
            vocab_size=self.vocab_size,
            byte_level=self.byte_level
        )
        
        # Обучение
        print("\n🔄 Начало обучения...")
        start_time = time.time()
        
        self.tokenizer.train(corpus)
        
        end_time = time.time()
        self.stats['training_time'] = end_time - start_time
        self.stats['final_vocab_size'] = self.tokenizer.vocab_size()
        self.stats['num_merges'] = self.tokenizer.merges_count()
        
        print(f"\n✅ Обучение завершено за {self.stats['training_time']:.2f} сек")
        print(f"   Итоговый размер словаря: {self.stats['final_vocab_size']}")
        print(f"   Выполнено слияний: {self.stats['num_merges']}")
        
        # Сохранение модели
        self.save_model(output_path)
        
        # Валидация
        if validate:
            self.validate(corpus[:1000])  # Валидация на первых 1000 примерах
        
        # Построение графиков
        self.plot_statistics(output_path)
        
        return self.tokenizer
    
    def save_model(self, output_path):
        """Сохранение модели в файлы"""
        print("\n💾 Сохранение модели...")
        
        vocab_path = output_path / "vocab.json"
        merges_path = output_path / "merges.txt"
        
        # Сохранение в текстовом формате
        self.tokenizer.save(vocab_path, merges_path)
        
        # Сохранение в бинарном формате
        binary_path = output_path / "model_trained.bin"
        self.tokenizer.save_binary(binary_path)
        
        print(f"   ✅ Словарь: {vocab_path}")
        print(f"   ✅ Слияния: {merges_path}")
        print(f"   ✅ Бинарная модель: {binary_path}")
        
        # Сохранение метаданных
        metadata = {
            'vocab_size': self.vocab_size,
            'byte_level': self.byte_level,
            'special_tokens': self.special_tokens,
            'stats': self.stats,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        meta_path = output_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Метаданные: {meta_path}")
    
    def validate(self, test_samples):
        """Валидация на тестовых примерах"""
        print("\n🧪 Валидация модели...")
        
        correct = 0
        total = len(test_samples)
        
        for i, text in enumerate(test_samples):
            # Кодирование и декодирование
            tokens = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(tokens)
            
            # Проверка (в byte-level режиме должно совпадать точно)
            if self.byte_level:
                if decoded == text:
                    correct += 1
            else:
                # В обычном режиме проверяем, что все символы присутствуют
                all_chars_present = all(c in decoded for c in text)
                if all_chars_present:
                    correct += 1
            
            if i < 5:  # Показываем первые 5 примеров
                print(f"\n  Пример {i+1}:")
                print(f"    Исходный:  {text[:50]}{'...' if len(text) > 50 else ''}")
                print(f"    Декод.:    {decoded[:50]}{'...' if len(decoded) > 50 else ''}")
                print(f"    Токенов:   {len(tokens)}")
        
        accuracy = (correct / total) * 100
        print(f"\n📊 Точность roundtrip: {accuracy:.2f}% ({correct}/{total})")
        
        self.stats['validation_accuracy'] = accuracy
    
    def plot_statistics(self, output_path):
        """Построение графиков статистики"""
        print("\n📈 Построение графиков...")
        
        try:
            import matplotlib.pyplot as plt
            
            # График частот токенов
            if hasattr(self.tokenizer, 'vocabulary'):
                vocab = self.tokenizer.vocabulary
                tokens = vocab.get_all_tokens()
                
                # Длины токенов
                lengths = [len(t) for t in tokens]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # 1. Распределение длин токенов
                axes[0, 0].hist(lengths, bins=50, color='skyblue', edgecolor='black')
                axes[0, 0].set_title('Распределение длин токенов')
                axes[0, 0].set_xlabel('Длина токена')
                axes[0, 0].set_ylabel('Количество')
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Типы токенов (ASCII, UTF-8, спец)
                ascii_count = sum(1 for t in tokens if all(ord(c) < 128 for c in t))
                utf8_count = len(tokens) - ascii_count
                special_count = sum(1 for t in tokens if t.startswith('<') and t.endswith('>'))
                
                axes[0, 1].pie(
                    [ascii_count, utf8_count, special_count],
                    labels=['ASCII', 'UTF-8', 'Специальные'],
                    autopct='%1.1f%%',
                    colors=['lightgreen', 'lightblue', 'gold']
                )
                axes[0, 1].set_title('Типы токенов')
                
                # 3. Прогресс обучения (если есть данные)
                if 'num_merges' in self.stats:
                    merges = list(range(self.stats['num_merges']))
                    vocab_growth = [256 + i for i in merges]  # Примерная оценка
                    axes[1, 0].plot(merges, vocab_growth, 'b-', linewidth=2)
                    axes[1, 0].set_title('Рост словаря')
                    axes[1, 0].set_xlabel('Номер слияния')
                    axes[1, 0].set_ylabel('Размер словаря')
                    axes[1, 0].grid(True, alpha=0.3)
                
                # 4. Статистика обучения
                stats_text = (
                    f"Размер словаря: {self.stats['final_vocab_size']}\n"
                    f"Слияний: {self.stats['num_merges']}\n"
                    f"Время обучения: {self.stats['training_time']:.2f} сек\n"
                    f"Корпус: {self.stats['corpus_size']/1024/1024:.2f} MB\n"
                    f"Уник. символов: {self.stats['unique_chars']}"
                )
                
                if 'validation_accuracy' in self.stats:
                    stats_text += f"\nТочность: {self.stats['validation_accuracy']:.1f}%"
                
                axes[1, 1].text(0.5, 0.5, stats_text,
                               horizontalalignment='center',
                               verticalalignment='center',
                               transform=axes[1, 1].transAxes,
                               fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                axes[1, 1].set_title('Статистика обучения')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Сохранение
                plot_path = output_path / "training_stats.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"   ✅ Графики сохранены: {plot_path}")
                
                plt.close()
            
        except ImportError:
            print("   ⚠️ matplotlib не установлен, графики не построены")
        except Exception as e:
            print(f"   ⚠️ Ошибка при построении графиков: {e}")
    
    def print_summary(self):
        """Вывод сводки обучения"""
        print("\n" + "=" * 60)
        print("📊 СВОДКА ОБУЧЕНИЯ")
        print("=" * 60)
        print(f"🎯 Размер словаря: {self.stats['final_vocab_size']} / {self.vocab_size}")
        print(f"🔗 Количество слияний: {self.stats['num_merges']}")
        print(f"⏱️  Время обучения: {self.stats['training_time']:.2f} сек")
        print(f"📁 Размер корпуса: {self.stats['corpus_size'] / 1024 / 1024:.2f} MB")
        print(f"🔤 Уникальных символов: {self.stats['unique_chars']}")
        
        if 'validation_accuracy' in self.stats:
            print(f"✅ Точность валидации: {self.stats['validation_accuracy']:.2f}%")
        
        print("=" * 60)

# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Обучение BPE токенизатора')
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Размер словаря (по умолчанию: 8000)')
    parser.add_argument('--corpus', type=str, 
                       default='data/corpus/train_code.txt',
                       help='Путь к корпусу для обучения')
    parser.add_argument('--output-dir', type=str, default='bpe/',
                       help='Директория для сохранения модели')
    parser.add_argument('--no-byte-level', action='store_false', dest='byte_level',
                       help='Отключить byte-level режим')
    parser.add_argument('--special', type=str,
                       help='Файл со специальными токенами (JSON)')
    parser.add_argument('--no-validate', action='store_false', dest='validate',
                       help='Отключить валидацию')
    parser.add_argument('--no-plot', action='store_false', dest='plot',
                       help='Отключить построение графиков')
    parser.add_argument('--verbose', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    # Загрузка специальных токенов
    special_tokens = None
    if args.special:
        with open(args.special, 'r') as f:
            special_tokens = json.load(f)
    
    # Создание тренера
    trainer = BPETrainer(
        vocab_size=args.vocab_size,
        byte_level=args.byte_level,
        special_tokens=special_tokens
    )
    
    # Обучение
    try:
        tokenizer = trainer.train(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            validate=args.validate
        )
        
        # Вывод сводки
        trainer.print_summary()
        
        # Дополнительная информация
        if args.verbose:
            print("\n🔍 Детальная информация:")
            print(f"   Тип токенизатора: {type(tokenizer).__name__}")
            print(f"   ID <UNK>: {tokenizer.unknown_token_id()}")
            
            # Показываем первые 20 токенов
            print("\n   Первые 20 токенов:")
            vocab = tokenizer.vocabulary
            for i in range(min(20, vocab.size())):
                token = vocab.id_to_token(i)
                if len(token) > 30:
                    token = token[:27] + "..."
                print(f"     {i:4d}: '{token}'")
        
        print("\n✅ Обучение успешно завершено!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())