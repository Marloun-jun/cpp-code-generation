#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# prepare_train_dataset.py - Подготовка данных для обучения модели
# ======================================================================
#
# @file prepare_train_dataset.py
# @brief Токенизация корпуса C++ кода с сохранением в формате PyTorch
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль выполняет токенизацию корпуса C++ кода с помощью
#          FastBPETokenizer (C++ реализация) и сохраняет результаты в
#          формате PyTorch для последующего обучения модели.
#
#          **Основные возможности:**
#
#          1. **Токенизация корпуса**
#             - Чтение текстовых файлов (train/val/test)
#             - Кодирование с помощью C++ BPE токенизатора
#             - Добавление BOS/EOS токенов
#
#          2. **Без обрезания последовательностей**
#             - Сохранение полной длины токенов
#             - Паддинг будет выполняться динамически в DataLoader
#             - Статистика длин для анализа
#
#          3. **Обработка ошибок**
#             - Пропуск проблемных строк
#             - Вывод статистики пропущенных примеров
#             - Индикация прогресса через tqdm
#
#          4. **Статистика данных**
#             - Количество примеров в каждой выборке
#             - Средняя/мин/макс длина последовательностей
#             - Общее количество токенов
#
#          5. **Сохранение результатов**
#             - Формат: PyTorch тензоры (.pt)
#             - Каждый пример сохраняется как отдельный тензор
#             - Совместимость с CppCodeDataModule
#
# @usage
#     python scripts/prepare_train_dataset.py
#
# @example
#     # Стандартный запуск
#     python prepare_train_dataset.py
#
# ======================================================================

import sys
import torch

from pathlib import Path
from tqdm import tqdm

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Добавляем путь к C++ биндингам
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'bpe_tokenizer_cpu' / 'bpe_cpp' / 'build'))
import bpe_tokenizer_cpp as bpe

# ======================================================================
# КЛАСС DataPreparator
# ======================================================================

class DataPreparator:
    """
    Класс для подготовки данных: токенизация и сохранение.
    
    **Особенности:**
    - Использует C++ FastBPETokenizer для высокой скорости
    - Сохраняет последовательности БЕЗ обрезания
    - Добавляет BOS (начало) и EOS (конец) токены
    
    Args:
        tokenizer_path (str или Path): Путь к директории с токенизатором
                                       (должна содержать cpp_vocab.json и cpp_merges.txt)
    
    **Специальные токены:**
    - BOS (Begin Of Sequence): ID = 2
    - EOS (End Of Sequence): ID = 3
    """
    
    def __init__(self, tokenizer_path):
        self.tokenizer = bpe.FastBPETokenizer()
        
        # Загружаем токенизатор
        vocab_path = Path(tokenizer_path) / 'cpp_vocab.json'
        merges_path = Path(tokenizer_path) / 'cpp_merges.txt'
        
        self.tokenizer.load(str(vocab_path), str(merges_path))
        self.bos_token_id = 2    # <BOS>
        self.eos_token_id = 3    # <EOS>
        
        print(f"Токенизатор загружен: {self.tokenizer.vocab_size} токенов")
    
    def tokenize_file(self, input_path, output_path, desc="Токенизация"):
        """
        Токенизация файла и сохранение результатов.
        
        Args:
            input_path (str или Path):  Путь к входному текстовому файлу
            output_path (str или Path): Путь для сохранения .pt файла
            desc (str):                 Описание для прогресс-бара
        
        Returns:
            list or None: Список тензоров с токенами или None при ошибке
        
        **Процесс:**
        1. Чтение строк из файла (построчно)
        2. Токенизация каждой строки
        3. Добавление BOS и EOS токенов
        4. Сохранение в формате PyTorch
        
        **Примечание:**
        - Пустые строки пропускаются
        - При ошибках токенизации пример пропускается
        - Последовательности НЕ обрезаются
        """
        
        # Читаем строки
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"{desc}: {len(texts)} примеров")
        
        tokens_list = []
        skipped = 0
        lengths = []
        
        for text in tqdm(texts, desc=desc):
            try:
                # Токенизация
                tokens = self.tokenizer.encode(text)
                
                if len(tokens) == 0:
                    skipped += 1
                    continue
                
                # Добавляем BOS и EOS токены
                tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
                
                lengths.append(len(tokens))
                tokens_list.append(torch.tensor(tokens, dtype=torch.long))
                
            except Exception as e:
                print(f"\nОшибка: {text[:50]}...")
                skipped += 1
                continue
        
        if not tokens_list:
            print(f"Нет валидных примеров в {input_path}!")
            return None
        
        # Сохраняем
        torch.save(tokens_list, output_path)
        
        # Статистика
        avg_len = sum(lengths) / len(lengths)
        print(f"Сохранено {output_path}")
        print(f"- Примеров:      {len(tokens_list)}")
        print(f"- Пропущено:     {skipped}")
        print(f"- Всего токенов: {sum(lengths)}")
        print(f"- Средняя длина: {avg_len:.1f}")
        print(f"- Мин/макс:      {min(lengths)} / {max(lengths)}")
        
        return tokens_list
    
    def prepare_all(self, data_dir, output_dir):
        """
        Подготовка всех выборок (train, val, test).
        
        Args:
            data_dir (str или Path):   Директория с исходными .txt файлами
            output_dir (str или Path): Директория для сохранения .pt файлов
        
        **Ожидаемая структура:**
        data_dir/
        ├── train_code.txt
        ├── val_code.txt
        └── test_code.txt
        """
        
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Обрабатываем каждую выборку
        splits = ['train', 'val', 'test']
        
        for split in splits:
            input_path = data_dir / f'{split}_code.txt'
            
            if not input_path.exists():
                print(f"Файл не найден: {input_path}!")
                continue
            
            output_path = output_dir / f'{split}_tokens.pt'
            self.tokenize_file(input_path, output_path, desc=f"{split.upper()}")


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция для подготовки данных."""
    
    # Пути
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'corpus'
    tokenizer_path = project_root.parent / 'bpe_tokenizer_cpu' / 'bpe_cpp' / 'models' / 'bpe_10000'
    output_dir = project_root / 'data' / 'tokenized'
    
    print("=" * 60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Исходные данные:     {data_dir}")
    print(f"Токенизатор:         {tokenizer_path}")
    print(f"Выходная директория: {output_dir}")
    print("=" * 60)
    
    # Проверяем существование
    if not data_dir.exists():
        print(f"Директория данных не найдена: {data_dir}")
        print("Создайте симлинки:")
        print("ln -sf ../../data/corpus data/corpus")
        return
    
    if not tokenizer_path.exists():
        print(f"Токенизатор не найден: {tokenizer_path}!")
        return
    
    # Создаем preparator
    preparator = DataPreparator(tokenizer_path=str(tokenizer_path))
    
    # Подготавливаем данные
    preparator.prepare_all(data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА!")
    print("=" * 60)
    print("ВНИМАНИЕ: Данные НЕ обрезаны!")
    print("Паддинг будет происходить динамически в DataLoader")
    print("=" * 60)


if __name__ == "__main__":
    main()