#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# data_module.py - PyTorch Lightning DataModule для работы с C++ кодом
# ======================================================================
#
# @file data_module.py
# @brief Модуль для загрузки и предобработки данных C++ кода для обучения LLM
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль предоставляет DataModule и Dataset для эффективной
#          загрузки токенизированного C++ кода в PyTorch Lightning.
#
#          **Основные возможности:**
#
#          1. **Загрузка данных**
#             - Чтение предварительно токенизированных .pt файлов
#             - Поддержка train/val/test разделений
#             - Вывод статистики по загруженным данным
#
#          2. **Динамический паддинг**
#             - Паддинг внутри батча для оптимальной памяти
#             - Обрезка последовательностей до максимальной длины
#             - Автоматическое создание attention mask
#
#          3. **Causal LM формат**
#             - Автоматическое преобразование для causal language modeling
#             - input = tokens[:-1], target = tokens[1:]
#             - Паддинг синхронно для input и target
#
#          4. **DataLoader оптимизации**
#             - pin_memory для ускорения передачи на GPU
#             - Настраиваемое количество workers
#             - Перемешивание для train датасета
#
#          5. **Статистика данных**
#             - Минимальная, максимальная и средняя длина последовательностей
#             - Отображение размера загруженных датасетов
#
# @usage
#     dm = CppCodeDataModule(
#         data_dir='./data',
#         batch_size=8,
#         num_workers=4,
#         max_len=1024
#     )
#     dm.setup()
#     train_loader = dm.train_dataloader()
#
# @example
#     # Базовое использование
#     dm = CppCodeDataModule(data_dir='./processed_data')
#     dm.setup()
#
#     # Обучение с PyTorch Lightning
#     trainer = pl.Trainer()
#     trainer.fit(model, dm)
#
#     # Пользовательские параметры
#     dm = CppCodeDataModule(
#         data_dir='./data',
#         batch_size=16,
#         num_workers=8,
#         max_len=2048
#     )
#
# ======================================================================

import torch
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ======================================================================
# КЛАСС CppCodeDataset
# ======================================================================

class CppCodeDataset(Dataset):
    """
    Dataset для загрузки токенизированного C++ кода.
    
    **Формат данных:**
    Ожидается, что файл .pt содержит список тензоров с токенами.
    Каждый элемент - тензор целых чисел (ID токенов).
    
    **Преобразование для Causal LM:**
    - input:  Все токены кроме последнего (tokens[:-1])
    - target: Все токены кроме первого (tokens[1:])
    
    Args:
        tokens_path (str или Path): Путь к файлу .pt с токенами
    """
    
    def __init__(self, tokens_path):
        self.tokens = torch.load(tokens_path)
        print(f"Загружено {len(self.tokens)} примеров")
    
    def __len__(self):
        """Возвращает количество примеров в датасете."""
        return len(self.tokens)
    
    def __getitem__(self, idx):
        """
        Возвращает пару (input, target) для causal language modeling.
        
        Args:
            idx (int): Индекс примера
            
        Returns:
            tuple: (input_tokens, target_tokens)
                   - input_tokens:  Токены для входа ([:-1])
                   - target_tokens: Токены для предсказания ([1:])
        """
        tokens = self.tokens[idx]
        # Для causal LM: input = tokens[:-1], target = tokens[1:]
        return tokens[:-1], tokens[1:]


# ======================================================================
# ФУНКЦИЯ COLLATE
# ======================================================================

def collate_fn(batch, max_len=1024):
    """
    Функция коллации для динамического паддинга внутри батча.
    
    Args:
        batch (list):  Список кортежей (input, target) из __getitem__
        max_len (int): Максимальная длина последовательности (обрезаем более длинные)
        
    Returns:
        dict: Словарь с тензорами:
              - input_ids (torch.Tensor):      Паддинговые входные токены
              - labels (torch.Tensor):         Паддинговые целевые токены
              - attention_mask (torch.Tensor): Маска внимания (1 - реальные, 0 - паддинг)
    
    **Процесс:**
    1. Разделение batch на inputs и targets
    2. Обрезка последовательностей, превышающих max_len
    3. Определение максимальной длины в текущем батче
    4. Синхронный паддинг нулями (<PAD> токен = 0)
    5. Создание attention_mask для игнорирования паддинга
    
    **Внимание:**
    - Токен <PAD> должен иметь ID = 0 в словаре токенизатора
    - Паддинг применяется синхронно к input и target
    """
    
    # Разделяем на input и target
    inputs, targets = zip(*batch)
    
    # Обрезаем слишком длинные последовательности
    inputs = [inp[:max_len] if len(inp) > max_len else inp for inp in inputs]
    targets = [tgt[:max_len] if len(tgt) > max_len else tgt for tgt in targets]
    
    # Находим максимальную длину в батче
    max_len_batch = max(len(x) for x in inputs)
    
    # Паддинг
    padded_inputs = []
    padded_targets = []
    attention_masks = []
    
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len_batch - len(inp)
        
        if pad_len > 0:
            # Паддинг нулями (0 = <PAD>)
            padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
            padded_tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
            # Маска внимания: 1 для реальных токенов, 0 для паддинга
            mask = torch.cat([torch.ones(len(inp)), torch.zeros(pad_len)])
        else:
            padded_inp = inp
            padded_tgt = tgt
            mask = torch.ones(len(inp))
        
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
        attention_masks.append(mask)
    
    return {
        'input_ids': torch.stack(padded_inputs),
        'labels': torch.stack(padded_targets),
        'attention_mask': torch.stack(attention_masks)
    }


# ======================================================================
# КЛАСС CppCodeDataModule
# ======================================================================

class CppCodeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule для управления данными C++ кода.
    
    **Структура директории:**
    data_dir/
    ├── train_tokens.pt    # Тренировочные данные
    ├── val_tokens.pt      # Валидационные данные
    └── test_tokens.pt     # Тестовые данные
    
    Args:
        data_dir (str или Path): Директория с .pt файлами
        batch_size (int):        Размер батча (по умолчанию: 8)
        num_workers (int):       Количество процессов для загрузки данных (по умолчанию: 4)
        max_len (int):           Максимальная длина последовательности (по умолчанию: 1024)
    
    **Особенности:**
    - Автоматический pin_memory для ускорения GPU
    - Перемешивание (shuffle) только для train даталоадера
    - Единая collate_fn для всех даталоадеров
    - Вывод статистики длин последовательностей при setup()
    
    @example
        # Инициализация
        dm = CppCodeDataModule(
            data_dir='./data',
            batch_size=16,
            num_workers=8,
            max_len=2048
        )
        
        # Подготовка данных
        dm.setup()
        
        # Получение даталоадеров
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
    """
    
    def __init__(self, data_dir, batch_size=8, num_workers=4, max_len=1024):
        super().__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
    
    def setup(self, stage=None):
        """
        Загрузка данных и вывод статистики.
        
        Args:
            stage (str, optional): Стадия обучения ('fit', 'validate', 'test', 'predict')
                                   В данном implementation не используется
        
        **Загружаемые файлы:**
        - train_tokens.pt - Тренировочные данные
        - val_tokens.pt   - Валидационные данные
        - test_tokens.pt  - Тестовые данные
        
        **Выводимая статистика:**
        - Минимальная длина последовательности
        - Максимальная длина последовательности
        - Средняя длина последовательности
        """
        self.train_dataset = CppCodeDataset(self.data_dir / 'train_tokens.pt')
        self.val_dataset = CppCodeDataset(self.data_dir / 'val_tokens.pt')
        self.test_dataset = CppCodeDataset(self.data_dir / 'test_tokens.pt')
        
        # Статистика по длинам
        if hasattr(self.train_dataset, 'tokens'):
            lengths = [len(t) for t in self.train_dataset.tokens]
            print(f"Статистика train:")
            print(f"- Мин длина:  {min(lengths)}")
            print(f"- Макс длина: {max(lengths)}")
            print(f"- Средняя:    {sum(lengths)/len(lengths):.1f}")
    
    def train_dataloader(self):
        """
        Создает DataLoader для тренировочных данных.
        
        Returns:
            DataLoader: Настроенный DataLoader с перемешиванием данных
        
        **Особенности:**
        - shuffle=True для стохастического градиентного спуска
        - pin_memory=True для ускорения передачи на GPU
        - Динамический паддинг через collate_fn
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.max_len),
            pin_memory=True
        )
    
    def val_dataloader(self):
        """
        Создает DataLoader для валидационных данных.
        
        Returns:
            DataLoader: Настроенный DataLoader без перемешивания
        
        **Особенности:**
        - shuffle=False для детерминированной валидации
        - pin_memory=True для ускорения передачи на GPU
        - Динамический паддинг через collate_fn
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.max_len),
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Создает DataLoader для тестовых данных.
        
        Returns:
            DataLoader: Настроенный DataLoader без перемешивания
        
        **Особенности:**
        - shuffle=False для детерминированного тестирования
        - pin_memory=True для ускорения передачи на GPU
        - Динамический паддинг через collate_fn
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: collate_fn(batch, self.max_len),
            pin_memory=True
        )