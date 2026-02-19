#!/usr/bin/env python3
# ======================================================================
# pytorch_integration.py - Модуль интеграции BPE токенизатора с PyTorch
# ======================================================================
#
# @file pytorch_integration.py
# @brief Модуль интеграции BPE токенизатора с PyTorch
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Предоставляет классы и функции для использования BPE токенизатора
#          в пайплайнах машинного обучения на PyTorch:
#          - Обертка токенизатора с поддержкой батчинга и паддинга
#          - Dataset для C++ кода
#          - DataLoader с коллацией
#          - Работа со специальными токенами (PAD, UNK, BOS, EOS)
#
# @usage from pytorch_integration import BPETokenizerWrapper, CodeDataset, create_dataloader
#
# @example
#   tokenizer = BPETokenizer.load('vocab.json', 'merges.txt')
#   wrapper = BPETokenizerWrapper(tokenizer, max_length=128)
#   dataset = CodeDataset(texts, wrapper)
#   dataloader = create_dataloader(texts, wrapper, batch_size=32)
#
# ======================================================================

import logging
import torch

from pathlib import Path
from typing import List, Dict, Union, Optional
from torch.utils.data import Dataset, DataLoader

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()           # pytorch_integration.py
BPE_PYTHON_DIR = CURRENT_FILE.parent               # bpe_python/
PROJECT_ROOT = BPE_PYTHON_DIR.parent               # cpp-bpe-tokenizer/

# ======================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def get_project_paths() -> Dict[str, Path]:
    """
    Получить пути проекта.
    
    Returns:
        Dict[str, Path]: Словарь с путями проекта
    """
    return {
        "project_root": PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "models_dir": BPE_PYTHON_DIR / 'models',
    }


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


# ======================================================================
# КЛАСС ОБЕРТКИ ТОКЕНИЗАТОРА
# ======================================================================

class BPETokenizerWrapper:
    """
    Обертка для использования BPE токенизатора с PyTorch.
    
    Предоставляет удобный интерфейс для:
    - Кодирования текста в тензоры input_ids и attention_mask
    - Декодирования тензоров обратно в текст
    - Работы со специальными токенами (PAD, UNK, BOS, EOS)
    - Пакетной обработки с паддингом
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Инициализация обертки.
        
        Args:
            tokenizer: Обученный экземпляр BPETokenizer
            max_length: Максимальная длина последовательности
            
        Example:
            >>> tokenizer = BPETokenizer.load('vocab.json', 'merges.txt')
            >>> wrapper = BPETokenizerWrapper(tokenizer, max_length=128)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Получаем ID специальных токенов
        self.pad_id = tokenizer.inverse_vocab.get('<PAD>', 0)
        self.unk_id = tokenizer.inverse_vocab.get('<UNK>', 1)
        self.bos_id = tokenizer.inverse_vocab.get('<BOS>', 2)
        self.eos_id = tokenizer.inverse_vocab.get('<EOS>', 3)
        
        logger.info(f"Инициализирована обертка:")
        logger.info(f"  max_length: {max_length}")
        logger.info(f"  PAD ID: {self.pad_id}, UNK ID: {self.unk_id}")
        logger.info(f"  BOS ID: {self.bos_id}, EOS ID: {self.eos_id}")
    
    # ======================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # ======================================================================
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Кодирование текста в тензоры PyTorch.
        
        Args:
            text: Входной текст
            add_special_tokens: Добавлять ли BOS/EOS токены
            padding: Добавлять ли паддинг ('max_length' или True)
            truncation: Обрезать ли до max_length
            return_tensors: Тип возвращаемых тензоров ('pt' или 'np')
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с полями:
                - input_ids: Тензор ID токенов
                - attention_mask: Маска внимания (1 для реальных токенов)
            
        Example:
            >>> encoded = wrapper.encode("int main()", padding=True)
            >>> encoded['input_ids'].shape
            torch.Size([128])
        """
        # Кодируем текст
        tokens = self.tokenizer.encode(text)
        
        # Добавляем специальные токены
        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]
        
        # Обрезаем при необходимости
        original_length = len(tokens)
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            logger.debug(f"Обрезано с {original_length} до {self.max_length}")
        
        # Создаем маску внимания
        attention_mask = [1] * len(tokens)
        
        # Добавляем паддинг
        if padding:
            pad_len = self.max_length - len(tokens)
            if pad_len > 0:
                tokens += [self.pad_id] * pad_len
                attention_mask += [0] * pad_len
            elif pad_len < 0 and padding == 'max_length':
                # При max_length паддинге обрезаем до max_length
                tokens = tokens[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
        
        # Конвертируем в тензоры
        result = {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        return result
    
    def decode(
        self, 
        tokens: Union[List[int], torch.Tensor], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Декодирование тензора обратно в текст.
        
        Args:
            tokens: Список ID токенов или тензор
            skip_special_tokens: Пропускать ли специальные токены
            
        Returns:
            str: Декодированный текст
            
        Example:
            >>> decoded = wrapper.decode(encoded['input_ids'])
            >>> print(decoded)
            "int main()"
        """
        # Конвертируем тензор в список
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # Фильтруем специальные токены
        if skip_special_tokens:
            filtered_tokens = []
            for token in tokens:
                if token not in [self.pad_id, self.bos_id, self.eos_id]:
                    filtered_tokens.append(token)
        else:
            filtered_tokens = tokens
        
        # Декодируем
        text = self.tokenizer.decode(filtered_tokens)
        
        return text
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Кодирование батча текстов.
        
        Args:
            texts: Список текстов
            add_special_tokens: Добавлять ли специальные токены
            padding: Добавлять ли паддинг
            truncation: Обрезать ли до max_length
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с батч-тензорами
        """
        # Кодируем каждый текст без паддинга
        encoded_batch = [self.encode(
            text, 
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=truncation
        ) for text in texts]
        
        if padding:
            # Находим максимальную длину в батче
            max_len = max(len(item['input_ids']) for item in encoded_batch)
            max_len = min(max_len, self.max_length)
            
            # Добавляем паддинг до max_len
            padded_batch = []
            for item in encoded_batch:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                
                pad_len = max_len - len(input_ids)
                if pad_len > 0:
                    input_ids = torch.cat([
                        input_ids,
                        torch.full((pad_len,), self.pad_id, dtype=torch.long)
                    ])
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.zeros(pad_len, dtype=torch.long)
                    ])
                
                padded_batch.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
            
            # Стекуем в батч
            result = {
                'input_ids': torch.stack([item['input_ids'] for item in padded_batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in padded_batch])
            }
        else:
            # Просто стекуем (все должны быть одинаковой длины)
            result = {
                'input_ids': torch.stack([item['input_ids'] for item in encoded_batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in encoded_batch])
            }
        
        return result
    
    def __call__(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Удобный вызов как функции."""
        return self.encode(text, **kwargs)


# ======================================================================
# КЛАСС DATASET
# ======================================================================

class CodeDataset(Dataset):
    """
    Dataset для кода на C++.
    
    Работает с BPETokenizerWrapper для подготовки данных для PyTorch.
    """
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer_wrapper: BPETokenizerWrapper,
                 max_length: Optional[int] = None):
        """
        Инициализация датасета.
        
        Args:
            texts: Список текстов (C++ код)
            tokenizer_wrapper: Обертка токенизатора
            max_length: Максимальная длина (если None, используется из wrapper)
        """
        self.texts = texts
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length or tokenizer_wrapper.max_length
        
        logger.info(f"Создан CodeDataset с {len(texts)} примерами")
    
    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает элемент датасета.
        
        Args:
            idx: Индекс элемента
            
        Returns:
            Dict[str, torch.Tensor]: Словарь с input_ids и attention_mask
        """
        text = self.texts[idx]
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        return encoded


# ======================================================================
# ФУНКЦИИ ДЛЯ СОЗДАНИЯ DATALOADER
# ======================================================================

def create_dataloader(
    texts: List[str],
    tokenizer_wrapper: BPETokenizerWrapper,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Создание DataLoader для PyTorch.
    
    Args:
        texts: Список текстов
        tokenizer_wrapper: Обертка токенизатора
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        num_workers: Количество рабочих процессов
        
    Returns:
        DataLoader: Настроенный DataLoader
        
    Example:
        >>> dataloader = create_dataloader(texts, wrapper, batch_size=32)
        >>> for batch in dataloader:
        ...     model(batch['input_ids'], batch['attention_mask'])
    """
    dataset = CodeDataset(texts, tokenizer_wrapper)
    
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Функция коллации для батча.
        
        Args:
            batch: Список элементов датасета
            
        Returns:
            Dict[str, torch.Tensor]: Батч тензоров
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Создан DataLoader: batch_size={batch_size}, "
                f"num_workers={num_workers}, shuffle={shuffle}")
    
    return dataloader


# ======================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ======================================================================

def example_usage() -> None:
    """
    Пример использования интеграции с PyTorch.
    
    Демонстрирует:
    - Загрузку модели
    - Создание обертки
    - Кодирование отдельных примеров
    - Создание DataLoader
    - Работу с батчами
    """
    print_header("ПРИМЕР ИСПОЛЬЗОВАНИЯ BPE TOKENIZER С PyTorch")
    
    try:
        # Импортируем токенизатор только для примера
        from tokenizer import BPETokenizer
        
        # Получаем пути
        paths = get_project_paths()
        
        # Загружаем обученный токенизатор
        model_size = 8000
        vocab_path = paths['models_dir'] / f'bpe_{model_size}' / 'vocab.json'
        merges_path = paths['models_dir'] / f'bpe_{model_size}' / 'merges.txt'
        
        if not vocab_path.exists() or not merges_path.exists():
            print(f"\nМодель bpe_{model_size} не найдена!")
            print(f"   Поиск по пути: {paths['models_dir']}")
            print("\n   Доступные модели:")
            for model_dir in sorted(paths['models_dir'].iterdir()):
                if model_dir.is_dir() and model_dir.name.startswith('bpe_'):
                    vocab = model_dir / 'vocab.json'
                    merges = model_dir / 'merges.txt'
                    status = '✓' if vocab.exists() and merges.exists() else '✗'
                    print(f"     {status} {model_dir.name}")
            return
        
        print(f"\nЗагрузка модели bpe_{model_size}...")
        tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
        print(f"Загружено {len(tokenizer.vocab)} токенов")
        
        # Создаем обертку
        print(f"\nСоздание обертки с max_length=128...")
        wrapper = BPETokenizerWrapper(tokenizer, max_length=128)
        
        # Пример текстов
        texts = [
            "#include <iostream>",
            "int main() { return 0; }",
            "std::vector<int> vec;",
            "template<typename T> class Vector {",
            "// комментарий на русском языке",
            "auto ptr = std::make_unique<int>(42);",
        ]
        
        print(f"\nТестовые тексты ({len(texts)} примеров):")
        for i, text in enumerate(texts, 1):
            preview = text[:50] + ('...' if len(text) > 50 else '')
            print(f"   {i}. {preview}")
        
        # Кодируем один пример
        print(f"\nКодирование первого примера с паддингом:")
        encoded = wrapper.encode(texts[0], padding=True)
        print(f"   input_ids shape: {encoded['input_ids'].shape}")
        print(f"   attention_mask shape: {encoded['attention_mask'].shape}")
        print(f"   input_ids[:20]: {encoded['input_ids'][:20].tolist()}...")
        print(f"   attention_mask[:20]: {encoded['attention_mask'][:20].tolist()}...")
        
        # Декодируем обратно
        print(f"\nДекодирование:")
        decoded = wrapper.decode(encoded['input_ids'])
        print(f"   Оригинал: {texts[0]}")
        print(f"   Декод:    {decoded}")
        print(f"   Совпадение: {'[OK]' if texts[0] == decoded else '[BAD]'}")
        
        # Создаем DataLoader
        print(f"\nСоздание DataLoader с batch_size=2...")
        dataloader = create_dataloader(texts, wrapper, batch_size=2, shuffle=False)
        
        # Проходим по батчам
        print(f"\nБатчи:")
        for i, batch in enumerate(dataloader):
            print(f"\n   Батч {i + 1}:")
            print(f"     input_ids shape: {batch['input_ids'].shape}")
            print(f"     attention_mask shape: {batch['attention_mask'].shape}")
            
            # Декодируем первый элемент батча
            first_text = wrapper.decode(batch['input_ids'][0])
            preview = first_text[:50] + ('...' if len(first_text) > 50 else '')
            print(f"     Первый текст: {preview}")
        
        print_header("ПРИМЕР УСПЕШНО ВЫПОЛНЕН")
        
    except ImportError as e:
        print(f"\nОшибка импорта: {e}")
        print("   Убедитесь, что tokenizer.py находится в той же директории")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================

if __name__ == '__main__':
    example_usage()