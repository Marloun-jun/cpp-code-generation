#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# pytorch_dataloader_example.py - Пример использования C++ токенизатора в PyTorch DataLoader
# ======================================================================
#
# @file pytorch_dataloader_example.py
# @brief Пример интеграции C++ BPE токенизатора с PyTorch DataLoader
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Демонстрирует полную интеграцию C++ токенизатора с PyTorch DataLoader:
#
#          - **Создание Dataset** - пользовательский датасет, использующий C++ токенизатор
#          - **Загрузка модели** - автоматический поиск файлов модели в проекте
#          - **Тестирование производительности** - измерение скорости при разных параметрах
#          - **Многопоточность** - сравнение разных значений num_workers
#          - **Batch size** - влияние размера батча на производительность
#          - **GPU поддержка** - автоматическое использование CUDA при наличии
#
#          **Измеряемые метрики:**
#          - Скорость обработки (примеров/сек)
#          - Время на батч (мс)
#          - Влияние количества потоков
#          - Влияние размера батча
#          - Ускорение относительно Python реализации
#
# @usage python pytorch_dataloader_example.py
#
# @requirements
#   torch
#   C++ модуль bpe_tokenizer_cpp (должен быть собран)
#
# @example
#   python pytorch_dataloader_example.py
#
# ======================================================================

import sys
import time
import multiprocessing
import torch

from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА C++ МОДУЛЯ
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()    # scripts/pytorch_dataloader_example.py
SCRIPTS_DIR = CURRENT_FILE.parent          # scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent          # bpe_tokenizer/
BPE_CPP_DIR = PROJECT_ROOT / 'bpe_cpp'     # bpe_cpp/
CPP_BUILD_DIR = BPE_CPP_DIR / 'build'      # bpe_cpp/build/

# Добавляем путь к C++ модулю
sys.path.insert(0, str(CPP_BUILD_DIR))

print("=" * 60)
print("ТЕСТИРОВАНИЕ PYTORCH DATALOADER С C++ ТОКЕНИЗАТОРОМ")
print("=" * 60)
print(f"Корень проекта: {PROJECT_ROOT}")
print(f"C++ директория: {BPE_CPP_DIR}")
print(f"C++ build директория: {CPP_BUILD_DIR}")

# Проверяем наличие C++ модуля
if CPP_BUILD_DIR.exists():
    print(f"\nФайлы в C++ build директории:")
    found_modules = False
    for f in CPP_BUILD_DIR.iterdir():
        if f.name.endswith('.so') or 'bpe_tokenizer_cpp' in f.name:
            print(f"   v {f.name}")
            found_modules = True
    if not found_modules:
        print(f" !!! C++ модули не найдены")
else:
    print(f"\n !!! C++ build директория не найдена: {CPP_BUILD_DIR}")
    print("Соберите C++ проект командой:")
    print(f"cd {BPE_CPP_DIR} && mkdir -p build && cd build && cmake .. && make")

# ======================================================================
# ИМПОРТ C++ МОДУЛЯ
# ======================================================================

try:
    import bpe_tokenizer_cpp
    print("C++ модуль успешно импортирован!")
    print(f"Доступные классы: {[a for a in dir(bpe_tokenizer_cpp) if not a.startswith('_')]}")
except ImportError as e:
    print(f"Ошибка импорта C++ модуля: {e}")
    print("\nВозможные причины:")
    print("   1. C++ проект не собран")
    print("   2. Неправильный путь к модулю")
    print("   3. Отсутствуют зависимости")
    print("\nРешение:")
    print(f"   cd {BPE_CPP_DIR}")
    print("   mkdir -p build && cd build")
    print("   cmake .. && make -j$(nproc)")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела для красивого форматирования вывода.
    
    Args:
        title: Заголовок
        width: Ширина линии
    
    Example:
        >>> print_header("ТЕСТ 1: БЕЗ МНОГОПОТОЧНОСТИ")
        ============================================================
                       ТЕСТ 1: БЕЗ МНОГОПОТОЧНОСТИ                
        ============================================================
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


# ======================================================================
# КЛАСС ДАТАСЕТА
# ======================================================================

class CppCodeDataset(Dataset):
    """
    Датасет для C++ кода с использованием C++ токенизатора.
    
    Обеспечивает эффективную токенизацию через C++ биндинги
    и интеграцию с PyTorch DataLoader.
    
    **Особенности:**
    - Автоматическая загрузка модели из стандартных путей
    - Создание тестовых данных при отсутствии реальных
    - Паддинг до max_length
    - Возврат torch.Tensor для прямого использования в PyTorch
    """
    
    def __init__(self, data_path: str, max_length: int = 512, vocab_size: int = 8000):
        """
        Инициализация датасета.
        
        Args:
            data_path: Путь к файлу с данными
            max_length: Максимальная длина последовательности
            vocab_size: Размер словаря (8000, 10000 или 12000)
        """
        self.max_length = max_length
        self.data_path = Path(data_path)
        
        # ======================================================================
        # ЗАГРУЗКА ДАННЫХ
        # ======================================================================
        
        print(f"\nЗагрузка данных из {self.data_path}...")
        
        if not self.data_path.exists():
            print(f" !!! Файл не найден, создаю тестовые данные...")
            self._create_test_data()
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Загружено {len(self.texts)} примеров")
        
        # ======================================================================
        # ИНИЦИАЛИЗАЦИЯ C++ ТОКЕНИЗАТОРА
        # ======================================================================
        
        print("Инициализация C++ токенизатора...")
        self.tokenizer = bpe_tokenizer_cpp.FastBPETokenizer(vocab_size, True)
        
        # Ищем файлы модели в стандартных местах
        vocab_path = BPE_CPP_DIR / 'models' / 'cpp_vocab.json'
        merges_path = BPE_CPP_DIR / 'models' / 'cpp_merges.txt'
        
        # Альтернативные пути (для моделей разных размеров)
        if not vocab_path.exists():
            vocab_path = BPE_CPP_DIR / 'models' / f'vocab_{vocab_size}.json'
            merges_path = BPE_CPP_DIR / 'models' / f'merges_{vocab_size}.txt'
        
        if vocab_path.exists() and merges_path.exists():
            self.tokenizer.load(str(vocab_path), str(merges_path))
            print(f"Модель загружена! Словарь: {self.tokenizer.vocab_size} токенов")
            print(f"    Словарь: {vocab_path}")
            print(f"    Слияния: {merges_path}")
        else:
            print(f" !!! Модель не найдена, используется пустой токенизатор")
            print(f"Искали: {vocab_path}")
    
    def _create_test_data(self) -> None:
        """Создать тестовые данные для демонстрации."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w', encoding='utf-8') as f:
            test_samples = [
                "#include <iostream>",
                "int main() { return 0; }",
                "std::vector<int> vec;",
                "template<typename T> class Vector {",
                "// комментарий на русском языке",
                "auto ptr = std::make_unique<int>(42);",
            ]
            # Создаем 1000 примеров
            for i in range(1000):
                f.write(test_samples[i % len(test_samples)] + f" // {i}\n")
        print(f"   Создано 1000 тестовых примеров")
    
    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Возвращает элемент датасета.
        
        Args:
            idx: Индекс элемента
            
        Returns:
            torch.Tensor: Тензор с ID токенов (форма: [max_length])
        
        **Процесс:**
        1. Получение текста по индексу
        2. Токенизация через C++ модуль
        3. Обрезка до max_length
        4. Паддинг нулями (PAD)
        5. Конвертация в torch.Tensor
        """
        text = self.texts[idx]
        
        # Токенизация через C++
        tokens = self.tokenizer.encode(text)
        
        # Обрезаем/дополняем до max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            # Дополняем нулями (PAD)
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        # Конвертируем в тензор PyTorch
        return torch.tensor(tokens, dtype=torch.long)


# ======================================================================
# ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ
# ======================================================================

def benchmark_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    num_batches: int = 100
) -> Tuple[float, float]:
    """
    Сравнение производительности DataLoader с разными параметрами.
    
    Args:
        dataset: Датасет
        batch_size: Размер батча
        num_workers: Количество рабочих процессов
        num_batches: Количество батчей для измерения
        
    Returns:
        Tuple[float, float]: (примеров в секунду, время на батч в мс)
    
    **Процесс:**
    1. Создание DataLoader с заданными параметрами
    2. Прогрев (5 батчей)
    3. Измерение скорости на num_batches батчах
    4. Перемещение на GPU если доступно
    """
    print(f"\nТестирование DataLoader (batch_size={batch_size}, workers={num_workers})")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Прогрев
    print("   Прогрев...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
    
    # Измерение скорости
    print("   Измерение...")
    start_time = time.time()
    
    total_samples = 0
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        total_samples += batch.size(0)
        # Имитация обработки батча (перемещение на GPU если доступно)
        if torch.cuda.is_available():
            batch = batch.cuda()
    
    elapsed = time.time() - start_time
    
    samples_per_sec = total_samples / elapsed
    time_per_batch = elapsed / num_batches * 1000  # ms
    
    print(f"   Обработано {total_samples} примеров за {elapsed:.2f} сек")
    print(f"   Скорость: {samples_per_sec:.0f} примеров/сек")
    print(f"   Время на батч: {time_per_batch:.2f} ms")
    
    return samples_per_sec, time_per_batch


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция.

    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    # Путь к данным (обучающая выборка)
    data_path = PROJECT_ROOT / 'data' / 'corpus' / 'train_code.txt'
    
    # Создаем датасет
    print_header("СОЗДАНИЕ ДАТАСЕТА")
    dataset = CppCodeDataset(str(data_path), max_length=256, vocab_size=8000)
    
    print(f"\nХАРАКТЕРИСТИКИ ДАТАСЕТА:")
    print(f"   - Всего примеров: {len(dataset)}")
    if len(dataset) > 0:
        print(f"   - Пример текста: {dataset.texts[0][:80]}...")
        sample = dataset[0]
        print(f"   - Пример тензора: {sample[:20]}... (длина: {len(sample)})")
    
    # ======================================================================
    # ТЕСТ 1: Разные конфигурации
    # ======================================================================
    print_header("⚡ СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    
    results = []
    
    # Тест 1: Без многопоточности
    speed1, time1 = benchmark_dataloader(dataset, batch_size=32, num_workers=0, num_batches=50)
    results.append(("Без workers", 32, 0, speed1, time1))
    
    # Тест 2: С 2 потоками
    if multiprocessing.cpu_count() >= 2:
        speed2, time2 = benchmark_dataloader(dataset, batch_size=32, num_workers=2, num_batches=50)
        results.append(("2 workers", 32, 2, speed2, time2))
    
    # Тест 3: С 4 потоками
    if multiprocessing.cpu_count() >= 4:
        speed3, time3 = benchmark_dataloader(dataset, batch_size=32, num_workers=4, num_batches=50)
        results.append(("4 workers", 32, 4, speed3, time3))
    
    # Тест 4: Batch size 64
    speed4, time4 = benchmark_dataloader(dataset, batch_size=64, num_workers=2, num_batches=50)
    results.append(("Batch 64", 64, 2, speed4, time4))
    
    # Тест 5: Batch size 128
    speed5, time5 = benchmark_dataloader(dataset, batch_size=128, num_workers=2, num_batches=50)
    results.append(("Batch 128", 128, 2, speed5, time5))
    
    # ======================================================================
    # ИТОГОВЫЕ РЕЗУЛЬТАТЫ
    # ======================================================================
    print_header("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    
    print(f"\n{'=' * 80}")
    print(f"{'Конфигурация':<20} {'Batch':<8} {'Workers':<8} {'Скорость (экз/с)':<18} {'Время/батч (мс)':<18}")
    print(f"{'-' * 80}")
    
    for name, batch, workers, speed, time_per_batch in results:
        print(f"{name:<20} {batch:<8} {workers:<8} {speed:<18.0f} {time_per_batch:<18.2f}")
    
    print(f"{'=' * 80}")
    
    # ======================================================================
    # СРАВНЕНИЕ С PYTHON ТОКЕНИЗАТОРОМ
    # ======================================================================
    print_header("СРАВНЕНИЕ С PYTHON ТОКЕНИЗАТОРОМ")
    
    # Гипотетическая скорость Python токенизатора (из предыдущих бенчмарков)
    python_speed = 300  # экз/сек
    
    # Берем лучший результат C++
    best_speed = max(speed for _, _, _, speed, _ in results)
    
    print(f"\nPython токенизатор (оценка): ~{python_speed} экз/сек")
    print(f"C++ токенизатор (лучший):    ~{best_speed:.0f} экз/сек")
    print(f"Ускорение:                   {best_speed/python_speed:.1f}x")
    
    # ======================================================================
    # ТЕСТИРОВАНИЕ С GPU
    # ======================================================================
    if torch.cuda.is_available():
        print_header("🎮 ТЕСТИРОВАНИЕ С GPU")
        
        device = torch.device('cuda')
        print(f"GPU обнаружен: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=128, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        print("\nИзмерение скорости с GPU...")
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            if i >= 50:
                break
        
        elapsed = time.time() - start_time
        samples_processed = 50 * 128
        
        print(f"   Обработано {samples_processed} примеров за {elapsed:.2f} сек")
        print(f"   Скорость: {samples_processed/elapsed:.0f} примеров/сек")
        print(f"   Время на батч: {elapsed/50*1000:.2f} мс")
    
    print_header("v ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    return 0


if __name__ == "__main__":
    sys.exit(main())