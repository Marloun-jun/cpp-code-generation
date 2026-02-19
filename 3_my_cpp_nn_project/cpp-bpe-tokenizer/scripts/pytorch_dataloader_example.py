#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
import sys

# Добавляем путь к C++ модулю
cpp_module_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'cpp', 'build'
))
sys.path.insert(0, cpp_module_path)

print(f"🔍 Поиск модуля в: {cpp_module_path}")
print(f"📂 Файлы в папке: {os.listdir(cpp_module_path)}")

# Теперь импорт должен работать
try:
    import bpe_tokenizer_cpp
    print("✅ Модуль успешно импортирован!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

class CppCodeDataset(Dataset):
    """Датасет для C++ кода с использованием C++ токенизатора"""
    
    def __init__(self, data_path, max_length=512, vocab_size=32000):
        """
        Args:
            data_path: путь к файлу с данными
            max_length: максимальная длина последовательности
            vocab_size: размер словаря
        """
        self.max_length = max_length
        
        # Загружаем данные
        print(f"📚 Загрузка данных из {data_path}...")
        with open(data_path, 'r') as f:
            self.texts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"✅ Загружено {len(self.texts)} примеров")
        
        # Инициализируем C++ токенизатор
        print("🔄 Инициализация C++ токенизатора...")
        self.tokenizer = bpe_tokenizer_cpp.FastBPETokenizer(vocab_size, True)
        
        # Загружаем обученную модель
        vocab_path = os.path.join(os.path.dirname(__file__), '..', 'bpe', 'vocab_trained.json')
        merges_path = os.path.join(os.path.dirname(__file__), '..', 'bpe', 'merges_trained.txt')
        
        if os.path.exists(vocab_path):
            self.tokenizer.load(vocab_path, merges_path)
            print(f"✅ Модель загружена! Словарь: {self.tokenizer.vocab_size}")
        else:
            print(f"⚠️ Модель не найдена, используется пустой токенизатор")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
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

def benchmark_dataloader(dataset, batch_size=32, num_workers=0, num_batches=100):
    """Сравнение производительности DataLoader"""
    
    print(f"\n🚀 Тестирование DataLoader (batch_size={batch_size}, workers={num_workers})")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Прогрев
    print("🔄 Прогрев...")
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
    
    # Измерение скорости
    print("⏱️ Измерение...")
    start_time = time.time()
    
    total_samples = 0
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        total_samples += batch.size(0)
        # Имитация обработки батча
        batch = batch.cuda() if torch.cuda.is_available() else batch
    
    elapsed = time.time() - start_time
    
    print(f"✅ Обработано {total_samples} примеров за {elapsed:.2f} сек")
    print(f"📊 Скорость: {total_samples/elapsed:.0f} примеров/сек")
    print(f"📊 Время на батч: {elapsed/num_batches*1000:.2f} ms")
    
    return total_samples / elapsed

def main():
    print("=" * 60)
    print("🔬 ТЕСТИРОВАНИЕ PYTORCH DATALOADER С C++ ТОКЕНИЗАТОРОМ")
    print("=" * 60)
    
    # Путь к данным
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'data', 'corpus', 'train_code.txt'
    )
    
    if not os.path.exists(data_path):
        print(f"❌ Файл не найден: {data_path}")
        print("Создаю тестовые данные...")
        
        # Создаем тестовые данные
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w') as f:
            for i in range(1000):
                f.write(f"int func{i}() {{ return {i}; }}\n")
        print(f"✅ Создано 1000 тестовых примеров")
    
    # Создаем датасет
    dataset = CppCodeDataset(data_path, max_length=256)
    
    print(f"\n📊 Характеристики датасета:")
    print(f"   - Всего примеров: {len(dataset)}")
    print(f"   - Пример: {dataset.texts[0][:50]}...")
    
    # Тест с разными параметрами
    print("\n" + "=" * 60)
    print("🚀 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    results = []
    
    # Тест 1: Без многопоточности
    speed1 = benchmark_dataloader(dataset, batch_size=32, num_workers=0)
    results.append(("Без workers", 32, 0, speed1))
    
    # Тест 2: С многопоточностью (если есть)
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    speed2 = benchmark_dataloader(dataset, batch_size=32, num_workers=2)
    results.append(("2 workers", 32, 2, speed2))
    
    # Тест 3: Разные batch sizes
    speed3 = benchmark_dataloader(dataset, batch_size=64, num_workers=2)
    results.append(("Batch 64", 64, 2, speed3))
    
    speed4 = benchmark_dataloader(dataset, batch_size=128, num_workers=2)
    results.append(("Batch 128", 128, 2, speed4))
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 60)
    print(f"{'Конфигурация':<20} {'Batch':<8} {'Workers':<8} {'Скорость':<12}")
    print("-" * 60)
    
    for name, batch, workers, speed in results:
        print(f"{name:<20} {batch:<8} {workers:<8} {speed:<12.0f} экз/сек")
    
    # Сравнение с гипотетическим Python токенизатором
    print("\n" + "=" * 60)
    print("🔍 СРАВНЕНИЕ С PYTHON ТОКЕНИЗАТОРОМ")
    print("=" * 60)
    print("Python токенизатор (оценка): ~300 экз/сек")
    print(f"C++ токенизатор:            ~{speed2:.0f} экз/сек")
    print(f"Ускорение:                   {speed2/300:.1f}x")
    
    # Пример работы с GPU
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("🚀 ТЕСТИРОВАНИЕ С GPU")
        print("=" * 60)
        
        device = torch.device('cuda')
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            if i >= 50:
                break
        elapsed = time.time() - start_time
        
        print(f"✅ Обработано 50 батчей на GPU за {elapsed:.2f} сек")
        print(f"📊 Скорость: {50*128/elapsed:.0f} примеров/сек")

if __name__ == "__main__":
    main()