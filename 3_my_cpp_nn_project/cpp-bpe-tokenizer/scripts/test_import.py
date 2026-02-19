#!/usr/bin/env python3
import os
import sys

print("=== Тест импорта ===\n")

# Текущая директория
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Текущая директория: {current_dir}")

# Путь к bpe
bpe_dir = os.path.join(current_dir, '..', 'bpe')
print(f"BPE директория: {bpe_dir}")
print(f"BPE директория существует: {os.path.exists(bpe_dir)}")

# Содержимое bpe директории
if os.path.exists(bpe_dir):
    print("\nФайлы в bpe директории:")
    for f in os.listdir(bpe_dir):
        print(f"  - {f}")
        
    # Проверяем наличие tokenizer.py
    tokenizer_path = os.path.join(bpe_dir, 'tokenizer.py')
    print(f"\ntokenizer.py существует: {os.path.exists(tokenizer_path)}")

# Добавляем пути
sys.path.insert(0, bpe_dir)
sys.path.insert(0, current_dir)

print(f"\nPython path:")
for p in sys.path[:5]:
    print(f"  - {p}")

# Пробуем импортировать
print("\nПробуем импортировать...")
try:
    from tokenizer import BPETokenizer
    print("✅ Импорт через 'from tokenizer import BPETokenizer' успешен")
except ImportError as e:
    print(f"❌ Ошибка: {e}")

try:
    import tokenizer
    print("✅ Импорт через 'import tokenizer' успешен")
    print(f"   tokenizer.__file__ = {tokenizer.__file__}")
except ImportError as e:
    print(f"❌ Ошибка: {e}")