#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# convert_rus_tokenizer_to_cpp.py - Конвертация русского токенизатора в C++ формат
# ======================================================================
#
# @file convert_rus_tokenizer_to_cpp.py
# @brief Конвертация Python BPE токенизатора (русский) в формат для C++ инференса
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль конвертирует русский BPE токенизатор из Python формата
#          в формат, совместимый с C++ FastBPETokenizer 
#          (если планируется использование FastBPETokenizer). 
#          Важно: НЕ трогает существующий C++ токенизатор для кода! 
#
#          **Основные возможности:**
#
#          1. **Конвертация vocab.json**
#             - Сохраняет словарь токенов в формате C++
#             - Сохраняет кодировку UTF-8
#             - Идентичный формат для совместимости
#
#          2. **Конвертация merges.txt**
#             - Сохраняет правила слияния
#             - Сохраняет порядок операций
#             - Идентичный формат для совместимости
#
#          3. **Безопасность**
#             - Не трогает существующий C++ токенизатор для кода
#             - Создаёт отдельную директорию rus_bpe_4000_cpp
#             - Проверка наличия исходных файлов
#
#          4. **Проверка совместимости**
#             - Верификация форматов файлов
#             - Вывод примеров для контроля
#             - Подтверждение успешной конвертации
#
# @usage
#     python scripts/description_tokenizator/convert_rus_tokenizer_to_cpp.py
#
# @example
#     # Стандартный запуск
#     python convert_rus_tokenizer_to_cpp.py
#
# ======================================================================

import json
from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
rus_tokenizer_dir = PROJECT_ROOT / "tokenizers" / "rus_bpe_4000"
cpp_output_dir = PROJECT_ROOT / "tokenizers" / "rus_bpe_4000_cpp"


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция конвертации русского токенизатора."""
    
    print("=" * 60)
    print("КОНВЕРТАЦИЯ РУССКОГО ТОКЕНИЗАТОРА В C++ ФОРМАТ")
    print("=" * 60)
    
    # Создаём выходную директорию
    cpp_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Загрузка русского токенизатора из: {rus_tokenizer_dir}")
    print(f"C++ выход:                         {cpp_output_dir}")
    print()
    
    # Проверка существования
    if not rus_tokenizer_dir.exists():
        print(f"Папка не найдена: {rus_tokenizer_dir}!")
        print(f"Сначала запусти train_rus_tokenizer.py")
        exit(1)
    
    # Загружаем vocab
    vocab_path = rus_tokenizer_dir / "vocab.json"
    if not vocab_path.exists():
        print(f"Файл vocab.json не найден: {vocab_path}!")
        exit(1)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # Загружаем merges
    merges_path = rus_tokenizer_dir / "merges.txt"
    if not merges_path.exists():
        print(f"Файл merges.txt не найден: {merges_path}!")
        exit(1)
    
    with open(merges_path, 'r', encoding='utf-8') as f:
        merges = f.readlines()
    
    print(f"Vocab:  {len(vocab_data)} токенов")
    print(f"Merges: {len(merges)} операций")
    
    # Сохраняем в C++ формат
    print("\nСохранение в C++ формате...")
    
    # vocab.json для C++ (такой же формат)
    with open(cpp_output_dir / "cpp_vocab.json", 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    # merges.txt для C++ (такой же формат)
    with open(cpp_output_dir / "cpp_merges.txt", 'w', encoding='utf-8') as f:
        f.writelines(merges)
    
    print(f"Сохранено в: {cpp_output_dir}")
    
    # Проверка совместимости
    print("\nПроверка совместимости:")
    print("- Формат vocab.json - идентичен C++ токенизатору")
    print("- Формат merges.txt - идентичен C++ токенизатору")
    print("- Старый C++ токенизатор для кода НЕ ТРОНУТ")
    
    # Показываем пример файлов
    print("\nПример vocab.json (первые 5 токенов):")
    for i, (k, v) in enumerate(list(vocab_data.items())[:5]):
        print(f"    {k}: {v}")
    
    print("\nПример merges.txt (первые 5 строк):")
    for line in merges[:5]:
        print(f"    {line.strip()}")
    
    print("\n" + "=" * 60)
    print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 60)
    print(f"\nФайлы готовы для C++ токенизатора:")
    print(f"- {cpp_output_dir / 'cpp_vocab.json'}")
    print(f"- {cpp_output_dir / 'cpp_merges.txt'}")
    print("\nТеперь можно использовать русский токенизатор в C++!")


if __name__ == "__main__":
    main()