#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# train_rus_tokenizer.py - Обучение BPE токенизатора для русского языка
# ======================================================================
#
# @file train_rus_tokenizer.py
# @brief Обучение BPE токенизатора на русских описаниях из датасета
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль обучает BPE токенизатор специально для русского языка
#          на основе описаний из датасета. Токенизатор будет использоваться
#          для кодирования инструкций при LoRA дообучении.
#
#          **Основные возможности:**
#
#          1. **Обучение BPE токенизатора**
#             - Настраиваемый размер словаря (3000-4000)
#             - Обучение на русских текстах
#             - Автоматическая сегментация слов
#
#          2. **Сохранение результатов**
#             - vocab.json (словарь токенов)
#             - merges.txt (правила слияния)
#             - Совместимость с Python BPETokenizer
#
#          3. **Тестирование токенизатора**
#             - Проверка кодирования/декодирования
#             - Визуализация результатов
#             - Верификация целостности
#
#          4. **Обработка ошибок**
#             - Проверка наличия входного файла
#             - Создание выходной директории
#             - Информативные сообщения
#
# @usage
#     python scripts/description_tokenizator/train_rus_tokenizer.py
#
# @example
#     # Стандартный запуск (размер словаря 4000)
#     python train_rus_tokenizer.py
#
#     # С другим размером словаря (измените VOCAB_SIZE в коде)
#
# ======================================================================

import sys
from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Добавляем путь к Python BPE токенизатору
PROJECT_ROOT = Path(__file__).parent.parent.parent
bpe_python_dir = PROJECT_ROOT.parent / "bpe_tokenizer_cpu" / "bpe_python"
sys.path.insert(0, str(bpe_python_dir))

from tokenizer import BPETokenizer

# ======================================================================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ======================================================================

VOCAB_SIZE = 4000    # Размер словаря

# ======================================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================================

input_path = PROJECT_ROOT / "data" / "rus_descriptions.txt"
output_dir = PROJECT_ROOT / "tokenizers" / f"rus_bpe_{VOCAB_SIZE}"

# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция обучения русского токенизатора."""
    
    print("=" * 60)
    print("ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРА ДЛЯ РУССКОГО ЯЗЫКА")
    print("=" * 60)
    print(f"Загрузка текстов из: {input_path}")
    print(f"Выходная директория: {output_dir}")
    print(f"Размер словаря:      {VOCAB_SIZE}")
    print()
    
    # Проверка существования входного файла
    if not input_path.exists():
        print(f"Файл не найден: {input_path}!")
        print(f"Сначала запусти extract_descriptions.py")
        exit(1)
    
    # Загрузка текстов
    with open(input_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Очистка от пустых строк и лишних пробелов
    texts = [t.strip() for t in texts if t.strip()]
    
    print(f"Загружено {len(texts)} строк")
    
    # Статистика текстов
    lengths = [len(t) for t in texts]
    print(f"\nСтатистика текстов:")
    print(f"- Средняя длина: {sum(lengths) / len(lengths):.1f} символов")
    print(f"- Мин:           {min(lengths)}")
    print(f"- Макс:          {max(lengths)}")
    
    # Обучение токенизатора
    print(f"\nОбучение BPE токенизатора (vocab_size={VOCAB_SIZE})...")
    print("Это может занять несколько минут...")
    
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.train(texts, verbose=True)
    
    # Сохранение
    print(f"\nСохранение токенизатора...")
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    tokenizer.save(str(vocab_path), str(merges_path))
    
    print(f"Токенизатор сохранён в: {output_dir}")
    print(f"- {vocab_path.name}: {len(tokenizer.vocab)} токенов")
    print(f"- {merges_path.name}: {tokenizer.merges_count()} операций слияния")
    
    # Тестирование
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ТОКЕНИЗАТОРА")
    print("=" * 60)
    
    test_texts = [
        "напиши программу которая выводит Hello World",
        "создай класс Person с полями имя и возраст",
        "напиши рекурсивную функцию факториала",
        "сделай функцию которая сортирует вектор целых чисел",
        "реализуй шаблонную функцию max для сравнения двух чисел",
    ]
    
    success_count = 0
    for i, text in enumerate(test_texts, 1):
        print(f"\nТест {i}:")
        print(f"- Текст: {text}")
        
        try:
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            print(f"- Токенов:   {len(tokens)}")
            print(f"- Декод.:    {decoded[:80]}...")
            
            is_match = (text == decoded)
            print(f"- Совпадает: {'ДА' if is_match else 'НЕТ'}")
            
            if is_match:
                success_count += 1
            else:
                print(f"- Оригинал:  {text}")
                print(f"- Декод:     {decoded}")
                
        except Exception as e:
            print(f"Ошибка: {e}!")
    
    # Итоги тестирования
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"- Успешно: {success_count}/{len(test_texts)}")
    print(f"- Процент: {success_count / len(test_texts) * 100:.1f}%")
    
    # Дополнительная информация
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ О ТОКЕНИЗАТОРЕ")
    print("=" * 60)
    print(f"- Размер словаря:     {len(tokenizer.vocab)}")
    print(f"- Количество слияний: {tokenizer.merges_count()}")
    print(f"- Формат:             совместим с Python BPETokenizer")
    print(f"\nДля конвертации в C++ формат запусти:")
    print(f"python convert_rus_tokenizer_to_cpp.py")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)


if __name__ == "__main__":
    main()