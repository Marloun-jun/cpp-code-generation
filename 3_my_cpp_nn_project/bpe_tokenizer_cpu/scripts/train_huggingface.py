#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# train_huggingface.py - Обучение HuggingFace BPE токенизатора
# ======================================================================
#
# @file train_huggingface.py
# @brief Обучение BPE токенизатора из библиотеки HuggingFace Tokenizers
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Обучает BPE токенизатор из библиотеки HuggingFace Tokenizers
#          на корпусе C++ кода. Созданный токенизатор используется как
#          эталон для сравнения с собственной реализацией.
#
#          **Особенности:**
#          - **Byte-level токенизация** - аналогично GPT-4
#          - **Специальные токены** - <PAD>, <UNK>, <BOS>, <EOS>
#          - **Настраиваемый размер словаря** - 8000, 10000, 12000
#          - **Автоматическое создание тестовых данных** при отсутствии корпуса
#
#          **Выходной файл:** `scripts/hf_tokenizer.json`
#
# @usage python train_huggingface.py [--vocab-size SIZE] [--data-path PATH]
#
# @example
#   python train_huggingface.py                          # обучение с параметрами по умолчанию
#   python train_huggingface.py --vocab-size 16000       # словарь 16000 токенов
#   python train_huggingface.py --data-path ../data/corpus/train_code.txt
#   python train_huggingface.py --output ./my_tokenizer.json
#
# ======================================================================

import sys
import argparse

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


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
        >>> print_header("ОБУЧЕНИЕ ТОКЕНИЗАТОРА")
        ============================================================
                         ОБУЧЕНИЕ ТОКЕНИЗАТОРА                   
        ============================================================
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


def get_project_paths() -> dict:
    """
    Получить пути проекта с учетом обновленной структуры.
    
    Returns:
        dict: Словарь с путями проекта
    """
    script_path = Path(__file__).resolve()    # scripts/train_huggingface.py
    scripts_dir = script_path.parent          # scripts/
    project_root = scripts_dir.parent         # bpe_tokenizer/
    
    return {
        "project_root": project_root,
        "scripts_dir": scripts_dir,
        "data_dir": project_root / 'data',
        "corpus_dir": project_root / 'data' / 'corpus',
        "output_file": scripts_dir / 'hf_tokenizer.json',
    }


def create_test_data(file_path: Path, num_samples: int = 1000) -> None:
    """
    Создать тестовые данные для обучения, если реальный корпус отсутствует.
    
    Args:
        file_path: Путь для сохранения тестовых данных
        num_samples: Количество тестовых примеров
    
    **Генерирует:** 1000 примеров C++ кода различных типов:
    - Include директивы
    - Функции main
    - STL контейнеры
    - Шаблоны
    - Комментарии
    - Умные указатели
    - Циклы и условия
    - Классы
    """
    print(f"\nСоздание тестовых данных ({num_samples} примеров)...")
    
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    test_samples = [
        "#include <iostream>",
        "int main() { return 0; }",
        "std::vector<int> vec = {1, 2, 3};",
        "template<typename T> class Vector {",
        "// Это комментарий на русском языке",
        "auto ptr = std::make_unique<int>(42);",
        "std::cout << \"Hello, world!\" << std::endl;",
        "for (int i = 0; i < 10; ++i) {",
        "if (condition) { do_something(); }",
        "class MyClass { public: void method(); };",
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            f.write(test_samples[i % len(test_samples)] + f" // {i}\n")
    
    print(f"Создано {num_samples} примеров в {file_path}")


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция обучения HuggingFace токенизатора.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(
        description='Обучение HuggingFace BPE токенизатора на C++ коде',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python train_huggingface.py                          # обучение с параметрами по умолчанию
  python train_huggingface.py --vocab-size 16000       # словарь 16000 токенов
  python train_huggingface.py --data-path ../data/corpus/train_code.txt
  python train_huggingface.py --output ./my_tokenizer.json
        """
    )
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Размер словаря (по умолчанию: 8000)')
    parser.add_argument('--data-path', type=str,
                       help='Путь к файлу с данными для обучения')
    parser.add_argument('--output', type=str,
                       help='Путь для сохранения токенизатора')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Минимальная частота токена (по умолчанию: 2)')
    
    args = parser.parse_args()
    
    print_header("ОБУЧЕНИЕ HUGGINGFACE BPE ТОКЕНИЗАТОРА")
    
    # Получаем пути
    paths = get_project_paths()
    
    # Определяем путь к данным
    if args.data_path:
        train_file = Path(args.data_path)
    else:
        train_file = paths['corpus_dir'] / 'train_code.txt'
    
    # Определяем выходной файл
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = paths['output_file']
    
    print(f"Корень проекта: {paths['project_root']}")
    print(f"Директория данных: {paths['corpus_dir']}")
    print(f"Входной файл: {train_file}")
    print(f"Выходной файл: {output_file}")
    print(f"Размер словаря: {args.vocab_size}")
    print(f"Минимальная частота: {args.min_frequency}")
    
    # Проверяем существование файла с данными
    if not train_file.exists():
        print(f"\n !!! Файл не найден: {train_file}")
        create_test_data(train_file, num_samples=1000)
    
    # Загружаем данные
    print(f"\nЗагрузка данных из {train_file}...")
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"   Загружено {len(lines)} строк")
        
        if len(lines) > 0:
            preview = lines[0][:80] + ('...' if len(lines[0]) > 80 else '')
            print(f"   Пример: {preview}")
        
    except Exception as e:
        print(f"x Ошибка загрузки данных: {e}")
        return 1
    
    # Создаем BPE токенизатор
    print("\nСоздание BPE токенизатора...")
    tokenizer = Tokenizer(models.BPE())
    
    # Настройка пре-токенизации (byte-level)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    print("   - Pre-tokenizer: ByteLevel (add_prefix_space=True)")
    print("   - Decoder: ByteLevel")
    print("   - Post-processor: ByteLevel")
    
    # Настройка тренера
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=args.min_frequency,
        show_progress=True
    )
    
    print(f"\nНачало обучения...")
    print(f"   Специальные токены: {special_tokens}")
    
    try:
        tokenizer.train_from_iterator(lines, trainer=trainer)
    except Exception as e:
        print(f"x Ошибка обучения: {e}")
        return 1
    
    # Получаем информацию о токенизаторе
    vocab_size = tokenizer.get_vocab_size()
    print(f"\nРезультаты обучения:")
    print(f"   Размер словаря: {vocab_size} токенов")
    
    # Тестирование на нескольких примерах
    print(f"\nТестирование:")
    test_texts = [
        "int main()",
        "std::cout << \"Hello\"",
        "template<typename T>",
        "// комментарий",
    ]
    
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"\n   {i+1}. Исходный:  {text}")
        print(f"      Токены:    {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")
        print(f"      Декод.:    {decoded}")
        print(f"      Совпадает: {'v' if text == decoded else 'x'}")
    
    # Сохраняем токенизатор
    print(f"\nСохранение токенизатора...")
    tokenizer.save(str(output_file))
    print(f"Токенизатор сохранен в {output_file}")
    
    # Показываем размер файла
    if output_file.exists():
        size_kb = output_file.stat().st_size / 1024
        print(f"Размер файла: {size_kb:.2f} KB")
    
    print_header("v ОБУЧЕНИЕ ЗАВЕРШЕНО")
    return 0


if __name__ == "__main__":
    sys.exit(main())