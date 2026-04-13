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
# @version 3.3.0
#
# @details Обучает BPE токенизатор из библиотеки HuggingFace Tokenizers
#          на корпусе C++ кода. Созданный токенизатор используется как
#          эталон для сравнения с собственной реализацией.
#
#          **Особенности:**
#          - **Byte-level токенизация**       - аналогично GPT-4
#          - **Специальные токены**           - <PAD>, <UNK>, <BOS>, <EOS>
#          - **Настраиваемый размер словаря** - 8000, 10000, 12000
#          - **Автоматическое создание тестовых данных** при отсутствии корпуса
#          - **Сохранение в формате совместимом с C++** (vocab.json + merges.txt)
#
#          **Выходные файлы:**
#          - `hf_tokenizer_{vocab_size}.json` - полный токенизатор
#          - `hf_vocab_{vocab_size}.json`     - словарь для C++
#          - `hf_merges_{vocab_size}.txt`     - слияния для C++
#
# @usage python train_huggingface.py [--vocab-size SIZE] [--data-path PATH]
#
# @example
#   python train_huggingface.py                       # Обучение с параметрами по умолчанию
#   python train_huggingface.py --vocab-size 10000    # Словарь 10000 токенов
#   python train_huggingface.py --data-path ../data/corpus/train_code.txt
#   python train_huggingface.py --output ./my_tokenizer.json
#
# ======================================================================

import sys
import argparse
import json

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
    bpe_cpp_dir = project_root / 'bpe_cpp'    # bpe_cpp/
    
    return {
        "project_root": project_root,
        "scripts_dir": scripts_dir,
        "data_dir": project_root / 'data',
        "corpus_dir": project_root / 'data' / 'corpus',
        "bpe_cpp_dir": bpe_cpp_dir,
        "hf_models_dir": bpe_cpp_dir / 'models' / 'hf',
    }


def create_test_data(file_path: Path, num_samples: int = 1000) -> None:
    """
    Создать тестовые данные для обучения, если реальный корпус отсутствует.
    
    Args:
        file_path:   Путь для сохранения тестовых данных
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


def save_cpp_compatible_files(tokenizer, output_dir: Path, vocab_size: int) -> None:
    """
    Сохраняет токенизатор в формате, совместимом с C++ реализацией.
    
    Args:
        tokenizer:  Обученный токенизатор HuggingFace
        output_dir: Директория для сохранения
        vocab_size: Размер словаря для имени файла
    """
    # Получаем словарь в формате {token: id}
    vocab = tokenizer.get_vocab()
    
    # Создаем словарь в формате {id: token} для C++
    cpp_vocab = {str(idx): token for token, idx in vocab.items()}
    
    # Сохраняем vocab.json
    vocab_file = output_dir / f'hf_vocab_{vocab_size}.json'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(cpp_vocab, f, indent=2, ensure_ascii=False)
    
    # Сохраняем merges.txt
    merges_file = output_dir / f'hf_merges_{vocab_size}.txt'
    with open(merges_file, 'w', encoding='utf-8') as f:
        # Записываем заголовок
        f.write("#version: 3.3.0\n")
        f.write("#sourced from: HuggingFace Tokenizers\n")
        
        # Получаем merges из токенизатора
        # Для этого нужно получить модель BPE
        if hasattr(tokenizer.model, 'get_merges'):
            merges = tokenizer.model.get_merges()
            for left, right in merges:
                f.write(f"{left} {right}\n")
    
    print(f"Словарь C++: {vocab_file}")
    print(f"Слияния C++: {merges_file}")


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
  python train_huggingface.py                       # Обучение с параметрами по умолчанию
  python train_huggingface.py --vocab-size 10000    # Словарь 10000 токенов
  python train_huggingface.py --data-path ../data/corpus/train_code.txt
  python train_huggingface.py --output ./my_tokenizer.json
        """
    )
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Размер словаря (по умолчанию: 10000)')
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
    
    # Определяем выходной файл и директорию
    if args.output:
        output_file = Path(args.output)
        output_dir = output_file.parent
    else:
        # Создаем директорию для HF моделей, если её нет
        paths['hf_models_dir'].mkdir(parents=True, exist_ok=True)
        output_dir = paths['hf_models_dir']
        output_file = output_dir / f'hf_tokenizer_{args.vocab_size}.json'
    
    # Создаем директорию для выходного файла, если её нет
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Корень проекта:        {paths['project_root']}")
    print(f"Директория данных:     {paths['corpus_dir']}")
    print(f"Директория HF моделей: {paths['hf_models_dir']}")
    print(f"Входной файл:          {train_file}")
    print(f"Выходной файл:         {output_file}")
    print(f"Размер словаря:        {args.vocab_size}")
    print(f"Минимальная частота:   {args.min_frequency}")
    
    # Проверяем существование файла с данными
    if not train_file.exists():
        print(f"\nФайл не найден: {train_file}!")
        create_test_data(train_file, num_samples=1000)
    
    # Загружаем данные
    print(f"\nЗагрузка данных из {train_file}...")
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Загружено {len(lines)} строк")
        
        if len(lines) > 0:
            preview = lines[0][:80] + ('...' if len(lines[0]) > 80 else '')
            print(f"Пример: {preview}")
        
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}!")
        return 1
    
    # Создаем BPE токенизатор
    print("\nСоздание BPE токенизатора...")
    tokenizer = Tokenizer(models.BPE())
    
    # add_prefix_space=False для точного roundtrip
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    
    print("- Pre-tokenizer:  ByteLevel (add_prefix_space=False)")
    print("- Decoder:        ByteLevel")
    print("- Post-processor: ByteLevel")
    
    # Настройка тренера
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=args.min_frequency,
        show_progress=True
    )
    
    print(f"\nНачало обучения...")
    print(f"Специальные токены: {special_tokens}")
    
    try:
        tokenizer.train_from_iterator(lines, trainer=trainer)
    except Exception as e:
        print(f"Ошибка обучения: {e}!")
        return 1
    
    # Получаем информацию о токенизаторе
    vocab_size = tokenizer.get_vocab_size()
    print(f"\nРезультаты обучения:")
    print(f"- Размер словаря: {vocab_size} токенов")
    
    # Тестирование на нескольких примерах
    print(f"\nТестирование:")
    test_texts = [
        "int main()",
        "std::cout << \"Hello\"",
        "template<typename T>",
        "// comment",
    ]
    
    all_match = True
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        
        match = (text == decoded)
        all_match = all_match and match
        
        print(f"\n{i+1}. Исходный:  {text}")
        print(f"- Токены:    {encoded.ids[:10]}{'...' if len(encoded.ids) > 10 else ''}")
        print(f"- Декод.:    {decoded}")
        print(f"- Совпадает: {'да' if match else 'нет'}")
        
        if not match:
            print(f"Проблема: исходный и декодированный отличаются!")
            print(f"Ожидалось: '{text}'")
            print(f"Получено:  '{decoded}'")
    
    if all_match:
        print(f"\nВсе тесты пройдены успешно! Roundtrip точность 100%!")
    else:
        print(f"\nВнимание: есть несовпадения в roundtrip тестах!")
    
    # Сохраняем токенизатор
    print(f"\nСохранение токенизатора...")
    tokenizer.save(str(output_file))
    print(f"Токенизатор сохранен в {output_file}")
    
    # Сохраняем в формате, совместимом с C++
    print(f"\nСохранение в формате C++...")
    save_cpp_compatible_files(tokenizer, output_dir, args.vocab_size)
    
    # Показываем размер файла
    if output_file.exists():
        size_kb = output_file.stat().st_size / 1024
        print(f"\nРазмер файла: {size_kb:.2f} КБ")
    
    # Показываем итоговую статистику
    print_header("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"\nИтоговая статистика:")
    print(f"- Размер словаря:      {vocab_size}")
    print(f"- Специальных токенов: {len(special_tokens)}")
    print(f"- Файл модели:         {output_file}")
    print(f"- Файл словаря C++:    {output_dir / f'hf_vocab_{args.vocab_size}.json'}")
    print(f"- Файл слияний C++:    {output_dir / f'hf_merges_{args.vocab_size}.txt'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())