#!/usr/bin/env python3
# ======================================================================
# trainer.py - Модуль для обучения BPE токенизатора на корпусе текстов
# ======================================================================
#
# @file trainer.py
# @brief Модуль для обучения BPE токенизатора на корпусе текстов
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Предоставляет функции для загрузки корпуса, обучения токенизатора
#          и тестирования результатов. Поддерживает:
#          - Загрузку корпуса из текстового файла
#          - Обучение BPE токенизатора с заданными параметрами
#          - Сохранение обученной модели
#          - Тестирование на примерах
#          - Валидацию входных файлов
#
# @usage python trainer.py [--corpus PATH] [--vocab-size N] [--output-dir PATH]
#
# @example
#   python trainer.py
#   python trainer.py --corpus ../data/corpus.txt --vocab-size 10000
#   python trainer.py --output-dir ./my_models
#
# ======================================================================

import sys
import logging
import argparse

from pathlib import Path
from typing import List, Optional, Union

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()           # bpe_python/trainer.py
BPE_PYTHON_DIR = CURRENT_FILE.parent               # bpe_python/
PROJECT_ROOT = BPE_PYTHON_DIR.parent               # cpp-bpe-tokenizer/

# Добавляем путь для импорта токенизатора (если нужно)
import sys
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}")
    print(f"Убедитесь, что файл tokenizer.py существует в {BPE_PYTHON_DIR}")
    sys.exit(1)


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


def get_project_paths() -> dict:
    """
    Получить пути проекта.
    
    Returns:
        dict: Словарь с путями проекта
    """
    return {
        "project_root": PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "models_dir": BPE_PYTHON_DIR / 'models',
        "default_corpus": PROJECT_ROOT / 'data' / 'corpus' / 'train_code.txt',
    }


def validate_corpus_file(corpus_path: Union[str, Path]) -> bool:
    """
    Проверить файл корпуса на читаемость и базовую структуру.
    
    Args:
        corpus_path: Путь к файлу корпуса
        
    Returns:
        bool: True если файл валиден, иначе False
    """
    corpus_path = Path(corpus_path)
    
    if not corpus_path.exists():
        logger.error(f"Файл не существует: {corpus_path}")
        return False
    
    if not corpus_path.is_file():
        logger.error(f"Путь не является файлом: {corpus_path}")
        return False
    
    # Проверяем, что файл не пустой и читается
    try:
        size_kb = corpus_path.stat().st_size / 1024
        if corpus_path.stat().st_size == 0:
            logger.error(f"Файл пуст: {corpus_path}")
            return False
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                logger.warning(f" !!! Первая строка файла пуста: {corpus_path}")
        
        logger.info(f"Файл корпуса валиден: {corpus_path.name} (размер: {size_kb:.1f} KB)")
        return True
        
    except UnicodeDecodeError:
        logger.error(f"Ошибка кодировки UTF-8 в файле: {corpus_path}")
        return False
    except Exception as e:
        logger.error(f"Ошибка при проверке файла {corpus_path}: {e}")
        return False


# ======================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ======================================================================

def load_corpus(corpus_path: Path, verbose: bool = True) -> List[str]:
    """
    Загрузить корпус из файла.
    
    Args:
        corpus_path: Путь к файлу корпуса
        verbose: Выводить прогресс загрузки
        
    Returns:
        List[str]: Список строк корпуса
        
    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если корпус пуст
    """
    if verbose:
        logger.info(f"Загрузка корпуса из {corpus_path}")
    
    corpus = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Пропускаем пустые строки
                    corpus.append(line)
                
                # Прогресс для больших файлов
                if verbose and line_num % 100000 == 0:
                    logger.info(f"   Загружено {line_num} строк...")
        
        if verbose:
            logger.info(f"   Загружено {len(corpus)} примеров")
            logger.info(f"   Размер корпуса: {corpus_path.stat().st_size / (1024*1024):.2f} MB")
        
        if not corpus:
            raise ValueError(f"Корпус пуст или не содержит непустых строк: {corpus_path}")
        
        return corpus
        
    except UnicodeDecodeError as e:
        raise ValueError(f"Ошибка кодировки файла {corpus_path}: {e}")


def train_from_corpus(
    corpus_path: Union[str, Path],
    vocab_size: int = 8000,
    byte_level: bool = True,
    special_tokens: Optional[List[str]] = None,
    output_dir: Union[str, Path] = './bpe',
    verbose: bool = True
) -> BPETokenizer:
    """
    Обучить BPE токенизатор из файла корпуса.
    
    Args:
        corpus_path: Путь к файлу с корпусом (построчно)
        vocab_size: Размер словаря
        byte_level: Использовать byte-level режим
        special_tokens: Специальные токены
        output_dir: Директория для сохранения результатов
        verbose: Подробный вывод
        
    Returns:
        BPETokenizer: Обученный экземпляр токенизатора
        
    Raises:
        FileNotFoundError: Если файл корпуса не найден
        ValueError: Если корпус пуст или размер словаря слишком мал
        
    Example:
        >>> tokenizer = train_from_corpus(
        ...     corpus_path='../data/corpus.txt',
        ...     vocab_size=10000,
        ...     output_dir='./models'
        ... )
    """
    # Преобразуем пути в объекты Path
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)
    
    # Проверяем существование файла
    if not corpus_path.exists():
        raise FileNotFoundError(f"Файл корпуса не найден: {corpus_path}")
    
    # Загружаем корпус
    corpus = load_corpus(corpus_path, verbose=verbose)
    
    # Инициализируем токенизатор
    if verbose:
        logger.info(f"Инициализация токенизатора:")
        logger.info(f"   • Размер словаря: {vocab_size}")
        logger.info(f"   • Byte-level: {byte_level}")
        if special_tokens:
            logger.info(f"   • Спецтокены: {', '.join(special_tokens)}")
    
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        special_tokens=special_tokens
    )
    
    # Обучаем
    if verbose:
        logger.info(f"Начало обучения...")
    
    tokenizer.train(corpus, verbose=verbose)
    
    if verbose:
        logger.info(f"   Обучение завершено")
        logger.info(f"   Итоговый размер словаря: {len(tokenizer.vocab)}")
        logger.info(f"   Выполнено слияний: {tokenizer.merges_count()}")
    
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем результаты
    vocab_path = output_dir / 'vocab.json'
    merges_path = output_dir / 'merges.txt'
    binary_path = output_dir / 'model.bin'
    
    tokenizer.save(str(vocab_path), str(merges_path))
    tokenizer.save_binary(str(binary_path))
    
    if verbose:
        logger.info(f"Модель сохранена в {output_dir}")
        logger.info(f"   • Словарь: {vocab_path.name}")
        logger.info(f"   • Слияния: {merges_path.name}")
        logger.info(f"   • Бинарная: {binary_path.name}")
        
        # Размеры файлов
        vocab_size_kb = vocab_path.stat().st_size / 1024
        merges_size_kb = merges_path.stat().st_size / 1024
        binary_size_kb = binary_path.stat().st_size / 1024
        logger.info(f"   • Размеры: {vocab_size_kb:.1f} KB / {merges_size_kb:.1f} KB / {binary_size_kb:.1f} KB")
    
    return tokenizer


def test_tokenizer(
    tokenizer: BPETokenizer,
    test_texts: List[str],
    show_tokens: int = 10
) -> None:
    """
    Тестирование токенизатора на примерах текстов.
    
    Args:
        tokenizer: Обученный токенизатор
        test_texts: Список тестовых текстов
        show_tokens: Количество первых токенов для отображения
        
    Example:
        >>> test_texts = ["int main() {", "std::cout << \"Hello\";"]
        >>> test_tokenizer(tokenizer, test_texts)
    """
    print_header("ТЕСТИРОВАНИЕ ТОКЕНИЗАТОРА")
    
    total = len(test_texts)
    passed = 0
    
    for i, text in enumerate(test_texts, 1):
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            
            is_match = (text == decoded)
            if is_match:
                passed += 1
            
            print(f"\n{i}. {'[OK]' if is_match else '[BAD]'} Текст: {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"   Токенов: {len(encoded)}")
            
            if show_tokens > 0 and encoded:
                # Показываем первые несколько токенов
                preview = encoded[:show_tokens]
                if len(encoded) > show_tokens:
                    preview.append('...')
                print(f"   ID: {preview}")
                
                # Показываем соответствующие токены
                tokens = []
                for idx in encoded[:show_tokens]:
                    if idx in tokenizer.vocab:
                        token = tokenizer.vocab[idx]
                        # Экранируем специальные символы для вывода
                        if token in tokenizer.special_tokens:
                            tokens.append(f"[{token}]")
                        elif len(token) > 20:
                            tokens.append(token[:17] + "...")
                        else:
                            tokens.append(token)
                if tokens:
                    print(f"   Токены: {', '.join(tokens)}{'...' if len(encoded) > show_tokens else ''}")
            
        except Exception as e:
            print(f"\n{i}. Ошибка при тестировании: {e}")
    
    print(f"\nРезультат: {passed}/{total} тестов пройдено ({passed/total*100:.1f}%)")


def get_default_corpus_path() -> Path:
    """
    Получить путь к корпусу по умолчанию.
    
    Returns:
        Path: Путь к корпусу по умолчанию
    """
    paths = get_project_paths()
    return paths['default_corpus']


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция для запуска обучения из командной строки.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(description='Обучение BPE токенизатора')
    parser.add_argument('--corpus', type=str, 
                       help='Путь к файлу корпуса')
    parser.add_argument('--vocab-size', '-v', type=int, default=8000,
                       help='Размер словаря (по умолчанию: 8000)')
    parser.add_argument('--output-dir', '-o', type=str, default='./bpe',
                       help='Директория для сохранения модели')
    parser.add_argument('--no-byte-level', action='store_false', dest='byte_level',
                       help='Отключить byte-level режим')
    parser.add_argument('--special', type=str,
                       help='Файл со специальными токенами (JSON)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Тихий режим (без подробностей)')
    
    args = parser.parse_args()
    
    print_header("ОБУЧЕНИЕ BPE ТОКЕНИЗАТОРА")
    
    # Получаем пути
    paths = get_project_paths()
    
    # Определяем путь к корпусу
    if args.corpus:
        corpus_path = Path(args.corpus)
    else:
        corpus_path = paths['default_corpus']
    
    # Загружаем специальные токены если указаны
    special_tokens = None
    if args.special:
        import json
        try:
            with open(args.special, 'r', encoding='utf-8') as f:
                special_tokens = json.load(f)
            logger.info(f"Загружены специальные токены: {', '.join(special_tokens)}")
        except Exception as e:
            logger.error(f"Ошибка загрузки специальных токенов: {e}")
            return 1
    else:
        # Токены по умолчанию для C++ кода
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CPP>', '<CODE>']
    
    # Проверяем корпус
    if not validate_corpus_file(corpus_path):
        logger.error(f"Файл корпуса не найден: {corpus_path}")
        logger.info(f"\nПоместите файл train_code.txt в:")
        logger.info(f"   {paths['default_corpus']}")
        return 1
    
    try:
        # Создаем выходную директорию
        output_dir = Path(args.output_dir)
        if args.output_dir == './bpe':
            # Если используется относительный путь, сохраняем в models/
            output_dir = paths['models_dir'] / f'bpe_{args.vocab_size}'
        
        # Обучение
        tokenizer = train_from_corpus(
            corpus_path=corpus_path,
            vocab_size=args.vocab_size,
            byte_level=args.byte_level,
            special_tokens=special_tokens,
            output_dir=output_dir,
            verbose=not args.quiet
        )
        
        # Тестирование на нескольких примерах
        if not args.quiet:
            test_texts = [
                "#include <iostream>",
                "int main() { return 0; }",
                'std::cout << "Hello, мир!" << std::endl;',
                "// Это комментарий на русском языке",
                "template<typename T> class Vector {",
                "auto result = std::make_unique<int[]>(size);",
                "for (int i = 0; i < 10; ++i) {",
                "std::vector<std::string> tokens;"
            ]
            test_tokenizer(tokenizer, test_texts, show_tokens=15)
        
        print_header("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
        logger.info(f"Модель сохранена в: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n !!! Обучение прервано пользователем")
        return 1
    except Exception as e:
        logger.exception(f"Критическая ошибка при обучении: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())