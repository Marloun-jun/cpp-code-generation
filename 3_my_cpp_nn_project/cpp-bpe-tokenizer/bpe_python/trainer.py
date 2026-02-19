"""
Модуль для обучения BPE токенизатора на корпусе текстов.

Предоставляет функции для загрузки корпуса, обучения токенизатора
и тестирования результатов.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional, Union
from tokenizer import BPETokenizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_from_corpus(
    corpus_path: Union[str, Path],
    vocab_size: int = 8000,
    byte_level: bool = True,
    special_tokens: Optional[List[str]] = None,
    output_dir: Union[str, Path] = './bpe'
) -> BPETokenizer:
    """
    Обучение BPE токенизатора из файла корпуса.

    Аргументы:
        corpus_path: Путь к файлу с корпусом (построчно).
        vocab_size: Размер словаря.
        byte_level: Использовать byte-level режим.
        special_tokens: Специальные токены.
        output_dir: Директория для сохранения результатов.

    Возвращает:
        Обученный экземпляр BPETokenizer.

    Исключения:
        FileNotFoundError: Если файл корпуса не найден.
        ValueError: Если корпус пуст или размер словаря слишком мал.

    Пример:
        >>> tokenizer = train_from_corpus(
        ...     corpus_path='3_my_cpp_nn_project/cpp-bpe-tokenizer/data/corpus.txt',
        ...     vocab_size=10000,
        ...     output_dir='./models'
        ... )
    """
    # Преобразуем пути в объекты Path для удобства
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)

    # Проверяем существование файла
    if not corpus_path.exists():
        raise FileNotFoundError(f"Файл корпуса не найден: {corpus_path}")

    logger.info(f"Загрузка корпуса из {corpus_path}...")

    # Загружаем корпус
    corpus = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Пропускаем пустые строки
                    corpus.append(line)

                # Прогресс для больших файлов
                if line_num % 100000 == 0:
                    logger.info(f"Загружено {line_num} строк...")

    except UnicodeDecodeError as e:
        raise ValueError(f"Ошибка кодировки файла {corpus_path}: {e}")

    if not corpus:
        raise ValueError(f"Корпус пуст или не содержит непустых строк: {corpus_path}")

    logger.info(f"Загружено {len(corpus)} примеров")

    # Инициализируем токенизатор
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        special_tokens=special_tokens
    )

    # Обучаем
    logger.info(f"Обучение BPE токенизатора (размер словаря={vocab_size})...")
    tokenizer.train(corpus, verbose=True)

    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем результаты
    vocab_path = output_dir / 'vocab.json'
    merges_path = output_dir / 'merges.txt'

    tokenizer.save(str(vocab_path), str(merges_path))
    logger.info(f"Токенизатор сохранен в {output_dir}")

    return tokenizer


def test_tokenizer(
    tokenizer: BPETokenizer,
    test_texts: List[str],
    show_tokens: int = 10
) -> None:
    """
    Тестирование токенизатора на примерах текстов.

    Аргументы:
        tokenizer: Обученный токенизатор.
        test_texts: Список тестовых текстов.
        show_tokens: Количество первых токенов для отображения.

    Пример:
        >>> test_texts = ["int main() {", "std::cout << \"Hello\";"]
        >>> test_tokenizer(tokenizer, test_texts)
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print("ТЕСТИРОВАНИЕ ТОКЕНИЗАТОРА")
    print(separator)

    for i, text in enumerate(test_texts, 1):
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)

            print(f"\n{i}. Оригинал: {text}")
            print(f"   Закодировано: {encoded[:show_tokens]}... "
                  f"(длина: {len(encoded)})")
            print(f"   Декодировано: {decoded}")
            print(f"   Совпадение: {'✓' if text == decoded else '✗'}")

            # Показываем первые несколько токенов
            if show_tokens > 0 and encoded:
                tokens = []
                for idx in encoded[:show_tokens]:
                    if idx in tokenizer.vocab:
                        token = tokenizer.vocab[idx]
                        # Экранируем специальные символы для вывода
                        if token in tokenizer.special_tokens:
                            tokens.append(f"[{token}]")
                        else:
                            tokens.append(repr(token))
                print(f"   Токены: {', '.join(tokens)}")

        except Exception as e:
            logger.error(f"Ошибка при тестировании текста '{text}': {e}")


def validate_corpus_file(corpus_path: Union[str, Path]) -> bool:
    """
    Проверка файла корпуса на читаемость и базовую структуру.

    Аргументы:
        corpus_path: Путь к файлу корпуса.

    Возвращает:
        True если файл валиден, иначе False.
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
        if corpus_path.stat().st_size == 0:
            logger.error(f"Файл пуст: {corpus_path}")
            return False

        with open(corpus_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line:
                logger.warning(f"Первая строка файла пуста: {corpus_path}")

        logger.info(f"Файл корпуса валиден: {corpus_path} "
                   f"(размер: {corpus_path.stat().st_size / 1024:.1f} KB)")
        return True

    except UnicodeDecodeError:
        logger.error(f"Ошибка кодировки UTF-8 в файле: {corpus_path}")
        return False
    except Exception as e:
        logger.error(f"Ошибка при проверке файла {corpus_path}: {e}")
        return False


def get_default_corpus_path() -> Path:
    """
    Получение пути к корпусу по умолчанию относительно проекта.

    Возвращает:
        Path объект с путем к корпусу.
    """
    # Текущий файл: bpe_python/trainer.py
    current_file = Path(__file__).resolve()
    bpe_python_dir = current_file.parent
    project_root = bpe_python_dir.parent
    
    return project_root / 'data' / 'corpus' / 'train_code.txt'


if __name__ == '__main__':
    # ================ НАСТРОЙКИ ================
    # Определяем пути автоматически
    current_file = Path(__file__).resolve()  # bpe_python/trainer.py
    bpe_python_dir = current_file.parent  # bpe_python/
    project_root = bpe_python_dir.parent  # cpp-bpe-tokenizer/
    
    # Путь к файлу корпуса
    CORPUS_PATH = project_root / 'data' / 'corpus' / 'train_code.txt'
    
    # Параметры обучения
    VOCAB_SIZE = 12000
    
    # Директория для сохранения модели (в bpe_python/models/bpe_8000/)
    OUTPUT_DIR = bpe_python_dir / 'models' / f'bpe_{VOCAB_SIZE}'
    # ============================================

    # Специальные токены для кода на C++
    SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CPP>', '<CODE>']

    # Создаем директорию для моделей если её нет
    (bpe_python_dir / 'models').mkdir(parents=True, exist_ok=True)

    # Проверяем существование файла корпуса
    if not validate_corpus_file(CORPUS_PATH):
        logger.error(
            f"Файл корпуса не найден по пути: {CORPUS_PATH}\n"
            f"Пожалуйста, поместите файл train_code.txt в:\n"
            f"{project_root / 'data' / 'corpus' / ''}"
        )
        sys.exit(1)

    try:
        # Обучение
        logger.info("=" * 60)
        logger.info("НАЧАЛО ОБУЧЕНИЯ BPE ТОКЕНИЗАТОРА")
        logger.info("=" * 60)
        logger.info(f"Размер словаря: {VOCAB_SIZE}")
        logger.info(f"Корпус: {CORPUS_PATH}")
        logger.info(f"Модель будет сохранена в: {OUTPUT_DIR}")
        logger.info("=" * 60)

        tokenizer = train_from_corpus(
            corpus_path=CORPUS_PATH,
            vocab_size=VOCAB_SIZE,
            byte_level=True,
            special_tokens=SPECIAL_TOKENS,
            output_dir=OUTPUT_DIR
        )

        # Тестирование
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

        logger.info("=" * 60)
        logger.info("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        logger.info(f"Модель сохранена в: {OUTPUT_DIR}")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Критическая ошибка при обучении: {e}")
        sys.exit(1)