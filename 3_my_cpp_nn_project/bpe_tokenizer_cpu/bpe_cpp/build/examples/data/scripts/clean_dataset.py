#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# clean_dataset.py - Подготовка и очистка датасета C++ кода для BPE
# ======================================================================
#
# @file clean_dataset.py
# @brief Подготовка и очистка датасета C++ кода для обучения BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Извлекает код на C++ из CSV файла, очищает и разделяет на:
#          - train_code.txt (обучающая выборка)   - 8000 примеров
#          - val_code.txt (валидационная выборка) - 800 примеров
#          - test_code.txt  (тестовая выборка)    - 500 примеров
#          Также создает полный корпус corpus.txt со всеми примерами.
#
#          **Процесс обработки:**
#          1. Чтение исходного CSV файла из `data/raw/`
#          2. Извлечение поля `code` из каждой строки
#          3. Очистка от кавычек и лишних пробелов
#          4. Сохранение полного корпуса в `data/corpus/corpus.txt`
#          5. Разделение на train/val/test с помощью sklearn
#          6. Сохранение отдельных файлов в `data/corpus/`
#
# @note Требуется библиотека scikit-learn: pip install scikit-learn
#
# @usage python clean_dataset.py
#
# @example
#   python clean_dataset.py
#   # После выполнения в data/corpus/ появятся файлы:
#   # - corpus.txt     (9322 строк)
#   # - train_code.txt (8000 строк)
#   # - val_code.txt   (800 строк)
#   # - test_code.txt  (500 строк)
#
# ======================================================================

from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split


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
        >>> print_header("ЭТАП 1: ЗАГРУЗКА ДАННЫХ")
        ============================================================
                       ЭТАП 1: ЗАГРУЗКА ДАННЫХ                    
        ============================================================
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

def get_project_root() -> Path:
    """
    Получить корневую директорию проекта.
    
    Returns:
        Path: Путь к корню проекта (bpe_tokenizer/)
    
    **Структура путей:**
    - `clean_dataset.py` находится в `data/scripts/`
    - Корень проекта на два уровня выше
    """
    current_file = Path(__file__).resolve()    # data/scripts/clean_dataset.py
    scripts_dir = current_file.parent          # data/scripts/
    data_dir = scripts_dir.parent              # data/
    project_root = data_dir.parent             # bpe_tokenizer/
    
    return project_root

def parse_csv_line(line: str) -> List[str]:
    """
    Разобрать строку CSV с учетом кавычек и экранирования.
    
    Args:
        line: Строка CSV
        
    Returns:
        List[str]: Список полей
    
    **Поддерживает:**
    - Кавычки внутри полей
    - Экранированные символы
    - Пустые поля
    """
    parts = []
    current = ''
    in_quotes = False
    escape_next = False
    
    for char in line:
        if escape_next:
            current += char
            escape_next = False
        elif char == '\\':
            escape_next = True
            current += char
        elif char == '"':
            in_quotes = not in_quotes
            current += char
        elif char == ',' and not in_quotes:
            parts.append(current)
            current = ''
        else:
            current += char
    parts.append(current)    # Последнее поле
    
    return parts

def clean_code_field(code: str) -> str:
    """
    Очистить поле с кодом от кавычек.
    
    Args:
        code: Сырое поле с кодом
        
    Returns:
        str: Очищенный код
    
    **Удаляет:**
    - Внешние кавычки в начале и конце
    - Лишние пробелы
    """
    # Убираем кавычки в начале и конце
    if code.startswith('"') and code.endswith('"'):
        code = code[1:-1]
    return code.strip()

# ======================================================================
# ОСНОВНОЙ КЛАСС
# ======================================================================

class CppDatasetPreparer:
    """
    Класс для подготовки датасета C++ кода.
    
    Извлекает код из CSV, очищает, сохраняет корпус и разделяет
    на обучающую, валидационную и тестовую выборки с заданными пропорциями.
    
    **Параметры разделения:**
    - Train:      8000 примеров (~86%)
    - Validation: 800 примеров (~8.5%)
    - Test:       500 примеров (~5.5%)
    """
    
    def __init__(self, project_root: str = "bpe_tokenizer"):
        """
        Инициализация подготовщика датасета.
        
        Args:
            project_root: Корневая директория проекта
        """
        self.root = Path(project_root)
        
        # ======================================================================
        # ИСПРАВЛЕНИЕ: Обновленные пути с учетом новой структуры
        # ======================================================================
        
        # Директории
        self.data_dir = self.root / 'data'            # data/
        self.raw_dir = self.data_dir / 'raw'          # data/raw/
        self.corpus_dir = self.data_dir / 'corpus'    # data/corpus/
        
        # Создаем необходимые директории
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Файлы
        self.input_csv = self.raw_dir / '2_cpp_code_generation_dataset.csv'    # data/raw/2_cpp_code_generation_dataset.csv
        self.corpus_file = self.corpus_dir / 'corpus.txt'                      # data/corpus/corpus.txt
        self.train_file = self.corpus_dir / 'train_code.txt'                   # data/corpus/train_code.txt
        self.val_file = self.corpus_dir / 'val_code.txt'                       # data/corpus/val_code.txt
        self.test_file = self.corpus_dir / 'test_code.txt'                     # data/corpus/test_code.txt
        
        print(f"Корневая директория: {self.root}")
        print(f"Директория данных:   {self.data_dir}")
        print(f"Входной CSV:         {self.input_csv}")
        print(f"Выходная директория: {self.corpus_dir}")
    
    # ======================================================================
    # ОСНОВНЫЕ МЕТОДЫ
    # ======================================================================
    
    def extract_codes_raw(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Извлечение данных поля code из CSV.
        
        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: 
                (все коды, train, val, test)
        
        **Процесс:**
        1. Проверка существования входного файла
        2. Чтение CSV построчно
        3. Парсинг с учетом кавычек
        4. Очистка поля code
        5. Сбор всех примеров
        6. Сохранение полного корпуса
        7. Разделение на train/val/test
        8. Сохранение отдельных файлов
        """
        print_header("ЭТАП 2: Подготовка датасета (RAW режим)")
        
        # Проверяем существование входного файла
        if not self.input_csv.exists():
            print(f"Входной файл не найден: {self.input_csv}!")
            print(f"Убедитесь, что файл существует в {self.raw_dir}")
            return [], [], [], []
        
        # Читаем файл
        print(f"\nЧтение {self.input_csv}...")
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Разбираем строки
        lines = content.strip().split('\n')
        print(f"Всего строк в файле: {len(lines)}")
        
        codes = []
        skipped = 0
        
        for i, line in enumerate(lines[1:], 1):    # Пропускаем заголовок
            line = line.strip()
            
            # Пропускаем пустые строки и комментарии
            if not line or line.startswith('#'):
                skipped += 1
                continue
            
            # Разбираем CSV
            parts = parse_csv_line(line)
            
            # Берем второе поле (code)
            if len(parts) >= 2:
                code = clean_code_field(parts[1])
                if code:    # Не пустой
                    codes.append(code)
            else:
                skipped += 1
            
            # Прогресс каждые 10000 строк
            if i % 10000 == 0:
                print(f"Обработано {i} строк, найдено {len(codes)} примеров кода...")
        
        total = len(codes)
        print(f"\nСтатистика:")
        print(f"- Всего примеров кода: {total}")
        print(f"- Пропущено строк: {skipped}")
        
        if total > 0:
            print(f"\nПервый пример:")
            preview = codes[0][:100] + ('...' if len(codes[0]) > 100 else '')
            print(f"    {preview}")
        
        # Сохраняем полный корпус
        self._save_corpus(codes)
        
        # Разделяем на train/val/test
        train, val, test = self._split_dataset(codes)
        
        # Сохраняем разделенные файлы
        self._save_splits(train, val, test)
        
        print_header("ПОДГОТОВКА ДАТАСЕТА ЗАВЕРШЕНА")
        print(f"Статистика:")
        print(f"- Всего примеров: {total}")
        print(f"- train:          {len(train)} ({len(train)/total*100:.1f}%)")
        print(f"- val:            {len(val)} ({len(val)/total*100:.1f}%)")
        print(f"- test:           {len(test)} ({len(test)/total*100:.1f}%)")
        print(f"\nФайлы сохранены в {self.corpus_dir}")
        
        return codes, train, val, test
    
    def _save_corpus(self, codes: List[str]) -> None:
        """
        Сохранить полный корпус в файл.
        
        Args:
            codes: Список кодов
        """
        print(f"\nСоздание {self.corpus_file}...")
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            for code in codes:
                f.write(code + '\n')
        print(f"Сохранено {len(codes)} строк")
        
        # Проверяем сохранение
        self._verify_corpus()
    
    def _verify_corpus(self) -> None:
        """
        Проверить сохраненный корпус.
        
        Проверяет:
        - Существование файла
        - Размер файла
        - Первую строку
        - Наличие экранированных символов
        """
        if not self.corpus_file.exists():
            print(f"X Файл не создан!")
            return
        
        size_kb = self.corpus_file.stat().st_size / 1024
        size_mb = size_kb / 1024
        print(f"Размер файла: {size_kb:.2f} KB ({size_mb:.2f} MB)")
        
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        print(f"\nПроверка corpus.txt:")
        preview = first_line[:100] + ('...' if len(first_line) > 100 else '')
        print(f"    {preview}")
        
        if '\\n' in first_line:
            print(f"Найден символ '\\n' (экранированный перевод строки)")
        else:
            print(f"Символ '\\n' отсутствует (код в одну строку)")
    
    def _split_dataset(self, codes: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Разделить датасет на train/val/test.
        
        Args:
            codes: Список всех кодов
            
        Returns:
            Tuple[List[str], List[str], List[str]]: train, val, test
        
        **Параметры разделения:**
        - test_size: 500 примеров
        - val_size:  800 примеров
        - Остальное: train (обычно ~8000)
        
        @note Использует train_test_split из sklearn с фиксированным random_state
              для воспроизводимости результатов.
        """
        print(f"\nРазделение на train, val и test...")
        
        # Параметры разделения
        test_size = 500
        val_size = 800
        
        # Сначала отделяем тест
        train_val, test = train_test_split(
            codes, 
            test_size=test_size, 
            random_state=42, 
            shuffle=True
        )
        
        # Потом отделяем валидацию от трейна
        train, val = train_test_split(
            train_val, 
            test_size=val_size, 
            random_state=42, 
            shuffle=True
        )
        
        print(f"- train: {len(train)} примеров")
        print(f"- val:   {len(val)} примеров")
        print(f"- test:  {len(test)} примеров")
        
        return train, val, test
    
    def _save_splits(self, train: List[str], val: List[str], test: List[str]) -> None:
        """
        Сохранить разделенные выборки в файлы.
        
        Args:
            train: Обучающая выборка
            val:   Валидационная выборка
            test:  Тестовая выборка
        """
        # Сохраняем train
        with open(self.train_file, 'w', encoding='utf-8') as f:
            for code in train:
                f.write(code + '\n')
        print(f"\nСохранен train: {len(train)} примеров -> {self.train_file.name}")
        
        # Сохраняем val
        with open(self.val_file, 'w', encoding='utf-8') as f:
            for code in val:
                f.write(code + '\n')
        print(f"Сохранен val:   {len(val)} примеров -> {self.val_file.name}")
        
        # Сохраняем test
        with open(self.test_file, 'w', encoding='utf-8') as f:
            for code in test:
                f.write(code + '\n')
        print(f"Сохранен test:  {len(test)} примеров -> {self.test_file.name}")


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция.

    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    try:
        # Определяем корень проекта
        project_root = get_project_root()
        print_header("ПОДГОТОВКА ДАТАСЕТА C++ КОДА")
        print(f"Корень проекта:      {project_root}")
        
        # Создаем подготовщик
        preparer = CppDatasetPreparer(project_root=str(project_root))
        
        # Запускаем извлечение
        codes, train, val, test = preparer.extract_codes_raw()
        
        if not codes:
            print("\nX Не удалось извлечь данные. Проверьте входной файл.")
            print(f"Ожидаемый файл: {preparer.input_csv}")
            return 1
        
        print(f"\nПодготовка датасета успешно завершена!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nОперация прервана пользователем!")
        return 1
    except Exception as e:
        print(f"\nОшибка: {e}!")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()