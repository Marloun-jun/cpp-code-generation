#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# inspect_vocab.py - Скрипт для анализа структуры vocab.json
# ======================================================================
#
# @file inspect_vocab.py
# @brief Скрипт для детального анализа структуры файла словаря BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Выполняет комплексный анализ файла словаря vocab.json для определения
#          его формата и проверки корректности. Помогает при отладке проблем
#          с загрузкой моделей в C++ и Python реализациях.
#
#          **Определяемые форматы:**
#
#          1) **Формат 1 - {token: id}** (Python формат обучения)
#             - Ключи    - Cтроки (токены)
#             - Значения - Числа (ID)
#             - Используется при обучении в Python
#
#          2) **Формат 2 - {id: token}** (Python формат сохранения)
#             - Ключи    - Числа (ID)   
#             - Значения - Строки (токены)
#             - Используется при сохранении модели через tokenizer.save()
#
#          3) **Формат 3                - {"tokens": [...]}** (C++ формат)
#             - Объект с полем "tokens" - массив строк
#             - ID соответствуют индексам в массиве
#             - Используется в C++ реализации
#
#          **Анализируемые метрики:**
#          - Типы ключей и значений
#          - Диапазон и непрерывность ID
#          - Наличие пропусков в нумерации
#          - Примеры токенов
#          - Наличие специальных символов
#
# @usage python inspect_vocab.py [путь_к_vocab.json]
#
# @example
#   python inspect_vocab.py    # Автоматический поиск
#   python inspect_vocab.py ../bpe_python/models/bpe_8000/vocab.json
#   python inspect_vocab.py ../../bpe_cpp/models/bpe_8000/cpp_vocab.json
#
# ======================================================================

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


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
        >>> print_header("СТАТИСТИКА ID")
        ============================================================
                           СТАТИСТИКА ID                          
        ============================================================
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


def safe_int(value: Any) -> int:
    """
    Безопасное преобразование в int.
    
    Args:
        value: Значение для преобразования
        
    Returns:
        int: Целое число или -1 при ошибке
    
    Поддерживает:
    - int (возвращается как есть)
    - str (пытается преобразовать)
    - другие типы возвращают -1
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return -1
    return -1


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ АНАЛИЗА
# ======================================================================

def inspect_vocab(file_path: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """
    Детальный анализ файла vocab.json.
    
    Args:
        file_path: Путь к файлу словаря
        
    Returns:
        Optional[Union[Dict, List]]: Загруженные данные или None при ошибке
    
    **Процесс анализа:**
    1. Проверка существования файла
    2. Чтение и показ первых 200 символов
    3. Парсинг JSON
    4. Определение типа данных (dict/list)
    5. Анализ структуры и формата
    6. Проверка непрерывности ID
    7. Вывод примеров токенов
    """
    file_path = Path(file_path)
    
    print_header(f"АНАЛИЗ ФАЙЛА: {file_path.name}")
    print(f"Полный путь: {file_path.absolute()}")
    
    if not file_path.exists():
        print(f"Файл не найден!")
        return None
    
    try:
        # Читаем файл
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_size_kb = len(content) / 1024
        print(f"Размер файла: {len(content)} байт ({file_size_kb:.2f} КБ)")
        
        # Показываем начало файла для отладки
        print(f"\nПервые 200 символов:")
        print("-" * 40)
        print(content[:200])
        print("-" * 40)
        
        # Парсим JSON
        data = json.loads(content)
        
        print(f"\nСТРУКТУРА ДАННЫХ:")
        print(f"- Тип: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"- Количество записей: {len(data)}")
            
            # Проверяем типы ключей и значений
            key_types = set(type(k).__name__ for k in data.keys())
            value_types = set(type(v).__name__ for v in data.values())
            
            print(f"- Типы ключей:        {', '.join(key_types)}")
            print(f"- Типы значений:      {', '.join(value_types)}")
            
            # Показываем примеры
            print(f"\nПРИМЕРЫ (первые 10 записей):")
            print("-" * 40)
            items = list(data.items())[:10]
            for i, (key, value) in enumerate(items):
                print(f"{i:2d}. Ключ: '{key}' ({type(key).__name__})")
                print(f"    Знач: '{value}' ({type(value).__name__})")
            print("-" * 40)
            
            # Определяем формат словаря
            print(f"\nФОРМАТ СЛОВАРЯ:")
            
            # ======================================================================
            # ФОРМАТ 1: {token: id} (Python формат обучения)
            # ======================================================================
            if all(isinstance(v, (int, str)) and str(v).isdigit() for v in data.values()) and \
               all(isinstance(k, str) for k in data.keys()):
                sample_token = next(iter(data.keys()))
                sample_id = data[sample_token]
                print(f"Формат: {{'токен': ID}} (Python формат обучения)")
                print(f"Пример: '{sample_token}' -> {sample_id}")
                
                # ID берутся из ЗНАЧЕНИЙ
                ids = []
                for key, value in data.items():
                    val_id = safe_int(value)
                    if val_id >= 0:
                        ids.append(val_id)
                
                if ids:
                    ids = sorted(set(ids))
                    max_id = max(ids)
                    min_id = min(ids)
                    unique_count = len(ids)
                    
                    print(f"\nСТАТИСТИКА ID (из значений):")
                    print(f"- Min ID:        {min_id}")
                    print(f"- Max ID:        {max_id}")
                    print(f"- Уникальных ID: {unique_count}")
                    print(f"- Диапазон:      {max_id - min_id + 1}")
                    print(f"- Пропусков:     {(max_id - min_id + 1) - unique_count}")
                    
                    if unique_count == max_id + 1:
                        print(f"\nID непрерывны от 0 до {max_id}")
                    else:
                        print(f"\nЕсть пропуски в ID!")
                        if max_id - min_id + 1 - unique_count > 0:
                            missing = []
                            for i in range(min_id, max_id + 1):
                                if i not in ids:
                                    missing.append(i)
                            print(f"Пропущенные ID: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            
            # ======================================================================
            # ФОРМАТ 2: {id: token} (Python формат сохранения)
            # ======================================================================
            elif all(isinstance(k, (int, str)) and str(k).isdigit() for k in data.keys()) and \
                 all(isinstance(v, str) for v in data.values()):
                sample_id = next(iter(data.keys()))
                sample_token = data[sample_id]
                print(f"Формат: {{ID: 'токен'}} (Python формат сохранения)")
                print(f"Пример: {sample_id} -> '{sample_token}'")
                
                # ID берутся из КЛЮЧЕЙ
                ids = []
                for key, value in data.items():
                    key_id = safe_int(key)
                    if key_id >= 0:
                        ids.append(key_id)
                
                if ids:
                    ids = sorted(set(ids))
                    max_id = max(ids)
                    min_id = min(ids)
                    unique_count = len(ids)
                    
                    print(f"\nСТАТИСТИКА ID (из ключей):")
                    print(f"- Min ID:        {min_id}")
                    print(f"- Max ID:        {max_id}")
                    print(f"- Уникальных ID: {unique_count}")
                    print(f"- Диапазон:      {max_id - min_id + 1}")
                    print(f"- Пропусков:     {(max_id - min_id + 1) - unique_count}")
                    
                    if unique_count == max_id + 1:
                        print(f"\nID непрерывны от 0 до {max_id}")
                    else:
                        print(f"\nЕсть пропуски в ID!")
                        if max_id - min_id + 1 - unique_count > 0:
                            missing = []
                            for i in range(min_id, max_id + 1):
                                if i not in ids:
                                    missing.append(i)
                            print(f"Пропущенные ID: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            
            # ======================================================================
            # ФОРМАТ 3: {"tokens": [...]} (C++ формат)
            # ======================================================================
            elif "tokens" in data and isinstance(data["tokens"], list):
                print(f"Формат:             {{\"tokens\": [...]}} (C++ формат)")
                print(f"Количество токенов: {len(data['tokens'])}")
                if len(data['tokens']) > 0:
                    print(f"Первый токен:       '{data['tokens'][0]}'")
                    print(f"Последний токен:    '{data['tokens'][-1]}'")
                
                # Для C++ формата ID соответствуют индексам в массиве
                tokens = data['tokens']
                print(f"\nСТАТИСТИКА ТОКЕНОВ:")
                print(f"- Всего токенов:      {len(tokens)}")
                
                # Длины токенов
                lengths = [len(t) for t in tokens]
                avg_len = sum(lengths) / len(lengths)
                max_len = max(lengths)
                min_len = min(lengths)
                
                print(f"- Мин. длина токена:  {min_len}")
                print(f"- Макс. длина токена: {max_len}")
                print(f"- Средняя длина:      {avg_len:.2f}")
                
                # Проверяем наличие специальных символов
                special_chars = [' ', '\n', '\t', '\r', '<', '>', '/', '\\', '(', ')', '{', '}', '[', ']']
                found_chars = []
                for i, token in enumerate(tokens[:1000]):    # Проверяем первые 1000
                    for char in special_chars:
                        if char in token:
                            found_chars.append((i, repr(token)))
                            break
                
                if found_chars:
                    print(f"\nНайдены специальные символы (первые 10):")
                    for idx, char_repr in found_chars[:10]:
                        print(f"    ID {idx}: {char_repr}")
                else:
                    print(f"\nСпециальные символы не найдены в первых 1000 токенах!")
            
            else:
                print(f"Не удалось определить формат!")
        
        elif isinstance(data, list):
            print(f"Длина массива: {len(data)}")
            
            print(f"\nПРИМЕРЫ (первые 10 токенов):")
            print("-" * 40)
            for i, token in enumerate(data[:10]):
                print(f"{i:2d}. '{token}'")
            print("-" * 40)
            
            print(f"\nЭто простой массив токенов")
            
            # Длины токенов для списка
            lengths = [len(t) for t in data]
            avg_len = sum(lengths) / len(lengths)
            max_len = max(lengths)
            min_len = min(lengths)
            
            print(f"\nСТАТИСТИКА ТОКЕНОВ:")
            print(f"- Мин. длина:    {min_len}")
            print(f"- Макс. длина:   {max_len}")
            print(f"- Средняя длина: {avg_len:.2f}")
        
        else:
            print(f"Содержимое: {data}")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"\nОШИБКА ПАРСИНГА JSON:")
        print(f"{e}")
        print(f"\nПозиция ошибки: строка {e.lineno}, колонка {e.colno}")
        print(f"Текст ошибки: {e.msg}")
        return None
    except Exception as e:
        print(f"\nОШИБКА: {e}!")
        return None


def find_vocab_file() -> Optional[Path]:
    """
    Поиск файла vocab.json в стандартных местах проекта.
    
    Returns:
        Optional[Path]: Путь к файлу или None
    
    **Поиск осуществляется в:**
    - C++ модели: bpe_cpp/models/bpe_8000/cpp_vocab.json, bpe_10000/cpp_vocab.json, bpe_12000/cpp_vocab.json
    - Python модели: bpe_python/models/bpe_8000/vocab.json, bpe_10000/vocab.json, bpe_12000/vocab.json
    """
    # Пути относительно текущей директории (scripts/)
    script_path = Path(__file__).resolve()
    scripts_dir = script_path.parent
    project_root = scripts_dir.parent
    
    possible_paths = [
        # Текущая директория
        scripts_dir / "cpp_vocab.json",
        
        # C++ модели (bpe_8000, bpe_10000, bpe_12000)
        project_root / "bpe_cpp" / "models" / "bpe_8000" / "cpp_vocab.json",
        project_root / "bpe_cpp" / "models" / "bpe_10000" / "cpp_vocab.json",
        project_root / "bpe_cpp" / "models" / "bpe_12000" / "cpp_vocab.json",
        
        # Python модели
        project_root / "bpe_python" / "models" / "bpe_8000" / "vocab.json",
        project_root / "bpe_python" / "models" / "bpe_10000" / "vocab.json",
        project_root / "bpe_python" / "models" / "bpe_12000" / "vocab.json",
        
        # Относительные пути для запуска из разных мест
        Path("bpe_cpp/models/bpe_8000/cpp_vocab.json"),
        Path("bpe_cpp/models/bpe_10000/cpp_vocab.json"),
        Path("bpe_cpp/models/bpe_12000/cpp_vocab.json"),
        Path("bpe_python/models/bpe_8000/vocab.json"),
        Path("bpe_python/models/bpe_10000/vocab.json"),
        Path("bpe_python/models/bpe_12000/vocab.json"),
        
        # Пути для запуска из корня проекта
        Path("bpe_tokenizer_cpu/bpe_cpp/models/bpe_8000/cpp_vocab.json"),
        Path("bpe_tokenizer_cpu/bpe_cpp/models/bpe_10000/cpp_vocab.json"),
        Path("bpe_tokenizer_cpu/bpe_cpp/models/bpe_12000/cpp_vocab.json"),
        Path("bpe_tokenizer_cpu/bpe_python/models/bpe_8000/vocab.json"),
        Path("bpe_tokenizer_cpu/bpe_python/models/bpe_10000/vocab.json"),
        Path("bpe_tokenizer_cpu/bpe_python/models/bpe_12000/vocab.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция.

    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    # Парсим аргументы командной строки
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Ищем файл автоматически
        found_path = find_vocab_file()
        if found_path:
            file_path = str(found_path)
            print(f"Найден файл по пути: {file_path}")
        else:
            print_header("ИНСПЕКТОР СЛОВАРЯ VOCAB.JSON")
            print("\nФайл vocab.json не найден в стандартных местах!")
            print("\nУкажите путь к файлу:")
            print("python inspect_vocab.py <path_to_vocab.json>")
            print("\nПримеры:")
            print("python inspect_vocab.py ../bpe_python/models/bpe_8000/vocab.json")
            print("python inspect_vocab.py ../../bpe_cpp/models/bpe_8000/cpp_vocab.json")
            print("python inspect_vocab.py bpe_tokenizer_cpu/bpe_cpp/models/bpe_8000/cpp_vocab.json")
            return 1
    
    # Анализируем файл
    data = inspect_vocab(file_path)
    
    if data is None:
        return 1
    
    print_header("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    return 0


if __name__ == "__main__":
    sys.exit(main())