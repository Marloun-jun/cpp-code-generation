#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# convert_vocab.py - Конвертация словаря BPE токенизатора из Python в C++ формат
# ======================================================================
#
# @file convert_vocab.py
# @brief Конвертирует словарь из Python формата в формат, оптимизированный для C++
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Этот скрипт преобразует словари токенизатора из Python-формата
#          в формат, удобный для использования в C++ реализации.
#
#          **Поддерживаемые входные форматы:**
#          - **ID -> Токен**    - {"0": "<PAD>", "1": "<UNK>", ...}
#          - **Токен -> ID**    - {"<PAD>": 0, "<UNK>": 1, ...}
#          - **Массив токенов** - ["<PAD>", "<UNK>", ...]
#
#          **Выходной формат (C++):**
#          ```json
#          {
#            "size":   8000,
#            "tokens": ["<PAD>", "<UNK>", "int", "main", ...]
#          }
#          ```
#
#          **Дополнительные возможности:**
#          - Автоматическое определение формата входных данных
#          - Добавление недостающих специальных символов (пробел, табуляция и т.д.)
#          - Сжатие словаря (удаление пропусков в ID)
#          - Валидация UTF-8 и анализ статистики
#          - Копирование файла слияний (merges.txt)
#          - Проверка непрерывности индексов
#          - Приведение к нужному размеру модели
#
# @usage python tools/convert_vocab.py [options]
#
# @options
#   --no-fill            Не заполнять пропуски, а переиндексировать (сжать ID)
#   --inspect-only       Только проанализировать словарь без конвертации
#   --format FORMAT      Принудительный формат (auto|id_to_token|token_to_id|array)
#   --model-size SIZE    Размер модели (8000, 10000, 12000) - по умолч. 8000
#   --input-dir DIR      Входная директория с Python моделями
#   --output-dir DIR     Выходная директория для C++ моделей
#   --verbose (-v)       Подробный вывод
#   --validate           Проверить токены на корректность
#   --strict             Строгий режим - прерывать при ошибках
#
# @example
#   python tools/convert_vocab.py                       # обычная конвертация
#   python tools/convert_vocab.py --no-fill             # сжатие словаря
#   python tools/convert_vocab.py --inspect-only        # только анализ
#   python tools/convert_vocab.py --model-size 8000     # модель 8000
#   python tools/convert_vocab.py --input-dir ../bpe_python/models
#
# ======================================================================

import json
import sys
import shutil
import argparse

from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any, Set

# ======================================================================
# Конфигурация
# ======================================================================

# Специальные токены с описанием их назначения
SPECIAL_TOKENS = {
    "<PAD>": "Токен для выравнивания последовательностей",
    "<UNK>": "Токен для неизвестных символов",
    "<BOS>": "Токен начала последовательности",
    "<EOS>": "Токен конца последовательности",
    "<MASK>": "Токен для маскирования (опционально)",
    "<CPP>": "Токен для обозначения C++ кода",
    "<CODE>": "Токен для обозначения блока кода"
}

# Специальные символы, которые должны быть в словаре для корректной работы
SPECIAL_CHARS = [
    ' ',     # пробел (ASCII 32)          - разделитель слов
    '\n',    # перевод строки (ASCII 10)  - конец строки
    '\t',    # табуляция (ASCII 9)        - отступы
    '\r',    # возврат каретки (ASCII 13) - Windows-совместимость
    '\\',    # обратный слеш              - escape-последовательности
    '\"',    # двойная кавычка            - строковые литералы
    '\'',    # одинарная кавычка          - символьные литералы
]

# Допустимые размеры моделей
VALID_MODEL_SIZES = {8000, 10000, 12000}

# ======================================================================
# Функции определения формата
# ======================================================================

def detect_format(data: Any) -> str:
    """
    Определяет формат словаря.
    
    Args:
        data: Загруженные JSON данные
    
    Returns:
        str: Один из: "id_to_token", "token_to_id", "array", "unknown"
    
    **Алгоритм определения:**
    1. Если данные - список -> "array"
    2. Если данные - словарь:
       - Проверяем ключи:    все ли они числа (строковые или int) -> "id_to_token"
       - Проверяем значения: все ли они числа (строковые или int) -> "token_to_id"
    3. Иначе -> "unknown"
    """
    if isinstance(data, list):
        return "array"
    
    if not isinstance(data, dict):
        return "unknown"
    
    # Проверяем ключи - все ли они числа (строковые или int)
    keys_are_numeric = True
    sample_keys = list(data.keys())[:100]
    
    for k in sample_keys:
        if isinstance(k, (int, float)):
            continue
        elif isinstance(k, str) and k.isdigit():
            continue
        else:
            keys_are_numeric = False
            break
    
    # Проверяем значения - все ли они строки (токены)
    sample_values = list(data.values())[:100]
    values_are_strings = all(isinstance(v, str) for v in sample_values)
    
    if keys_are_numeric and values_are_strings:
        return "id_to_token"    # {"0": "<PAD>", "1": "<UNK>", ...}
    
    # Проверяем обратный формат (токен -> ID)
    keys_are_strings = all(isinstance(k, str) for k in sample_keys)
    values_are_numeric = all(isinstance(v, (int, str)) and str(v).isdigit() 
                            for v in sample_values)
    
    if keys_are_strings and values_are_numeric:
        return "token_to_id"    # {"<PAD>": 0, "<UNK>": 1, ...}
    
    return "unknown"


def check_index_continuity(data: Dict, format_type: str) -> Tuple[bool, List[int], int]:
    """
    Проверяет непрерывность индексов в словаре.
    
    Args:
        data:        Исходные данные
        format_type: Тип формата
    
    Returns:
        Tuple[bool, List[int], int]: (есть ли пропуски, список отсутствующих индексов, максимальный ID)
    
    **Важность:**
    Пропуски в индексах приведут к пустым строкам в C++ массиве и смещению всех
    последующих токенов. Например, если нет индекса 100, то токен с ID 101
    в C++ станет индексом 100, что вызовет несоответствие при токенизации.
    """
    if format_type == "id_to_token":
        # {"0": "<PAD>", "1": "<UNK>", ...}
        indices = []
        for key in data.keys():
            if isinstance(key, (int, float)):
                indices.append(int(key))
            elif isinstance(key, str) and key.isdigit():
                indices.append(int(key))
        
        if not indices:
            return False, [], -1
        
        indices.sort()
        max_idx = max(indices)
        expected = set(range(max_idx + 1))
        actual = set(indices)
        missing = sorted(expected - actual)
        
        return len(missing) > 0, missing, max_idx
    
    elif format_type == "token_to_id":
        # {"<PAD>": 0, "<UNK>": 1, ...}
        indices = []
        for value in data.values():
            if isinstance(value, (int, float)):
                indices.append(int(value))
            elif isinstance(value, str) and value.isdigit():
                indices.append(int(value))
        
        if not indices:
            return False, [], -1
        
        indices.sort()
        max_idx = max(indices)
        expected = set(range(max_idx + 1))
        actual = set(indices)
        missing = sorted(expected - actual)
        
        return len(missing) > 0, missing, max_idx
    
    return False, [], -1


def validate_vocab_size(tokens: List[str], expected_size: int, strict: bool = False) -> bool:
    """
    Проверяет соответствие размера словаря ожидаемому.
    
    Args:
        tokens:        Список токенов
        expected_size: Ожидаемый размер
        strict:        Строгий режим - не изменять размер
    
    Returns:
        bool: True если размер соответствует или был успешно скорректирован
    
    **Действия:**
    - Если токенов меньше - добавляет заглушки <EXTRA_N>
    - Если токенов больше - обрезает до нужного размера
    """
    actual_size = len(tokens)
    
    if actual_size == expected_size:
        return True
    
    print(f"\nНЕСООТВЕТСТВИЕ РАЗМЕРА СЛОВАРЯ!")
    print(f"   Ожидалось: {expected_size} токенов")
    print(f"   Получено:  {actual_size} токенов")
    
    if strict:
        print("   Строгий режим: прерывание выполнения")
        return False
    
    if actual_size < expected_size:
        # Добавляем заглушки до нужного размера
        needed = expected_size - actual_size
        print(f"   Добавляется {needed} заглушек <EXTRA_N>")
        for i in range(actual_size, expected_size):
            tokens.append(f"<EXTRA_{i}>")
        print(f"   Новый размер: {len(tokens)}")
        return True
    else:
        # Обрезаем до нужного размера
        extra = actual_size - expected_size
        print(f"   Удаляется {extra} лишних токенов (с конца)")
        del tokens[expected_size:]
        print(f"   Новый размер: {len(tokens)}")
        return True

# ======================================================================
# Функции конвертации
# ======================================================================

def convert_id_to_token_format(data: Dict, verbose: bool = False) -> Tuple[Dict, List[int]]:
    """
    Конвертирует формат {"id": "token"} в массив токенов.
    
    Args:
        data:    Словарь в формате ID -> токен
        verbose: Подробный вывод
    
    Returns:
        Tuple[Dict, List[int]]:
           (конвертированные данные, список пропущенных ID)
    
    **Процесс:**
    1. Извлекаем все пары ID -> токен
    2. Находим максимальный ID
    3. Создаем массив токенов длиной max_id + 1
    4. Заполняем массив по индексам
    5. Отмечаем пропущенные позиции
    """
    print("Обнаружен формат: ID -> ТОКЕН")
    
    # Находим максимальный ID
    max_id = -1
    id_to_token = {}
    skipped = 0
    
    for key, token in data.items():
        # Конвертируем ключ в число
        if isinstance(key, (int, float)):
            numeric_id = int(key)
        elif isinstance(key, str) and key.isdigit():
            numeric_id = int(key)
        else:
            if verbose:
                print(f"Пропускаем нечисловой ключ: '{key}'")
            skipped += 1
            continue
        
        id_to_token[numeric_id] = token
        max_id = max(max_id, numeric_id)
    
    print(f"Найдено записей:      {len(id_to_token)}")
    if skipped > 0:
        print(f"Пропущено записей:    {skipped}")
    print(f"Максимальный ID:      {max_id}")
    
    # Создаем массив токенов
    tokens = [''] * (max_id + 1)
    
    for id_val, token in id_to_token.items():
        tokens[id_val] = token
    
    # Проверяем пропуски
    missing = [i for i, t in enumerate(tokens) if not t]
    print(f"Пропусков в ID: {len(missing)}")
    
    if missing and verbose:
        print(f"Примеры пропусков: {missing[:10]}")
    
    return {
        "size": len(tokens),
        "tokens": tokens
    }, missing


def convert_token_to_id_format(data: Dict, verbose: bool = False) -> Tuple[Dict, List[int]]:
    """
    Конвертирует формат {"token": id} в массив токенов.
    
    Args:
        data:    Словарь в формате токен -> ID
        verbose: Подробный вывод
    
    Returns:
        Tuple[Dict, List[int]]:
            (конвертированные данные, список пропущенных ID)
    
    **Процесс:**
    1. Извлекаем все пары токен -> ID
    2. Находим максимальный ID
    3. Создаем массив токенов длиной max_id + 1
    4. Заполняем массив по индексам
    5. Отмечаем пропущенные позиции
    """
    print("Обнаружен формат: ТОКЕН -> ID")
    
    token_to_id = {}
    max_id = -1
    skipped = 0
    
    for token, id_val in data.items():
        # Конвертируем значение в число
        if isinstance(id_val, (int, float)):
            numeric_id = int(id_val)
        elif isinstance(id_val, str) and id_val.isdigit():
            numeric_id = int(id_val)
        else:
            if verbose:
                print(f"Пропускаем запись с нечисловым ID: {token} -> {id_val}")
            skipped += 1
            continue
        
        token_to_id[token] = numeric_id
        max_id = max(max_id, numeric_id)
    
    print(f"Найдено записей:      {len(token_to_id)}")
    if skipped > 0:
        print(f"Пропущено записей:    {skipped}")
    print(f"Максимальный ID:      {max_id}")
    
    # Создаем массив токенов
    tokens = [''] * (max_id + 1)
    
    for token, id_val in token_to_id.items():
        tokens[id_val] = token
    
    # Проверяем пропуски
    missing = [i for i, t in enumerate(tokens) if not t]
    print(f"Пропусков в ID: {len(missing)}")
    
    if missing and verbose:
        print(f"Примеры пропусков: {missing[:10]}")
    
    return {
        "size": len(tokens),
        "tokens": tokens
    }, missing


def convert_array_format(data: List, verbose: bool = False) -> Dict:
    """
    Конвертирует формат [token1, token2] в нужный формат.
    
    Args:
        data:    Массив токенов
        verbose: Подробный вывод
    
    Returns:
        Dict: Конвертированные данные
    
    **Особенности:**
    - Индекс в массиве становится ID токена
    - Специальные токены должны быть в начале
    """
    print("Обнаружен формат: МАССИВ токенов")
    print(f"Количество токенов: {len(data)}")
    
    # Проверяем, есть ли специальные токены в начале
    special_found = []
    for i, token in enumerate(data[:10]):
        if token in SPECIAL_TOKENS:
            special_found.append(token)
    
    if special_found:
        print(f"Найдены специальные токены: {', '.join(special_found)}")
    
    return {
        "size": len(data),
        "tokens": data
    }

# ======================================================================
# Функции пост-обработки
# ======================================================================

def add_special_chars(tokens: List[str], verbose: bool = False) -> List[str]:
    """
    Добавляет специальные символы, которых может не быть в словаре.
    
    Args:
        tokens:  Исходный список токенов
        verbose: Подробный вывод
    
    Returns:
        List[str]: Обновленный список токенов
    
    **Зачем это нужно:**
    C++ токенизатор ожидает наличия базовых символов в словаре,
    иначе они будут заменены на <UNK>, что приведет к потере информации.
    """
    print("\nПроверка наличия специальных символов...")
    added = 0
    existing = set(tokens)
    
    for char in SPECIAL_CHARS:
        if char not in existing:
            tokens.append(char)
            added += 1
            # Показываем символ в читаемом виде
            display = repr(char).strip("'")
            print(f"Добавлен специальный символ: {display}")
    
    if added > 0:
        print(f"Добавлено специальных символов: {added}")
    else:
        print("Все специальные символы уже присутствуют!")
    
    return tokens


def compress_tokens(tokens: List[str], missing: List[int], verbose: bool = False) -> Dict:
    """
    Сжимает токены, удаляя пропуски и переиндексируя.
    
    Args:
        tokens:  Исходный массив токенов с пропусками
        missing: Список индексов пропусков
        verbose: Подробный вывод
    
    Returns:
        Dict: Сжатый словарь без пропусков
    
    **Важно:** После сжатия индексы токенов изменятся!
    Например, токен с ID 100 может стать ID 95.
    Это критически важно для совместимости с предобученными моделями!
    """
    print(f"\nСжатие токенов (удаление пропусков)...")
    
    # Создаем список только с реальными токенами
    real_tokens = []
    old_to_new = {}
    
    new_idx = 0
    for old_idx, token in enumerate(tokens):
        if token:    # если токен не пустой
            real_tokens.append(token)
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    print(f"Было токенов (с пропусками): {len(tokens)}")
    print(f"Стало реальных токенов:      {len(real_tokens)}")
    print(f"Удалено пропусков:           {len(tokens) - len(real_tokens)}")
    
    if len(tokens) > 0:
        print(f"Экономия места:              {(1 - len(real_tokens)/len(tokens))*100:.1f}%")
    
    # Показываем примеры переиндексации
    if verbose:
        print("\nПримеры переиндексации (старый ID -> новый ID):")
        examples_shown = 0
        for old_idx in sorted(old_to_new.keys())[:10]:
            print(f"  {old_idx:4d} -> {old_to_new[old_idx]:4d}")
            examples_shown += 1
        
        if len(old_to_new) > 10:
            print(f"  ... и еще {len(old_to_new) - 10}")
    
    return {
        "size": len(real_tokens),
        "tokens": real_tokens,
        "old_to_new_mapping": old_to_new
    }


def fill_missing_with_placeholders(tokens: List[str], missing: List[int]) -> List[str]:
    """
    Заполняет пропуски заглушками вместо сжатия.
    
    Args:
        tokens:  Массив токенов с пропусками
        missing: Список индексов пропусков
    
    Returns:
        List[str]: Массив с заполненными пропусками
    
    **Недостатки этого подхода:**
    - Увеличивает размер словаря
    - Создает "дыры" в индексации
    - Может привести к путанице при отладке
    """
    print(f"\nЗаполнение пропусков заглушками...")
    
    for i in missing:
        if not tokens[i]:
            tokens[i] = f"<MISSING_{i}>"
    
    print(f"Заполнено пропусков: {len(missing)}")
    return tokens


def validate_tokens(tokens: List[str]) -> List[str]:
    """
    Проверяет токены на корректность UTF-8.
    
    Args:
        tokens: Список токенов для проверки
    
    Returns:
        List[str]: Список проблемных токенов
    
    **Проверяет:**
    - Пустые токены (кроме специальных)
    - Слишком длинные токены (>1000 символов)
    - Некорректный UTF-8
    - Дубликаты токенов
    """
    issues = []
    seen_tokens = set()
    
    for i, token in enumerate(tokens):
        if not token:
            issues.append(f"ID {i}: пустой токен!")
            continue
        
        # Проверяем на дубликаты
        if token in seen_tokens:
            issues.append(f"ID {i}: дубликат токена '{token}'!")
        else:
            seen_tokens.add(token)
        
        # Проверяем длину
        if len(token) > 1000:
            issues.append(f"ID {i}: слишком длинный токен ({len(token)} символов)!")
        
        # Проверяем на некорректный UTF-8
        try:
            token.encode('utf-8').decode('utf-8')
        except UnicodeError:
            issues.append(f"ID {i}: некорректный UTF-8!")
        
        # Проверяем управляющие символы (кроме разрешенных)
        if len(token) == 1:
            c = token[0]
            if ord(c) < 32 and c not in SPECIAL_CHARS:
                issues.append(f"ID {i}: неразрешенный управляющий символ ASCII {ord(c)}")
    
    return issues

# ======================================================================
# Функции анализа
# ======================================================================

def inspect_vocab_file(file_path: Path, verbose: bool = False) -> Optional[Any]:
    """
    Анализирует структуру vocab.json и выводит информацию.
    
    Args:
        file_path: Путь к файлу словаря
        verbose:   Подробный вывод
    
    Returns:
        Optional[Any]: Загруженные данные или None при ошибке
    
    **Выводит:**
    - Первые 200 символов файла для предпросмотра
    - Тип данных (dict/list)
    - Количество записей
    - Примеры первых 5 записей
    - Определенный формат
    - Статистика типов ключей
    """
    try:
        # Сначала читаем небольшой кусок для предпросмотра
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(500)
            
        print(f"\nАнализ файла: {file_path}")
        print(f"Первые 200 символов:")
        print("-" * 40)
        print(sample[:200])
        print("-" * 40)
        
        # Читаем полностью
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nСтруктура данных:")
        print(f"- тип: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"- количество записей: {len(data)}")
            
            # Показываем примеры первых 5 записей
            items = list(data.items())[:5]
            for k, v in items:
                k_type = type(k).__name__
                v_type = type(v).__name__
                print(f"  '{k}' ({k_type}) -> '{v}' ({v_type})")
            
            # Определяем формат
            format_type = detect_format(data)
            print(f"- формат: {format_type}")
            
            # Анализируем типы ключей
            key_types = Counter(type(k).__name__ for k in data.keys())
            print(f"- типы ключей: {dict(key_types)}")
            
            # Проверяем непрерывность индексов
            has_gaps, missing, max_idx = check_index_continuity(data, format_type)
            if has_gaps:
                print(f"- пропуски в индексах: {len(missing)}")
                if verbose:
                    print(f"    отсутствуют: {missing[:20]}")
            
        elif isinstance(data, list):
            print(f"- количество элементов: {len(data)}")
            print(f"- первые 5: {data[:5]}")
        
        return data
        
    except Exception as e:
        print(f"Ошибка анализа: {e}")
        return None


def print_statistics(tokens: List[str], title: str = "Статистика"):
    """
    Выводит статистику по токенам.
    
    Args:
        tokens: Список токенов
        title:  Заголовок
    
    **Статистика:**
    - Общее количество токенов
    - Средняя/мин/макс длина
    - Количество односимвольных/многосимвольных
    - Количество специальных токенов и символов
    """
    print(f"\n{title}:")
    print(f"Всего токенов: {len(tokens)}")
    
    # Статистика по длинам
    lengths = [len(t) for t in tokens if t]
    if lengths:
        print(f"Средняя длина: {sum(lengths)/len(lengths):.1f} символов")
        print(f"Макс. длина:   {max(lengths)} символов")
        print(f"Мин. длина:    {min(lengths)} символов")
    
    # Статистика по типам
    single_char = sum(1 for t in tokens if len(t) == 1)
    multi_char = sum(1 for t in tokens if len(t) > 1)
    special_tokens = sum(1 for t in tokens if t in SPECIAL_TOKENS)
    special_chars = sum(1 for t in tokens if t in SPECIAL_CHARS)
    
    print(f"Односимвольных:       {single_char}")
    print(f"Многосимвольных:      {multi_char}")
    print(f"Специальных токенов:  {special_tokens}")
    print(f"Специальных символов: {special_chars}")
    
    # Проверяем наличие базовых символов
    missing_chars = [c for c in SPECIAL_CHARS if c not in tokens]
    if missing_chars:
        print(f"Отсутствуют символы: {[repr(c) for c in missing_chars]}!")


def save_mapping_file(mapping: Dict[int, int], output_dir: Path, model_size: int) -> None:
    """
    Сохраняет маппинг старых ID в новые ID в отдельный файл.
    
    Args:
        mapping:    Словарь {старый_id: новый_id}
        output_dir: Директория для сохранения
        model_size: Размер модели
    """
    mapping_path = output_dir / f"id_mapping_{model_size}.json"
    
    # Конвертируем ключи в строки для JSON
    str_mapping = {str(k): v for k, v in mapping.items()}
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(str_mapping, f, indent=2)
    
    print(f"Маппинг ID сохранен в: {mapping_path}")

# ======================================================================
# Основная функция
# ======================================================================

def main() -> int:
    """Точка входа в программу"""
    
    parser = argparse.ArgumentParser(
        description='Преобразовать словарь BPE в формат C++',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/convert_vocab.py                      # обычная конвертация
  python tools/convert_vocab.py --no-fill            # сжатие словаря
  python tools/convert_vocab.py --inspect-only       # только анализ
  python tools/convert_vocab.py --model-size 8000    # модель 8000
  python tools/convert_vocab.py --input-dir ../bpe_python/models
        """
    )
    
    parser.add_argument('--no-fill', action='store_true',
                       help='Не заполнять пропуски, а переиндексировать (сжать ID)')
    
    parser.add_argument('--inspect-only', action='store_true',
                       help='Только проанализировать словарь без конвертации')
    
    parser.add_argument('--format', choices=['auto', 'id_to_token', 'token_to_id', 'array'],
                       default='auto', help='Принудительный формат словаря')
    
    parser.add_argument('--model-size', type=int, choices=sorted(VALID_MODEL_SIZES),
                       default=8000, help='Размер модели (8000, 10000, 12000)')
    
    parser.add_argument('--input-dir', type=str, default='../bpe_python/models',
                    help='Входная директория с Python моделями')
    
    parser.add_argument('--output-dir', type=str, default='models',
                   help='Выходная директория для C++ моделей')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    parser.add_argument('--validate', action='store_true',
                       help='Проверить токены на корректность')
    
    parser.add_argument('--strict', action='store_true',
                       help='Строгий режим - прерывать при ошибках')
    
    args = parser.parse_args()
    
    print("============================================================")
    print("КОНВЕРТАЦИЯ СЛОВАРЯ BPE ИЗ PYTHON В C++ ФОРМАТ")
    print("============================================================")
    
    # Определяем пути относительно скрипта
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent    # bpe_cpp/
    
    # Формируем пути с учетом размера модели
    python_vocab = project_root / args.input_dir / f"bpe_{args.model_size}" / "vocab.json"
    python_merges = project_root / args.input_dir / f"bpe_{args.model_size}" / "merges.txt"
    output_dir = project_root / args.output_dir / f"bpe_{args.model_size}"
    
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nПути к файлам:")
    print(f"- python словарь: {python_vocab}")
    print(f"- python слияния: {python_merges}")
    print(f"- выходная папка: {output_dir}")
    
    if not python_vocab.exists():
        print(f"\nФайл не найден: {python_vocab}!")
        print(f"Проверьте путь или укажите другой размер модели через --model-size")
        print(f"\nДоступные размеры: {sorted(VALID_MODEL_SIZES)}")
        return 1
    
    if args.inspect_only:
        inspect_vocab_file(python_vocab, args.verbose)
        return 0
    
    # Анализируем
    print("\n" + "-" * 40)
    print("АНАЛИЗ ИСХОДНОГО СЛОВАРЯ")
    print("-" * 40)
    data = inspect_vocab_file(python_vocab, args.verbose)
    
    if data is None:
        print("Не удалось загрузить словарь!")
        return 1
    
    # Определяем формат
    if args.format != 'auto':
        format_type = args.format
        print(f"\nПринудительный формат: {format_type}")
    else:
        format_type = detect_format(data)
        print(f"\nОпределенный формат: {format_type}")
    
    if format_type == "unknown":
        print("Не удалось определить формат словаря!")
        return 1
    
    # Проверяем непрерывность индексов
    has_gaps, missing_indices, max_idx = check_index_continuity(data, format_type)
    if has_gaps:
        print(f"\nОБНАРУЖЕНЫ ПРОПУСКИ В ИНДЕКСАХ!")
        print(f"   Отсутствуют индексы: {missing_indices[:20]}")
        if len(missing_indices) > 20:
            print(f"   ... и еще {len(missing_indices) - 20}")
        print(f"   Максимальный ID: {max_idx}")
        print(f"   Это может вызвать смещение токенов в C++ версии!")
    
    # Конвертируем в зависимости от формата
    if format_type == "id_to_token":
        cpp_data, missing = convert_id_to_token_format(data, args.verbose)
    elif format_type == "token_to_id":
        cpp_data, missing = convert_token_to_id_format(data, args.verbose)
    elif format_type == "array":
        cpp_data = convert_array_format(data, args.verbose)
        missing = []
    else:
        print("Неподдерживаемый формат!")
        return 1
    
    # Извлекаем токены
    tokens = cpp_data["tokens"]
    
    # Добавляем специальные символы
    tokens = add_special_chars(tokens, args.verbose)
    
    # Применяем сжатие или заполнение
    mapping = None
    if args.no_fill and missing:
        # Сжимаем словарь (удаляем пропуски)
        compressed = compress_tokens(tokens, missing, args.verbose)
        tokens = compressed["tokens"]
        mapping = compressed.get("old_to_new_mapping")
    elif missing and not args.no_fill:
        # Заполняем пропуски заглушками
        tokens = fill_missing_with_placeholders(tokens, missing)
    
    # Проверяем и корректируем размер
    if not validate_vocab_size(tokens, args.model_size, args.strict):
        print("Ошибка валидации размера словаря")
        return 1
    
    # Валидация токенов
    if args.validate:
        print("\n" + "-" * 40)
        print("ВАЛИДАЦИЯ ТОКЕНОВ")
        print("-" * 40)
        issues = validate_tokens(tokens)
        if issues:
            print(f"Найдено проблем: {len(issues)}")
            for issue in issues[:20]:
                print(f"  - {issue}")
            if len(issues) > 20:
                print(f"  ... и еще {len(issues) - 20}")
            
            if args.strict:
                print("Строгий режим: прерывание из-за ошибок валидации")
                return 1
        else:
            print("Валидация успешна - проблем не найдено!")
    
    # Формируем чистый вывод для C++
    cpp_output = {
        "size": len(tokens),
        "tokens": tokens
    }
    
    # Сохраняем основной словарь
    output_path = output_dir / "cpp_vocab.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cpp_output, f, indent=2, ensure_ascii=False)
    
    print(f"\nКонвертировано в: {output_path}")
    
    # Сохраняем маппинг если есть
    if mapping and args.verbose:
        save_mapping_file(mapping, output_dir, args.model_size)
    
    # Копируем merges.txt
    if python_merges.exists():
        shutil.copy2(python_merges, output_dir / "cpp_merges.txt")
        print(f"Скопированы слияния: {output_dir / 'cpp_merges.txt'}")
    else:
        print(f"Файл слияний не найден: {python_merges}!")
    
    # Выводим статистику
    print("\n" + "-" * 40)
    print_statistics(tokens, f"ИТОГОВЫЙ СЛОВАРЬ (модель {args.model_size})")
    
    # Показываем примеры токенов
    print(f"\nПримеры токенов (первые 20):")
    for i in range(min(20, len(tokens))):
        token = tokens[i]
        if token.startswith('<MISSING_'):
            print(f"{i:4d}: [ПРОПУСК] {token}")
        elif token.startswith('<EXTRA_'):
            print(f"{i:4d}: [ДОБАВЛЕН] {token}")
        elif token in SPECIAL_TOKENS:
            print(f"{i:4d}: [СПЕЦ] '{token}'")
        elif len(token) == 1:
            if token.isprintable():
                print(f"{i:4d}: '{token}' (ASCII: {ord(token)})")
            else:
                display = repr(token).strip("'")
                print(f"{i:4d}: {display} (ASCII: {ord(token)})")
        else:
            display = token if len(token) < 40 else token[:37] + "..."
            print(f"{i:4d}: '{display}'")
    
    # Итоговые рекомендации
    print("\n" + "============================================================")
    print("ИТОГ:")
    print("============================================================")
    
    if has_gaps and not args.no_fill:
        print("\nРЕКОМЕНДАЦИЯ: В словаре есть пропуски!")
        print("Используйте --no-fill для сжатия словаря и удаления пропусков:")
        print(f"   python {sys.argv[0]} --no-fill --model-size {args.model_size}")
    elif args.no_fill and mapping:
        print("\nСловарь сжат, пропуски удалены.")
        print("Индексы токенов изменены. Маппинг сохранен в id_mapping.json")
    
    if len(tokens) != args.model_size:
        print(f"\nРазмер словаря ({len(tokens)}) не соответствует ожидаемому ({args.model_size})")
        print("Проверьте исходные данные или используйте другой --model-size")
    
    print("\nКонвертация завершена успешно!")
    return 0


if __name__ == "__main__":
    sys.exit(main())