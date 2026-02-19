#!/usr/bin/env python3
# ======================================================================
# inspect_vocab.py - Скрипт для анализа структуры vocab.json
# ======================================================================
#
# @file inspect_vocab.py
# @brief Скрипт для анализа структуры файла словаря vocab.json
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Выполняет детальный анализ файла словаря BPE токенизатора:
#          - Проверяет структуру JSON
#          - Анализирует типы ключей и значений
#          - Проверяет непрерывность ID
#          - Выводит примеры токенов
#
# @usage python inspect_vocab.py [путь_к_vocab.json]
#
# @example
#   python inspect_vocab.py
#   python inspect_vocab.py ../bpe/vocab.json
#   python inspect_vocab.py ../../models/cpp_vocab.json
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
    Вывести заголовок раздела.
    
    Args:
        title: Заголовок
        width: Ширина линии
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
        print(f"Размер файла: {len(content)} байт ({file_size_kb:.2f} KB)")
        
        # Показываем начало файла для отладки
        print(f"\nПервые 200 символов:")
        print("-" * 40)
        print(content[:200])
        print("-" * 40)
        
        # Парсим JSON
        data = json.loads(content)
        
        print(f"\nСТРУКТУРА ДАННЫХ:")
        print(f"   Тип: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"   Количество записей: {len(data)}")
            
            # Проверяем типы ключей и значений
            key_types = set(type(k).__name__ for k in data.keys())
            value_types = set(type(v).__name__ for v in data.values())
            
            print(f"   Типы ключей: {', '.join(key_types)}")
            print(f"   Типы значений: {', '.join(value_types)}")
            
            # Показываем примеры
            print(f"\nПРИМЕРЫ (первые 10 записей):")
            print("-" * 40)
            items = list(data.items())[:10]
            for i, (key, value) in enumerate(items):
                print(f"   {i:2d}. Ключ: '{key}' ({type(key).__name__})")
                print(f"      Знач: '{value}' ({type(value).__name__})")
            print("-" * 40)
            
            # Анализ ID
            ids = []
            for key, value in data.items():
                # Пробуем получить ID из ключа
                key_id = safe_int(key)
                if key_id >= 0:
                    ids.append(key_id)
                
                # Пробуем получить ID из значения
                val_id = safe_int(value)
                if val_id >= 0:
                    ids.append(val_id)
            
            if ids:
                ids = sorted(set(ids))  # Уникальные ID
                max_id = max(ids)
                min_id = min(ids)
                unique_count = len(ids)
                
                print(f"\nСТАТИСТИКА ID:")
                print(f"   Min ID: {min_id}")
                print(f"   Max ID: {max_id}")
                print(f"   Уникальных ID: {unique_count}")
                print(f"   Диапазон: {max_id - min_id + 1}")
                print(f"   Пропусков: {(max_id - min_id + 1) - unique_count}")
                
                if unique_count == max_id + 1:
                    print(f"\nID непрерывны от 0 до {max_id}")
                else:
                    print(f"\n !!! Есть пропуски в ID")
                    
                    # Показываем пропуски
                    if max_id - min_id + 1 - unique_count > 0:
                        missing = []
                        for i in range(min_id, max_id + 1):
                            if i not in ids:
                                missing.append(i)
                        print(f"   Пропущенные ID: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            
            # Проверяем формат словаря
            print(f"\nФОРМАТ СЛОВАРЯ:")
            
            # Формат 1: {token: id}
            if all(isinstance(v, (int, str)) for v in data.values()) and \
               all(isinstance(k, str) for k in data.keys()):
                sample_token = next(iter(data.keys()))
                sample_id = data[sample_token]
                print(f"   Похоже на формат: {{'токен': ID}}")
                print(f"   Пример: '{sample_token}' -> {sample_id}")
            
            # Формат 2: {id: token}
            elif all(isinstance(k, (int, str)) and str(k).isdigit() for k in data.keys()) and \
                 all(isinstance(v, str) for v in data.values()):
                sample_id = next(iter(data.keys()))
                sample_token = data[sample_id]
                print(f"   Похоже на формат: {{ID: 'токен'}}")
                print(f"   Пример: {sample_id} -> '{sample_token}'")
            
            # Формат 3: {"tokens": [...]}
            elif "tokens" in data and isinstance(data["tokens"], list):
                print(f"   Похоже на формат: {{\"tokens\": [...]}}")
                print(f"   Количество токенов: {len(data['tokens'])}")
                if len(data['tokens']) > 0:
                    print(f"      Первый токен: '{data['tokens'][0]}'")
            
            else:
                print(f"Не удалось определить формат")
        
        elif isinstance(data, list):
            print(f"   Длина массива: {len(data)}")
            
            print(f"\nПРИМЕРЫ (первые 10 токенов):")
            print("-" * 40)
            for i, token in enumerate(data[:10]):
                print(f"   {i:2d}. '{token}'")
            print("-" * 40)
            
            print(f"\nЭто простой массив токенов")
        
        else:
            print(f"Содержимое: {data}")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"\nОШИБКА ПАРСИНГА JSON:")
        print(f"   {e}")
        print(f"\n   Позиция ошибки: строка {e.lineno}, колонка {e.colno}")
        print(f"   Текст ошибки: {e.msg}")
        return None
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        return None


def find_vocab_file() -> Optional[Path]:
    """
    Поиск файла vocab.json в стандартных местах.
    
    Returns:
        Optional[Path]: Путь к файлу или None
    """
    # Возможные пути относительно скрипта
    possible_paths = [
        Path("bpe/vocab.json"),
        Path("cpp/models/cpp_vocab.json"),
        Path("../bpe/vocab.json"),
        Path("../../bpe/vocab.json"),
        Path("models/cpp_vocab.json"),
        Path("vocab.json"),
    ]
    
    # Добавляем пути относительно корня проекта
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    project_paths = [
        project_root / "bpe" / "vocab.json",
        project_root / "cpp" / "models" / "cpp_vocab.json",
        project_root / "models" / "cpp_vocab.json",
    ]
    
    for path in possible_paths + project_paths:
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
            print(f"🔍 Найден файл по пути: {file_path}")
        else:
            print_header("ИНСПЕКТОР СЛОВАРЯ VOCAB.JSON")
            print("\n   Файл vocab.json не найден в стандартных местах!")
            print("\n   Укажите путь к файлу:")
            print("   python inspect_vocab.py <path_to_vocab.json>")
            print("\n   Примеры:")
            print("   python inspect_vocab.py ../bpe/vocab.json")
            print("   python inspect_vocab.py ../../models/cpp_vocab.json")
            return 1
    
    # Анализируем файл
    data = inspect_vocab(file_path)
    
    if data is None:
        return 1
    
    print_header("АНАЛИЗ ЗАВЕРШЕН")
    return 0


if __name__ == "__main__":
    sys.exit(main())