#!/usr/bin/env python3
# cpp/tools/convert_vocab.py - Конвертация словаря BPE токенизатора
# 
# @author Ваше Имя
# @date 2024
# @version 1.0.0
#
# @details Конвертирует словарь из Python формата в C++ формат.
#          Поддерживает различные форматы входных данных:
#          - {"id": "token"}  (ID как ключи, токены как значения)
#          - {"token": id}    (токены как ключи, ID как значения)
#          - [token1, token2] (массив токенов)
#          
#          Результат сохраняется в формате, оптимизированном для C++:
#          {
#            "size": 32000,
#            "tokens": ["<PAD>", "<UNK>", ...]
#          }

import json
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

# ======================================================================
# Конфигурация
# ======================================================================

SPECIAL_TOKENS = {
    "<PAD>": "Токен для выравнивания последовательностей",
    "<UNK>": "Токен для неизвестных символов",
    "<BOS>": "Токен начала последовательности",
    "<EOS>": "Токен конца последовательности",
    "<MASK>": "Токен для маскирования (опционально)"
}

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
        return "id_to_token"  # {"0": "<PAD>", "1": "<UNK>", ...}
    
    # Проверяем обратный формат (токен -> ID)
    keys_are_strings = all(isinstance(k, str) for k in sample_keys)
    values_are_numeric = all(isinstance(v, (int, str)) and str(v).isdigit() 
                            for v in sample_values)
    
    if keys_are_strings and values_are_numeric:
        return "token_to_id"  # {"<PAD>": 0, "<UNK>": 1, ...}
    
    return "unknown"

# ======================================================================
# Функции конвертации
# ======================================================================

def convert_id_to_token_format(data: Dict) -> Tuple[Dict, List[int]]:
    """
    Конвертирует формат {"id": "token"} в массив токенов.
    
    Args:
        data: Словарь в формате ID -> токен
    
    Returns:
        Tuple[Dict, List[int]]: (конвертированные данные, список пропущенных ID)
    """
    print("📦 Обнаружен формат: ID -> ТОКЕН")
    
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
            print(f"⚠️  Пропускаем нечисловой ключ: '{key}'")
            skipped += 1
            continue
        
        id_to_token[numeric_id] = token
        max_id = max(max_id, numeric_id)
    
    print(f"   Найдено записей: {len(id_to_token)}")
    print(f"   Пропущено записей: {skipped}")
    print(f"   Максимальный ID: {max_id}")
    
    # Создаем массив токенов
    tokens = [''] * (max_id + 1)
    
    for id_val, token in id_to_token.items():
        tokens[id_val] = token
    
    # Проверяем пропуски
    missing = [i for i, t in enumerate(tokens) if not t]
    print(f"   Пропусков в ID: {len(missing)}")
    
    return {
        "size": len(tokens),
        "tokens": tokens,
        "format": "id_to_token",
        "original_max_id": max_id
    }, missing

def convert_token_to_id_format(data: Dict) -> Tuple[Dict, List[int]]:
    """
    Конвертирует формат {"token": id} в массив токенов.
    
    Args:
        data: Словарь в формате токен -> ID
    
    Returns:
        Tuple[Dict, List[int]]: (конвертированные данные, список пропущенных ID)
    """
    print("📦 Обнаружен формат: ТОКЕН -> ID")
    
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
            print(f"⚠️  Пропускаем запись с нечисловым ID: {token} -> {id_val}")
            skipped += 1
            continue
        
        token_to_id[token] = numeric_id
        max_id = max(max_id, numeric_id)
    
    print(f"   Найдено записей: {len(token_to_id)}")
    print(f"   Пропущено записей: {skipped}")
    print(f"   Максимальный ID: {max_id}")
    
    # Создаем массив токенов
    tokens = [''] * (max_id + 1)
    
    for token, id_val in token_to_id.items():
        tokens[id_val] = token
    
    # Проверяем пропуски
    missing = [i for i, t in enumerate(tokens) if not t]
    print(f"   Пропусков в ID: {len(missing)}")
    
    return {
        "size": len(tokens),
        "tokens": tokens,
        "format": "token_to_id",
        "original_max_id": max_id
    }, missing

def convert_array_format(data: List) -> Dict:
    """
    Конвертирует формат [token1, token2] в нужный формат.
    
    Args:
        data: Массив токенов
    
    Returns:
        Dict: Конвертированные данные
    """
    print("📦 Обнаружен формат: МАССИВ токенов")
    print(f"   Количество токенов: {len(data)}")
    
    # Проверяем, есть ли специальные токены в начале
    special_found = []
    for i, token in enumerate(data[:10]):
        if token in SPECIAL_TOKENS:
            special_found.append(token)
    
    if special_found:
        print(f"   Найдены специальные токены: {', '.join(special_found)}")
    
    return {
        "size": len(data),
        "tokens": data,
        "format": "array"
    }

# ======================================================================
# Функции пост-обработки
# ======================================================================

def compress_tokens(tokens: List[str], missing: List[int]) -> Dict:
    """
    Сжимает токены, удаляя пропуски и переиндексируя.
    
    Args:
        tokens: Исходный массив токенов с пропусками
        missing: Список индексов пропусков
    
    Returns:
        Dict: Сжатый словарь
    """
    print(f"\n🔄 Сжатие токенов (удаление пропусков)...")
    
    # Оставляем только реальные токены
    real_tokens = [t for t in tokens if t]
    
    # Создаем карту старых ID -> новые ID
    old_to_new = {}
    new_idx = 0
    for old_idx, token in enumerate(tokens):
        if token:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    print(f"   Было токенов (с пропусками): {len(tokens)}")
    print(f"   Стало реальных токенов: {len(real_tokens)}")
    print(f"   Удалено пропусков: {len(tokens) - len(real_tokens)}")
    print(f"   Экономия места: {(1 - len(real_tokens)/len(tokens))*100:.1f}%")
    
    return {
        "size": len(real_tokens),
        "tokens": real_tokens,
        "old_to_new_mapping": old_to_new,
        "compressed": True
    }

def validate_tokens(tokens: List[str]) -> List[str]:
    """
    Проверяет токены на корректность UTF-8.
    
    Args:
        tokens: Список токенов для проверки
    
    Returns:
        List[str]: Список проблемных токенов
    """
    issues = []
    
    for i, token in enumerate(tokens):
        if not token:
            continue
            
        # Проверяем на пустые токены
        if not token.strip() and token not in SPECIAL_TOKENS:
            issues.append(f"ID {i}: пустой токен")
        
        # Проверяем длину
        if len(token) > 1000:
            issues.append(f"ID {i}: слишком длинный токен ({len(token)} символов)")
        
        # Проверяем на некорректный UTF-8
        try:
            token.encode('utf-8').decode('utf-8')
        except UnicodeError:
            issues.append(f"ID {i}: некорректный UTF-8")
    
    return issues

# ======================================================================
# Функции анализа
# ======================================================================

def inspect_vocab_file(file_path: Path) -> Optional[Any]:
    """
    Анализирует структуру vocab.json и выводит информацию.
    
    Args:
        file_path: Путь к файлу словаря
    
    Returns:
        Optional[Any]: Загруженные данные или None при ошибке
    """
    try:
        # Сначала читаем небольшой кусок для предпросмотра
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(500)
            
        print(f"\n🔍 Анализ файла: {file_path}")
        print(f"📝 Первые 200 символов:")
        print("-" * 40)
        print(sample[:200])
        print("-" * 40)
        
        # Читаем полностью
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📊 Структура данных:")
        print(f"   Тип: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"   Количество записей: {len(data)}")
            
            # Показываем примеры первых 5 записей
            items = list(data.items())[:5]
            for k, v in items:
                print(f"   '{k}' ({type(k).__name__}) -> '{v}' ({type(v).__name__})")
            
            # Определяем формат
            format_type = detect_format(data)
            print(f"   Формат: {format_type}")
            
            # Анализируем типы ключей
            key_types = Counter(type(k).__name__ for k in data.keys())
            print(f"   Типы ключей: {dict(key_types)}")
            
        elif isinstance(data, list):
            print(f"   Количество элементов: {len(data)}")
            print(f"   Первые 5: {data[:5]}")
        
        return data
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return None

def print_statistics(tokens: List[str], title: str = "Статистика"):
    """
    Выводит статистику по токенам.
    
    Args:
        tokens: Список токенов
        title: Заголовок
    """
    print(f"\n📊 {title}:")
    print(f"   Всего токенов: {len(tokens)}")
    
    # Статистика по длинам
    lengths = [len(t) for t in tokens if t]
    if lengths:
        print(f"   Средняя длина: {sum(lengths)/len(lengths):.1f} символов")
        print(f"   Макс. длина: {max(lengths)} символов")
        print(f"   Мин. длина: {min(lengths)} символов")
    
    # Статистика по типам
    single_char = sum(1 for t in tokens if len(t) == 1)
    multi_char = sum(1 for t in tokens if len(t) > 1)
    special = sum(1 for t in tokens if t in SPECIAL_TOKENS)
    
    print(f"   Односимвольных: {single_char}")
    print(f"   Многосимвольных: {multi_char}")
    print(f"   Специальных токенов: {special}")

# ======================================================================
# Основная функция
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert BPE vocabulary to C++ format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python tools/convert_vocab.py                    # обычная конвертация
  python tools/convert_vocab.py --no-fill          # сжатие словаря
  python tools/convert_vocab.py --inspect-only     # только анализ
  python tools/convert_vocab.py --format array     # принудительный формат
  python tools/convert_vocab.py --output custom.json
        """
    )
    
    parser.add_argument('--no-fill', action='store_true',
                       help='Не заполнять пропуски, а переиндексировать (сжать ID)')
    
    parser.add_argument('--inspect-only', action='store_true',
                       help='Только проанализировать словарь без конвертации')
    
    parser.add_argument('--format', choices=['auto', 'id_to_token', 'token_to_id', 'array'],
                       default='auto', help='Принудительный формат словаря')
    
    parser.add_argument('--output', '-o', type=str, default='cpp_vocab.json',
                       help='Имя выходного файла (по умолчанию: cpp_vocab.json)')
    
    # ИСПРАВЛЕНО: Путь к входному словарю
    parser.add_argument('--input-vocab', type=str, 
                       default='/home/john/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe_python/models/bpe_8000/vocab.json',
                       help='Путь к входному файлу словаря')
    
    # ИСПРАВЛЕНО: Путь к входному файлу слияний
    parser.add_argument('--input-merges', type=str,
                       default='/home/john/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe_python/models/bpe_8000/merges.txt',
                       help='Путь к входному файлу слияний')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    parser.add_argument('--validate', action='store_true',
                       help='Проверить токены на корректность')
    
    args = parser.parse_args()
    
    print("🔄 Конвертация словаря и слияний для C++...")
    print("-" * 50)
    
    # ИСПРАВЛЕНО: Используем абсолютные пути для входных файлов
    python_vocab = Path(args.input_vocab)
    python_merges = Path(args.input_merges)
    
    if not python_vocab.exists():
        print(f"❌ Файл не найден: {python_vocab}")
        print(f"   Проверьте путь: /home/john/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe_python/models/bpe_8000/vocab.json")
        return 1
    
    # Анализируем
    print("\n🔍 Анализ исходного словаря:")
    data = inspect_vocab_file(python_vocab)
    
    if args.inspect_only:
        return 0
    
    if data is None:
        return 1
    
    # Определяем формат
    if args.format != 'auto':
        format_type = args.format
        print(f"   Принудительный формат: {format_type}")
    else:
        format_type = detect_format(data)
    
    # Конвертируем в зависимости от формата
    if format_type == "id_to_token" or format_type == "id_to_token":
        cpp_vocab_data, missing = convert_id_to_token_format(data)
    elif format_type == "token_to_id":
        cpp_vocab_data, missing = convert_token_to_id_format(data)
    elif format_type == "array":
        cpp_vocab_data = convert_array_format(data)
        missing = []
    else:
        print("❌ Не удалось определить формат словаря")
        return 1
    
    # Извлекаем токены
    if isinstance(cpp_vocab_data, dict) and "tokens" in cpp_vocab_data:
        tokens = cpp_vocab_data["tokens"]
    else:
        print("❌ Ошибка: не удалось извлечь токены")
        return 1
    
    # Применяем сжатие если нужно
    if args.no_fill and missing:
        cpp_vocab_data = compress_tokens(tokens, missing)
    elif missing:
        # Заполняем пропуски заглушками
        print(f"\n📝 Заполнение пропусков заглушками...")
        for i in missing:
            if not tokens[i]:
                tokens[i] = f"<MISSING_{i}>"
        print(f"   Заполнено пропусков: {len(missing)}")
    
    # Валидация
    if args.validate:
        issues = validate_tokens(tokens if 'tokens' in cpp_vocab_data else cpp_vocab_data)
        if issues:
            print(f"\n⚠️ Найдены проблемы:")
            for issue in issues[:10]:
                print(f"   • {issue}")
            if len(issues) > 10:
                print(f"   ... и еще {len(issues) - 10}")
        else:
            print(f"\n✅ Валидация успешна - проблем не найдено")
    
    # ИСПРАВЛЕНО: Путь для сохранения результата
    output_dir = Path('/home/john/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe_cpp/models/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / args.output
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cpp_vocab_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Конвертировано в: {output_path}")
    
    # Копируем merges
    if python_merges.exists():
        shutil.copy2(python_merges, output_dir / "cpp_merges.txt")
        print(f"✅ Скопированы слияния: {output_dir / 'cpp_merges.txt'}")
    else:
        print(f"⚠️ Файл слияний не найден: {python_merges}")
    
    # Выводим статистику
    if 'tokens' in cpp_vocab_data:
        print_statistics(cpp_vocab_data['tokens'], "Итоговый словарь")
    
    # Показываем примеры
    print(f"\n📝 Примеры токенов (первые 20):")
    tokens_to_show = cpp_vocab_data['tokens'] if 'tokens' in cpp_vocab_data else cpp_vocab_data
    
    for i in range(min(20, len(tokens_to_show))):
        token = tokens_to_show[i]
        if token.startswith('<MISSING_'):
            print(f"   {i:4d}: [ПРОПУСК]")
        elif token in SPECIAL_TOKENS:
            print(f"   {i:4d}: '{token}' (специальный)")
        elif len(token) == 1:
            if token.isprintable():
                print(f"   {i:4d}: '{token}' (ASCII: {ord(token)})")
            else:
                print(f"   {i:4d}: [непечатный символ {ord(token)}]")
        else:
            display = token if len(token) < 40 else token[:37] + "..."
            print(f"   {i:4d}: '{display}'")
    
    if args.no_fill:
        print(f"\n✅ Словарь сжат до {len(tokens_to_show)} реальных токенов!")
    elif missing:
        print(f"\n💡 Совет: Используйте --no-fill для сжатия словаря")
        print(f"   python {sys.argv[0]} --no-fill")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())