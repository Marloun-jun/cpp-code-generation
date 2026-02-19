# scripts/inspect_vocab.py
#!/usr/bin/env python3

"""
Скрипт для анализа структуры vocab.json
"""

import json
import sys
from pathlib import Path

def inspect_vocab(file_path):
    """Детальный анализ vocab.json"""
    print(f"\n🔍 Анализ файла: {file_path}")
    print("=" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📏 Размер файла: {len(content)} байт")
        print(f"📝 Первые 200 символов:\n{content[:200]}")
        
        data = json.loads(content)
        
        print(f"\n📊 Структура данных:")
        print(f"   Тип: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"   Количество записей: {len(data)}")
            
            # Проверяем типы ключей и значений
            key_types = set(type(k).__name__ for k in data.keys())
            value_types = set(type(v).__name__ for v in data.values())
            
            print(f"   Типы ключей: {key_types}")
            print(f"   Типы значений: {value_types}")
            
            # Показываем примеры
            print(f"\n📋 Примеры (первые 10):")
            items = list(data.items())[:10]
            for i, (token, token_id) in enumerate(items):
                print(f"   {i:2d}. '{token}' -> {token_id} (тип: {type(token_id).__name__})")
            
            # Проверяем непрерывность ID
            if all(isinstance(v, (int, str)) for v in data.values()):
                ids = []
                for v in data.values():
                    if isinstance(v, str):
                        try:
                            ids.append(int(v))
                        except:
                            ids.append(-1)
                    else:
                        ids.append(v)
                
                ids = [i for i in ids if i >= 0]
                if ids:
                    max_id = max(ids)
                    min_id = min(ids)
                    unique_ids = len(set(ids))
                    
                    print(f"\n📈 Статистика ID:")
                    print(f"   Min ID: {min_id}")
                    print(f"   Max ID: {max_id}")
                    print(f"   Уникальных ID: {unique_ids}")
                    print(f"   Диапазон: {max_id - min_id + 1}")
                    print(f"   Пропусков: {(max_id - min_id + 1) - unique_ids}")
                    
                    if unique_ids == max_id + 1:
                        print(f"✅ ID непрерывны от 0 до {max_id}")
                    else:
                        print(f"⚠️  Есть пропуски в ID")
        
        elif isinstance(data, list):
            print(f"   Длина массива: {len(data)}")
            print(f"\n📋 Примеры (первые 10):")
            for i, token in enumerate(data[:10]):
                print(f"   {i:2d}. '{token}'")
        
        else:
            print(f"   Содержимое: {data}")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка парсинга JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None

def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # По умолчанию ищем в стандартных местах
        paths = [
            "bpe/vocab.json",
            "cpp/models/cpp_vocab.json",
            "../bpe/vocab.json",
            "../../bpe/vocab.json"
        ]
        
        for path in paths:
            if Path(path).exists():
                file_path = path
                break
        else:
            print("❌ Файл vocab.json не найден!")
            print("   Укажите путь: python inspect_vocab.py <path_to_vocab.json>")
            return 1
    
    inspect_vocab(file_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())