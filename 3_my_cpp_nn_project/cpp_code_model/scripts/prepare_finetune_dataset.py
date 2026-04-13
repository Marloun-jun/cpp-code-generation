#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# prepare_finetune_dataset.py - Подготовка датасета для LoRA дообучения
# ======================================================================
#
# @file prepare_finetune_dataset.py
# @brief Парсинг CSV датасета с инструкциями и кодом с аугментацией
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль подготавливает датасет для LoRA дообучения модели
#          на русские инструкции. Выполняет парсинг CSV файла, очистку
#          экранированных символов и аугментацию описаний.
#
#          **Основные возможности:**
#
#          1. **Парсинг CSV**
#             - Извлечение description и code из CSV
#             - Обработка кавычек и экранирования
#             - Пропуск заголовков и комментариев
#
#          2. **Очистка кода**
#             - Замена экранированных кавычек (\\" -> ")
#             - Замена экранированных переносов (\\n -> \n)
#             - Удаление лишних экранирований
#
#          3. **Аугментация данных**
#             - Замена слов синонимами
#             - Создание вариаций описаний
#             - Настраиваемый процент аугментации
#
#          4. **Сохранение в формате JSONL**
#             - Формат: {"text": "описание\n\nкод"}
#             - Совместимость с LoRA обучением
#             - Перемешивание данных
#
#          5. **Статистика и проверка**
#             - Количество обработанных примеров
#             - Проверка на наличие экранирования
#             - Просмотр первых примеров
#
# @usage
#     python scripts/prepare_finetune_dataset.py
#
# @example
#     # Стандартный запуск
#     python prepare_finetune_dataset.py
#
#     # С отключённой аугментацией (измените параметры в коде)
#
# ======================================================================

import re
import json
import random

from pathlib import Path

# ======================================================================
# ПАРАМЕТРЫ АУГМЕНТАЦИИ
# ======================================================================

ENABLE_AUGMENTATION = True    # Включить аугментацию
AUGMENT_RATIO = 0.7           # 70% примеров будут аугментированы
NUM_VARIATIONS = 2            # 2 вариации на пример

# ======================================================================
# СИНОНИМЫ ДЛЯ АУГМЕНТАЦИИ
# ======================================================================

SYNONYMS = {
    'напиши': ['создай', 'реализуй', 'напиши код', 'разработай', 'сделай', 'напиши программу'],
    'программу': ['программу', 'код', 'функцию', 'скрипт', 'приложение', 'алгоритм'],
    'выводит': ['печатает', 'выводит на экран', 'отображает', 'показывает', 'выводит в консоль'],
    'находит': ['ищет', 'определяет', 'вычисляет', 'находит', 'определяет'],
    'сумму': ['сумму', 'сложение', 'общую сумму', 'суммарное значение'],
    'создай': ['напиши', 'реализуй', 'сделай', 'разработай'],
    'функцию': ['функцию', 'метод', 'процедуру', 'алгоритм'],
    'рекурсивную': ['рекурсивную', 'циклическую', 'итеративную'],
    'класс': ['класс', 'структуру', 'объект', 'тип данных'],
    'вектор': ['вектор', 'массив', 'список', 'динамический массив'],
}


# ======================================================================
# ФУНКЦИИ ОБРАБОТКИ
# ======================================================================

def clean_code(code):
    """
    Очистка кода от экранированных символов.
    
    Args:
        code (str): Исходный код с экранированиями
        
    Returns:
        str: Очищенный код
    
    **Замены:**
    - \\" -> " (экранированные кавычки)
    - \\n -> \n (экранированные переносы)
    - Удаление лишних обратных слешей
    """
    # Заменяем экранированные кавычки на обычные
    code = code.replace('\\"', '"')
    # Заменяем экранированные переносы на реальные (если есть)
    code = code.replace('\\n', '\n')
    # Убираем лишние экранирования
    code = re.sub(r'\\([^\\])', r'\1', code)
    return code


def generate_variations(description, num=2):
    """
    Генерирует вариации описания с помощью синонимов.
    
    Args:
        description (str): Исходное описание
        num (int):         Количество вариаций для генерации
        
    Returns:
        list: Список вариаций описания
    
    **Алгоритм:**
    1. Разбиваем описание на слова
    2. Для каждого слова проверяем, есть ли синонимы
    3. Заменяем случайные слова синонимами
    4. Собираем новое описание
    """
    variations = []
    words = description.split()
    
    for _ in range(num):
        new_words = words.copy()
        # Заменяем случайные слова синонимами
        for i, word in enumerate(new_words):
            word_lower = word.lower()
            for orig, syns in SYNONYMS.items():
                if orig in word_lower:
                    new_words[i] = word.replace(orig, random.choice(syns))
                    break
        
        new_desc = ' '.join(new_words)
        if new_desc != description:
            variations.append(new_desc)
    
    return variations


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция подготовки датасета."""
    
    # Пути
    PROJECT_ROOT = Path(__file__).parent.parent
    csv_path = PROJECT_ROOT / "data" / "raw" / "2_cpp_code_generation_dataset.csv"
    output_path = PROJECT_ROOT / "data" / "instruction_train.jsonl"
    
    print("=" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ LoRA")
    print("=" * 60)
    print(f"Парсинг CSV: {csv_path}")
    
    if not csv_path.exists():
        print(f"Файл не найден: {csv_path}!")
        print("Убедитесь, что файл находится в data/raw/")
        exit(1)
    
    # Читаем весь файл
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Всего строк: {len(lines)}")
    
    data = []
    skipped = 0
    
    for line_num, line in enumerate(lines, 1):
        line = line.rstrip('\n')
        
        # Пропускаем первую строку (заголовок)
        if line_num == 1:
            continue
        
        # Пропускаем комментарии
        if line.startswith('#'):
            continue
        
        # Пропускаем пустые строки
        if not line.strip():
            continue
        
        # Должна начинаться с кавычки
        if not line.startswith('"'):
            skipped += 1
            continue
        
        try:
            # Находим description (первое поле в кавычках)
            desc_start = 1
            desc_end = line.find('"', desc_start)
            if desc_end == -1:
                skipped += 1
                continue
            
            description = line[desc_start:desc_end]
            
            # Остаток строки после description
            rest = line[desc_end + 1:]
            
            # Находим начало code (следующая кавычка)
            code_start = rest.find('"')
            if code_start == -1:
                skipped += 1
                continue
            
            # Находим конец code (закрывающую кавычку)
            code_end = -1
            pos = code_start + 1
            while pos < len(rest):
                if rest[pos] == '"':
                    if pos + 1 >= len(rest) or rest[pos + 1] == ',':
                        code_end = pos
                        break
                pos += 1
            
            if code_end == -1:
                code_end = rest.rfind('"')
                if code_end == -1:
                    skipped += 1
                    continue
            
            code = rest[code_start + 1:code_end]
            
            # Заменяем \n на реальные переносы строк
            code = code.replace('\\n', '\n')
            
            # ОЧИСТКА ОТ ЭКРАНИРОВАНИЯ
            code = clean_code(code)
            
            # Сохраняем
            data.append({
                'description': description,
                'code': code
            })
            
        except Exception as e:
            print(f"Ошибка в строке {line_num}: {e}!")
            skipped += 1
    
    print(f"\nНайдено примеров: {len(data)}")
    print(f"Пропущено:        {skipped}")
    
    # ============ АУГМЕНТАЦИЯ ============
    if ENABLE_AUGMENTATION:
        print("\n" + "=" * 60)
        print("АУГМЕНТАЦИЯ ДАТАСЕТА")
        print("=" * 60)
        
        augmented_data = []
        
        for item in data:
            # Оригинал
            augmented_data.append(item)
            
            # Вариации (только для части примеров)
            if random.random() < AUGMENT_RATIO:
                variations = generate_variations(item['description'], NUM_VARIATIONS)
                for var_desc in variations:
                    augmented_data.append({
                        'description': var_desc,
                        'code': item['code']
                    })
        
        print(f"Размер датасета: {len(data)} -> {len(augmented_data)}")
        data = augmented_data
        random.shuffle(data)  # Перемешиваем
    
    # ============ СОХРАНЕНИЕ ============
    print("\n" + "=" * 60)
    print("СОХРАНЕНИЕ")
    print("=" * 60)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        for item in data:
            text = f"{item['description']}\n\n{item['code']}"
            out.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')
    
    print(f"Сохранено {len(data)} примеров")
    print(f"{output_path}")
    
    # ============ ПРОВЕРКА ============
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ПЕРВЫХ 3 ПРИМЕРОВ")
    print("=" * 60)
    
    for i, item in enumerate(data[:3]):
        print(f"\nПример {i+1}:")
        print(f"Описание: {item['description'][:100]}...")
        print(f"Код (первые 150 символов):")
        print(f"{item['code'][:150]}...")
        
        # Проверка на экранирование
        if '\\"' in item['code']:
            print(f"ВНИМАНИЕ: Обнаружены экранированные кавычки!")
        else:
            print(f"Экранирование отсутствует")
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"\nСтатистика:")
    print(f"- Всего примеров:      {len(data)}")
    print(f"- Аугментация:         {'Включена' if ENABLE_AUGMENTATION else 'Выключена'}")
    print(f"- Вариаций на пример:  {NUM_VARIATIONS}")
    print(f"- Процент аугментации: {AUGMENT_RATIO * 100:.0f}%")


if __name__ == "__main__":
    main()