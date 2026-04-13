#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# p_4_descriptions.py - Проверка качества описаний в датасете C++ кода
# ======================================================================
#
# @file p_4_descriptions.py
# @brief Выборочная проверка поля description в датасете C++ кода
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Выполняет случайную выборочную проверку качества описаний программ:
#          - Начинаются ли с правильного глагола (напиши, создай, реализуй и т.д.)
#          - Длина описания (не слишком короткое и не слишком длинное)
#          - Отсутствие кода в описании (код должен быть только в поле code)
#          - Корректность формата строки
#
#          **Критерии качества:**
#          - Начинается с глагола из списка valid_verbs
#          - Длина от 10 до 500 символов
#          - Не содержит элементов C++ кода (#include, std::, ; и т.д.)
#          - Корректный формат CSV
#
# @usage python p_4_descriptions.py
#
# @example
#   python p_4_descriptions.py
#   # Результаты:
#   # - Статистика проблем в выборке
#   # - Примеры хороших описаний
#   # - Рекомендации по улучшению
#
# ======================================================================

import sys
import random
from typing import List, Tuple, Dict


# ======================================================================
# КОНСТАНТЫ
# ======================================================================

# Глаголы, с которых должны начинаться описания
VALID_VERBS = [
    # Инфинитивы (что сделать?)
    'написать', 'создать', 'реализовать', 'разработать', 
    'показать', 'сделать', 'добавить', 'открыть',
    'выполнить', 'считать', 'прочитать', 'записать',
    'использовать', 'проверить', 'вычислить', 'найти',
    'продемонстрировать', 'представить', 'привести', 
    'описать', 'объяснить',
    
    # Императивы (сделай)
    'напиши', 'создай', 'реализуй', 'разработай',
    'покажи', 'сделай', 'добавь', 'открой',
    'выполни', 'считай', 'прочитай', 'запиши',
    'используй', 'проверь', 'вычисли', 'найди',
    'продемонстрируй', 'представь', 'приведи',
    'опиши', 'объясни'
]

# Индикаторы наличия кода в описании (чего быть не должно)
CODE_INDICATORS = [
    '#include', 'usung namespace std;', 'int main()', 'cout <<', 
    'std::cout <<', ';', '{', '}', 'return'
]

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

def print_subheader(title: str, width: int = 60) -> None:
    """
    Вывести подзаголовок раздела.
    
    Args:
        title: Подзаголовок
        width: Ширина линии
    """
    print(f"\n{'-' * width}")
    print(f"{title}")
    print(f"{'-' * width}")

# ======================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ======================================================================

def check_descriptions(filename: str, sample_size: int = 200) -> Dict[str, int]:
    """
    Выборочная проверка качества описаний (description) в датасете.
    
    Args:
        filename:       Путь к файлу датасета
        sample_size:    Количество случайных примеров для проверки
        
    Returns:
        Dict[str, int]:
            Статистика проблем:
            - wrong_verb   - Не начинаются с глагола
            - too_short    - Слишком короткие (<10 символов)
            - too_long     - Слишком длинные (>500 символов)
            - code_in_desc - Содержат код C++
            - bad_format   - Ошибки формата CSV
    
    **Процесс:**
    1. Сбор всех валидных строк датасета
    2. Случайная выборка sample_size строк
    3. Проверка каждой строки по критериям
    4. Вывод статистики и примеров
    """
    print_header("ВЫБОРОЧНАЯ ПРОВЕРКА ОПИСАНИЙ")
    print(f"Проверяем {sample_size} случайных примеров")
    print("=" * 60)
    
    # Собираем все строки с данными
    data_lines: List[Tuple[int, str]] = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.rstrip('\n')
                
                # Пропускаем пустые строки и комментарии
                if not raw_line.strip() or raw_line.strip().startswith('#'):
                    continue
                
                # Проверяем базовую структуру
                if not raw_line.startswith('"') or not raw_line.endswith('"'):
                    continue
                
                data_lines.append((line_num, raw_line))
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден!")
        return {}
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}!")
        return {}
    
    print(f"Всего строк с данными: {len(data_lines)}")
    
    # Берём случайную выборку
    if sample_size > len(data_lines):
        sample_size = len(data_lines)
        print(f"Размер выборки уменьшен до {sample_size} (максимум)")
    
    sample = random.sample(data_lines, sample_size)
    
    print(f"\nПроверяем {sample_size} случайных строк...")
    
    # Критерии проверки
    issues = {
        'wrong_verb': 0,      # Не начинается с глагола
        'too_short': 0,       # Слишком короткое
        'too_long': 0,        # Слишком длинное
        'code_in_desc': 0,    # Код в описании
        'bad_format': 0       # Плохое форматирование
    }
    
    # Проверяем каждую строку в выборке
    for i, (line_num, line) in enumerate(sample, 1):
        try:
            # Парсим строку (упрощённый парсинг CSV)
            parts = line.split('","')
            
            if len(parts) < 5:
                issues['bad_format'] += 1
                continue
            
            # Первое поле - описание (убираем начальную кавычку)
            description = parts[0][1:] if parts[0].startswith('"') else parts[0]
            
            # ==================================================================
            # Проверка 1: Начинается ли с глагола?
            # ==================================================================
            starts_with_verb = any(
                description.lower().startswith(verb) 
                for verb in VALID_VERBS
            )
            
            if not starts_with_verb:
                issues['wrong_verb'] += 1
                
                # Показываем примеры первых 3 проблем
                if issues['wrong_verb'] <= 3:
                    print(f"\nСтрока {line_num}: не начинается с глагола!")
                    preview = description[:100] + "..." if len(description) > 100 else description
                    print(f"Описание: {preview}")
            
            # ==================================================================
            # Проверка 2: Длина описания
            # ==================================================================
            if len(description) < 10:
                issues['too_short'] += 1
            elif len(description) > 500:
                issues['too_long'] += 1
            
            # ==================================================================
            # Проверка 3: Есть ли код в описании?
            # ==================================================================
            if any(indicator in description for indicator in CODE_INDICATORS):
                issues['code_in_desc'] += 1
                
        except Exception as e:
            issues['bad_format'] += 1
    
    # ======================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ======================================================================
    
    print_subheader("РЕЗУЛЬТАТЫ ПРОВЕРКИ ОПИСАНИЙ")
    
    total_issues = sum(issues.values())
    
    if total_issues == 0:
        print("ВСЕ описания в выборке корректны!")
    else:
        print(f"Найдено проблем в {total_issues} из {sample_size} проверенных строк:")
        print(f"- Не начинаются с глагола: {issues['wrong_verb']}")
        print(f"- Слишком короткие (<10):  {issues['too_short']}")
        print(f"- Слишком длинные (>500):  {issues['too_long']}")
        print(f"- Содержат код:            {issues['code_in_desc']}")
        print(f"- Ошибки формата:          {issues['bad_format']}")
    
    # ======================================================================
    # ПРИМЕРЫ ХОРОШИХ ОПИСАНИЙ
    # ======================================================================
    
    print_subheader("ПРИМЕРЫ ХОРОШИХ ОПИСАНИЙ")
    
    good_examples = []
    for line_num, line in sample[:20]:    # Просматриваем первые 20 из выборки
        try:
            parts = line.split('","')
            if len(parts) >= 5:
                description = parts[0][1:] if parts[0].startswith('"') else parts[0]
                
                # Проверяем, хорошее ли описание
                if (len(description) >= 20 and 
                    len(description) <= 200 and
                    any(description.lower().startswith(verb) for verb in VALID_VERBS) and
                    not any(indicator in description for indicator in CODE_INDICATORS)):
                    good_examples.append((line_num, description))
        except:
            continue
    
    if good_examples:
        for i, (line_num, desc) in enumerate(good_examples[:3], 1):
            print(f"\n  {i}. Строка {line_num}:")
            preview = desc[:120] + "..." if len(desc) > 120 else desc
            print(f"     {preview}")
    else:
        print("Хороших примеров не найдено в выборке!")
    
    return issues


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================

def main() -> int:
    """
    Основная функция.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    
    print(f"Анализируемый файл: {filename}")
    
    # Проверяем описания
    issues = check_descriptions(filename, sample_size=200)
    
    if not issues:
        return 1
    
    # ======================================================================
    # РЕКОМЕНДАЦИИ
    # ======================================================================
    
    print_header("РЕКОМЕНДАЦИИ ПО ОПИСАНИЯМ")
    
    recommendations = []
    
    if issues.get('wrong_verb', 0) > 0:
        recommendations.append(
            "Убедитесь, что описания начинаются с глагола:\n"
            "- 'напиши', 'создай', 'реализуй', 'разработай'\n"
            "- 'написать', 'создать', 'реализовать', 'разработать'"
        )
    
    if issues.get('too_short', 0) > 0:
        recommendations.append(
            "Описания должны быть информативными (минимум 10 символов)"
        )
    
    if issues.get('too_long', 0) > 0:
        recommendations.append(
            "Описания не должны быть слишком длинными (максимум 500 символов)"
        )
    
    if issues.get('code_in_desc', 0) > 0:
        recommendations.append(
            "Не включайте код C++ в описание:\n"
            "- Код должен быть только в поле 'code'\n"
            "- Описание должно описывать, что делает программа, а не содержать код"
        )
    
    if issues.get('bad_format', 0) > 0:
        recommendations.append(
            "Проверьте формат CSV строк (должно быть ровно 5 полей в кавычках)"
        )
    
    if recommendations:
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\nВсе описания соответствуют требованиям!")
    
    print(f"\nПроцент проблем: {sum(issues.values())/200*100:.1f}%")
    
    return 0 if sum(issues.values()) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())