#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# p_3_topic.py - Категоризация и анализ тем (topic) в датасете C++ кода
# ======================================================================
#
# @file p_3_topic.py
# @brief Анализ поля topic в датасете: извлечение уникальных тем и проверка баланса
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Выполняет анализ категоризации тем в датасете:
#          - Извлечение всех уникальных тем в порядке их появления в файле
#          - Проверка на опечатки и дубликаты (регистр, разделители)
#          - Анализ баланса между стилями using_namespace_std и explicit_std для каждой темы
#          - Выявление проблемных тем с недостаточным количеством примеров
#
#          Формат датасета:
#          "description","code","style","topic","keywords"
#
#          Поле topic находится на предпоследней позиции (индекс -2)
#
# @usage python p_3_topic.py
#
# @example
#   python p_3_topic.py
#   # Результаты:
#   # - Список всех уникальных тем в порядке появления
#   # - Статистика баланса стилей для каждой темы
#   # - Рекомендации по исправлению
#
# ======================================================================

import sys
from typing import List, Dict


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

def extract_unique_topics_in_order(filename: str) -> List[str]:
    """
    Извлекает уникальные темы в порядке их появления в датасете.
    
    Args:
        filename: Путь к файлу датасета
        
    Returns:
        List[str]: Список уникальных тем в порядке появления
    
    **Процесс:**
    1. Чтение файла построчно
    2. Пропуск пустых строк и комментариев
    3. Проверка базовой структуры (начинается и заканчивается кавычкой)
    4. Разделение по ","
    5. Извлечение предпоследнего поля (индекс -2)
    6. Сохранение уникальных тем в порядке первого появления
    
    **Дополнительно:**
        Проверка на возможные опечатки:
        - Дубликаты с разным регистром
        - Разные разделители (подчеркивание vs дефис)
        - Похожие названия
    """
    print_header("АНАЛИЗ КАТЕГОРИЗАЦИИ ТЕМ (TOPIC)")
    print("- Поле    - Второе с конца (перед ключевыми словами)")
    print("- Порядок - Как идут в датасете (сверху вниз)")
    print("=" * 60)
    
    topics = []            # Для сохранения порядка
    seen_topics = set()    # Для проверки уникальности
    line_num = 0
    data_line_num = 0
    parse_errors = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line_num += 1
                raw_line = raw_line.rstrip('\n')
                
                # Пропускаем пустые строки и комментарии
                if not raw_line.strip() or raw_line.strip().startswith('#'):
                    continue
                
                data_line_num += 1
                
                # Проверяем базовую структуру
                if not raw_line.startswith('"') or not raw_line.endswith('"'):
                    continue
                
                # Разделяем по ","
                parts = raw_line.split('","')
                
                if len(parts) >= 5:      # Должно быть минимум 5 полей
                    topic = parts[-2]    # -1 это последнее, -2 предпоследнее
                    
                    if topic not in seen_topics:
                        seen_topics.add(topic)
                        topics.append(topic)
                else:
                    parse_errors += 1
                
                # Показываем прогресс каждые 1000 строк
                if data_line_num % 1000 == 0:
                    print(f"Обработано {data_line_num} строк...")
                    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден!")
        return []
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}!")
        return []
    
    print(f"\nСТАТИСТИКА:")
    print(f"- Всего строк обработано: {data_line_num}")
    print(f"- Ошибок парсинга:        {parse_errors}")
    print(f"- Уникальных тем найдено: {len(topics)}")
    
    print_subheader("ТЕМЫ В ПОРЯДКЕ ПОЯВЛЕНИЯ")
    for i, topic in enumerate(topics, 1):
        print(f"  {i:3}. {topic}")
    
    # ======================================================================
    # ПРОВЕРКА НА ОПЕЧАТКИ
    # ======================================================================
    
    print_subheader("ПРОВЕРКА НА ОПЕЧАТКИ")
    
    possible_issues = []
    
    # Проверяем похожие названия (без сортировки, в порядке появления)
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            t1, t2 = topics[i].lower(), topics[j].lower()
            
            # Проверяем разные варианты опечаток
            if t1 == t2:
                possible_issues.append(
                    f"Дубликат (разный регистр): '{topics[i]}' и '{topics[j]}'")
            elif t1.replace('_', '') == t2.replace('_', ''):
                possible_issues.append(
                    f"Разные разделители: '{topics[i]}' и '{topics[j]}'")
            elif t1.replace('_', '-') == t2.replace('_', '-'):
                possible_issues.append(
                    f"Разные разделители (дефис/подчёркивание): '{topics[i]}' и '{topics[j]}'")
            elif abs(len(t1) - len(t2)) <= 2 and (t1 in t2 or t2 in t1):
                possible_issues.append(
                    f"Возможная опечатка: '{topics[i]}' и '{topics[j]}'")
    
    if possible_issues:
        print(f"Найдены возможные проблемы ({len(possible_issues)}):")
        for issue in possible_issues[:5]:    # Показываем первые 5
            print(f"- {issue}")
        if len(possible_issues) > 5:
            print(f"... и еще {len(possible_issues) - 5} проблем")
    else:
        print("Очевидных проблем не найдено!")
    
    return topics

def check_topic_balance_simple(filename: str, topics: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Проверяет баланс между стилями для каждой темы.
    
    Args:
        filename: Путь к файлу датасета
        topics:   Список тем для проверки
        
    Returns:
        Dict[str, Dict[str, int]]: Статистика по темам:
            {
                "тема": {
                    "using":    Количество с using_namespace_std,
                    "explicit": Количество с explicit_std
                }
            }
    
    **Критерии:**
    - Хорошо:         Минимум 2 примера каждого стиля
    - Проблема:       Отсутствие одного из стилей
    - Предупреждение: Мало примеров (менее 2)
    """
    print_header("ПРОВЕРКА БАЛАНСА ПО СТИЛЯМ")
    print("Для каждой темы проверяется наличие примеров в обоих стилях")
    print("=" * 60)
    
    # Инициализируем статистику
    topic_stats = {topic: {'using': 0, 'explicit': 0} for topic in topics}
    style_errors = 0
    line_num = 0
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line_num += 1
                raw_line = raw_line.rstrip('\n')
                
                # Пропускаем пустые строки и комментарии
                if not raw_line.strip() or raw_line.strip().startswith('#'):
                    continue
                
                # простой парсинг split
                parts = raw_line.split('","')
                
                if len(parts) < 5:
                    continue
                
                try:
                    # Извлекаем нужные поля
                    # parts[0] - Описание (убираем начальную кавычку)
                    # parts[1] - Код
                    # parts[2] - Стиль
                    # parts[3] - Тема
                    # parts[4] - Ключевые слова (убираем конечную кавычку)
                    style = parts[2]
                    topic = parts[3]
                    
                    # Проверяем стиль
                    if 'using_namespace_std' in style:
                        style_type = 'using'
                    elif 'explicit_std' in style:
                        style_type = 'explicit'
                    else:
                        style_errors += 1
                        # Добавим вывод проблемных строк
                        if style_errors <= 4:    # Покажем первые 4
                            print(f"Строка {line_num}: неопределённый стиль '{style}'!")
                        continue
                    
                    # Обновляем статистику
                    if topic in topic_stats:
                        topic_stats[topic][style_type] += 1
                        
                except IndexError:
                    continue
                    
    except Exception as e:
        print(f"Ошибка при анализе баланса: {e}!")
        return topic_stats
    
    # ======================================================================
    # АНАЛИЗ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    problems = []
    good_topics = []
    warning_topics = []
    
    for topic in topics:
        using_count = topic_stats[topic]['using']
        explicit_count = topic_stats[topic]['explicit']
        total = using_count + explicit_count
        
        if total == 0:
            problems.append(f"Тема '{topic}': не найдена в датасете (возможно ошибка парсинга)")
        elif using_count == 0:
            problems.append(f"Тема '{topic}': {explicit_count} explicit, НЕТ using_namespace_std")
        elif explicit_count == 0:
            problems.append(f"Тема '{topic}': {using_count} using, НЕТ explicit_std")
        elif using_count < 2 or explicit_count < 2:
            warning_topics.append((topic, using_count, explicit_count))
        else:
            good_topics.append((topic, using_count, explicit_count))
    
    # ======================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ======================================================================
    
    if good_topics:
        print_subheader("ТЕМЫ С ХОРОШИМ БАЛАНСОМ:")
        for topic, using, explicit in good_topics[:10]:    # Показываем первые 10
            print(f"- {topic:25} using={using:3}, explicit={explicit:3}")
        if len(good_topics) > 10:
            print(f"... и еще {len(good_topics) - 10} тем")
    
    if warning_topics:
        print_subheader("ТЕМЫ С МАЛЫМ КОЛИЧЕСТВОМ ПРИМЕРОВ:")
        for topic, using, explicit in warning_topics:
            print(f"- {topic:25} using={using:3}, explicit={explicit:3} (нужно больше)")
    
    if problems:
        print_subheader("ПРОБЛЕМНЫЕ ТЕМЫ:")
        print(f"Найдено проблем: {len(problems)}")
        for problem in problems[:10]:
            print(f"- {problem}")
        if len(problems) > 10:
            print(f"... и еще {len(problems) - 10} проблем")
    else:
        print("\nВсе темы имеют примеры в обоих стилях!")
    
    print(f"\nОшибок определения стиля: {style_errors}")
    
    return topic_stats


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
    
    # Извлекаем темы в порядке появления
    topics = extract_unique_topics_in_order(filename)
    
    if not topics:
        print("Не удалось извлечь темы из файла!")
        return 1
    
    # Проверяем баланс по стилям
    stats = check_topic_balance_simple(filename, topics)
    
    # ======================================================================
    # РЕЗУЛЬТАТ
    # ======================================================================
    
    print_header("РЕЗУЛЬТАТ:")
    
    if 10 <= len(topics) <= 50:
        print(f"Количество тем: {len(topics)} (идеально: 20-50)")
    elif len(topics) < 20:
        print(f"Мало тем: {len(topics)} (рекомендуется 20-50)")
        print("Возможно, нужно добавить больше разнообразных тем!")
    else:
        print(f"Много тем: {len(topics)} (рекомендуется не более 50)")
        print("Рассмотрите объединение похожих тем!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())