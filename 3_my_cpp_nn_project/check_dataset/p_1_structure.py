#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# p_1_structure.py - Проверка структурной целостности датасета
# ======================================================================
#
# @file p_1_structure.py
# @brief Проверка структурной целостности CSV датасета C++ кода
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Выполняет комплексную проверку структуры датасета:
#          - Наличие ровно 5 полей в каждой строке
#          - Корректное экранирование кавычек
#          - Правильные значения полей (style, topic, keywords)
#          - Отсутствие пустых обязательных полей
#
#          Формат датасета:
#          description,code,style,topic,keywords
#          "описание","код","using_namespace_std|explicit_std","тема","ключевые_слова"
#
# @usage python p_1_structure.py [путь_к_файлу.csv]
#
# @example
#   python p_1_structure.py
#   python p_1_structure.py ../data/raw/2_cpp_code_generation_dataset.csv
#
# ======================================================================

import sys
from datetime import datetime
from typing import List, Dict


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 70) -> None:
    """
    Вывести заголовок раздела.
    
    Args:
        title:    Заголовок
        width:    Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

def print_subheader(title: str, width: int = 70) -> None:
    """
    Вывести подзаголовок раздела.
    
    Args:
        title:    Подзаголовок
        width:    Ширина линии
    """
    print(f"\n{'-' * width}")
    print(f"{title}")
    print(f"{'-' * width}")

def extract_fields(line: str) -> List[str]:
    """
    Извлекает поля из CSV строки с учетом кавычек и экранирования.
    
    Args:
        line: Строка CSV
        
    Returns:
        List[str]:    Список извлеченных полей
    
    **Поддерживает:**
    - Кавычки внутри полей
    - Экранированные символы
    - Пустые поля
    """
    fields = []
    current_field = []
    in_quotes = False
    escaped = False
    
    for char in line:
        if escaped:
            escaped = False
            current_field.append(char)
            continue
            
        if char == '\\':
            escaped = True
            current_field.append(char)
            continue
            
        if char == '"':
            in_quotes = not in_quotes
            current_field.append(char)
            continue
            
        if char == ',' and not in_quotes:
            fields.append(''.join(current_field))
            current_field = []
        else:
            current_field.append(char)
    
    # добавляем последнее поле
    if current_field:
        fields.append(''.join(current_field))
    
    return fields

def save_error_report(filename: str, results: Dict) -> None:
    """
    Сохраняет подробный отчет об ошибках в файл.
    
    Args:
        filename:    Имя исходного файла
        results:     Результаты проверки со списком ошибок
    
    **Создает файл:** `structure_report_<имя_файла>.txt`
    """
    report_filename = f"structure_report_{filename.split('/')[-1]}.txt"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ОТЧЕТ О ПРОВЕРКЕ СТРУКТУРЫ ДАТАСЕТА\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Файл: {filename}\n")
            f.write(f"Время проверки: {datetime.now()}\n\n")
            
            f.write("СТАТИСТИКА:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Всего строк в файле: {results['total_lines']}\n")
            f.write(f"Пропущено комментариев: {results['comments_skipped']}\n")
            f.write(f"Пропущено заголовков: {results['header_skipped']}\n")
            f.write(f"Валидных строк: {results['valid_rows']}\n")
            f.write(f"Невалидных строк: {results['invalid_rows']}\n")
            f.write(f"Всего ошибок: {len(results['errors'])}\n\n")
            
            # группируем ошибки по категориям
            error_categories = {
                'description': [],
                'code': [],
                'style': [],
                'topic': [],
                'keywords': [],
                'structure': [],
                'other': []
            }
            
            for error in results['errors']:
                if 'description' in error.lower():
                    error_categories['description'].append(error)
                elif 'code' in error.lower():
                    error_categories['code'].append(error)
                elif 'style' in error.lower():
                    error_categories['style'].append(error)
                elif 'topic' in error.lower():
                    error_categories['topic'].append(error)
                elif 'keywords' in error.lower():
                    error_categories['keywords'].append(error)
                elif 'количество полей' in error or 'кавычкой' in error:
                    error_categories['structure'].append(error)
                else:
                    error_categories['other'].append(error)
            
            f.write("РАСПРЕДЕЛЕНИЕ ОШИБОК ПО КАТЕГОРИЯМ:\n")
            f.write("-" * 40 + "\n")
            for category, errors in error_categories.items():
                if errors:
                    f.write(f"{category.capitalize()}: {len(errors)} ошибок\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("ПОДРОБНЫЙ СПИСОК ОШИБОК:\n")
            f.write("=" * 70 + "\n\n")
            
            for i, error in enumerate(results['errors'], 1):
                f.write(f"{i}. {error}\n")
        
        print(f"\nПолный отчет сохранен в: {report_filename}")
        
    except Exception as e:
        print(f"!!! Не удалось сохранить отчет: {e}")

# ======================================================================
# ОСНОВНЫЕ ФУНКЦИИ ПРОВЕРКИ
# ======================================================================

def analyze_sample_data(filename: str, sample_size: int = 5) -> None:
    """
    Анализирует несколько примеров строк для демонстрации структуры.
    
    Args:
        filename:       Путь к файлу датасета
        sample_size:    Количество примеров для анализа
    """
    print_header("АНАЛИЗ ПРИМЕРОВ СТРОК")
    
    samples = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.rstrip('\n')
                
                # пропускаем пустые строки, комментарии и заголовок
                if (not line or 
                    line.startswith('#') or 
                    line == 'description,code,style,topic,keywords'):
                    continue
                
                if len(samples) < sample_size:
                    samples.append((line_num, line))
                else:
                    break
        
        for i, (line_num, line) in enumerate(samples, 1):
            print(f"\nПример {i} (Строка {line_num}):")
            print("-" * 40)
            
            # показываем сокращенную строку
            preview = line[:150] + "..." if len(line) > 150 else line
            print(f"Строка: {preview}")
            print(f"Длина: {len(line)} символов")
            
            # считаем кавычки и запятые
            quote_count = line.count('"')
            comma_count = line.count(',')
            
            print(f"    \" Кавычек: {quote_count} "
                  f"(должно быть четное число: {'V' if quote_count % 2 == 0 else 'X'})")
            print(f"    , Запятых: {comma_count} "
                  f"(должно быть 4: {'V' if comma_count == 4 else 'X'})")
            
            # пытаемся извлечь поля
            try:
                fields = extract_fields(line)
                print(f"    Извлечено полей: {len(fields)} "
                      f"(должно быть 5: {'V' if len(fields) == 5 else 'X'})")
                
                if len(fields) >= 3:
                    style_value = fields[2].strip('"')
                    print(f"    Style значение: '{style_value}'")
                
                if len(fields) >= 5:
                    print(f"    Последние 2 поля:")
                    topic_preview = fields[3][:50] + "..." if len(fields[3]) > 50 else fields[3]
                    keywords_preview = fields[4][:50] + "..." if len(fields[4]) > 50 else fields[4]
                    print(f"    - topic: {topic_preview}")
                    print(f"    - keywords: {keywords_preview}")
                    
            except Exception as e:
                print(f"X Ошибка при анализе: {e}")
                
    except Exception as e:
        print(f"X Ошибка при анализе примеров: {e}")

def check_dataset_structure_final(filename: str) -> bool:
    """
    Проверяет, что каждая строка имеет ровно 5 полей в правильном формате.
    
    Args:
        filename:    Путь к файлу датасета
        
    Returns:
        bool:    True если все проверки пройдены, False при наличии ошибок
    
    **Проверяет:**
    1. Строка начинается и заканчивается кавычкой
    2. Ровно 4 запятые вне кавычек (5 полей)
    3. Все поля извлекаются корректно
    4. Поле style имеет допустимое значение
    5. Поля topic и keywords не пустые
    """
    print_header("ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТА")
    print(f"Файл: {filename}")
    print("-" * 70)
    
    results = {
        'total_lines': 0,
        'comments_skipped': 0,
        'header_skipped': 0,
        'valid_rows': 0,
        'invalid_rows': 0,
        'errors': []
    }
    
    field_names = ["description", "code", "style", "topic", "keywords"]
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                results['total_lines'] += 1
                line = line.rstrip('\n')
                
                # 1. Пропускаем пустые строки и комментарии
                if not line or line.startswith('#'):
                    results['comments_skipped'] += 1
                    continue
                
                # 2. Пропускаем заголовок
                if line_num == 1 and line == 'description,code,style,topic,keywords':
                    results['header_skipped'] += 1
                    print("Заголовок CSV найден и пропущен")
                    continue
                
                # 3. Проверяем базовую структуру
                if not (line.startswith('"') and line.endswith('"')):
                    results['errors'].append(
                        f"Строка {line_num}: Не начинается или не заканчивается кавычкой")
                    results['invalid_rows'] += 1
                    continue
                
                # 4. Проверяем количество полей (должно быть ровно 5)
                # считаем запятые вне кавычек
                comma_count = 0
                in_quotes = False
                escaped = False
                
                for i, char in enumerate(line):
                    if escaped:
                        escaped = False
                        continue
                    if char == '\\':
                        escaped = True
                        continue
                    if char == '"':
                        in_quotes = not in_quotes
                        continue
                    if char == ',' and not in_quotes:
                        comma_count += 1
                
                # должно быть 4 запятые, разделяющие 5 полей
                if comma_count != 4:
                    results['errors'].append(
                        f"Строка {line_num}: Неправильное количество полей "
                        f"({comma_count + 1} вместо 5)")
                    results['invalid_rows'] += 1
                    continue
                
                # 5. Пытаемся извлечь все 5 полей
                try:
                    fields = extract_fields(line)
                    
                    if len(fields) != 5:
                        results['errors'].append(
                            f"Строка {line_num}: Удалось извлечь только {len(fields)} полей")
                        results['invalid_rows'] += 1
                        continue
                    
                    # 6. Проверяем каждое поле
                    all_fields_valid = True
                    
                    # поле 0: description - должно быть в кавычках
                    if not (fields[0].startswith('"') and fields[0].endswith('"')):
                        results['errors'].append(
                            f"Строка {line_num}: Поле description не в кавычках")
                        all_fields_valid = False
                    
                    # поле 1: code - должно быть в кавычках
                    if not (fields[1].startswith('"') and fields[1].endswith('"')):
                        results['errors'].append(
                            f"Строка {line_num}: Поле code не в кавычках")
                        all_fields_valid = False
                    
                    # поле 2: style - должно быть 'using_namespace_std' или 'explicit_std'
                    style_value = fields[2].strip('"')
                    if style_value not in ['using_namespace_std', 'explicit_std']:
                        results['errors'].append(
                            f"Строка {line_num}: Недопустимое значение style: '{style_value}'")
                        all_fields_valid = False
                    
                    # поле 3: topic - не должно быть пустым
                    if not fields[3].strip('"'):
                        results['errors'].append(
                            f"Строка {line_num}: Пустое поле topic")
                        all_fields_valid = False
                    
                    # поле 4: keywords - не должно быть пустым
                    if not fields[4].strip('"'):
                        results['errors'].append(
                            f"Строка {line_num}: Пустое поле keywords")
                        all_fields_valid = False
                    
                    if all_fields_valid:
                        results['valid_rows'] += 1
                    else:
                        results['invalid_rows'] += 1
                        
                except Exception as e:
                    results['errors'].append(
                        f"Строка {line_num}: Ошибка при разборе полей: {str(e)}")
                    results['invalid_rows'] += 1
                
                # показываем прогресс каждые 1000 строк
                if line_num % 1000 == 0:
                    print(f"Обработано {line_num} строк...")
                    
    except FileNotFoundError:
        print(f"X Ошибка: Файл '{filename}' не найден!")
        return False
    except Exception as e:
        print(f"X Ошибка при чтении файла: {e}")
        return False
    
    # выводим результаты
    print_subheader("РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print(f"Всего строк в файле: {results['total_lines']}")
    print(f"Пропущено комментариев: {results['comments_skipped']}")
    print(f"Пропущено заголовков: {results['header_skipped']}")
    print(f"Валидных строк: {results['valid_rows']}")
    print(f"Невалидных строк: {results['invalid_rows']}")
    print(f"Найдено ошибок: {len(results['errors'])}")
    
    if results['errors']:
        print_subheader("ДЕТАЛИ ОШИБОК")
        
        # группируем ошибки по типам
        error_types = {}
        for error in results['errors']:
            error_type = error.split(':')[1].strip().split()[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("Статистика по типам ошибок:")
        for err_type, count in sorted(error_types.items()):
            print(f"  • {err_type}: {count}")
        
        print(f"\nПервые 10 ошибок:")
        for i, error in enumerate(results['errors'][:10]):
            print(f"  {i+1}. {error}")
        
        if len(results['errors']) > 10:
            print(f"... и еще {len(results['errors']) - 10} ошибок")
        
        # сохраняем полный отчет
        save_error_report(filename, results)
        
        print_header("!!! ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ")
        return False
    
    else:
        print_header("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print(f"Все {results['valid_rows']} строк имеют правильную структуру")
        return True


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================

def main() -> int:
    """
    Основная функция.
    
    Returns:
        int:    0 при успехе, 1 при ошибке
    """
    # определяем путь к файлу
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv"
    
    print(f"Анализируемый файл: {filename}")
    
    # анализируем несколько примеров
    analyze_sample_data(filename, 3)
    
    # запускаем полную проверку
    is_valid = check_dataset_structure_final(filename)
    
    if is_valid:
        print("\nДатасет готов к использованию!")
        return 0
    else:
        print("\n!!! Требуются исправления!")
        print("Сначала исправьте найденные ошибки структуры.")
        return 1


if __name__ == "__main__":
    sys.exit(main())