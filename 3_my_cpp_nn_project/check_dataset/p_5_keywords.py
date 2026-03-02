#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# p_5_keywords.py - Проверка ключевых слов в датасете C++ кода
# ======================================================================
#
# @file p_5_keywords.py
# @brief Комплексная проверка поля keywords в датасете C++ кода
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Выполняет всестороннюю проверку и исправление ключевых слов в датасете:
#
#          1. **Базовые проверки:**
#             - Наличие ключевых слов (не пустые)
#             - Количество ключевых слов (2-15 оптимально)
#             - Формат CSV (корректность разбора)
#
#          2. **Проверка префиксов в зависимости от стиля:**
#             - В `explicit_std` ключевые слова из std должны иметь префикс `std::`
#             - В `using_namespace_std` ключевые слова из std НЕ должны иметь префикс `std::`
#
#          3. **Проверка соответствия коду (только для нарушителей):**
#             - Если в explicit_std слово без std::, проверяем его в коде
#             - Если в коде оно используется с std:: → ОШИБКА НЕСООТВЕТСТВИЯ
#             - Если в коде тоже без std:: → просто предупреждение
#
#          4. **Автоматическое исправление (опционально):**
#             - Добавление std:: к ключевым словам в explicit_std, если в коде они с std::
#             - Создание исправленной копии датасета
#
#          5. **Проверка консистентности:**
#             - Для одной темы ключевые слова должны быть одинаковыми в обоих стилях
#             - (с учётом правил префиксов и только для слов, присутствующих в коде)
#
# @usage python p_5_keywords.py
#
# @example
#   python p_5_keywords.py                    # только проверка
#   python p_5_keywords.py --fix              # проверка + исправление
#   python p_5_keywords.py --fix --output fixed_dataset.csv
#
# ======================================================================

import sys
import os
import random
import argparse
from typing import List, Tuple, Dict, Set


# ======================================================================
# КОНСТАНТЫ
# ======================================================================

# ключевые слова, которые ВСЕГДА требуют std:: в explicit_std
ALWAYS_STD_KEYWORDS = {
    # IO streams
    'cout', 'endl', 'cin', 'cerr', 'clog',
    'ostream', 'istream', 'iostream', 'fstream', 'sstream',
    'stringstream', 'ostringstream', 'istringstream',
    
    # strings
    'string', 'wstring', 'u16string', 'u32string',
    'string_view', 'wstring_view',
    
    # STL containers
    'vector', 'array', 'deque', 'forward_list', 'list',
    'set', 'multiset', 'unordered_set', 'unordered_multiset',
    'map', 'multimap', 'unordered_map', 'unordered_multimap',
    'stack', 'queue', 'priority_queue',
    
    # STL algorithms
    'sort', 'find', 'copy', 'transform', 'accumulate',
    'max_element', 'min_element', 'reverse', 'unique',
    'binary_search', 'lower_bound', 'upper_bound',
    'generate',
    
    # smart pointers
    'unique_ptr', 'shared_ptr', 'weak_ptr',
    
    # threading
    'thread', 'mutex', 'lock_guard', 'unique_lock',
    'condition_variable', 'future', 'promise', 'async',
    
    # utilities
    'pair', 'tuple', 'optional', 'variant', 'any',
    'function', 'bind', 'ref', 'cref',
    
    # chrono
    'chrono', 'steady_clock', 'system_clock', 'high_resolution_clock',
    'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
    
    # filesystem
    'filesystem', 'path', 'directory_iterator', 'recursive_directory_iterator',
    
    # other std
    'regex', 'random', 'ratio', 'complex', 'valarray',
    'bitset', 'type_info', 'type_index', 'bad_cast',
    'bad_alloc', 'exception', 'runtime_error', 'logic_error',
    
    # common patterns
    'getline', 'stoi', 'stod', 'to_string', 'move', 'forward'
}

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 70) -> None:
    """Вывести заголовок раздела."""
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

def print_subheader(title: str, width: int = 70) -> None:
    """Вывести подзаголовок раздела."""
    print(f"\n{'-' * width}")
    print(f"{title}")
    print(f"{'-' * width}")

def is_std_library_keyword(keyword: str) -> bool:
    """
    Проверяет, относится ли ключевое слово к стандартной библиотеке C++.
    
    Args:
        keyword:    Проверяемое ключевое слово (без префикса std::)
        
    Returns:
        bool:    True если слово должно иметь префикс std:: в explicit_std
    """
    base_keyword = keyword[5:] if keyword.startswith('std::') else keyword
    return base_keyword in ALWAYS_STD_KEYWORDS

def keyword_in_code(keyword: str, code: str) -> bool:
    """
    Проверяет, присутствует ли ключевое слово в коде.
    
    Args:
        keyword:    Ключевое слово для поиска
        code:       Код программы
        
    Returns:
        bool:       True если слово найдено в коде
    """
    # ищем как отдельное слово
    patterns = [
        f" {keyword} ", f"{keyword} ", f" {keyword}",
        f"({keyword}", f"{keyword})", f"<{keyword}>",
        f"{keyword};", f"{keyword},", f"{keyword}.",
        f"->{keyword}", f"::{keyword}",
    ]
    
    for pattern in patterns:
        if pattern in code:
            return True
    
    # проверяем начало и конец строки
    if code.startswith(keyword) or code.endswith(keyword):
        return True
    
    return False

def extract_fields(line: str) -> List[str]:
    """
    Извлекает поля из CSV строки с учётом кавычек.
    
    Args:
        line:    Строка CSV
        
    Returns:
        List[str]:    Список полей
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
    
    if current_field:
        fields.append(''.join(current_field))
    
    return fields

# ======================================================================
# ФУНКЦИЯ ИСПРАВЛЕНИЯ
# ======================================================================

def fix_keywords_in_file(input_filename: str, output_filename: str) -> Dict:
    """
    Исправляет ключевые слова в датасете.
    
    Args:
        input_filename:     Исходный файл
        output_filename:    Файл для сохранения исправленной версии
        
    Returns:
        Dict:    Статистика исправлений
    
    **Что исправляется:**
    - В explicit_std: добавление std:: к ключевым словам, если в коде они используются с std::
    - В using_namespace_std: пока ничего не исправляем (может быть добавлено позже)
    """
    print_header("ИСПРАВЛЕНИЕ КЛЮЧЕВЫХ СЛОВ")
    print(f"Исходный файл: {input_filename}")
    print(f"Исправленный файл: {output_filename}")
    
    stats = {
        'total_lines': 0,
        'fixed_lines': 0,
        'fixed_keywords': 0,
        'skipped_lines': 0,
        'errors': 0
    }
    
    fixes = []    # для детального отчёта
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as fin, \
             open(output_filename, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                line = line.rstrip('\n')
                stats['total_lines'] += 1
                
                # пропускаем пустые строки и комментарии
                if not line or line.startswith('#'):
                    fout.write(line + '\n')
                    stats['skipped_lines'] += 1
                    continue
                
                try:
                    # парсим строку
                    fields = extract_fields(line)
                    
                    if len(fields) < 5:
                        # недостаточно полей, сохраняем как есть
                        fout.write(line + '\n')
                        stats['errors'] += 1
                        continue
                    
                    # извлекаем поля
                    desc = fields[0]
                    code = fields[1]
                    style = fields[2]
                    topic = fields[3]
                    keywords_field = fields[4]
                    
                    # убираем внешние кавычки у keywords для обработки
                    if keywords_field.startswith('"') and keywords_field.endswith('"'):
                        keywords_field = keywords_field[1:-1]
                    
                    # разбиваем ключевые слова
                    keywords = [k.strip() for k in keywords_field.split(',') if k.strip()]
                    original_keywords = keywords.copy()
                    
                    # анализируем и исправляем
                    line_fixed = False
                    line_fixes = []
                    
                    if 'explicit_std' in style:
                        for i, kw in enumerate(keywords):
                            base_kw = kw[5:] if kw.startswith('std::') else kw
                            
                            # если это std-ключевое слово и оно без std::
                            if is_std_library_keyword(base_kw) and not kw.startswith('std::'):
                                # проверяем, используется ли в коде с std::
                                if f"std::{base_kw}" in code:
                                    # исправляем: добавляем std::
                                    keywords[i] = f"std::{base_kw}"
                                    line_fixed = True
                                    stats['fixed_keywords'] += 1
                                    line_fixes.append(f"{kw} → std::{base_kw}")
                    
                    # если были исправления, обновляем строку
                    if line_fixed:
                        stats['fixed_lines'] += 1
                        fixes.append({
                            'line': line_num,
                            'style': style,
                            'fixes': line_fixes
                        })
                        
                        # собираем исправленную строку
                        new_keywords_field = ', '.join(keywords)
                        
                        # восстанавливаем кавычки
                        new_line = f'{desc},{code},{style},{topic},"{new_keywords_field}"'
                        fout.write(new_line + '\n')
                    else:
                        fout.write(line + '\n')
                        
                except Exception as e:
                    print(f"X Ошибка в строке {line_num}: {e}")
                    fout.write(line + '\n')
                    stats['errors'] += 1
    
    except Exception as e:
        print(f"X Критическая ошибка: {e}")
        return stats
    
    # выводим статистику
    print_subheader("СТАТИСТИКА ИСПРАВЛЕНИЙ:")
    print(f"- всего строк обработано: {stats['total_lines']}")
    print(f"- строк с исправлениями: {stats['fixed_lines']}")
    print(f"- исправлено ключевых слов: {stats['fixed_keywords']}")
    print(f"- пропущено (пустые/комментарии): {stats['skipped_lines']}")
    print(f"- ошибок: {stats['errors']}")
    
    if fixes:
        print_subheader("ПРИМЕРЫ ИСПРАВЛЕНИЙ:")
        for fix in fixes[:5]:
            print(f"\nСтрока {fix['line']} ({fix['style']}):")
            for f in fix['fixes']:
                print(f"     • {f}")
        if len(fixes) > 5:
            print(f"\n... и еще {len(fixes) - 5} строк с исправлениями")
    
    return stats

# ======================================================================
# ОСНОВНЫЕ ФУНКЦИИ ПРОВЕРКИ
# ======================================================================

def check_keywords(filename: str, sample_size: int = 10000) -> Dict[str, int]:
    """
    Выборочная проверка ключевых слов в датасете.
    
    Args:
        filename:       Путь к файлу датасета
        sample_size:    Количество случайных примеров для проверки
        
    Returns:
        Dict[str, int]:    Статистика проблем
    """
    print_header("ПРОВЕРКА КЛЮЧЕВЫХ СЛОВ (KEYWORDS)")
    print(f"Проверяем {sample_size} случайных примеров")
    print("=" * 70)
    
    # собираем все строки с данными
    data_lines: List[Tuple[int, str]] = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.rstrip('\n')
                
                if not raw_line.strip() or raw_line.strip().startswith('#'):
                    continue
                
                if not raw_line.startswith('"') or not raw_line.endswith('"'):
                    continue
                
                data_lines.append((line_num, raw_line))
    except FileNotFoundError:
        print(f"X Ошибка: Файл '{filename}' не найден!")
        return {}
    except Exception as e:
        print(f"X Ошибка при чтении файла: {e}")
        return {}
    
    print(f"Всего строк с данными: {len(data_lines)}")
    
    if sample_size > len(data_lines):
        sample_size = len(data_lines)
        print(f"!!! Размер выборки уменьшен до {sample_size} (максимум)")
    
    sample = random.sample(data_lines, sample_size)
    
    print(f"\nПроверяем {sample_size} случайных строк...")
    
    issues = {
        'no_keywords': 0,
        'too_few': 0,
        'too_many': 0,
        'wrong_prefix_explicit': 0,
        'wrong_prefix_using': 0,
        'mismatch_explicit': 0,
        'mismatch_using': 0,
        'format_issues': 0
    }
    
    prefix_examples = []
    mismatch_examples = []
    
    for line_num, line in sample:
        try:
            fields = extract_fields(line)
            
            if len(fields) < 5:
                issues['format_issues'] += 1
                continue
            
            code = fields[1]
            style = fields[2]
            
            keywords_field = fields[4]
            if keywords_field.startswith('"') and keywords_field.endswith('"'):
                keywords_field = keywords_field[1:-1]
            
            keywords = [k.strip() for k in keywords_field.split(',') if k.strip()]
            
            if not keywords:
                issues['no_keywords'] += 1
                continue
            
            if len(keywords) < 2:
                issues['too_few'] += 1
            elif len(keywords) > 15:
                issues['too_many'] += 1
            
            if 'explicit_std' in style:
                for kw in keywords:
                    base_kw = kw[5:] if kw.startswith('std::') else kw
                    if is_std_library_keyword(base_kw) and not kw.startswith('std::'):
                        issues['wrong_prefix_explicit'] += 1
                        
                        if len(prefix_examples) < 5:
                            prefix_examples.append({
                                'line': line_num,
                                'style': 'explicit_std',
                                'keyword': kw,
                                'base': base_kw
                            })
                        
                        if f"std::{base_kw}" in code:
                            issues['mismatch_explicit'] += 1
                            if len(mismatch_examples) < 5:
                                mismatch_examples.append({
                                    'line': line_num,
                                    'style': 'explicit_std',
                                    'keyword': kw,
                                    'problem': f"в keywords без std::, но в коде используется 'std::{base_kw}'"
                                })
                        
            elif 'using_namespace_std' in style:
                for kw in keywords:
                    if kw.startswith('std::'):
                        base_kw = kw[5:]
                        if is_std_library_keyword(base_kw):
                            issues['wrong_prefix_using'] += 1
                            
                            if len(prefix_examples) < 5:
                                prefix_examples.append({
                                    'line': line_num,
                                    'style': 'using_namespace_std',
                                    'keyword': kw,
                                    'base': base_kw
                                })
                            
                            if base_kw in code and f"std::{base_kw}" not in code:
                                issues['mismatch_using'] += 1
                                if len(mismatch_examples) < 5:
                                    mismatch_examples.append({
                                        'line': line_num,
                                        'style': 'using_namespace_std',
                                        'keyword': kw,
                                        'problem': f"в keywords с std::, но в коде используется '{base_kw}' без std::"
                                    })
                        
        except Exception:
            issues['format_issues'] += 1
    
    print_subheader("РЕЗУЛЬТАТЫ ПРОВЕРКИ КЛЮЧЕВЫХ СЛОВ")
    
    total_issues = sum(issues.values())
    
    if total_issues == 0:
        print("ВСЕ ключевые слова в выборке корректны!")
    else:
        print(f"!!! Найдено проблем в {total_issues} из {sample_size} проверенных строк:")
        print(f"- нет ключевых слов: {issues['no_keywords']}")
        print(f"- слишком мало (<2): {issues['too_few']}")
        print(f"- слишком много (>15): {issues['too_many']}")
        print(f"- explicit_std без std:: у std-ключевых: {issues['wrong_prefix_explicit']}")
        print(f"  └─ из них несоответствие коду (в коде с std::): {issues['mismatch_explicit']} ← ИСПРАВЛЯЕМЫЕ")
        print(f"- using_namespace_std с std:: {issues['wrong_prefix_using']}")
        print(f"  └─ из них несоответствие коду (в коде без std::): {issues['mismatch_using']}")
        print(f"- проблемы формата: {issues['format_issues']}")
    
    if prefix_examples:
        print_subheader("ПРИМЕРЫ НАРУШЕНИЙ ПРЕФИКСОВ:")
        for ex in prefix_examples:
            if 'explicit_std' in ex['style']:
                print(f"\nСтрока {ex['line']} ({ex['style']}):")
                print(f"- '{ex['keyword']}' должно быть с std:: (как 'std::{ex['base']}')")
            else:
                print(f"\nСтрока {ex['line']} ({ex['style']}):")
                print(f"- '{ex['keyword']}' не должно быть с std:: (должно быть '{ex['base']}')")
    
    if mismatch_examples:
        print_subheader("ПРИМЕРЫ НЕСООТВЕТСТВИЯ КОДУ (ИСПРАВЛЯЕМЫЕ):")
        for ex in mismatch_examples:
            print(f"\nСтрока {ex['line']} ({ex['style']}):")
            print(f"- {ex['problem']}")
            if 'explicit_std' in ex['style']:
                print(f"-> будет исправлено на: std::{ex['keyword']}")
    
    return issues

def check_keyword_consistency(filename: str) -> Tuple[List[str], Dict]:
    """
    Проверяет консистентность ключевых слов для одинаковых тем.
    
    Args:
        filename:    Путь к файлу датасета
        
    Returns:
        Tuple[List[str], Dict]:    Список неконсистентных тем и детальная статистика
    """
    print_subheader("ПРОВЕРКА КОНСИСТЕНТНОСТИ КЛЮЧЕВЫХ СЛОВ")
    print("(учитываются только ключевые слова, реально присутствующие в коде)")
    
    topic_data: Dict[str, Dict[str, List[Set[str]]]] = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.rstrip('\n')
                
                if not raw_line.strip() or raw_line.strip().startswith('#'):
                    continue
                
                if not raw_line.startswith('"') or not raw_line.endswith('"'):
                    continue
                
                try:
                    fields = extract_fields(raw_line)
                    
                    if len(fields) < 5:
                        continue
                    
                    code = fields[1]
                    style = fields[2]
                    topic = fields[3]
                    
                    keywords_field = fields[4]
                    if keywords_field.startswith('"') and keywords_field.endswith('"'):
                        keywords_field = keywords_field[1:-1]
                    
                    all_keywords = [k.strip() for k in keywords_field.split(',') if k.strip()]
                    
                    # оставляем только те ключевые слова, которые есть в коде
                    present_keywords = set()
                    for kw in all_keywords:
                        base_kw = kw[5:] if kw.startswith('std::') else kw
                        if base_kw in code or f"std::{base_kw}" in code:
                            present_keywords.add(kw)
                    
                    if not present_keywords:
                        continue
                    
                    if topic not in topic_data:
                        topic_data[topic] = {'using': [], 'explicit': []}
                    
                    if 'using_namespace_std' in style:
                        topic_data[topic]['using'].append(present_keywords)
                    elif 'explicit_std' in style:
                        topic_data[topic]['explicit'].append(present_keywords)
                        
                except Exception:
                    continue
                    
    except Exception as e:
        print(f"X Ошибка при проверке консистентности: {e}")
        return [], {}
    
    print("\nАнализ консистентности ключевых слов по темам:\n")
    
    topic_stats = {}
    inconsistent_topics = []
    consistent_topics = []
    
    for topic, styles in topic_data.items():
        if not styles['using'] or not styles['explicit']:
            topic_stats[topic] = {
                'status': 'skipped',
                'reason': 'нет обоих стилей',
                'using_count': len(styles['using']),
                'explicit_count': len(styles['explicit'])
            }
            continue
        
        explicit_sets = []
        for kw_set in styles['explicit']:
            clean_set = set()
            for kw in kw_set:
                if kw.startswith('std::'):
                    clean_set.add(kw[5:])
                else:
                    clean_set.add(kw)
            explicit_sets.append(clean_set)
        
        using_sets = styles['using']
        
        using_combined = set().union(*using_sets) if using_sets else set()
        explicit_combined = set().union(*explicit_sets) if explicit_sets else set()
        
        is_consistent = True
        reason = ""
        
        if using_combined != explicit_combined:
            is_consistent = False
            only_in_using = using_combined - explicit_combined
            only_in_explicit = explicit_combined - using_combined
            
            if only_in_using:
                reason += f"только в using: {only_in_using}"
            if only_in_explicit:
                if reason:
                    reason += "; "
                reason += f"только в explicit: {only_in_explicit}"
        
        topic_stats[topic] = {
            'status': 'inconsistent' if not is_consistent else 'consistent',
            'using_keywords': sorted(using_combined),
            'explicit_keywords': sorted(explicit_combined),
            'using_count': len(styles['using']),
            'explicit_count': len(styles['explicit']),
            'differences': reason if not is_consistent else 'нет'
        }
        
        if not is_consistent:
            inconsistent_topics.append(topic)
        else:
            consistent_topics.append(topic)
    
    print(f"Всего тем с обоими стилями: {len(consistent_topics) + len(inconsistent_topics)}")
    print(f"- консистентных тем: {len(consistent_topics)}")
    print(f"- неконсистентных тем: {len(inconsistent_topics)}")
    
    if inconsistent_topics:
        print("\nДетали по неконсистентным темам (первые 5):")
        for topic in inconsistent_topics[:5]:
            stats = topic_stats[topic]
            print(f"\n- {topic}")
            print(f"    using:    {', '.join(stats['using_keywords'][:5])}")
            print(f"    explicit: {', '.join(stats['explicit_keywords'][:5])}")
            print(f"    различия: {stats['differences']}")
        
        if len(inconsistent_topics) > 5:
            print(f"\n... и еще {len(inconsistent_topics) - 5} тем")
    
    return inconsistent_topics, topic_stats


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================

def main() -> int:
    """
    Основная функция.
    
    Returns:
        int:    0 при успехе, 1 при ошибке
    """
    # определяем путь к файлу относительно текущей директории
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, '2_cpp_code_generation_dataset.csv')
    
    parser = argparse.ArgumentParser(description='Проверка ключевых слов в датасете C++ кода')
    parser.add_argument('filename', nargs='?',
                       default=default_path,
                       help='Путь к файлу датасета')
    parser.add_argument('--fix', action='store_true',
                       help='Исправить найденные проблемы (добавить std:: в explicit_std)')
    parser.add_argument('--output', '-o', default='reports/fixed_dataset.csv',
                       help='Имя файла для сохранения исправленной версии (по умолчанию: reports/fixed_dataset.csv)')
    parser.add_argument('--sample', type=int, default=10000,
                       help='Размер выборки для проверки (по умолчанию: 10000)')
    
    args = parser.parse_args()
    
    filename = args.filename
    print(f"Анализируемый файл: {filename}")
    
    # проверяем ключевые слова
    issues = check_keywords(filename, sample_size=args.sample)
    
    if not issues:
        return 1
    
    # проверяем консистентность
    inconsistent, topic_stats = check_keyword_consistency(filename)
    
    # ======================================================================
    # ИСПРАВЛЕНИЕ (если запрошено)
    # ======================================================================
    
    if args.fix and issues.get('mismatch_explicit', 0) > 0:
        print_header("ЗАПУСК ИСПРАВЛЕНИЯ")
        print(f"Будет исправлено {issues['mismatch_explicit']} случаев несоответствия в explicit_std")
        
        # создаем папку reports, если её нет
        os.makedirs('reports', exist_ok=True)
        fix_stats = fix_keywords_in_file(filename, args.output)
        
        if fix_stats['fixed_lines'] > 0:
            print(f"\nИсправления применены. Сохранено в: {args.output}")
            print(f"Рекомендуется проверить исправленный файл и перезапустить проверку:")
            print(f"python p_5_keywords.py {args.output}")
    
    # ======================================================================
    # РЕКОМЕНДАЦИИ
    # ======================================================================
    
    print_header("РЕКОМЕНДАЦИИ ПО КЛЮЧЕВЫМ СЛОВАМ")
    
    if issues.get('no_keywords', 0) > 0:
        print("\n* Добавьте ключевые слова для всех примеров")
        print("Каждая программа должна иметь хотя бы одно ключевое слово")
    
    if issues.get('too_few', 0) > 0 or issues.get('too_many', 0) > 0:
        print("\n* Оптимальное количество ключевых слов: 2-15")
        print("Ключевые слова должны отражать основные конструкции программы")
    
    if issues.get('wrong_prefix_explicit', 0) > 0:
        print(f"\n* В explicit_std найдено {issues['wrong_prefix_explicit']} слов без std::")
        print(f"Из них {issues['mismatch_explicit']} можно ИСПРАВИТЬ АВТОМАТИЧЕСКИ (добавить std::)")
        if not args.fix and issues['mismatch_explicit'] > 0:
            print(f"\nДля автоматического исправления выполните:")
            print(f"python p_5_keywords.py --fix")
    
    if issues.get('wrong_prefix_using', 0) > 0:
        print(f"\n* В using_namespace_std найдено {issues['wrong_prefix_using']} слов с std::")
        print(f"Из них {issues['mismatch_using']} не соответствуют коду (требуют ручного исправления)")
    
    if inconsistent:
        print(f"\n* Найдено {len(inconsistent)} неконсистентных тем")
        print("Для каждой темы ключевые слова должны быть одинаковыми в обоих стилях")
        print("(с учётом префикса std:: для std-ключевых слов)")
    
    print("\nИтоговая статистика:")
    total_issues = sum(issues.values())
    print(f"- всего проблем в выборке: {total_issues}")
    print(f"- неконсистентных тем: {len(inconsistent)}")
    print(f"- автоматически исправляемых: {issues.get('mismatch_explicit', 0)}")
    
    return 0 if total_issues == 0 and not inconsistent else 1


if __name__ == "__main__":
    sys.exit(main())