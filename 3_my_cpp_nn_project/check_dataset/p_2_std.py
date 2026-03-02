#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# p_2_std.py - Балансировка стилей using_namespace_std и explicit_std
# ======================================================================
#
# @file p_2_std.py
# @brief Проверка и исправление стилевых нарушений в C++ коде
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Анализирует и исправляет противоречия между стилями
#          `using_namespace_std` и `explicit_std`. Если программа
#          использует `using_namespace_std`, то все обращения к
#          стандартной библиотеке должны быть без префикса `std::`.
#
#          **Правила:**
#          - `using_namespace_std` → без префикса `std::` (например, `cout`, `vector`)
#          - `explicit_std` → с префиксом `std::` (например, `std::cout`, `std::vector`)
#
#          **Исключения:** Есть ряд примеров (строки 9031, 9474, 10307, 10313, 10349,
#          10370, 10632, 11076, 11264, 11933, 12610, 13025), которые содержат
#          смешанный код и не должны корректироваться автоматически. Проверенно вручную
#
# @usage python p_2_std.py
#
# @example
#   python p_2_std.py          # только проверка нарушений
#   # Для исправления раскомментировать вызов fix_std_violations()
#
# ======================================================================

import sys
from typing import List


# ======================================================================
# КОНСТАНТЫ
# ======================================================================

# список строк, которые НЕ должны корректироваться (смешанный код). Проверенно вручную
EXCLUDED_LINES = [
    9031, 9474, 10307, 10313, 10349, 10370,
    10632, 11076, 11264, 11933, 12610, 13025
]

# словарь замен для std::префиксов
STD_REPLACEMENTS = {
    # ввод/вывод
    'std::cout': 'cout',
    'std::endl': 'endl',
    'std::cin': 'cin',
    'std::clog': 'clog',
    'std::cerr': 'cerr',
    
    # строки
    'std::string': 'string',
    'std::to_string': 'to_string',
    'std::stoi': 'stoi',
    'std::stol': 'stol',
    'std::stoll': 'stoll',
    'std::stof': 'stof',
    'std::stod': 'stod',
    
    # контейнеры
    'std::vector': 'vector',
    'std::array': 'array',
    'std::list': 'list',
    'std::forward_list': 'forward_list',
    'std::deque': 'deque',
    'std::map': 'map',
    'std::unordered_map': 'unordered_map',
    'std::set': 'set',
    'std::unordered_set': 'unordered_set',
    'std::pair': 'pair',
    'std::tuple': 'tuple',
    'std::span': 'span',
    'std::bitset': 'bitset',
    'std::stack': 'stack',
    'std::queue': 'queue',
    'std::priority_queue': 'priority_queue',
    
    # алгоритмы
    'std::sort': 'sort',
    'std::find': 'find',
    'std::count': 'count',
    'std::copy': 'copy',
    'std::move': 'move',
    'std::swap': 'swap',
    'std::transform': 'transform',
    'std::accumulate': 'accumulate',
    'std::remove_if': 'remove_if',
    'std::unique': 'unique',
    'std::reverse': 'reverse',
    'std::rotate': 'rotate',
    'std::partition': 'partition',
    'std::binary_search': 'binary_search',
    'std::lower_bound': 'lower_bound',
    'std::upper_bound': 'upper_bound',
    'std::equal_range': 'equal_range',
    'std::min': 'min',
    'std::max': 'max',
    'std::minmax': 'minmax',
    'std::clamp': 'clamp',
    'std::abs': 'abs',
    'std::pow': 'pow',
    'std::sqrt': 'sqrt',
    'std::sin': 'sin',
    'std::cos': 'cos',
    'std::tan': 'tan',
    
    # умные указатели
    'std::unique_ptr': 'unique_ptr',
    'std::shared_ptr': 'shared_ptr',
    'std::weak_ptr': 'weak_ptr',
    'std::make_unique': 'make_unique',
    'std::make_shared': 'make_shared',
    
    # многопоточность
    'std::thread': 'thread',
    'std::mutex': 'mutex',
    'std::lock_guard': 'lock_guard',
    'std::unique_lock': 'unique_lock',
    'std::condition_variable': 'condition_variable',
    'std::future': 'future',
    'std::promise': 'promise',
    'std::async': 'async',
    'std::atomic': 'atomic',
    
    # файлы
    'std::ifstream': 'ifstream',
    'std::ofstream': 'ofstream',
    'std::fstream': 'fstream',
    'std::stringstream': 'stringstream',
    'std::istringstream': 'istringstream',
    'std::ostringstream': 'ostringstream',
    
    # исключения
    'std::exception': 'exception',
    'std::runtime_error': 'runtime_error',
    'std::logic_error': 'logic_error',
    'std::invalid_argument': 'invalid_argument',
    'std::out_of_range': 'out_of_range',
    'std::bad_alloc': 'bad_alloc',
    'std::bad_cast': 'bad_cast',
    'std::bad_typeid': 'bad_typeid',
    
    # типы и traits
    'std::true_type': 'true_type',
    'std::false_type': 'false_type',
    'std::is_arithmetic': 'is_arithmetic',
    'std::is_arithmetic_v': 'is_arithmetic_v',
    'std::is_integral': 'is_integral',
    'std::is_floating_point': 'is_floating_point',
    'std::is_same': 'is_same',
    'std::is_convertible': 'is_convertible',
    'std::enable_if': 'enable_if',
    'std::conditional': 'conditional',
    'std::integral_constant': 'integral_constant',
    'std::type_info': 'type_info',
    'std::numeric_limits': 'numeric_limits',
    
    # sequence generators
    'std::integer_sequence': 'integer_sequence',
    'std::make_integer_sequence': 'make_integer_sequence',
    'std::index_sequence': 'index_sequence',
    'std::make_index_sequence': 'make_index_sequence',
    'std::index_sequence_for': 'index_sequence_for',
    
    # прочие
    'std::any': 'any',
    'std::variant': 'variant',
    'std::visit': 'visit',
    'std::optional': 'optional',
    'std::nullopt': 'nullopt',
    'std::byte': 'byte',
    'std::function': 'function',
    'std::bind': 'bind',
    'std::ref': 'ref',
    'std::cref': 'cref',
    'std::boolalpha': 'boolalpha',
    'std::noboolalpha': 'noboolalpha',
    'std::showbase': 'showbase',
    'std::hex': 'hex',
    'std::dec': 'dec',
    'std::oct': 'oct',
    
    # character classification
    'std::isupper': 'isupper',
    'std::islower': 'islower',
    'std::isdigit': 'isdigit',
    'std::isalpha': 'isalpha',
    'std::isalnum': 'isalnum',
    'std::isspace': 'isspace',
    'std::ispunct': 'ispunct',
    'std::isprint': 'isprint',
    'std::isgraph': 'isgraph',
    'std::iscntrl': 'iscntrl',
    
    # math
    'std::isnan': 'isnan',
    'std::isinf': 'isinf',
    'std::isfinite': 'isfinite',
    'std::isnormal': 'isnormal',
    'std::signbit': 'signbit',
}

# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела.
    
    Args:
        title:    Заголовок
        width:    Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

def is_excluded_line(line_num: int) -> bool:
    """
    Проверяет, является ли строка исключением (не должна исправляться).
    
    Args:
        line_num:    Номер строки
        
    Returns:
        bool:    True если строка в списке исключений
    """
    return line_num in EXCLUDED_LINES

# ======================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ======================================================================

def find_std_in_using_namespace_std(filename: str) -> List[int]:
    """
    Находит строки, где используется `using_namespace_std` вместе с `std::`.
    
    Args:
        filename:    Путь к файлу датасета
        
    Returns:
        List[int]:    Список номеров строк с нарушениями
    
    **Нарушение:** В строках с `using_namespace_std` не должно быть префикса `std::`,
    так как пространство имен уже подключено.
    """
    violations = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.rstrip('\n')
                
                # пропускаем пустые строки и комментарии
                if not raw_line.strip() or raw_line.strip().startswith('#'):
                    continue
                
                # проверяем наличие обоих признаков
                if 'using_namespace_std' in raw_line and 'std::' in raw_line:
                    violations.append(line_num)
                    
    except FileNotFoundError:
        print(f"X Ошибка: Файл '{filename}' не найден!")
        return []
    except Exception as e:
        print(f"X Ошибка при чтении файла: {e}")
        return []
    
    return violations

def fix_std_violations(input_filename: str, output_filename: str) -> int:
    """
    Исправляет нарушения стиля, удаляя префикс `std::` в строках с `using_namespace_std`.
    
    Args:
        input_filename:     Исходный файл
        output_filename:    Файл для сохранения исправленной версии
        
    Returns:
        int:    Количество исправленных строк
    
    **Важно:** Строки из списка EXCLUDED_LINES не исправляются!
    
    **Пример:**
    ```cpp
    // Было:
    std::cout << "Hello" << std::endl;
    
    // Стало:
    cout << "Hello" << endl;
    """
    fixes_applied = 0
    skipped_excluded = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as fin, \
            open(output_filename, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                # проверяем, нужно ли исправлять эту строку
                if 'using_namespace_std' in line and 'std::' in line:
                    # пропускаем исключения
                    if is_excluded_line(line_num):
                        print(f"Строка {line_num} пропущена (исключение)")
                        skipped_excluded += 1
                    else:
                        # применяем все замены из словаря
                        for std_pattern, replacement in STD_REPLACEMENTS.items():
                            line = line.replace(std_pattern, replacement)
                        fixes_applied += 1
                        print(f"Строка {line_num} исправлена")
                
                fout.write(line)

        print(f"\nСтатистика:")
        print(f" - исправлено строк: {fixes_applied}")
        print(f" - пропущено (исключения): {skipped_excluded}")
        print(f" - сохранено в: {output_filename}")

        return fixes_applied

    except FileNotFoundError:
        print(f"X Ошибка: Файл '{input_filename}' не найден!")
        return 0
    except Exception as e:
        print(f"X Ошибка при обработке файла: {e}")
        return 0

def analyze_style_distribution(filename: str) -> None:
    """
    Анализирует распределение стилей в датасете.
    
    Args:
        filename:    Путь к файлу датасета
    
    **Выводит:**
    - Количество строк с using_namespace_std
    - Количество строк с explicit_std
    - Процентное соотношение
    """
    using_std_count = 0
    explicit_std_count = 0
    total_valid = 0

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # пропускаем заголовок
                if line_num == 1 and 'description' in line:
                    continue
                
                if 'using_namespace_std' in line:
                    using_std_count += 1
                    total_valid += 1
                elif 'explicit_std' in line:
                    explicit_std_count += 1
                    total_valid += 1

        if total_valid > 0:
            print_header("РАСПРЕДЕЛЕНИЕ СТИЛЕЙ")
            print(f" - using_namespace_std: {using_std_count} ({using_std_count/total_valid*100:.1f}%)")
            print(f" - explicit_std: {explicit_std_count} ({explicit_std_count/total_valid*100:.1f}%)")
            print(f" - всего примеров: {total_valid}")

    except Exception as e:
        print(f"X Ошибка при анализе: {e}")


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================

def main() -> int:
    """
    Основная функция.
    
    Returns:
        int:    0 при успехе, 1 при ошибке
    """
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    
    print_header("ПРОВЕРКА СТИЛЕЙ STD")
    print(f"Анализируемый файл: {filename}")
    
    # анализируем распределение стилей
    analyze_style_distribution(filename)
    
    # ищем нарушения
    violations = find_std_in_using_namespace_std(filename)
    
    if violations:
        print_header("!!! НАЙДЕНЫ НАРУШЕНИЯ")
        print(f"Всего нарушений: {len(violations)}")
        print(f"\nНомера строк с using_namespace_std где есть std:::")
        
        # выводим номера строк группами для удобства
        violations_str = ", ".join(map(str, violations))
        print(f"  {violations_str}")
        
        print(f"\nИсключения (не исправлять):")
        excluded_str = ", ".join(map(str, EXCLUDED_LINES))
        print(f"  {excluded_str}")
        
        # предлагаем исправление
        print(f"\nДля исправления нарушений выполните:")
        print(f"fix_std_violations('{filename}', 'fixed_dataset.csv')")
        
        return 1
    else:
        print_header("НАРУШЕНИЙ НЕ НАЙДЕНО!")
        print("Все строки с using_namespace_std корректны")
        return 0


if __name__ == "__main__":
    sys.exit(main())


# ======================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ ДЛЯ ИСПРАВЛЕНИЯ
# ======================================================================

    """
    Раскомментируйте для исправления файла:

    fix_std_violations(
        '3_my_cpp_nn_project/check_dataset/1_cpp_code_generation_dataset.csv',
        '3_my_cpp_nn_project/check_dataset/1_cpp_code_generation_dataset_fixed.csv'
    )
    """

# Примечание: строки 9031, 9474, 10307, 10313, 10349, 10370, 10632, 11076, 11264, 11933, 12610, 13025
# Строки содержат смешанный код и не должны корректироваться автоматически.