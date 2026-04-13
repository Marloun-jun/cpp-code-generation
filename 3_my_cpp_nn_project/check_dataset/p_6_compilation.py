#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# p_6_compilation.py - Продвинутое тестирование компиляции C++ кода
# ======================================================================
#
# @file p_6_compilation.py
# @brief Комплексное тестирование компиляции C++ кода из датасета
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Выполняет продвинутое тестирование компиляции C++ кода из датасета.
#          В отличие от предыдущих версий, этот тестер ТОЛЬКО ПРОВЕРЯЕТ код,
#          не пытаясь его исправить, что даёт реальную оценку качества.
#
#          **Основные возможности:**
#          - **Загрузка и парсинг датасета** - Надёжный пошаговый парсер CSV
#          - **Статистика датасета**         - Анализ тем, стилей, длины кода
#          - **Компиляция кода**             - Проверка на реальном компиляторе (g++/clang++)
#          - **Параллельное тестирование**   - Многопоточная компиляция
#          - **Классификация ошибок**        - Анализ типов ошибок компиляции
#          - **Детальные отчеты**            - Сохранение результатов в JSON
#          - **Итоговая оценка**             - Качество кода в процентах
#
#          **Режимы тестирования:**
#          - Быстрый тест (50 примеров)
#          - Средний тест (500 примеров)
#          - Полный тест (все примеры)
#          - Выборочный тест по теме
#          - Тест конкретных строк
#
# @usage python p_6_compilation.py
#
# @example
#   python p_6_compilation.py
#   # Интерактивное меню позволит выбрать режим тестирования
#
# ======================================================================

import os
import random
import subprocess
import tempfile
import time
import datetime
import json
import concurrent.futures
import hashlib

from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


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


def clean_input(prompt: str) -> str:
    """
    Очистка ввода пользователя от непечатаемых символов.
    
    Args:
        prompt: Приглашение для ввода
        
    Returns:
        str: Очищенная строка ввода
    """
    try:
        user_input = input(prompt).strip()
        # Убираем непечатаемые символы
        user_input = ''.join(char for char in user_input if char.isprintable())
        return user_input
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception as e:
        print(f"Ошибка ввода: {str(e)}!")
        return ""

# ======================================================================
# КЛАССЫ ДАННЫХ
# ======================================================================

@dataclass
class CompilationResult:
    """
    Результат компиляции одного примера.
    
    Attributes:
        success:       Успешность компиляции
        line_number:   Номер строки в исходном файле
        topic:         Тема примера
        style:         Стиль (using_namespace_std/explicit_std)
        description:   Описание программы
        error_message: Сообщение об ошибке (если есть)
        warnings:      Предупреждения компилятора
        compile_time:  Время компиляции в секундах
        code_hash:     Хеш кода для отслеживания дубликатов
        code_snippet:  Короткий сниппет кода для отладки
    """
    success: bool
    line_number: int
    topic: str
    style: str
    description: str
    error_message: str = ""
    warnings: str = ""
    compile_time: float = 0.0
    code_hash: str = ""
    code_snippet: str = ""

# ======================================================================
# ЗАГРУЗЧИК ДАТАСЕТА
# ======================================================================

class DatasetLoader:
    """
    Загрузчик и анализатор датасета C++ кода.
    
    Выполняет:
    - Пошаговый парсинг CSV с учётом экранирования
    - Валидацию структуры кода
    - Определение стиля (using/explicit)
    - Сбор статистики по темам, стилям, длине кода
    - Обнаружение дубликатов
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Инициализация загрузчика датасета.
        
        Args:
            debug_mode:    Включить отладочный вывод
        """
        self.dataset: List[Dict] = []
        self.topics_styles = defaultdict(lambda: {'using': [], 'explicit': []})
        self.stats = {
            'total_lines': 0,
            'loaded_examples': 0,
            'skipped_lines': 0,
            'parsing_errors': 0,
            'field_errors': 0,
            'style_errors': 0,
            'malformed_code': 0,
            'unknown_styles': defaultdict(int)
        }
        self.debug_mode = debug_mode
    
    def parse_csv_step_by_step(self, line: str) -> Optional[List[str]]:
        """
        Пошаговый парсинг CSV строки с учётом экранирования.
        
        Args:
            line: Строка CSV для парсинга
            
        Returns:
            Optional[List[str]]: Список полей или None при ошибке
        
        **Алгоритм:**
        1. Проверка базовой структуры (кавычки в начале и конце)
        2. Поиск первого разделителя '","'
        3. Извлечение поля description
        4. Поиск всех позиций '","' в оставшейся строке
        5. Извлечение полей с конца (code, style, topic, keywords)
        6. Очистка кавычек
        """
        if not line or not line.strip():
            return None
        
        line = line.strip()
        
        # Убираем BOM если есть (BOM - спецсимвол из стандарта Unicode, который добавляется в начале файла)
        if line.startswith('\ufeff'):
            line = line[1:]
        
        # Проверяем, что строка начинается и заканчивается кавычками
        if not line.startswith('"') or not line.endswith('"'):
            if self.debug_mode:
                print("Строка не начинается/заканчивается кавычками!")
            return None
        
        # Убираем внешние кавычки
        line = line[1:-1]
        fields = []
        
        try:
            # Шаг 1: Извлекаем поле 1 (description)
            # Ищем первую '","' в строке
            first_sep = line.find('","')
            if first_sep == -1:
                if self.debug_mode:
                    print("Не найден первый разделитель '","'!")
                return None
            
            field1 = line[:first_sep+1]    # Включаем закрывающую кавычку
            line = line[first_sep+3:]      # Убираем '","' и поле 1
            
            # Убираем кавычки с поля 1
            if field1.startswith('"') and field1.endswith('"'):
                field1 = field1[1:-1]
            fields.append(field1)
            
            # Теперь у нас строка начинается с поля 2 (code)
            # Нужно найти поле 3, 4, 5 с конца
            # Находим все позиции '","' в оставшейся строке
            positions = []
            pos = 0
            while True:
                idx = line.find('","', pos)
                if idx == -1:
                    break
                positions.append(idx)
                pos = idx + 3
            
            # Должно быть минимум 3 разделителя (между 2-3, 3-4, 4-5)
            if len(positions) < 3:
                if self.debug_mode:
                    print(f"Недостаточно полей (найдено {len(positions)} разделителей)!")
                return None
            
            # Разделяем поля с конца:
            # Поле 5: после последнего '","'
            last_sep = positions[-1]
            field5 = line[last_sep+3:]    # После '","'
            line = line[:last_sep+1]      # До закрывающей кавычки поля 4
            
            # Поле 4: после предпоследнего '","'
            second_last_sep = positions[-2]
            field4 = line[second_last_sep+3:]    # После '","'
            line = line[:second_last_sep+1]      # До закрывающей кавычки поля 3
            
            # Поле 3: после третьего с конца '","'
            third_last_sep = positions[-3]
            field3 = line[third_last_sep+3:]    # После '","'
            field2 = line[:third_last_sep+1]    # До закрывающей кавычки поля 2
            
            # Очищаем кавычки у полей
            def clean_field(field):
                if field.startswith('"') and field.endswith('"'):
                    return field[1:-1]
                return field
            
            fields.append(clean_field(field2))    # Поле 2: code
            fields.append(clean_field(field3))    # Поле 3: style
            fields.append(clean_field(field4))    # Поле 4: topic
            fields.append(clean_field(field5))    # Поле 5: keywords
            
            if self.debug_mode and len(fields) == 5:
                print(f"Успешно распарсено 5 полей:")
                print(f"- Поле 1 (desc):        {fields[0][:50]}...")
                print(f"- Поле 2 (code):        {fields[1][:50]}...")
                print(f"- Поле 3 (style):       {fields[2]}")
                print(f"- Поле 4 (topic):       {fields[3]}")
                print(f"- Поле 5 (keywords):    {fields[4][:50]}...")
            
            return fields
            
        except Exception as e:
            if self.debug_mode:
                print(f"Ошибка пошагового парсинга: {str(e)}!")
                import traceback
                traceback.print_exc()
            return None

    def validate_code_structure(self, code: str) -> bool:
        """
        Базовая проверка структуры кода.
        
        Args:
            code: Код для проверки
            
        Returns:
            bool: True если код имеет корректную структуру
        
        **Проверяет:**
        - Код не слишком короткий (<10 символов)
        - Не слишком много строк (>500)
        - Отсутствие бинарных данных
        """
        if not code or len(code) < 10:
            return False
        
        # Проверяем на явно мусорные данные
        if code.count('\n') > 500:    # Слишком много строк
            return False
        
        # Проверяем на бинарные данные
        if any(ord(c) < 32 and c not in '\n\r\t' for c in code[:1000]):
            return False
        
        return True
    
    def determine_style_type(self, style: str, line_num: int) -> Optional[str]:
        """
        Определение типа стиля (using/explicit).
        
        Args:
            style:    Строка со стилем
            line_num: Номер строки (для отладки)
            
        Returns:
            Optional[str]: 'using', 'explicit' или None если стиль не определён
        """
        style = style.strip()
        
        # Проверяем, что это не фрагмент кода
        if len(style) > 100 or any(x in style for x in ['<<', '>>', 'cout', 'printf', ';', '{', '}']):
            if self.debug_mode and line_num <= 20:
                print(f"Строка {line_num}: подозрительный стиль (возможно код): {style[:50]}...")
            return None
        
        style_lower = style.lower()
        
        # Проверяем using стиль
        if 'using_namespace_std' in style_lower or 'using namespace std' in style_lower:
            return 'using'
        
        # Проверяем explicit стиль
        if 'explicit_std' in style_lower or 'explicit std' in style_lower:
            return 'explicit'
        
        # Неизвестный стиль
        if line_num <= 20:  # Логируем только первые 20 ошибок
            print(f"Строка {line_num}: неизвестный стиль '{style[:50]}...'")
        
        return None
    
    def load_dataset(self, filename: str, max_lines: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """
        Загрузка и анализ датасета.
        
        Args:
            filename:  Путь к файлу датасета
            max_lines: Ограничение на количество загружаемых строк
            
        Returns:
            Tuple[List[Dict], Dict]:    Загруженный датасет и статистика по темам/стилям
        """
        print_header(f"ЗАГРУЗКА ДАТАСЕТА")
        print(f"Файл:          {filename}")
        print(f"Режим отладки: {'ВКЛ' if self.debug_mode else 'ВЫКЛ'}")
        
        try:
            filepath = Path(filename)
            if not filepath.exists():
                print(f"Файл не найден: {filename}!")
                return [], defaultdict(lambda: {'using': [], 'explicit': []})
            
            # Определяем кодировку
            encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'latin-1']
            content = None
            for encoding in encodings:
                try:
                    with open(filename, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"Кодировка: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print("Не удалось определить кодировку файла!")
                return [], defaultdict(lambda: {'using': [], 'explicit': []})
            
            lines = content.splitlines()
            self.stats['total_lines'] = len(lines)
            
            if max_lines and max_lines < len(lines):
                lines = lines[:max_lines]
                print(f"Ограничение: загружаем первые {max_lines} строк")
            
            if not lines:
                print("Файл пуст!")
                return [], defaultdict(lambda: {'using': [], 'explicit': []})
            
            # Определяем заголовки
            headers = self.parse_csv_step_by_step(lines[0])
            if headers:
                print(f"Заголовки: {headers}")
                expected_headers = ['description', 'code', 'style', 'topic', 'keywords']
                if len(headers) >= len(expected_headers):
                    print("Структура заголовков корректна!")
                else:
                    print(f"Неожиданное количество столбцов: {len(headers)}")
            
            # Пропускаем заголовки
            lines = lines[1:]
            print(f"Строк для обработки: {len(lines)}")
            
            # Обработка строк
            print("Обработка строк...")
            start_time = time.time()
            
            for line_num, line in enumerate(lines, 2):
                self._process_line(line, line_num)
                
                # Прогресс
                if line_num % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = line_num / elapsed if elapsed > 0 else 0
                    print(f"Обработано: {line_num}/{len(lines)} строк ({rate:.1f} строк/с)...")
            
            elapsed = time.time() - start_time
            print(f"Обработка завершена за {elapsed:.1f} секунд")
            
            self._print_stats()
            return self.dataset, self.topics_styles
            
        except Exception as e:
            print(f"Критическая ошибка загрузки: {str(e)}!")
            import traceback
            traceback.print_exc()
            return [], defaultdict(lambda: {'using': [], 'explicit': []})
    
    def _process_line(self, line: str, line_num: int) -> None:
        """
        Обработка одной строки датасета.
        
        Args:
            line:     Строка для обработки
            line_num: Номер строки
        """
        line = line.strip()
        
        # Пропускаем пустые строки и комментарии
        if not line:
            self.stats['skipped_lines'] += 1
            return
        
        if line.startswith('#'):
            self.stats['skipped_lines'] += 1
            return
        
        parts = self.parse_csv_step_by_step(line)
        if not parts:
            self.stats['parsing_errors'] += 1
            if line_num <= 20 and self.debug_mode:
                print(f"Строка {line_num}: ошибка парсинга CSV!")
            return
        
        if len(parts) < 4:    # Нужно минимум 4 поля
            self.stats['field_errors'] += 1
            if line_num <= 20 and self.debug_mode:
                print(f"Строка {line_num}: недостаточно полей ({len(parts)} из 4)")
            return
        
        try:
            description = parts[0].strip()
            code = parts[1].strip()
            style = parts[2].strip()
            topic = parts[3].strip()
            keywords = parts[4].strip() if len(parts) > 4 else ""
            
            # Проверяем структуру кода
            if not self.validate_code_structure(code):
                self.stats['malformed_code'] += 1
                if line_num <= 20 and self.debug_mode:
                    print(f"Строка {line_num}: неправильная структура кода!")
                return
            
            # Определяем тип стиля
            style_type = self.determine_style_type(style, line_num)
            if not style_type:
                self.stats['style_errors'] += 1
                self.stats['unknown_styles'][style[:50]] += 1
                return
            
            # Создаем хеш кода для отслеживания дубликатов
            code_hash = hashlib.md5(code.encode()).hexdigest()[:16]
            
            # Создаем короткий сниппет кода для отладки
            code_lines = code.split('\n')
            if len(code_lines) > 0:
                first_line = code_lines[0][:100]
                code_snippet = first_line + ('...' if len(first_line) == 100 else '')
                if len(code_lines) > 1:
                    code_snippet += f" [+{len(code_lines)-1} строк]"
            else:
                code_snippet = "Пустой код!"
            
            example = {
                'line': line_num,
                'description': description,
                'code': code,    # Код как есть, без изменений
                'style': style,
                'topic': topic,
                'keywords': keywords,
                'style_type': style_type,
                'code_hash': code_hash,
                'code_snippet': code_snippet,
                'code_length': len(code),
                'lines_in_code': code.count('\n') + 1
            }
            
            self.dataset.append(example)
            self.topics_styles[topic][style_type].append(example)
            self.stats['loaded_examples'] += 1
            
        except Exception as e:
            print(f"Ошибка обработки строки {line_num}: {str(e)}!")
    
    def _print_stats(self) -> None:
        """Вывод статистики загрузки датасета."""
        print_subheader("СТАТИСТИКА ЗАГРУЗКИ ДАТАСЕТА:")
        
        print(f"- Всего строк в файле: {self.stats['total_lines']}")
        print(f"- Успешно загружено:   {self.stats['loaded_examples']}")
        
        if self.stats['skipped_lines'] > 0:
            print(f"- Пропущено (пустые/комментарии): {self.stats['skipped_lines']}")
        
        if self.stats['parsing_errors'] > 0:
            print(f"- Ошибок парсинга CSV: {self.stats['parsing_errors']}")
        
        if self.stats['field_errors'] > 0:
            print(f"- Недостаточно полей:  {self.stats['field_errors']}")
        
        if self.stats['style_errors'] > 0:
            print(f"- Неизвестных стилей:  {self.stats['style_errors']}")
            
            # Показываем топ-10 неизвестных стилей
            if self.stats['unknown_styles']:
                print(f"\nТОП-10 НЕИЗВЕСТНЫХ СТИЛЕЙ:")
                for style, count in sorted(self.stats['unknown_styles'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]:
                    print(f"- '{style}': {count} раз")
        
        if self.stats['malformed_code'] > 0:
            print(f"- Неправильная структура кода: {self.stats['malformed_code']}")
        
        if self.dataset:
            # Статистика по стилям
            using_count = sum(len(self.topics_styles[t]['using']) for t in self.topics_styles)
            explicit_count = sum(len(self.topics_styles[t]['explicit']) for t in self.topics_styles)
            
            print(f"\nСТАТИСТИКА ПО СТИЛЯМ:")
            print(f"- using namespace std: {using_count} ({using_count/len(self.dataset)*100:.1f}%)")
            print(f"- explicit std:        {explicit_count} ({explicit_count/len(self.dataset)*100:.1f}%)")
            
            # Статистика по темам
            topic_counts = Counter([ex['topic'] for ex in self.dataset])
            print(f"\nСТАТИСТИКА ПО ТЕМАМ:")
            print(f"- Уникальных тем: {len(topic_counts)}")
            print(f"- Топ-5 тем по количеству примеров:")
            
            for topic, count in topic_counts.most_common(5):
                percentage = count / len(self.dataset) * 100
                print(f"    {topic}: {count} примеров ({percentage:.1f}%)")
            
            # Статистика по длине кода
            code_lengths = [ex['code_length'] for ex in self.dataset]
            avg_length = sum(code_lengths) / len(code_lengths)
            max_length = max(code_lengths)
            min_length = min(code_lengths)
            
            print(f"\nСТАТИСТИКА ПО ДЛИНЕ КОДА:")
            print(f"- Средняя длина: {avg_length:.0f} символов")
            print(f"- Минимальная:   {min_length} символов")
            print(f"- Максимальная:  {max_length} символов")
            
            # Распределение по длине
            bins = [0, 100, 500, 1000, 2000, 5000, 10000, float('inf')]
            bin_labels = ['<100', '100-500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '>10000']
            length_dist = Counter()
            
            for length in code_lengths:
                for i, bin_max in enumerate(bins[1:]):
                    if length <= bin_max:
                        length_dist[bin_labels[i]] += 1
                        break
            
            print(f"\nРаспределение по длине:")
            for label in bin_labels:
                if label in length_dist:
                    count = length_dist[label]
                    percentage = count / len(code_lengths) * 100
                    print(f"- {label}: {count} ({percentage:.1f}%)")
            
            # Проверка дубликатов
            code_hashes = [ex['code_hash'] for ex in self.dataset]
            unique_hashes = set(code_hashes)
            duplicates = len(code_hashes) - len(unique_hashes)
            
            if duplicates > 0:
                print(f"\nВОЗМОЖНЫЕ ДУБЛИКАТЫ:")
                print(f"- Найдено: {duplicates} возможных дубликатов")
                print(f"- Уникальных примеров: {len(unique_hashes)}")
            
            print(f"\nДАТАСЕТ ГОТОВ К ТЕСТИРОВАНИЮ!")

# ======================================================================
# КОМПИЛЯТОР C++
# ======================================================================

class CppCompiler:
    """
    Класс для компиляции C++ кода.
    
    Поддерживает:
    - Автоматический поиск доступного компилятора (g++/clang++)
    - Разные версии компиляторов
    - Настройку стандарта C++
    - Таймаут компиляции
    - Очистку сообщений об ошибках
    
    **Важно:** Класс ТОЛЬКО проверяет код, не внося никаких исправлений.
    """
    
    COMPILER_VERSIONS = {
        'g++': ['g++', 'g++-13', 'g++-12', 'g++-11', 'g++-10'],
        'clang++': ['clang++', 'clang++-15', 'clang++-14', 'clang++-13']
    }
    
    def __init__(self, compiler_type: str = 'g++', std: str = 'c++17', timeout: int = 15):
        """
        Инициализация компилятора.
        
        Args:
            compiler_type: Тип компилятора ('g++' или 'clang++')
            std:           Стандарт C++ (c++11, c++14, c++17, c++20)
            timeout:       Таймаут компиляции в секундах
        """
        self.compiler_type = compiler_type
        self.std = std
        self.timeout = timeout
        self.compiler_path = self._find_compiler()
        self._print_compiler_info()
    
    def _find_compiler(self) -> Optional[str]:
        """
        Поиск доступного компилятора в системе.
        
        Returns:
            Optional[str]: Путь к компилятору или None если не найден
        """
        if self.compiler_type not in self.COMPILER_VERSIONS:
            print(f"Неизвестный тип компилятора: {self.compiler_type}!")
            return None
        
        for compiler in self.COMPILER_VERSIONS[self.compiler_type]:
            try:
                result = subprocess.run(
                    [compiler, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    return compiler
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None
    
    def _print_compiler_info(self) -> bool:
        """
        Вывод информации о найденном компиляторе.
        
        Returns:
            bool: True если компилятор найден
        """
        if not self.compiler_path:
            print(f"Компилятор {self.compiler_type} не найден!")
            print("Проверьте установку компилятора или используйте другой тип!")
            return False
        
        try:
            result = subprocess.run(
                [self.compiler_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version_line = result.stdout.split('\n')[0]
            print(f"\nКомпилятор: {self.compiler_path}")
            print(f"- Версия:       {version_line}")
            print(f"- Стандарт C++: {self.std}")
            print(f"- Таймаут:      {self.timeout} секунд")
            return True
        except Exception as e:
            print(f"Ошибка проверки компилятора: {str(e)}!")
            return False
    
    def compile_code(self, code: str, temp_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Компиляция кода (БЕЗ ИСПРАВЛЕНИЙ).
        
        Args:
            code:     Код для компиляции
            temp_dir: Временная директория (опционально)
            
        Returns:
            Dict[str, Any]: Результат компиляции
        """
        start_time = time.time()
        
        # Только базовая проверка, без исправлений
        if not code or len(code.strip()) < 10:
            return {
                'success': False,
                'error': 'Код слишком короткий или пустой!',
                'warnings': '',
                'compile_time': 0,
                'details': {'precheck_failed': True}
            }
        
        # Проверка на явно некорректный код
        if self._is_malformed_code(code):
            return {
                'success': False,
                'error': 'Некорректная структура кода!',
                'warnings': '',
                'compile_time': 0,
                'details': {'malformed_code': True}
            }
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cpp',
            dir=temp_dir,
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)    # Пишем код как есть
            temp_file = f.name
        
        result = self._run_compilation(temp_file, start_time)
        
        # Очистка временного файла
        try:
            os.unlink(temp_file)
        except:
            pass
        
        return result
    
    def _is_malformed_code(self, code: str) -> bool:
        """
        Проверка на явно некорректный код.
        
        Args:
            code: Код для проверки
            
        Returns:
            bool: True если код некорректен
        """
        # Проверяем на наличие нулевых байтов
        if '\x00' in code:
            return True
        
        # Проверяем на очень длинные строки без переносов
        lines = code.split('\n')
        for line in lines[:50]:    # Проверяем первые 50 строк
            if len(line) > 1000 and '{' not in line and '}' not in line:
                return True
        
        return False
    
    def _run_compilation(self, temp_file: str, start_time: float) -> Dict[str, Any]:
        """
        Запуск компиляции временного файла.
        
        Args:
            temp_file:  Путь к временному файлу
            start_time: Время начала для измерения
            
        Returns:
            Dict[str, Any]: Результат компиляции
        """
        try:
            # Простая компиляция без оптимизаций
            compile_cmd = [
                self.compiler_path,
                f'-std={self.std}',
                '-fsyntax-only',
                temp_file
            ]
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='replace'
            )
            
            compile_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'error': '',
                    'warnings': result.stderr[:500] if result.stderr else '',
                    'compile_time': compile_time,
                    'details': {
                        'has_warnings': bool(result.stderr)
                    }
                }
            else:
                error_msg = self._clean_error_message(result.stderr, temp_file)
                return {
                    'success': False,
                    'error': error_msg[:1000],
                    'warnings': '',
                    'compile_time': compile_time,
                    'details': {
                        'return_code': result.returncode
                    }
                }
                
        except subprocess.TimeoutExpired:
            compile_time = time.time() - start_time
            return {
                'success': False,
                'error': f'Таймаут компиляции ({self.timeout} секунд)!',
                'warnings': '',
                'compile_time': compile_time,
                'details': {'timeout': True}
            }
        except Exception as e:
            compile_time = time.time() - start_time
            return {
                'success': False,
                'error': f'Системная ошибка: {str(e)}!',
                'warnings': '',
                'compile_time': compile_time,
                'details': {'system_error': True}
            }
    
    def _clean_error_message(self, error: str, temp_file: str) -> str:
        """
        Очистка сообщения об ошибке от путей к временным файлам.
        
        Args:
            error:     Исходное сообщение об ошибке
            temp_file: Путь к временному файлу для замены
            
        Returns:
            str: Очищенное сообщение
        """
        if not error:
            return "Неизвестная ошибка компиляции!"
        
        # Убираем путь к временному файлу
        error = error.replace(temp_file, 'source.cpp')
        
        # Оставляем только первые 10 строк
        lines = error.split('\n')
        return '\n'.join(lines[:10])

# ======================================================================
# ТЕСТЕР КОМПИЛЯЦИИ
# ======================================================================

class CompilationTester:
    """
    Основной класс тестирования (ТОЛЬКО проверка компиляции).
    
    Выполняет:
    - Пакетную компиляцию примеров
    - Параллельную обработку
    - Сбор статистики
    - Классификацию ошибок
    - Сохранение отчетов
    """
    
    def __init__(self, dataset: List[Dict], compiler: CppCompiler):
        """
        Инициализация тестера.
        
        Args:
            dataset:  Загруженный датасет
            compiler: Экземпляр компилятора
        """
        self.dataset = dataset
        self.compiler = compiler
        self.results = []
    
    def run_test(self, test_name: str, sample_size: Optional[int] = None, 
                 max_workers: int = 4) -> Dict[str, Any]:
        """
        Запуск теста компиляции.
        
        Args:
            test_name:   Название теста
            sample_size: Размер выборки (если None - все примеры)
            max_workers: Максимальное количество потоков
            
        Returns:
            Dict[str, Any]: Результаты тестирования
        """
        print_header(f"ТЕСТ КОМПИЛЯЦИИ: {test_name}")
        
        # Подготовка выборки
        if sample_size and sample_size < len(self.dataset):
            test_samples = random.sample(self.dataset, sample_size)
            print(f"Выборка: {sample_size} случайных примеров из {len(self.dataset)}")
        else:
            test_samples = self.dataset
            print(f"Тестирование всех {len(self.dataset)} примеров")
        
        # Статистика выборки
        self._print_sample_stats(test_samples)
        
        print(f"\nНАСТРОЙКИ ТЕСТА:")
        print(f"- Компилятор: {self.compiler.compiler_path}")
        print(f"- Потоков:    {max_workers}")
        print(f"- Таймаут:    {self.compiler.timeout} секунд")
        print(f"- Режим:      ТОЛЬКО ПРОВЕРКА (без исправлений)")
        
        # Запуск тестирования
        start_time = time.time()
        self.results = self._run_compilation_batch(test_samples, max_workers)
        total_time = time.time() - start_time
        
        # Анализ результатов
        return self._analyze_results(test_name, test_samples, total_time)
    
    def _print_sample_stats(self, samples: List[Dict]) -> None:
        """
        Вывод статистики выборки.
        
        Args:
            samples: Выборка примеров
        """
        topics = Counter([ex['topic'] for ex in samples])
        styles = Counter([ex['style_type'] for ex in samples])
        
        print(f"\nСТАТИСТИКА ВЫБОРКИ:")
        print(f"- Уникальных тем:     {len(topics)}")
        print(f"- Стиль 'using':      {styles.get('using', 0)} ({styles.get('using', 0)/len(samples)*100:.1f}%)")
        print(f"- Стиль 'explicit':   {styles.get('explicit', 0)} ({styles.get('explicit', 0)/len(samples)*100:.1f}%)")
        
        # Статистика по длине кода
        lengths = [ex['code_length'] for ex in samples]
        print(f"- Средняя длина кода: {sum(lengths)/len(lengths):.0f} символов")
        print(f"- Минимальная длина:  {min(lengths)} символов")
        print(f"- Максимальная длина: {max(lengths)} символов")
    
    def _run_compilation_batch(self, samples: List[Dict], max_workers: int) -> List[CompilationResult]:
        """
        Запуск пакетной компиляции.
        
        Args:
            samples:     Список примеров для компиляции
            max_workers: Максимальное количество потоков
            
        Returns:
            List[CompilationResult]: Результаты компиляции
        """
        results = []
        
        if len(samples) > 100 and max_workers > 1:
            results = self._run_parallel(samples, max_workers)
        else:
            results = self._run_sequential(samples)
        
        return results
    
    def _run_parallel(self, samples: List[Dict], max_workers: int) -> List[CompilationResult]:
        """
        Параллельное выполнение тестов.
        
        Args:
            samples:     Список примеров
            max_workers: Количество потоков
            
        Returns:
            List[CompilationResult]: Результаты компиляции
        """
        print(f"Запуск параллельной компиляции ({max_workers} потоков)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_example = {
                executor.submit(self._compile_single, example): example
                for example in samples
            }
            
            results = []
            completed = 0
            total = len(samples)
            
            for future in concurrent.futures.as_completed(future_to_example):
                example = future_to_example[future]
                
                try:
                    result = future.result(timeout=self.compiler.timeout + 5)
                    results.append(result)
                except concurrent.futures.TimeoutError:
                    results.append(CompilationResult(
                        success=False,
                        line_number=example['line'],
                        topic=example['topic'],
                        style=example['style'],
                        description=example['description'],
                        error_message='Таймаут выполнения теста',
                        code_snippet=example['code_snippet']
                    ))
                except Exception as e:
                    results.append(CompilationResult(
                        success=False,
                        line_number=example['line'],
                        topic=example['topic'],
                        style=example['style'],
                        description=example['description'],
                        error_message=f'X Ошибка тестирования: {str(e)}',
                        code_snippet=example['code_snippet']
                    ))
                
                completed += 1
                if completed % max(1, total // 20) == 0 or completed == total:
                    percentage = completed / total * 100
                    print(f"\rПрогресс: {completed}/{total} ({percentage:.1f}%)...", end='')
            
            print()    # Новая строка после прогресс-бара
            return results
    
    def _run_sequential(self, samples: List[Dict]) -> List[CompilationResult]:
        """
        Последовательное выполнение тестов.
        
        Args:
            samples: Список примеров
            
        Returns:
            List[CompilationResult]: Результаты компиляции
        """
        print("Запуск последовательной компиляции...")
        
        results = []
        total = len(samples)
        
        for i, example in enumerate(samples, 1):
            result = self._compile_single(example)
            results.append(result)
            
            if i % max(1, total // 20) == 0 or i == total:
                percentage = i / total * 100
                print(f"\rПрогресс: {i}/{total} ({percentage:.1f}%)...", end='')
        
        print()    # Новая строка после прогресс-бара
        return results
    
    def _compile_single(self, example: Dict) -> CompilationResult:
        """
        Компиляция одного примера (БЕЗ ИСПРАВЛЕНИЙ).
        
        Args:
            example: Пример для компиляции
            
        Returns:
            CompilationResult: Результат компиляции
        """
        compile_result = self.compiler.compile_code(example['code'])
        
        return CompilationResult(
            success=compile_result['success'],
            line_number=example['line'],
            topic=example['topic'],
            style=example['style'],
            description=example['description'],
            error_message=compile_result['error'],
            warnings=compile_result['warnings'],
            compile_time=compile_result['compile_time'],
            code_hash=example['code_hash'],
            code_snippet=example['code_snippet']
        )
    
    def _analyze_results(self, test_name: str, samples: List[Dict], total_time: float) -> Dict[str, Any]:
        """
        Анализ результатов тестирования.
        
        Args:
            test_name:  Название теста
            samples:    Тестируемые примеры
            total_time: Общее время выполнения
            
        Returns:
            Dict[str, Any]: Детальные результаты
        """
        print(f"\nВРЕМЯ ВЫПОЛНЕНИЯ: {total_time:.1f} секунд")
        print(f"Среднее время на пример: {total_time/len(samples):.3f} секунд")
        
        # Базовая статистика
        successful = sum(1 for r in self.results if r.success)
        failed = len(self.results) - successful
        success_rate = successful / len(self.results) * 100
        
        print_subheader("ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"- Всего примеров:         {len(self.results)}")
        print(f"- Успешно скомпилированы: {successful} ({success_rate:.2f}%)")
        print(f"- Ошибок компиляции:      {failed}")
        
        if failed > 0:
            self._analyze_errors()
        
        # Детальная статистика
        details = self._collect_detailed_stats()
        
        # Сохранение отчета
        report_path = self._save_report(test_name, samples, total_time, details)
        
        return {
            'success_rate': success_rate,
            'successful': successful,
            'failed': failed,
            'total': len(self.results),
            'total_time': total_time,
            'report_path': report_path,
            'details': details
        }
    
    def _analyze_errors(self) -> None:
        """
        Детальный анализ ошибок компиляции.
        """
        errors = [r for r in self.results if not r.success]
        
        print(f"\nАНАЛИЗ ОШИБОК КОМПИЛЯЦИИ ({len(errors)}):")
        
        # Классификация ошибок
        error_categories = Counter()
        error_examples = []
        
        for error in errors:
            error_msg = error.error_message.lower()
            category = self._categorize_error(error_msg)
            error_categories[category] += 1
            
            # Сохраняем примеры для каждой категории
            if len(error_examples) < 5:
                error_examples.append({
                    'line': error.line_number,
                    'topic': error.topic,
                    'category': category,
                    'message': error.error_message[:150] + '...' if len(error.error_message) > 150 else error.error_message
                })
        
        print(f"\nРАСПРЕДЕЛЕНИЕ ОШИБОК:")
        for category, count in error_categories.most_common():
            percentage = count / len(errors) * 100
            print(f"- {category}: {count} ({percentage:.1f}%)")
        
        # ошибки по темам
        topic_errors = Counter([e.topic for e in errors])
        if topic_errors:
            print(f"\nОШИБКИ ПО ТЕМАМ (ТОП-5):")
            for topic, count in topic_errors.most_common(5):
                percentage = count / len(errors) * 100
                print(f"- {topic}: {count} ({percentage:.1f}%)")
        
        # примеры ошибок
        print(f"\nПРИМЕРЫ ОШИБОК:")
        for i, example in enumerate(error_examples, 1):
            print(f"\n{i}. Строка {example['line']} - {example['topic']}")
            print(f"    Категория: {example['category']}")
            print(f"    Ошибка:    {example['message']}")
    
    def _categorize_error(self, error_msg: str) -> str:
        """
        Классификация ошибок по типу.
        
        Args:
            error_msg: Сообщение об ошибке
            
        Returns:
            str: Категория ошибки
        """
        error_msg = error_msg.lower()
        
        if 'expected' in error_msg:
            return 'syntax_error'
        elif 'undefined reference' in error_msg:
            return 'undefined_reference'
        elif 'not declared' in error_msg:
            return 'undeclared_identifier'
        elif 'invalid' in error_msg and 'character' in error_msg:
            return 'invalid_character'
        elif 'missing' in error_msg:
            return 'missing_symbol'
        elif 'timeout' in error_msg:
            return 'timeout'
        elif 'error:' in error_msg:
            return 'compiler_error'
        elif 'warning:' in error_msg:
            return 'warning_treated_as_error'
        else:
            return 'other_error'
    
    def _collect_detailed_stats(self) -> Dict[str, Any]:
        """
        Сбор детальной статистики.
        
        Returns:
            Dict[str, Any]: Детальная статистика
        """
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Время компиляции
        compile_times = [r.compile_time for r in self.results if r.compile_time > 0]
        
        # Статистика по стилям
        style_stats = defaultdict(lambda: {'total': 0, 'success': 0})
        for result in self.results:
            example = next((ex for ex in self.dataset if ex['line'] == result.line_number), None)
            if example:
                style = example['style_type']
                style_stats[style]['total'] += 1
                if result.success:
                    style_stats[style]['success'] += 1
        
        details = {
            'compile_time': {
                'average': sum(compile_times) / len(compile_times) if compile_times else 0,
                'max': max(compile_times) if compile_times else 0,
                'min': min(compile_times) if compile_times else 0,
                'total': sum(compile_times) if compile_times else 0
            },
            'styles': {
                style: {
                    'total': stats['total'],
                    'success': stats['success'],
                    'rate': stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
                }
                for style, stats in style_stats.items()
            },
            'error_count': len(failed),
            'success_count': len(successful)
        }
        
        return details
    
    def _save_report(self, test_name: str, samples: List[Dict], 
                    total_time: float, details: Dict) -> str:
        """
        Сохранение отчета в JSON.
        
        Args:
            test_name:  Название теста
            samples:    Тестируемые примеры
            total_time: Общее время выполнения
            details:    Детальная статистика
            
        Returns:
            str: Путь к сохраненному отчету
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path("3_my_cpp_nn_project/check_dataset/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Очищаем имя файла от недопустимых символов
        safe_test_name = "".join(c for c in test_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_test_name = safe_test_name.replace(' ', '_')
        
        report_file = report_dir / f"compilation_report_{safe_test_name}_{timestamp}.json"
        
        try:
            report_data = {
                'test_name': test_name,
                'timestamp': datetime.datetime.now().isoformat(),
                'summary': {
                    'total_examples': len(samples),
                    'successful': sum(1 for r in self.results if r.success),
                    'failed': sum(1 for r in self.results if not r.success),
                    'success_rate': sum(1 for r in self.results if r.success) / len(self.results) * 100,
                    'total_time': total_time,
                    'average_time_per_example': total_time / len(samples)
                },
                'compiler_info': {
                    'path': self.compiler.compiler_path,
                    'type': self.compiler.compiler_type,
                    'std': self.compiler.std,
                    'timeout': self.compiler.timeout
                },
                'sample_info': {
                    'size': len(samples),
                    'topics_count': len(set(ex['topic'] for ex in samples)),
                    'using_style_count': sum(1 for ex in samples if ex['style_type'] == 'using'),
                    'explicit_style_count': sum(1 for ex in samples if ex['style_type'] == 'explicit')
                },
                'details': details,
                'errors': [
                    {
                        'line_number': r.line_number,
                        'topic': r.topic,
                        'style': r.style,
                        'description': r.description[:100] + '...' if len(r.description) > 100 else r.description,
                        'error_message': r.error_message[:200] + '...' if len(r.error_message) > 200 else r.error_message,
                        'code_snippet': r.code_snippet
                    }
                    for r in self.results if not r.success
                ]
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nПОЛНЫЙ ОТЧЕТ СОХРАНЕН:")
            print(f"{report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"\nНе удалось сохранить отчет: {str(e)}!")
            return ""

# ======================================================================
# ФУНКЦИЯ ИТОГОВОЙ ОЦЕНКИ
# ======================================================================

def print_final_evaluation(success_rate: float, total_examples: int, failed_count: int) -> None:
    """
    Вывод итоговой оценки качества кода.
    
    Args:
        success_rate:   Процент успешной компиляции
        total_examples: Общее количество примеров
        failed_count:   Количество ошибок
    """
    print_header("ИТОГОВАЯ ОЦЕНКА КАЧЕСТВА КОДА:")
    
    if success_rate >= 99.5:
        print("ВЫДАЮЩЕЕСЯ КАЧЕСТВО!")
        print(f"{success_rate:.2f}% успешной компиляции")
        print("Код генерируется практически идеально")
    elif success_rate >= 98:
        print("ОТЛИЧНОЕ КАЧЕСТВО!")
        print(f"{success_rate:.2f}% успешной компиляции")
        print("Осталось исправить совсем немного примеров")
    elif success_rate >= 95:
        print("ХОРОШЕЕ КАЧЕСТВО")
        print(f"{success_rate:.2f}% успешной компиляции")
        print("Большая часть кода компилируется успешно")
    elif success_rate >= 90:
        print("УДОВЛЕТВОРИТЕЛЬНОЕ КАЧЕСТВО")
        print(f"{success_rate:.2f}% успешной компиляции")
        print("Есть над чем поработать")
    else:
        print("ТРЕБУЕТСЯ СЕРЬЕЗНАЯ ДОРАБОТКА")
        print(f"{success_rate:.2f}% успешной компиляции")
        print("Необходимо значительно улучшить генерацию кода")
    
    print(f"\nИТОГИ:")
    print(f"- Всего примеров: {total_examples}")
    print(f"- Успешно:        {total_examples - failed_count}")
    print(f"- Ошибок:         {failed_count}")
    
    if failed_count > 0:
        print(f"\nРЕКОМЕНДАЦИИ:")
        if success_rate >= 99:
            print("- Исправьте оставшиеся примеры вручную")
            print("- Проанализируйте отчет об ошибках")
        elif success_rate >= 95:
            print("- Сфокусируйтесь на исправлении ошибок компиляции")
            print("- Проанализируйте типичные ошибки")
            print("- Рассмотрите дообучение на проблемных примерах")
        else:
            print("- Проведите глубокий анализ ошибок")
            print("- Пересмотрите подход к генерации кода")
            print("- Рассмотрите добавление пост-обработки")
    
    print('=' * 60)


# ======================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> None:
    """Главная функция программы."""
    print_header("ПРОДВИНУТЫЙ ТЕСТЕР КОМПИЛЯЦИИ C++ КОДА ВЕРСИИ 3.3.0")  
    print("Особенности:")
    print("- Только проверка - без автоматических исправлений")
    print("- Детальная статистика и анализ")
    print("- Подробные отчеты в JSON")
    print("- Классификация ошибок")
    print("- Многопоточное тестирование")
    print("=" * 60)
    
    try:
        # Спрашиваем режим отладки
        debug_mode = False
        debug_input = clean_input("\nВключить режим отладки загрузки? (y/n): ").lower()
        if debug_input in ['y', 'д', 'yes', 'да']:
            debug_mode = True
            print("Режим отладки ВКЛЮЧЕН!")
        
        # Настройка компилятора
        print("\nНастройка компилятора...")
        compiler = CppCompiler(compiler_type='g++', std='c++17', timeout=15)
        
        if not compiler.compiler_path:
            print("Не удалось найти компилятор. Программа завершена.")
            return
        
        # Загрузка датасета
        filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
        loader = DatasetLoader(debug_mode=debug_mode)
        
        print("\nЗАГРУЗКА ДАТАСЕТА...")
        dataset, _ = loader.load_dataset(filename)
        
        if not dataset:
            print("Не удалось загрузить датасет!")
            return
        
        print(f"\nДАТАСЕТ ЗАГРУЖЕН:")
        print(f"- Примеров:       {len(dataset)}")
        print(f"- Уникальных тем: {len(set(ex['topic'] for ex in dataset))}")
        
        # Создание тестера
        tester = CompilationTester(dataset, compiler)
        
        # Меню тестирования
        while True:
            print_header("МЕНЮ ТЕСТИРОВАНИЯ:")
            print("1. Быстрый тест (50 случайных примеров)")
            print("2. Средний тест (500 примеров)")
            print("3. Полный тест (все примеры)")
            print("4. Выборочный тест по теме")
            print("5. Тест конкретных строк")
            print("D. Включить/выключить отладку")
            print("Q. Выход")
            print('=' * 60)
            
            try:
                choice = clean_input("\nВыберите вариант: ").lower()
                
                if choice == 'q' or choice == 'й':
                    print("\nВыход из программы...")
                    break
                
                elif choice == 'd':
                    debug_mode = not debug_mode
                    print(f"Режим отладки {'ВКЛЮЧЕН' if debug_mode else 'ВЫКЛЮЧЕН'}")
                    
                    # Перезагружаем датасет с новым режимом отладки
                    if debug_mode:
                        print("\nПерезагрузка датасета с отладкой...")
                        loader = DatasetLoader(debug_mode=True)
                        dataset, _ = loader.load_dataset(filename)
                        tester = CompilationTester(dataset, compiler)
                
                elif choice == '1':
                    print("\nЗАПУСК БЫСТРОГО ТЕСТА...")
                    result = tester.run_test("Быстрый тест (50 примеров)", sample_size=50, max_workers=4)
                    print_final_evaluation(result['success_rate'], result['total'], result['failed'])
                
                elif choice == '2':
                    print("\nЗАПУСК СРЕДНЕГО ТЕСТА...")
                    result = tester.run_test("Средний тест (500 примеров)", sample_size=500, max_workers=8)
                    print_final_evaluation(result['success_rate'], result['total'], result['failed'])
                
                elif choice == '3':
                    print("\nЗАПУСК ПОЛНОГО ТЕСТА...")
                    confirm = clean_input(f"Тестировать все {len(dataset)} примеров? Это может занять время. (y/n): ").lower()
                    if confirm in ['y', 'д']:
                        result = tester.run_test("Полный тест", max_workers=8)
                        print_final_evaluation(result['success_rate'], result['total'], result['failed'])
                    else:
                        print("Тест отменен")
                
                elif choice == '4':
                    # выбор темы
                    topics = sorted(set(ex['topic'] for ex in dataset))
                    print("\nВЫБОР ТЕМЫ:")
                    
                    for i, topic in enumerate(topics[:25], 1):
                        count = sum(1 for ex in dataset if ex['topic'] == topic)
                        print(f"{i:2}. {topic} ({count} примеров)")
                    
                    if len(topics) > 25:
                        print(f"... и еще {len(topics)-25} тем")
                    
                    try:
                        topic_input = clean_input("\nНомер темы (0 для отмены): ")
                        if not topic_input:
                            continue
                        
                        topic_num = int(topic_input)
                        if 1 <= topic_num <= len(topics):
                            selected_topic = topics[topic_num - 1]
                            topic_examples = [ex for ex in dataset if ex['topic'] == selected_topic]
                            
                            print(f"\nТЕСТ ТЕМЫ: {selected_topic}")
                            print(f"Примеров: {len(topic_examples)}")
                            
                            topic_tester = CompilationTester(topic_examples, compiler)
                            result = topic_tester.run_test(
                                f"Тест темы: {selected_topic}", 
                                max_workers=4
                            )
                            print_final_evaluation(result['success_rate'], result['total'], result['failed'])
                        
                        elif topic_num == 0:
                            continue
                        else:
                            print("Неверный номер темы!")
                    
                    except ValueError:
                        print("Неверный ввод. Введите число.")
                
                elif choice == '5':
                    print("\nТЕСТ КОНКРЕТНЫХ СТРОК")
                    lines_input = clean_input("Введите номера строк через запятую (например: 100,150,200): ")
                    
                    if not lines_input:
                        print("Не указаны номера строк!")
                        continue
                    
                    try:
                        line_numbers = [int(x.strip()) for x in lines_input.split(',') if x.strip()]
                        
                        if not line_numbers:
                            print("Не указаны номера строк!")
                            continue
                        
                        selected_examples = []
                        missing_lines = []
                        
                        for line_num in line_numbers:
                            example = next((ex for ex in dataset if ex['line'] == line_num), None)
                            if example:
                                selected_examples.append(example)
                            else:
                                missing_lines.append(line_num)
                        
                        if missing_lines:
                            print(f"Не найдены строки: {missing_lines}!")
                        
                        if selected_examples:
                            print(f"\nТЕСТ {len(selected_examples)} ВЫБРАННЫХ СТРОК")
                            line_tester = CompilationTester(selected_examples, compiler)
                            result = line_tester.run_test(
                                f"Тест строк: {lines_input}", 
                                max_workers=2
                            )
                            print_final_evaluation(result['success_rate'], result['total'], result['failed'])
                        else:
                            print("Не найдено ни одного примера по указанным строкам!")
                    
                    except ValueError:
                        print("Неверный формат ввода. Используйте формат: 100,150,200")
                
                else:
                    print("Неверный выбор. Пожалуйста, введите 1-5, D или Q")
            
            except (EOFError, KeyboardInterrupt):
                print("\n\nПрограмма завершена пользователем!")
                break
            
            except Exception as e:
                print(f"\Ошибка: {str(e)}!")
                continue
    
    except KeyboardInterrupt:
        print("\n\nПрограмма завершена пользователем!")
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}!")
        import traceback
        traceback.print_exc()


# ======================================================================
# ТОЧКА ВХОДА
# ======================================================================

if __name__ == "__main__":
    main()