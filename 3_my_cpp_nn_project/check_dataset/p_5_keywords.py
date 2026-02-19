import random

def is_std_library_keyword(keyword):
# Проверка, относится ли ключевое слово к стандартной библиотеке C++ и должно ли иметь префикс std:: в explicit_std
    # ключевые слова, которые ВСЕГДА требуют std:: в explicit_std
    always_std_keywords = {
        # IO Streams
        'cout', 'endl', 'cin', 'cerr', 'clog',
        'ostream', 'istream', 'iostream', 'fstream', 'sstream',
        'stringstream', 'ostringstream', 'istringstream',
        # Strings
        'string', 'wstring', 'u16string', 'u32string',
        'string_view', 'wstring_view',
        # STL Containers
        'vector', 'array', 'deque', 'forward_list', 'list',
        'set', 'multiset', 'unordered_set', 'unordered_multiset',
        'map', 'multimap', 'unordered_map', 'unordered_multimap',
        'stack', 'queue', 'priority_queue',
        # STL Algorithms
        'sort', 'find', 'copy', 'transform', 'accumulate',
        'max_element', 'min_element', 'reverse', 'unique',
        'binary_search', 'lower_bound', 'upper_bound',
        'generate',  # ← как std::generate
        # Smart Pointers
        'unique_ptr', 'shared_ptr', 'weak_ptr',
        # Threading
        'thread', 'mutex', 'lock_guard', 'unique_lock',
        'condition_variable', 'future', 'promise', 'async',
        # Utilities
        'pair', 'tuple', 'optional', 'variant', 'any',
        'function', 'bind', 'ref', 'cref',
        # Chrono
        'chrono', 'steady_clock', 'system_clock', 'high_resolution_clock',
        'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
        # Filesystem
        'filesystem', 'path', 'directory_iterator', 'recursive_directory_iterator',
        # Other std
        'regex', 'random', 'ratio', 'complex', 'valarray',
        'bitset', 'type_info', 'type_index', 'bad_cast',
        'bad_alloc', 'exception', 'runtime_error', 'logic_error',
        # Common patterns
        'getline', 'stoi', 'stod', 'to_string', 'move', 'forward'
    }
    # слова, которые МОГУТ быть без std:: в зависимости от контекста
    contextual_keywords = {
        'begin', 'end', 'rbegin', 'rend', 'cbegin', 'cend',
        'iterator', 'const_iterator', 'reverse_iterator',
        'size', 'empty', 'front', 'back', 'push_back', 'pop_back',
        'insert', 'erase', 'clear', 'data'
    }
    return keyword in always_std_keywords

def check_keywords(filename, sample_size=100):
# Проверка ключевых слов (keywords)
    print("=" * 70)
    print("ПРОВЕРКА КЛЮЧЕВЫХ СЛОВ (KEYWORDS)")
    print(f"Проверяем {sample_size} случайных примеров")
    print("=" * 70)
    # собираем все строки с данными
    data_lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip() or raw_line.strip().startswith('#'):
                continue
            if not raw_line.startswith('"') or not raw_line.endswith('"'):
                continue
            data_lines.append((line_num, raw_line))
    print(f"Всего строк с данными: {len(data_lines)}")
    # берём случайную выборку
    if sample_size > len(data_lines):
        sample_size = len(data_lines)
    sample = random.sample(data_lines, sample_size)
    print(f"\nПроверяем {sample_size} случайных строк...")
    print("=" * 70)
    issues = {
        'no_keywords': 0,           # нет ключевых слов
        'too_few': 0,               # слишком мало ключевых слов (<2)
        'too_many': 0,              # слишком много ключевых слов (>15)
        'wrong_prefix_explicit': 0, # в explicit_std у std-ключевых нет std::
        'wrong_prefix_using': 0,    # в using_namespace_std у std-ключевых есть std::
        'format_issues': 0          # проблемы с форматом
    }
    # для отладки: собираем примеры проблем
    debug_examples = []
    # проверяем каждую строку в выборке
    for i, (line_num, line) in enumerate(sample, 1):
        try:
            # парсим строку
            parts = line.split('","')
            if len(parts) < 5:
                issues['format_issues'] += 1
                continue
            # извлекаем поля
            style = parts[2]
            # безопасное извлечение ключевых слов
            keywords_field = parts[4]
            if keywords_field.endswith('"'):
                keywords_field = keywords_field[:-1]  # Убираем последнюю кавычку
            # разбиваем ключевые слова по запятой
            keywords = [k.strip() for k in keywords_field.split(',') if k.strip()]
            # Проверка 1: Есть ли ключевые слова?
            if not keywords:
                issues['no_keywords'] += 1
                continue
            # Проверка 2: Количество ключевых слов
            if len(keywords) < 2:
                issues['too_few'] += 1
            elif len(keywords) > 15:
                issues['too_many'] += 1
            # Проверка 3: Префиксы в зависимости от стиля
            if 'explicit_std' in style:
                # в explicit_std ключевые слова из std должны начинаться с std::
                missing_std = []
                for kw in keywords:
                    # если это ключевое слово из std и не начинается с std::
                    if is_std_library_keyword(kw) and not kw.startswith('std::'):
                        missing_std.append(kw)
                if missing_std:
                    issues['wrong_prefix_explicit'] += 1
                    # для отладки сохраняем примеры
                    if len(debug_examples) < 3:
                        debug_examples.append({
                            'line': line_num,
                            'style': 'explicit_std',
                            'missing_std': missing_std[:3],
                            'all_keywords': keywords[:10]
                        })
                    if issues['wrong_prefix_explicit'] <= 2:
                        print(f"Строка {line_num}: explicit_std, но у std-ключевых нет std::")
                        print(f"Проблемные: {missing_std[:3]}")
                        print(f"Все ключевые слова: {', '.join(keywords[:5])}...")
            elif 'using_namespace_std' in style:
                # в using_namespace_std ключевые слова НЕ должны начинаться с std::
                extra_std = []
                for kw in keywords:
                    # если ключевое слово начинается с std::
                    if kw.startswith('std::'):
                        # проверяем, является ли оно std-ключевым словом
                        base_kw = kw[5:]  # Убираем 'std::'
                        if is_std_library_keyword(base_kw):
                            extra_std.append(kw)
                if extra_std:
                    issues['wrong_prefix_using'] += 1
                    if issues['wrong_prefix_using'] <= 2:
                        print(f"Строка {line_num}: using_namespace_std, но есть std::")
                        print(f"Лишние std::: {extra_std[:3]}")
                        print(f"Все ключевые слова: {', '.join(keywords[:5])}...")
        except Exception as e:
            issues['format_issues'] += 1
    # выводим статистику
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ КЛЮЧЕВЫХ СЛОВ:")
    print("=" * 70)
    total_issues = sum(issues.values())
    if total_issues == 0:
        print("ВСЕ ключевые слова в выборке корректны!")
    else:
        print(f"Найдено проблем в {total_issues} из {sample_size} проверенных строк:")
        print(f" - Нет ключевых слов: {issues['no_keywords']}")
        print(f" - Слишком мало (<2): {issues['too_few']}")
        print(f" - Слишком много (>15): {issues['too_many']}")
        print(f" - explicit_std без std:: у std-ключевых: {issues['wrong_prefix_explicit']}")
        print(f" - using_namespace_std с std::: {issues['wrong_prefix_using']}")
        print(f" - Проблемы формата: {issues['format_issues']}")
    # показываем подробные примеры для отладки
    if debug_examples:
        print("\n" + "=" * 70)
        print("ПОДРОБНЫЕ ПРИМЕРЫ ПРОБЛЕМ:")
        print("=" * 70)
        for example in debug_examples:
            print(f"\nСтрока {example['line']} ({example['style']}):")
            print(f"Ключевые слова без std::: {example['missing_std']}")
            print(f"Все ключевые слова: {example['all_keywords']}")
    # выводим примеры хороших ключевых слов
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ХОРОШИХ КЛЮЧЕВЫХ СЛОВ:")
    print("=" * 70)
    good_examples = []
    for line_num, line in sample[:10]:  # первые 10 из выборки
        try:
            parts = line.split('","')
            if len(parts) >= 5:
                style = parts[2]
                keywords_field = parts[4]
                if keywords_field.endswith('"'):
                    keywords_field = keywords_field[:-1]
                keywords = [k.strip() for k in keywords_field.split(',') if k.strip()]
                # проверяем, хорошие ли ключевые слова
                is_good = True
                if 'explicit_std' in style:
                    # проверяем, что все std-ключевые слова имеют std::
                    for kw in keywords:
                        if is_std_library_keyword(kw) and not kw.startswith('std::'):
                            is_good = False
                            break
                elif 'using_namespace_std' in style:
                    # проверяем, что нет std:: у std-ключевых слов
                    for kw in keywords:
                        if kw.startswith('std::'):
                            base_kw = kw[5:]
                            if is_std_library_keyword(base_kw):
                                is_good = False
                                break
                if is_good and 2 <= len(keywords) <= 15:
                    good_examples.append((line_num, style, keywords))
        except:
            continue
    for i, (line_num, style, keywords) in enumerate(good_examples[:3], 1):
        print(f"{i}. Строка {line_num} ({style}):")
        print(f"   {', '.join(keywords[:8])}" + ("..." if len(keywords) > 8 else ""))
    return issues

def check_keyword_consistency(filename):
# Проверка консистентности ключевых слов для одинаковых тем
    print("\n" + "=" * 70)
    print("ПРОВЕРКА КОНСИСТЕНТНОСТИ КЛЮЧЕВЫХ СЛОВ:")
    print("=" * 70)
    # собираем ключевые слова по темам и стилям
    topic_keywords = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip() or raw_line.strip().startswith('#'):
                continue
            if not raw_line.startswith('"') or not raw_line.endswith('"'):
                continue
            try:
                parts = raw_line.split('","')
                if len(parts) < 5:
                    continue
                style = parts[2]
                topic = parts[3]
                keywords_field = parts[4]
                if keywords_field.endswith('"'):
                    keywords_field = keywords_field[:-1]
                keywords = frozenset(k.strip() for k in keywords_field.split(',') if k.strip())
                if topic not in topic_keywords:
                    topic_keywords[topic] = {'using': set(), 'explicit': set()}
                if 'using_namespace_std' in style:
                    topic_keywords[topic]['using'].add(keywords)
                elif 'explicit_std' in style:
                    topic_keywords[topic]['explicit'].add(keywords)
            except Exception:
                continue
    # анализируем консистентность
    print("\nАнализ консистентности ключевых слов по темам:")
    inconsistent_topics = []
    for topic, styles in topic_keywords.items():
        # для тем, где есть оба стиля
        if styles['using'] and styles['explicit']:
            # преобразуем explicit ключевые слова (убираем std::) для сравнения
            explicit_without_std = set()
            for kw_set in styles['explicit']:
                clean_set = frozenset()
                for kw in kw_set:
                    if kw.startswith('std::'):
                        clean_set |= frozenset([kw[5:]])  # убираем std::
                    else:
                        clean_set |= frozenset([kw])
                explicit_without_std.add(clean_set)
            # сравниваем с using (там std:: не должно быть)
            if styles['using'] != explicit_without_std:
                inconsistent_topics.append(topic)
    if inconsistent_topics:
        print(f"\nНайдено {len(inconsistent_topics)} тем с неконсистентными ключевыми словами:")
        for topic in inconsistent_topics[:5]:
            print(f" - {topic}")
        if len(inconsistent_topics) > 5:
            print(f" ... и еще {len(inconsistent_topics) - 5} тем")
    else:
        print("Все темы имеют консистентные ключевые слова между стилями")
    return inconsistent_topics

if __name__ == "__main__":
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    # проверяем ключевые слова
    issues = check_keywords(filename, sample_size=100)
    # проверяем консистентность
    inconsistent = check_keyword_consistency(filename)
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ ПО КЛЮЧЕВЫМ СЛОВАМ:")
    print("=" * 70)
    if issues['wrong_prefix_explicit'] > 0:
        print("1. В explicit_std ключевые слова из стандартной библиотеки")
        print("   должны начинаться с 'std::' (например, 'std::cout', 'std::endl')")
        print("   Но базовые конструкции (if, for, int) оставлять без std::")
    if issues['wrong_prefix_using'] > 0:
        print("2. В using_namespace_std ключевые слова НЕ должны начинаться с 'std::'")
        print("   Используйте 'cout,endl,cin' вместо 'std::cout,std::endl,std::cin'")
    if inconsistent:
        print("3. Для одной темы ключевые слова должны быть одинаковыми")
        print("   (с учётом префикса std:: для std-ключевых слов в разных стилях)")