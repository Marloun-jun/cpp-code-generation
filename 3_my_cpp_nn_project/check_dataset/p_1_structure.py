def check_dataset_structure_final(filename):
# Проверка, что каждая строка имеет ровно 5 полей в правильном формате
    print("=" * 70)
    print("ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТА")
    print("=" * 70)
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
                    print("✓ Заголовок CSV найден и пропущен")
                    continue
                # 3. Проверяем базовую структуру
                if not (line.startswith('"') and line.endswith('"')):
                    results['errors'].append(f"Строка {line_num}: Не начинается или не заканчивается кавычкой")
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
                    results['errors'].append(f"Строка {line_num}: Неправильное количество полей ({comma_count + 1} вместо 5)")
                    results['invalid_rows'] += 1
                    continue
                # 5. Пытаемся извлечь все 5 полей
                try:
                    fields = extract_fields(line)
                    if len(fields) != 5:
                        results['errors'].append(f"Строка {line_num}: Удалось извлечь только {len(fields)} полей")
                        results['invalid_rows'] += 1
                        continue
                    # 6. Проверяем каждое поле
                    all_fields_valid = True
                    # Поле 0: description - должно быть в кавычках
                    if not (fields[0].startswith('"') and fields[0].endswith('"')):
                        results['errors'].append(f"Строка {line_num}: Поле description не в кавычках")
                        all_fields_valid = False
                    # Поле 1: code - должно быть в кавычках
                    if not (fields[1].startswith('"') and fields[1].endswith('"')):
                        results['errors'].append(f"Строка {line_num}: Поле code не в кавычках")
                        all_fields_valid = False
                    # Поле 2: style - должно быть 'using_namespace_std' или 'explicit_std'
                    style_value = fields[2].strip('"')
                    if style_value not in ['using_namespace_std', 'explicit_std']:
                        results['errors'].append(f"Строка {line_num}: Недопустимое значение style: '{style_value}'")
                        all_fields_valid = False
                    # Поле 3: topic - не должно быть пустым
                    if not fields[3].strip('"'):
                        results['errors'].append(f"Строка {line_num}: Пустое поле topic")
                        all_fields_valid = False
                    # Поле 4: keywords - не должно быть пустым
                    if not fields[4].strip('"'):
                        results['errors'].append(f"Строка {line_num}: Пустое поле keywords")
                        all_fields_valid = False
                    if all_fields_valid:
                        results['valid_rows'] += 1
                    else:
                        results['invalid_rows'] += 1
                except Exception as e:
                    results['errors'].append(f"Строка {line_num}: Ошибка при разборе полей: {str(e)}")
                    results['invalid_rows'] += 1
                # показываем прогресс каждые 1000 строк
                if line_num % 1000 == 0:
                    print(f"  Обработано {line_num} строк...")
    except FileNotFoundError:
        print(f"Ошибка: Файл '{filename}' не найден!")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    # выводим результаты
    print("-" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
    print("-" * 70)
    print(f"Всего строк в файле: {results['total_lines']}")
    print(f"Пропущено комментариев: {results['comments_skipped']}")
    print(f"Пропущено заголовков: {results['header_skipped']}")
    print(f"Валидных строк: {results['valid_rows']}")
    print(f"Невалидных строк: {results['invalid_rows']}")
    print(f"Найдено ошибок: {len(results['errors'])}")
    if results['errors']:
        print("\n" + "-" * 70)
        print("ДЕТАЛИ ОШИБОК:")
        print("-" * 70)
        # группируем ошибки по типам
        error_types = {}
        for error in results['errors']:
            error_type = error.split(':')[1].strip().split()[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        print("Статистика по типам ошибок:")
        for err_type, count in sorted(error_types.items()):
            print(f"  {err_type}: {count}")
        print(f"\nПервые 10 ошибок:")
        for i, error in enumerate(results['errors'][:10]):
            print(f"{i+1}. {error}")
        if len(results['errors']) > 10:
            print(f"... и еще {len(results['errors']) - 10} ошибок")
        # сохраняем полный отчет
        save_error_report(filename, results)
        print("\n" + "=" * 70)
        print("ТРЕБУЮТСЯ ИСПРАВЛЕНИЯ")
        print("=" * 70)
        return False
    else:
        print("\n" + "=" * 70)
        print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print(f"Все {results['valid_rows']} строк имеют правильную структуру")
        print("=" * 70)
        return True

def extract_fields(line):
# Извлекаем поля из CSV строки
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

def save_error_report(filename, results):
# Сохранение отчета об ошибках
    report_filename = f"structure_report_{filename.split('/')[-1]}.txt"
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("ОТЧЕТ О ПРОВЕРКЕ СТРУКТУРЫ ДАТАСЕТА\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Файл: {filename}\n")
            f.write(f"Время проверки: {__import__('datetime').datetime.now()}\n\n")
            f.write("СТАТИСТИКА:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Всего строк в файле: {results['total_lines']}\n")
            f.write(f"Пропущено комментариев: {results['comments_skipped']}\n")
            f.write(f"Пропущено заголовков: {results['header_skipped']}\n")
            f.write(f"Валидных строк: {results['valid_rows']}\n")
            f.write(f"Невалидных строк: {results['invalid_rows']}\n")
            f.write(f"Всего ошибок: {len(results['errors'])}\n\n")
            # группируем ошибки
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
        print(f"Не удалось сохранить отчет: {e}")

def analyze_sample_data(filename, sample_size=5):
# Анализ несколько примеров строк для демонстрации структуры
    print("\n" + "=" * 70)
    print("АНАЛИЗ ПРИМЕРОВ СТРОК:")
    print("=" * 70)
    samples = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.rstrip('\n')
                if not line or line.startswith('#') or line == 'description,code,style,topic,keywords':
                    continue
                if len(samples) < sample_size:
                    samples.append((line_num, line))
                else:
                    break
        for i, (line_num, line) in enumerate(samples, 1):
            print(f"\nПример {i} (Строка {line_num}):")
            print("-" * 40)
            # показываем структуру
            if len(line) > 150:
                preview = line[:150] + "..."
            else:
                preview = line
            print(f"Строка: {preview}")
            print(f"Длина: {len(line)} символов")
            # считаем кавычки и запятые
            quote_count = line.count('"')
            comma_count = line.count(',')
            print(f"Кавычек: {quote_count} (должно быть четное число: {'✓' if quote_count % 2 == 0 else '✗'})")
            print(f"Запятых: {comma_count} (должно быть 4: {'✓' if comma_count == 4 else '✗'})")
            # пытаемся извлечь поля
            try:
                fields = extract_fields(line)
                print(f"Извлечено полей: {len(fields)} (должно быть 5: {'✓' if len(fields) == 5 else '✗'})")
                if len(fields) >= 3:
                    print(f"Style значение: {fields[2].strip('"')}")
                if len(fields) >= 5:
                    print(f"Последние 2 поля:")
                    print(f"topic: {fields[3][:50]}...")
                    print(f"keywords: {fields[4][:50]}...")
            except Exception as e:
                print(f"Ошибка при анализе: {e}")
    except Exception as e:
        print(f"Ошибка при анализе примеров: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv"
    # анализируем несколько примеров
    analyze_sample_data(filename, 3)
    # запускаем полную проверку
    is_valid = check_dataset_structure_final(filename)
    if is_valid:
        print("Датасет готов к использованию!")
    else:
        print("Требуются исправления!")
        print("Сначала исправьте найденные ошибки структуры.")
    print("=" * 70)