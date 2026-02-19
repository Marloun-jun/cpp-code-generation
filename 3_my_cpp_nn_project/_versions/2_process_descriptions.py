import csv

def create_sorted_descriptions_dataset(input_file, output_file):
    # Создает новый CSV файл с правильными номерами строк и описаниями, отсортированными по длине
    try:
        # Читаем файл построчно для сохранения настоящих номеров строк
        with open(input_file, 'r', encoding='utf-8', newline='') as infile:
            all_lines = infile.readlines()
        print("Анализ и сбор описаний с сохранением номеров строк")
        processed_count = 0
        skipped_empty = 0
        skipped_comments = 0
        skipped_header = 0
        # Собираем все описания в список для сортировки
        descriptions_data = []
        # Обрабатываем каждую строку с сохранением исходного номера
        for line_num, line in enumerate(all_lines, 1):  # Начинаем с 1 как в текстовых редакторах
            line = line.strip()
            # Пропускаем полностью пустые строки
            if not line:
                skipped_empty += 1
                continue
            # Пропускаем строки, начинающиеся с '#'
            if line.startswith('#'):
                skipped_comments += 1
                continue
            # Пропускаем заголовок CSV (первая непустая строка)
            if line_num == 1 and line.startswith('description,code,style,topic,keywords'):
                skipped_header += 1
                continue
            # Парсим CSV строку
            try:
                # Используем csv.reader для корректного парсинга полей
                reader = csv.reader([line])
                fields = next(reader)
                # Проверяем, что есть поле description (первое поле)
                if len(fields) > 0:
                    description = fields[0].strip()
                   # Пропускаем пустые описания
                    if not description:
                        skipped_empty += 1
                        continue
                    # Сохраняем данные с ИСХОДНЫМ номером строки
                    descriptions_data.append({
                        'line_number': line_num,  # Исходный номер строки из файла
                        'description': description,
                        'length': len(description)
                    })
                    processed_count += 1
                    # Выводим прогресс каждые 100 строк
                    if processed_count % 100 == 0:
                        print(f"Собрано {processed_count} описаний ")
                else:
                    skipped_empty += 1
            except Exception as e:
                print(f"Ошибка парсинга строки {line_num}: {e}")
                continue
        print("Сортировка описаний по длине (от коротких к длинным)")
        # Сортируем по длине описания (от коротких к длинным)
        descriptions_data.sort(key=lambda x: x['length'])
        print("Сохранение отсортированных данных")
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['original_line_number', 'description', 'length'])
            for item in descriptions_data:
                writer.writerow([item['line_number'], item['description'], item['length']])
        # Статистика
        if descriptions_data:
            min_length = descriptions_data[0]['length']
            max_length = descriptions_data[-1]['length']
            avg_length = sum(item['length'] for item in descriptions_data) / len(descriptions_data)
            print("\nПОДРОБНАЯ СТАТИСТИКА:")
            print(f"   Всего строк в исходном файле: {len(all_lines)}")
            print(f"   Успешно обработано: {processed_count}")
            print(f"   Пропущено пустых строк: {skipped_empty}")
            print(f"   Пропущено комментариев: {skipped_comments}")
            print(f"   Пропущено заголовков: {skipped_header}")
            print(f"\nСТАТИСТИКА ПО ДЛИНАМ ОПИСАНИЙ:")
            print(f"   Самое короткое описание: {min_length} символов")
            print(f"   Самое длинное описание: {max_length} символов")
            print(f"   Средняя длина: {avg_length:.1f} символов")
            # Показываем примеры с правильными номерами строк
            print(f"\nПРИМЕРЫ ОПИСАНИЙ:")
            if len(descriptions_data) >= 5:
                print(f"   Строка {descriptions_data[0]['line_number']} (короткое): \"{descriptions_data[0]['description'][:80]}...\"")
                mid_index = len(descriptions_data) // 2
                print(f"   Строка {descriptions_data[mid_index]['line_number']} (среднее): \"{descriptions_data[mid_index]['description'][:80]}...\"")
                print(f"   Строка {descriptions_data[-1]['line_number']} (длинное): \"{descriptions_data[-1]['description'][:80]}...\"")
        print(f"\nФайл сохранен: {output_file}")
        print("   Структура файла: original_line_number, description, length")
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Упрощенная версия с правильными номерами строк
def create_sorted_descriptions_simple(input_file, output_file):
    # Упрощенная версия с правильными номерами строк
    try:
        with open(input_file, 'r', encoding='utf-8', newline='') as infile:
            all_lines = infile.readlines()
        descriptions_data = []
        for line_num, line in enumerate(all_lines, 1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('description,code,'):
                continue
            try:
                reader = csv.reader([line])
                fields = next(reader)
                if len(fields) > 0 and fields[0].strip():
                    description = fields[0].strip()
                    descriptions_data.append({
                        'line_number': line_num,  # Правильный номер из исходного файла
                        'description': description,
                        'length': len(description)
                    })
            except:
                continue
        # Сортируем по длине
        descriptions_data.sort(key=lambda x: x['length'])
        # Сохраняем
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['original_line_number', 'description', 'length'])
            for item in descriptions_data:
                writer.writerow([item['line_number'], item['description'], item['length']])
        print(f"Готово! Создан файл: {output_file}")
        print(f"Обработано записей: {len(descriptions_data)}")
        print(f"Отсортировано по длине (от {descriptions_data[0]['length']} до {descriptions_data[-1]['length']} символов)")
        print(f"Номера строк соответствуют исходному файлу '{input_file}'")
    except FileNotFoundError:
        print(f"Файл '{input_file}' не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")

# Запуск программы
if __name__ == "__main__":
    input_file = "3_my_cpp_nn_project/1_cpp_code_generation_dataset.csv"
    output_file = "3_my_cpp_nn_project/descriptions_sorted_by_length.csv"
    print("Запуск программы для создания отсортированного dataset")
    print("=" * 60)
    print("Описания отсортированы по длине (от коротких к длинным)")
    print("Номера строк соответствуют исходному файлу")
    print("=" * 60)
    # Выберите версию: подробную или упрощенную
    version = input("Выберите версию (1 - подробная, 2 - упрощенная): ").strip()
    if version == "1":
        create_sorted_descriptions_dataset(input_file, output_file)
    else:
        create_sorted_descriptions_simple(input_file, output_file)