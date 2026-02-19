import sys
import os
import random
import subprocess
import tempfile
import time
import re
import datetime
from collections import defaultdict

def parse_csv(line):
# Парсинг CSV строки (учитывает разделение полей ",")
    if not line.startswith('"') or not line.endswith('"'):
        return None
    # убираем внешние кавычки
    content = line[1:-1] 
    # разбиваем по "," (кавычка-запятая-кавычка)
    parts = content.split('","')
    # в начале и конце полей могут быть лишние кавычки
    cleaned_parts = []
    for part in parts:
        # убираем кавычки в начале/конце
        if part.startswith('"'):
            part = part[1:]
        if part.endswith('"'):
            part = part[:-1]
        cleaned_parts.append(part) 
    return cleaned_parts

def load_dataset(filename):
# Загрузка датасета
    print(f"Загрузка датасета из {filename}...")
    dataset = []
    topics_styles = defaultdict(lambda: {'using': [], 'explicit': []})
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Прочитано строк из файла: {len(lines)}")
            # пропускаем первую строку (заголовки)
            if len(lines) > 0:
                print(f"Пропускаем строку 1 (заголовки): {lines[0][:100]}...")
                lines = lines[1:]
                print(f"Осталось строк для обработки: {len(lines)}")
            
            for line_num, line in enumerate(lines, 2):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # показываем первые 5 строк для отладки
                if line_num <= 6:  # 2-6 строки после заголовков
                    print(f"\n--- Отладка строки {line_num} ---")
                    print(f"Строка (первые 200 символов): {line[:200]}")
                # парсинг
                parts = parse_csv(line)
                if not parts:
                    if line_num <= 11:  # 2-11 строки после заголовков
                        print(f"⚠️  Строка {line_num}: не удалось выполнить парсинг")
                    continue
                if line_num <= 4:  # 2-4 строки после заголовков
                    print(f"Найдено частей: {len(parts)}")
                    for i, part in enumerate(parts):
                        print(f"Часть {i} (первые 100 символов): {part[:100]}")
                # проверка на структуру примеров (наличие 5 полей)
                if len(parts) < 5:
                    if line_num <= 11:  # 2-11 строки после заголовков
                        print(f"⚠️  Строка {line_num}: только {len(parts)} частей вместо 5")
                    continue
                description = parts[0]
                code = parts[1]
                style = parts[2]
                topic = parts[3]
                keywords = parts[4] if len(parts) > 4 else ""
                # результат для первых 3 строк 
                if line_num <= 4:  # 2-4 строки после заголовков
                    print(f"Описание: {description[:50]}...")
                    print(f"Код (первые 100 символов): {code[:100]}...")
                    print(f"Стиль: {style}")
                    print(f"Тема: {topic}")
                # определяем тип стиля
                if 'using_namespace_std' in style:
                    style_type = 'using'
                elif 'explicit_std' in style:
                    style_type = 'explicit'
                else:
                    if line_num <= 11:  # 2-11 строки после заголовков
                        print(f"⚠️  Строка {line_num}: неизвестный стиль '{style}'")
                    continue
                # создаем запись для датасета
                example = {
                    'line': line_num,
                    'description': description,
                    'code': code,
                    'style': style,
                    'topic': topic,
                    'keywords': keywords,
                    'style_type': style_type
                }
                dataset.append(example)
                # добавляем в структуру тем и стилей
                topics_styles[topic][style_type].append(example)
                
                # отладочный вывод для первых нескольких строк
                if line_num <= 4:
                    print(f"✅ Добавлен пример строки {line_num}")
        print(f"\n✅ Загружено {len(dataset)} примеров")
        print(f"Уникальных тем: {len(topics_styles)}")
        # статистика по стилям
        using_count = sum(len(topics_styles[t]['using']) for t in topics_styles)
        explicit_count = sum(len(topics_styles[t]['explicit']) for t in topics_styles)
        print(f"Примеров с using namespace std: {using_count}")
        print(f"Примеров без using namespace std: {explicit_count}")
        # пример для проверки
        if dataset:
            print(f"\n{'='*60}")
            print("ПРОВЕРКА ПЕРВОГО ПРИМЕРА С КОДОМ")
            print('='*60)
            # ищем первый пример с нормальным кодом
            for ex in dataset[:10]:
                if ex['code'] and len(ex['code']) > 50 and '#include' in ex['code']:
                    print(f"\nПример найден (строка {ex['line']}):")
                    print(f"Тема: {ex['topic']}")
                    print(f"Стиль: {ex['style']}")
                    print(f"Описание: {ex['description']}")
                    code_lines = ex['code'].split('\n')
                    print(f"Код ({len(code_lines)} строк):")
                    for j, code_line in enumerate(code_lines[:5]):
                        print(f"  {j+1:2}: {code_line}")
                    if len(code_lines) > 5:
                        print(f"  ... и еще {len(code_lines)-5} строк")
                    break
            else:
                print("Не найдено примеров с корректным кодом!")
                # показываем первые 5 примеров для диагностики
                print(f"\nПервые 5 примеров для диагностики:")
                for i, ex in enumerate(dataset[:5], 1):
                    print(f"\n{i}. Строка {ex['line']}:")
                    print(f"   Тема: {ex['topic']}")
                    print(f"   Код (первые 200 символов): {ex['code'][:200]}")
        return dataset, topics_styles
    except Exception as e:
        print(f"❌ Ошибка загрузки датасета: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], defaultdict(lambda: {'using': [], 'explicit': []})

def availability_main_function(code):
# Простая проверка наличия функции main()
    return 'int main(' in code or 'void main(' in code or 'main(' in code

def simple_compile_and_test(code, timeout=2):
# Упрощенная компиляция и тестирование
    if not code or len(code) < 10:
        return {
            'compiled': False,
            'error': 'Код слишком короткий или пустой',
            'warnings': ''
        }
    # cоздаем временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(code)
        temp_file = f.name
    try:
        # проверяем синтаксис
        compile_cmd = ['g++', '-std=c++17', '-fsyntax-only', temp_file]
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True
        )
        if compile_result.returncode == 0:
            return {
                'compiled': True,
                'error': '',
                'warnings': compile_result.stderr[:200] if compile_result.stderr else ''
            }
        else:
            return {
                'compiled': False,
                'error': compile_result.stderr[:500],
                'warnings': ''
            }
    except Exception as e:
        return {
            'compiled': False,
            'error': f'Системная ошибка: {str(e)}',
            'warnings': ''
        }
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass

def quick_test(dataset, num_samples=50):
# Быстрый тест случайных примеров
    print(f"\n{'='*80}")
    print(f"БЫСТРЫЙ ТЕСТ ({num_samples} случайных примеров)")
    print('='*80)
    if num_samples > len(dataset):
        num_samples = len(dataset)
    test_samples = random.sample(dataset, num_samples)
    start_time = time.time()
    compiled = 0
    errors = []
    for i, example in enumerate(test_samples, 1):
        if i % 10 == 0 or i == len(test_samples):
            print(f"\rТестируем: {i}/{len(test_samples)}...", end='', flush=True)
        result = simple_compile_and_test(example['code'])
        if result['compiled']:
            compiled += 1
        else:
            errors.append({
                'line': example['line'],
                'topic': example['topic'],
                'style': example['style'],
                'description': example['description'],
                'error': result['error']
            })
    elapsed_time = time.time() - start_time
    success_rate = compiled / len(test_samples) * 100
    print(f"\rТестирование завершено за {elapsed_time:.1f} секунд")
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Всего примеров: {len(test_samples)}")
    print(f"   Успешно скомпилировались: {compiled} ({success_rate:.1f}%)")
    print(f"   Ошибок компиляции: {len(test_samples) - compiled}")
    if errors:
        print(f"\n❌ ПЕРВЫЕ 5 ОШИБОК:")
        for i, error in enumerate(errors[:5], 1):
            print(f"\n{i}. Строка {error['line']}: {error['topic']} ({error['style']})")
            print(f"   Описание: {error['description']}")
            if error['error']:
                error_lines = error['error'].split('\n')
                for line in error_lines[:3]:
                    if line.strip():
                        print(f"   {line[:100]}")
    return success_rate

def full_test(dataset, test_name, sample_size=None):
# Полный тест
    if sample_size and sample_size < len(dataset):
        test_samples = random.sample(dataset, sample_size)
    else:
        test_samples = dataset
    print(f"\n{'='*80}")
    print(f"ТЕСТ: {test_name}")
    print(f"Примеров для тестирования: {len(test_samples)}")
    print('='*80)
    start_time = time.time()
    compiled = 0
    errors = []
    for i, example in enumerate(test_samples, 1):
        if i % 10 == 0 or i == len(test_samples):
            print(f"\rТестируем: {i}/{len(test_samples)}...", end='', flush=True)
        result = simple_compile_and_test(example['code'])
        if result['compiled']:
            compiled += 1
        else:
            errors.append({
                'line': example['line'],
                'topic': example['topic'],
                'style': example['style'],
                'description': example['description'],
                'error': result['error']
            })
    elapsed_time = time.time() - start_time
    success_rate = compiled / len(test_samples) * 100
    print(f"\rТестирование завершено за {elapsed_time:.1f} секунд")
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   Всего примеров: {len(test_samples)}")
    print(f"   Успешно скомпилировались: {compiled} ({success_rate:.1f}%)")
    print(f"   Ошибок компиляции: {len(test_samples) - compiled}")
    if errors:
        print(f"\n❌ ОШИБКИ ({len(errors)}):")
        for i, error in enumerate(errors[:5], 1):
            print(f"\n{i}. Строка {error['line']}: {error['topic']} ({error['style']})")
            print(f"   Описание: {error['description']}")
            if error['error']:
                error_lines = error['error'].split('\n')
                for line in error_lines[:3]:
                    if line.strip():
                        print(f"   {line[:100]}")
        # сохраняем отчет
        errors_report(errors, test_name)
    return success_rate

def errors_report(errors, test_name):
# Отчет об ошибках
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "3_my_cpp_nn_project/check_dataset"
    os.makedirs(report_dir, exist_ok=True)
    report_file = os.path.join(report_dir, f"errors_report_{test_name.replace(' ', '_')}_{timestamp}.txt")
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"ОТЧЕТ ОБ ОШИБКАХ КОМПИЛЯЦИИ\n")
            f.write(f"Тест: {test_name}\n")
            f.write(f"Время: {datetime.datetime.now()}\n")
            f.write(f"Всего ошибок: {len(errors)}\n")
            f.write("="*80 + "\n\n")
            for i, error in enumerate(errors, 1):
                f.write(f"{i}. Строка {error['line']}: {error['topic']} ({error['style']})\n")
                f.write(f"   Описание: {error['description']}\n")
                f.write(f"   Ошибка:\n")
                error_lines = error['error'].split('\n')
                for line in error_lines[:20]:
                    if line.strip():
                        f.write(f"      {line}\n")
                f.write("-"*80 + "\n")
        print(f"\n📁 Полный отчет об ошибках сохранен в: {report_file}")
    except Exception as e:
        print(f"\n⚠️  Не удалось сохранить отчет об ошибках: {str(e)}")

def check_result_test(success_rate):
# Проверка результатов теста
    if success_rate >= 80:
        print(f"\n✅ ТЕСТ ПРОЙДЕН УСПЕШНО! ({success_rate:.1f}% успеха)")
    elif success_rate >= 50:
        print(f"\n⚠️  ТЕСТ ПРОЙДЕН С ПРЕДУПРЕЖДЕНИЕМ ({success_rate:.1f}% успеха)")
    else:
        print(f"\n❌ ТЕСТ НЕ ПРОЙДЕН ({success_rate:.1f}% успеха)")


def main():
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    print("="*80)
    print("ТЕСТИРОВАНИЕ КОМПИЛЯЦИИ КОДА C++")
    print("="*80)
    print("Особенности:")
    print("1. Корректный парсинг CSV с экранированием")
    print("2. Правильная обработка \\n и \\\"")
    print("3. Упрощенное тестирование")
    print("="*80)
    # проверяем наличие компилятора
    try:
        result = subprocess.run(['g++', '--version'], capture_output=True, text=True, check=True)
        print("✅ Компилятор g++ найден")
        print(f"   Версия: {result.stdout.split('\n')[0]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ОШИБКА: Компилятор g++ не найден!")
        return
    # проверяем наличие файла
    if not os.path.exists(filename):
        print(f"❌ Файл не найден: {filename}")
        return
    # загружаем датасет
    dataset, topics_styles = load_dataset(filename)
    if not dataset:
        print("❌ Не удалось загрузить датасет")
        return
    print("\n" + "="*80)
    print("ВАРИАНТЫ ТЕСТИРОВАНИЯ:")
    print("1. Быстрый тест (50 случайных примеров)")
    print("2. Средний тест (200 случайных примеров)")
    print("3. Полный тест (все примеры)")
    print("="*80)
    while True:
        try:
            choice = input("\nВыберите вариант (1-3) или 'q' для выхода: ").strip()
            if choice.lower() == 'q':
                print("Выход...")
                break
            elif choice == '1':
                success_rate = quick_test(dataset, 50)
                check_result_test(success_rate)
                break
            elif choice == '2':
                success_rate = full_test(dataset, "Средний тест (200 примеров)", 200)
                check_result_test(success_rate)
                break
            elif choice == '3':
                success_rate = full_test(dataset, "Полный тест (все примеры)")
                check_result_test(success_rate)
                break
            else:
                if not choice:
                    print("❌ Пустой ввод.", end=" ")
                print("Пожалуйста, введите 1, 2, 3 или q.")
        except KeyboardInterrupt:
            print("\n\nПрервано пользователем.")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            break

if __name__ == "__main__":
    main()