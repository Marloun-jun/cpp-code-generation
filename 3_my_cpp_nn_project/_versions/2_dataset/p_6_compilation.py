import sys
import os
import random
import subprocess
import tempfile
import time
from collections import defaultdict

def fix_code_from_csv(code):
# Исправляет код из CSV: заменяет все escape-последовательности
    code = code.replace('\\\\\\\\', '\\\\')
    code = code.replace('\\"', '"')
    code = code.replace('\\n', '\n')
    code = code.replace('\\t', '\t')
    code = code.replace('\\\\', '\\')
    code = code.replace('&lt;', '<')
    code = code.replace('&gt;', '>')
    code = code.replace('&amp;', '&')
    return code

def compile_and_test(code, timeout=3):
# Компилирует и тестирует один пример кода
    # Исправляем код
    fixed_code = fix_code_from_csv(code)
    # Проверяем разные способы ввода
    lower_code = fixed_code.lower()
    needs_input = any(x in lower_code for x in [
        'cin >>', 'std::cin >>', 'getline(', 'scanf(', 'std::getline',
        'getchar()', 'getc(', 'fgetc(', 'getch()', 'getche()',  # Дополнительные функции ввода
        'std::cin.get(', 'cin.get(',  # Метод get
        'ввод', 'input',  # По описанию на русском/английском
    ])
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(fixed_code)
        temp_file = f.name
    try:
        # Компилируем
        compile_cmd = ['g++', '-std=c++17', '-Wall', '-Wextra', temp_file, '-o', temp_file + '.exe']
        
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True
        )
        # Если есть ошибки компиляции (не предупреждения)
        if compile_result.returncode != 0:
            return {
                'compiled': False,
                'ran': False,
                'output': '',
                'error': compile_result.stderr[:500],
                'warnings': '',
                'needed_input': needs_input,
                'code': fixed_code[:200]
            }
        # Определяем, есть ли реальные ошибки (не предупреждения)
        has_errors = any(x in compile_result.stderr.lower() for x in ['error:', 'ошибка:'])
        if has_errors:
            return {
                'compiled': False,
                'ran': False,
                'output': '',
                'error': compile_result.stderr[:500],
                'warnings': '',
                'needed_input': needs_input,
                'code': fixed_code[:200]
            }
        # Есть только предупреждения
        warnings = compile_result.stderr if compile_result.stderr.strip() else ''
        # Запускаем с тестовым вводом если нужно
        try:
            # Всегда подаем небольшой ввод на всякий случай
            # Многие учебные программы ожидают ввод, даже если он не обнаружен анализом
            test_input = ""
            # Определяем тип ввода по содержимому программы
            if needs_input:
                # Детальный анализ типа ввода
                if "символ" in lower_code or "char" in lower_code:
                    test_input = "A\n"  # Символ
                elif "число" in lower_code or "number" in lower_code or "int" in lower_code:
                    test_input = "42\n"
                elif "строка" in lower_code or "string" in lower_code:
                    test_input = "Test\n"
                elif "массив" in lower_code or "array" in lower_code:
                    test_input = "5\n1 2 3 4 5\n"
                elif "матриц" in lower_code or "matrix" in lower_code:
                    test_input = "2\n3\n1 2 3\n4 5 6\n"
                elif "возраст" in lower_code or "age" in lower_code:
                    test_input = "25\n"
                elif "мин" in lower_code or "minute" in lower_code:
                    test_input = "125\n"
                elif "час" in lower_code or "hour" in lower_code:
                    test_input = "48\n"
                else:
                    test_input = "1\n"  # Значение по умолчанию
            else:
                # Даже если анализ не показал need_input, 
                # некоторые программы все равно могут ожидать ввод
                # Особенно программы с выводом ASCII кода символа
                test_input = "X\n"  # Минимальный ввод на всякий случай
            
            run_result = subprocess.run(
                [temp_file + '.exe'],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                'compiled': True,
                'ran': True,
                'output': run_result.stdout,
                'error': '',
                'warnings': warnings,
                'needed_input': needs_input,
                'test_input_used': test_input,
                'code': fixed_code[:200]
            }
        except subprocess.TimeoutExpired:
            return {
                'compiled': True,
                'ran': False,
                'output': '',
                'error': f'Таймаут выполнения (ввод{" требовался" if needs_input else " не требовался"}, подан: {repr(test_input)})',
                'warnings': warnings,
                'needed_input': needs_input,
                'test_input_used': test_input,
                'code': fixed_code[:200]
            }
        except Exception as e:
            return {
                'compiled': True,
                'ran': False,
                'output': '',
                'error': f'Ошибка запуска: {str(e)}',
                'warnings': warnings,
                'needed_input': needs_input,
                'test_input_used': test_input if 'test_input' in locals() else '',
                'code': fixed_code[:200]
            }
    except Exception as e:
        return {
            'compiled': False,
            'ran': False,
            'output': '',
            'error': f'Системная ошибка: {str(e)}',
            'warnings': '',
            'needed_input': needs_input,
            'code': fixed_code[:200]
        }
    finally:
        # Удаляем временные файлы
        try:
            os.unlink(temp_file)
            if os.path.exists(temp_file + '.exe'):
                os.unlink(temp_file + '.exe')
        except:
            pass

def load_dataset(filename):
# Загружает весь датасет в память
    print(f"Загрузка датасета из {filename}...")
    dataset = []
    topics_styles = defaultdict(lambda: {'using': [], 'explicit': []})
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('"') and line.endswith('"'):
                parts = line[1:-1].split('","')
                if len(parts) >= 5:
                    description = parts[0]
                    code = parts[1]
                    style = parts[2]
                    topic = parts[3]
                    keywords = parts[4]
                    # Определяем тип стиля
                    if 'using_namespace_std' in style:
                        style_type = 'using'
                    elif 'explicit_std' in style:
                        style_type = 'explicit'
                    else:
                        continue
                    example = {
                        'line': line_num,
                        'description': description[:100],
                        'code': code,
                        'style': style_type,
                        'topic': topic,
                        'keywords': keywords
                    }
                    dataset.append(example)
                    topics_styles[topic][style_type].append(example)
    print(f"Загружено {len(dataset)} примеров")
    print(f"Уникальных тем: {len(topics_styles)}")
    return dataset, topics_styles

def option1_quick_test(dataset, num_samples=20):
# Вариант 1: Быстрый тест N случайных примеров
    print("\n" + "="*80)
    print(f"БЫСТРЫЙ ТЕСТ ({num_samples} случайных примеров)")
    print("="*80)
    if num_samples > len(dataset):
        num_samples = len(dataset)
    samples = random.sample(dataset, num_samples)
    return run_tests(samples, f"Быстрый тест ({num_samples} примеров)")

def option2_full_test(topics_styles, samples_per_pair=2):
# Вариант 2: Полный тест (по N примеров на каждую пару тема×стиль)
    print("\n" + "="*80)
    print(f"ПОЛНЫЙ ТЕСТ (по {samples_per_pair} примера на каждую пару тема×стиль)")
    print("="*80)
    test_samples = []
    for topic, styles in topics_styles.items():
        for style_type in ['using', 'explicit']:
            examples = styles[style_type]
            if examples:
                if len(examples) >= samples_per_pair:
                    selected = random.sample(examples, samples_per_pair)
                else:
                    selected = examples
                test_samples.extend(selected)
    print(f"Будет протестировано: {len(test_samples)} примеров")
    print(f"Пар тема×стиль: {len(topics_styles) * 2}")
    return run_tests(test_samples, f"Полный тест ({len(test_samples)} примеров)")

def option3_limited_test(dataset, max_samples=100):
# Вариант 3: Ограниченный тест (максимум N случайных примеров)
    print("\n" + "="*80)
    print(f"ОГРАНИЧЕННЫЙ ТЕСТ (максимум {max_samples} случайных примеров)")
    print("="*80)
    if max_samples > len(dataset):
        max_samples = len(dataset)
    samples = random.sample(dataset, max_samples)
    return run_tests(samples, f"Ограниченный тест ({max_samples} примеров)")

def run_tests(samples, test_name):
# Запускает тестирование набора примеров
    print(f"\nНачинаем тестирование...")
    print("-"*80)
    start_time = time.time()
    results = {
        'total': len(samples),
        'compiled': 0,
        'ran': 0,
        'failed_compile': 0,
        'failed_run': 0,
        'with_warnings': 0,
        'by_topic': defaultdict(lambda: {'total': 0, 'compiled': 0, 'ran': 0}),
        'by_style': {'using': {'total': 0, 'compiled': 0, 'ran': 0}, 
                     'explicit': {'total': 0, 'compiled': 0, 'ran': 0}},
        'errors': []
    }
    for i, example in enumerate(samples, 1):
        print(f"\rТестируем пример {i}/{len(samples)}...", end='', flush=True)
        topic = example['topic']
        style = example['style']
        # Обновляем статистику
        results['by_topic'][topic]['total'] += 1
        results['by_style'][style]['total'] += 1
        # Тестируем
        test_result = compile_and_test(example['code'])
        if test_result['compiled']:
            results['compiled'] += 1
            results['by_topic'][topic]['compiled'] += 1
            results['by_style'][style]['compiled'] += 1
            if test_result['warnings']:
                results['with_warnings'] += 1
            if test_result['ran']:
                results['ran'] += 1
                results['by_topic'][topic]['ran'] += 1
                results['by_style'][style]['ran'] += 1
            else:
                results['failed_run'] += 1
                results['errors'].append({
                    'type': 'runtime',
                    'line': example['line'],
                    'topic': topic,
                    'style': style,
                    'description': example['description'],
                    'error': test_result['error'],
                    'code_preview': test_result['code']
                })
        else:
            results['failed_compile'] += 1
            results['errors'].append({
                'type': 'compile',
                'line': example['line'],
                'topic': topic,
                'style': style,
                'description': example['description'],
                'error': test_result['error'],
                'code_preview': test_result['code']
            })
    elapsed_time = time.time() - start_time
    print(f"\rТестирование завершено за {elapsed_time:.1f} секунд")
    # Выводим результаты
    print_results(results, test_name)
    return results

def print_results(results, test_name):
# Выводит результаты тестирования
    print("\n" + "="*80)
    print(f"РЕЗУЛЬТАТЫ: {test_name}")
    print("="*80)
    total = results['total']
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего примеров: {total}")
    print(f"   Успешно скомпилировались: {results['compiled']} ({results['compiled']/total*100:.1f}%)")
    print(f"   Успешно запустились: {results['ran']} ({results['ran']/total*100:.1f}%)")
    print(f"   С предупреждениями: {results['with_warnings']} ({results['with_warnings']/total*100:.1f}%)")
    print(f"   Ошибок компиляции: {results['failed_compile']}")
    print(f"   Ошибок выполнения: {results['failed_run']}")
    print(f"\n🎭 ПО СТИЛЯМ:")
    for style in ['using', 'explicit']:
        stats = results['by_style'][style]
        if stats['total'] > 0:
            compile_rate = stats['compiled'] / stats['total'] * 100
            run_rate = stats['ran'] / stats['total'] * 100
            print(f"   {style.upper():10}: {stats['total']:4} примеров")
            print(f"     Скомпилировались: {stats['compiled']:4} ({compile_rate:5.1f}%)")
            print(f"     Запустились:      {stats['ran']:4} ({run_rate:5.1f}%)")
    # Топ-10 тем по успешности
    print(f"\n🏆 ТОП-10 ТЕМ ПО УСПЕШНОСТИ КОМПИЛЯЦИИ:")
    topics_sorted = sorted(
        [(t, s) for t, s in results['by_topic'].items() if s['total'] >= 2],
        key=lambda x: x[1]['compiled'] / x[1]['total'] if x[1]['total'] > 0 else 0,
        reverse=True
    )[:10]
    for topic, stats in topics_sorted:
        if stats['total'] > 0:
            rate = stats['compiled'] / stats['total'] * 100
            print(f"   {topic:25}: {stats['compiled']}/{stats['total']} ({rate:5.1f}%)")
    # Худшие темы (если есть ошибки)
    problematic = [(t, s) for t, s in results['by_topic'].items() 
                   if s['total'] > 0 and s['compiled'] < s['total']]
    if problematic:
        print(f"\n⚠️  ПРОБЛЕМНЫЕ ТЕМЫ (есть ошибки компиляции):")
        for topic, stats in problematic[:5]:
            if stats['total'] > 0:
                error_rate = (stats['total'] - stats['compiled']) / stats['total'] * 100
                print(f"   {topic:25}: {stats['compiled']}/{stats['total']} ({error_rate:5.1f}% ошибок)")
    # Показываем примеры ошибок
    if results['errors']:
        print(f"\n❌ ОШИБКИ ({len(results['errors'])}):")
        print("-"*80)
        compile_errors = [e for e in results['errors'] if e['type'] == 'compile']
        runtime_errors = [e for e in results['errors'] if e['type'] == 'runtime']
        if compile_errors:
            print(f"Ошибки компиляции ({len(compile_errors)}):")
            for error in compile_errors[:3]:
                print(f"\n  Строка {error['line']}: {error['topic']} ({error['style']})")
                print(f"  Описание: {error['description']}")
                print(f"  Ошибка: {error['error'][:200]}...")
                if 'needed_input' in error and error['needed_input']:
                    print(f"  Требовался ввод: ДА")
        if runtime_errors:
            print(f"\nОшибки выполнения ({len(runtime_errors)}):")
            for error in runtime_errors[:2]:
                print(f"\n  Строка {error['line']}: {error['topic']} ({error['style']})")
                print(f"  Описание: {error['description']}")
                print(f"  Ошибка: {error['error']}")
    # Сохраняем отчет
    save_report(results, test_name)

def save_report(results, test_name):
# Сохраняет отчет в файл
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"3_my_cpp_nn_project/check_dataset/compilation_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ОТЧЕТ О ТЕСТИРОВАНИИ КОМПИЛЯЦИИ\n")
        f.write(f"Тест: {test_name}\n")
        f.write(f"Время: {datetime.datetime.now()}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Всего примеров: {results['total']}\n")
        f.write(f"Успешно скомпилировались: {results['compiled']} ({results['compiled']/results['total']*100:.1f}%)\n")
        f.write(f"Успешно запустились: {results['ran']} ({results['ran']/results['total']*100:.1f}%)\n")
        f.write(f"С предупреждениями: {results['with_warnings']}\n")
        f.write(f"Ошибок компиляции: {results['failed_compile']}\n")
        f.write(f"Ошибок выполнения: {results['failed_run']}\n\n")
        # Детальные ошибки
        if results['errors']:
            f.write("\n" + "="*80 + "\n")
            f.write("ДЕТАЛЬНЫЙ СПИСОК ОШИБОК\n")
            f.write("="*80 + "\n\n")
            for i, error in enumerate(results['errors'], 1):
                f.write(f"{i}. [{error['type'].upper()}] Строка {error['line']}\n")
                f.write(f"   Тема: {error['topic']}, Стиль: {error['style']}\n")
                f.write(f"   Описание: {error['description']}\n")
                f.write(f"   Ошибка: {error['error'][:500]}\n")
                f.write(f"   Код (фрагмент): {error['code_preview']}\n")
                f.write("-"*60 + "\n")
    print(f"\n📁 Подробный отчет сохранен в: {report_file}")


def main():
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    print("="*80)
    print("ТЕСТИРОВАНИЕ КОМПИЛЯЦИИ КОДА C++")
    print("="*80)
    # Проверяем наличие компилятора
    try:
        subprocess.run(['g++', '--version'], capture_output=True, check=True)
        print("✅ Компилятор g++ найден")
    except:
        print("❌ ОШИБКА: Компилятор g++ не найден!")
        print("Установите g++ или настройте PATH")
        return
    # Загружаем датасет
    dataset, topics_styles = load_dataset(filename)
    if not dataset:
        print("❌ Не удалось загрузить датасет")
        return
    print("\n" + "="*80)
    print("ВАРИАНТЫ ТЕСТИРОВАНИЯ:")
    print("1. Быстрый тест (20 случайных примеров)")
    print("2. Полный тест (по 2 примера на каждую пару тема×стиль)")
    print("3. Ограниченный тест (100 случайных примеров)")
    print("="*80)
    while True:
        choice = input("\nВыберите вариант (1-3) или 'q' для выхода: ").strip().lower()
        if choice == 'q':
            print("Выход...")
            break
        
        elif choice == '1':
            results = option1_quick_test(dataset, 20)
            if results and results['compiled'] / results['total'] >= 0.8:
                print("\n✅ ТЕСТ ПРОЙДЕН УСПЕШНО!")
            break
        elif choice == '2':
            results = option2_full_test(topics_styles, 2)
            if results and results['compiled'] / results['total'] >= 0.8:
                print("\n✅ ТЕСТ ПРОЙДЕН УСПЕШНО!")
            break
        elif choice == '3':
            results = option3_limited_test(dataset, 100)
            if results and results['compiled'] / results['total'] >= 0.8:
                print("\n✅ ТЕСТ ПРОЙДЕН УСПЕШНО!")
            break
        else:
            print("Некорректный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()