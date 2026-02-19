import random

def check_descriptions(filename, sample_size=200):
# Выборочная проверка описаний (description)
    print("=" * 70)
    print("ВЫБОРОЧНАЯ ПРОВЕРКА ОПИСАНИЙ")
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
    # критерии проверки
    issues = {
        'wrong_verb': 0,      # не начинается с глагола
        'too_short': 0,       # слишком короткое
        'too_long': 0,        # слишком длинное
        'code_in_desc': 0,    # код в описании
        'bad_format': 0       # плохое форматирование
    }
    # глаголы, с которых должны начинаться описания
    valid_verbs = [
        'напиши', 'создай', 'реализуй', 'разработай', 'покажи','сделай', 'добавь', 
        'написать', 'создать', 'реализовать', 'разработать', 'показать', 'сделать', 'добавить',
        'продемонстрируй', 'представь', 'приведи','опиши',  'объясни',
        'продемонстрировать', 'представить', 'привести', 'описать', 'объяснить',
        'открой', 'открыть', 'выполни', 'выполнить', 'считай', 'считать',
        'прочитай', 'прочитать', 'запиши', 'записать', 'используй', 'использовать',
        'проверь', 'проверить', 'вычисли', 'вычислить', 'найди', 'найти'
    ]
    # проверяем каждую строку в выборке
    for i, (line_num, line) in enumerate(sample, 1):
        try:
            # парсим строку
            parts = line.split('","')
            if len(parts) < 5:
                issues['bad_format'] += 1
                continue
            # первое поле - описание (убираем начальную кавычку)
            description = parts[0][1:]
            # Проверка 1: Начинается ли с глагола?
            first_word = description.split()[0].lower() if description.split() else ""
            starts_with_verb = any(description.lower().startswith(verb) for verb in valid_verbs)
            if not starts_with_verb:
                issues['wrong_verb'] += 1
                # показываем пример
                if issues['wrong_verb'] <= 3:   # первые 3 примера
                    print(f"Строка {line_num}: не начинается с глагола")
                    print(f"Описание: {description[:100]}...")
            # Проверка 2: Длина описания
            if len(description) < 10:
                issues['too_short'] += 1
            elif len(description) > 500:
                issues['too_long'] += 1
            # Проверка 3: Есть ли код в описании?
            code_indicators = ['#include', 'int main()', 'cout <<', 'std::', ';', '{', '}']
            if any(indicator in description for indicator in code_indicators):
                issues['code_in_desc'] += 1
        except Exception as e:
            issues['bad_format'] += 1
    # выводим статистику
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ ОПИСАНИЙ:")
    print("=" * 70)
    total_issues = sum(issues.values())
    if total_issues == 0:
        print("ВСЕ описания в выборке корректны!")
    else:
        print(f"Найдено проблем в {total_issues} из {sample_size} проверенных строк:")
        print(f" - Не начинаются с глагола: {issues['wrong_verb']}")
        print(f" - Слишком короткие (<10 символов): {issues['too_short']}")
        print(f" - Слишком длинные (>500 символов): {issues['too_long']}")
        print(f" - Содержат код: {issues['code_in_desc']}")
        print(f" - Ошибки формата: {issues['bad_format']}")
    # выводим примеры хороших описаний
    print("\n" + "=" * 70)
    print("ПРИМЕРЫ ХОРОШИХ ОПИСАНИЙ:")
    print("=" * 70)
    good_examples = []
    for line_num, line in sample[:5]:  # первые 5 из выборки
        try:
            parts = line.split('","')
            if len(parts) >= 5:
                description = parts[0][1:]
                # проверяем, хорошее ли описание
                if (len(description) >= 20 and len(description) <= 200 and
                    any(description.lower().startswith(verb) for verb in valid_verbs)):
                    good_examples.append((line_num, description))
        except:
            continue
    for i, (line_num, desc) in enumerate(good_examples[:3], 1):
        print(f"{i}. Строка {line_num}:")
        print(f"   {desc[:120]}..." if len(desc) > 120 else f"   {desc}")
    return issues


if __name__ == "__main__":
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    # проверяем описания
    issues = check_descriptions(filename, sample_size=200)
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ ПО ОПИСАНИЯМ:")
    print("=" * 70)
    if issues['wrong_verb'] > 0:
        print("1. Убедитесь, что описания начинаются с глагола:")
        print("   'напиши', 'создай', 'реализуй', 'разработай', и т.д.")
    if issues['too_short'] > 0:
        print("2. Описания должны быть информативными (минимум 10 символов)")
    if issues['code_in_desc'] > 0:
        print("3. Не включайте код C++ в описание")
        print("   Код должен быть только в поле 'code'")
    print("\nСледующий шаг: проверка ключевых слов (keywords)")