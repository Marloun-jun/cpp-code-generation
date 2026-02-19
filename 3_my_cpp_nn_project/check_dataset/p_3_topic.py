def extract_unique_topics_in_order(filename):
# Провека поля topic на соответствие структуры dataset
    print("=" * 60)
    print("АНАЛИЗ КАТЕГОРИЗАЦИИ ТЕМ (TOPIC)")
    print("Поле: второе с конца (перед ключевыми словами)")
    print("Порядок: как идут в датасете (сверху вниз)")
    print("=" * 60)
    topics = []             # для сохранения порядка
    seen_topics = set()     # для проверки уникальности
    line_num = 0
    data_line_num = 0
    parse_errors = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line_num += 1
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip() or raw_line.strip().startswith('#'):
                continue
            data_line_num += 1
            # проверяем базовую структуру
            if not raw_line.startswith('"') or not raw_line.endswith('"'):
                continue
            parts = raw_line.split('","')
            if len(parts) >= 5:     # должно быть минимум 5 полей
                topic = parts[-2]   # -1 это последнее, -2 предпоследнее
                if topic not in seen_topics:
                    seen_topics.add(topic)
                    topics.append(topic)
            else:
                parse_errors += 1
            if data_line_num % 1000 == 0:
                print(f"  Обработано {data_line_num} строк...")
    print("\n" + "=" * 60)
    print(f"ВСЕГО УНИКАЛЬНЫХ ТЕМ: {len(topics)}")
    print(f"Строк обработано: {data_line_num}")
    print(f"Ошибок парсинга: {parse_errors}")
    print("Порядок: сверху вниз по датасету")
    print("=" * 60)
    # выводим все темы в порядке появления
    for i, topic in enumerate(topics, 1):
        print(f"{i:3}. {topic}")
    # быстрая проверка очевидных опечаток
    print("\n" + "=" * 60)
    print("БЫСТРАЯ ПРОВЕРКА НА ОПЕЧАТКИ:")
    print("=" * 60)
    # проверяем похожие названия (без сортировки, в порядке появления)
    possible_issues = []
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            t1, t2 = topics[i].lower(), topics[j].lower()
            # проверяем разные варианты опечаток
            if (t1 == t2):
                possible_issues.append(f"Дубликат (разный регистр): '{topics[i]}' и '{topics[j]}'")
            elif t1.replace('_', '') == t2.replace('_', ''):
                possible_issues.append(f"Разные разделители: '{topics[i]}' и '{topics[j]}'")
            elif t1.replace('_', '-') == t2.replace('_', '-'):
                possible_issues.append(f"Разные разделители (дефис/подчёркивание): '{topics[i]}' и '{topics[j]}'")
            elif abs(len(t1) - len(t2)) <= 2 and (t1 in t2 or t2 in t1):
                possible_issues.append(f"Возможная опечатка: '{topics[i]}' и '{topics[j]}'")
    if possible_issues:
        print("Найдены возможные проблемы:")
        for issue in possible_issues[:5]:   # показываем первые 5
            print(f"  {issue}")
        if len(possible_issues) > 5:
            print(f"  ... и еще {len(possible_issues) - 5} проблем")
    else:
        print("Очевидных проблем не найдено")
    return topics

def check_topic_balance_simple(filename, topics):
# Проверка, что каждая тема представлена в обоих стилях (простая версия)
    print("\n" + "=" * 60)
    print("ПРОВЕРКА БАЛАНСА ПО СТИЛЯМ (для каждой темы):")
    print("=" * 60)
    # инициализируем статистику
    topic_stats = {topic: {'using': 0, 'explicit': 0} for topic in topics}
    style_errors = 0
    line_num = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line_num += 1
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip() or raw_line.strip().startswith('#'):
                continue
            # парсим простым split
            parts = raw_line.split('","')
            if len(parts) < 5:
                continue
            try:
                # Извлекаем нужные поля
                # parts[0] - описание (убираем начальную кавычку)
                # parts[1] - код
                # parts[2] - стиль
                # parts[3] - тема
                # parts[4] - ключевые слова (убираем конечную кавычку)
                style = parts[2]
                topic = parts[3]
                # проверяем стиль
                if 'using_namespace_std' in style:
                    style_type = 'using'
                elif 'explicit_std' in style:
                    style_type = 'explicit'
                else:
                    style_errors += 1
                    continue
                # обновляем статистику
                if topic in topic_stats:
                    topic_stats[topic][style_type] += 1
            except IndexError:
                continue
    # выводим результаты
    problems = []
    good_topics = []
    for topic in topics:
        using_count = topic_stats[topic]['using']
        explicit_count = topic_stats[topic]['explicit']
        total = using_count + explicit_count
        if total == 0:
            problems.append(f"Тема '{topic}': не найдена в датасете (возможно ошибка парсинга)")
        elif using_count == 0:
            problems.append(f"Тема '{topic}': {explicit_count} примеров, НЕТ using_namespace_std")
        elif explicit_count == 0:
            problems.append(f"Тема '{topic}': {using_count} примеров, НЕТ explicit_std")
        elif using_count < 2 or explicit_count < 2:
            problems.append(f"Тема '{topic}': using={using_count}, explicit={explicit_count} (мало примеров)")
        else:
            good_topics.append(topic)
    # выводим хорошие темы
    if good_topics:
        print("\nТемы с хорошим балансом:")
        for topic in good_topics[:10]:  # показываем первые 10
            using = topic_stats[topic]['using']
            explicit = topic_stats[topic]['explicit']
            print(f"  {topic:25}: using={using:3}, explicit={explicit:3}")
        if len(good_topics) > 10:
            print(f"  ... и еще {len(good_topics) - 10} тем")
    # выводим проблемы
    if problems:
        print(f"\nНайдено проблем: {len(problems)}")
        for problem in problems[:10]:
            print(f"  {problem}")
        if len(problems) > 10:
            print(f"  ... и еще {len(problems) - 10} проблем")
    else:
        print("\nВсе темы имеют примеры в обоих стилях с достаточным количеством")
    print(f"\nОшибок определения стиля: {style_errors}")
    return topic_stats


if __name__ == "__main__":
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    # извлекаем темы в порядке появления
    topics = extract_unique_topics_in_order(filename)
    # проверяем баланс по стилям
    if topics:
        stats = check_topic_balance_simple(filename, topics)
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    if 10 <= len(topics) <= 30:
        print(f"Количество тем: {len(topics)} (идеально: 10-30)")
    elif len(topics) < 10:
        print(f"Мало тем: {len(topics)} (рекомендуется 10-30)")
        print("Возможно, нужно добавить больше разнообразных тем")
    else:
        print(f"Много тем: {len(topics)} (рекомендуется 10-30)")
        print("Рассмотрите объединение похожих тем")
    print("\nСледующие шаги:")
    print("1. Исправить опечатки в названиях тем (если найдены)")
    print("2. Добавить недостающие примеры для тем без одного из стилей")
    print("3. Сбалансировать количество примеров для каждой темы")