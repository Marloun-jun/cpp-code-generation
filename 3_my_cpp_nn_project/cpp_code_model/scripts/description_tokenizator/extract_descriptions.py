#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# extract_descriptions.py - Извлечение описаний из CSV для обучения токенизатора
# ======================================================================
#
# @file extract_descriptions.py
# @brief Извлечение полей description из CSV датасета для обучения русского BPE
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль извлекает только поля description из CSV датасета
#          для последующего обучения русского BPE токенизатора.
#
#          **Основные возможности:**
#
#          1. **Извлечение описаний**
#             - Парсинг CSV с ручным поиском кавычек
#             - Сохранение только поля description
#             - Пропуск заголовков и комментариев
#
#          2. **Фильтрация данных**
#             - Пропуск строк с комментариями (начинаются с #)
#             - Пропуск пустых строк
#             - Пропуск заголовка CSV
#
#          3. **Обработка ошибок**
#             - Проверка существования файла
#             - Обработка исключений при парсинге
#             - Вывод предупреждений
#
#          4. **Статистика**
#             - Количество извлечённых описаний
#             - Средняя длина описаний
#             - Минимальная и максимальная длина
#
# @usage
#     python scripts/description_tokenizator/extract_descriptions.py
#
# @example
#     # Стандартный запуск
#     python extract_descriptions.py
#
# ======================================================================

from pathlib import Path

# ======================================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================================

# Правильные пути: поднимаемся на 3 уровня до cpp_code_model
PROJECT_ROOT = Path(__file__).parent.parent.parent
csv_path = PROJECT_ROOT / "data" / "raw" / "2_cpp_code_generation_dataset.csv"
output_path = PROJECT_ROOT / "data" / "rus_descriptions.txt"

# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main():
    """Главная функция извлечения описаний."""
    
    print("=" * 60)
    print("ИЗВЛЕЧЕНИЕ ОПИСАНИЙ ИЗ CSV")
    print("=" * 60)
    print(f"Чтение CSV:    {csv_path}")
    print(f"Выходной файл: {output_path}")
    print()
    
    # Проверка существования
    if not csv_path.exists():
        print(f"Файл не найден: {csv_path}!")
        print("Убедитесь, что файл находится в data/raw/")
        exit(1)
    
    descriptions = []
    skipped = 0
    
    # Читаем построчно, пропуская комментарии
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Пропускаем пустые строки
            if not line:
                skipped += 1
                continue
            
            # Пропускаем строки, начинающиеся с #
            if line.startswith('#'):
                continue
            
            # Пропускаем первую строку (заголовок)
            if line.startswith('description'):
                continue
            
            # Теперь парсим CSV строку
            try:
                # Находим description (первое поле в кавычках)
                if line.startswith('"'):
                    # Находим закрывающую кавычку description
                    desc_end = line.find('"', 1)
                    if desc_end != -1:
                        description = line[1:desc_end]
                        if description:
                            descriptions.append(description)
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"Ошибка в строке {line_num}: {e}!")
                skipped += 1
    
    print(f"Извлечено {len(descriptions)} описаний")
    if skipped > 0:
        print(f"Пропущено строк: {skipped}!")
    
    # Сохраняем в текстовый файл
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for desc in descriptions:
            f.write(desc + '\n')
    
    print(f"Сохранено в: {output_path}")
    
    # Статистика
    lengths = [len(d) for d in descriptions]
    if lengths:
        print(f"\nСтатистика:")
        print(f"- Средняя длина: {sum(lengths) / len(lengths):.1f} символов")
        print(f"- Мин:           {min(lengths)}")
        print(f"- Макс:          {max(lengths)}")
        
        # Дополнительная статистика
        print(f"\nРаспределение длин:")
        ranges = [(0, 50), (51, 100), (101, 200), (201, 500), (501, 1000)]
        for low, high in ranges:
            count = sum(1 for l in lengths if low <= l <= high)
            if count > 0:
                print(f"   {low}-{high}: {count} ({count / len(lengths) * 100:.1f}%)")
    else:
        print("\nНет данных для статистики!")
    
    print("\n" + "=" * 60)
    print("ИЗВЛЕЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"\nТеперь можно запустить train_rus_tokenizer.py для обучения")
    print(f" на извлечённых описаниях.")


if __name__ == "__main__":
    main()