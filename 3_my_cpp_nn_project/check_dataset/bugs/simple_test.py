import csv

# Простой тест чтобы понять проблему
def simple_test():
    filename = "3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv"
    print("Простая проверка строки 10:")
    print("=" * 60)
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # строка 10
    line10 = lines[9]  # Индексация с 0
    print(f"Сырая строка 10: {repr(line10[:200])}")
    # разные способы парсинга
    print(f"\n1. Стандартный csv.reader():")
    reader = csv.reader([line10])
    row = next(reader)
    print(f"   Поле code: {repr(row[1][:100])}")
    print(f"\n2. csv.reader() с escapechar=None:")
    reader = csv.reader([line10], escapechar=None)
    row = next(reader)
    print(f"   Поле code: {repr(row[1][:100])}")
    print(f"\n3. Ручной парсинг:")
    # ищем второе поле в CSV
    if '","' in line10:
        parts = line10.split('","', 2)
        if len(parts) > 1:
            code = parts[1].split('","')[0] if '","' in parts[1] else parts[1]
            print(f"   Поле code: {repr(code[:100])}")
    print(f"\nВывод: если разные методы показывают разное - проблема в парсинге CSV")

if __name__ == "__main__":
    simple_test()
