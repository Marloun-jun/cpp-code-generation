from pathlib import Path
from sklearn.model_selection import train_test_split


class CppDatasetPreparer:
# Извлечение и сохранение данных поля code (программы на с++)
    def __init__(self, project_root: str = "3_my_cpp_nn_project/cpp-bpe-tokenizer"):
        self.root = Path(project_root)
        # директории
        self.raw_dir = self.root / 'data' / 'raw'
        self.corpus_dir = self.root / 'data' / 'corpus'
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.input_csv = self.raw_dir / '2_cpp_code_generation_dataset.csv'
        self.corpus_file = self.corpus_dir / 'corpus.txt'
        self.train_file = self.corpus_dir / 'train_code.txt'
        self.val_file = self.corpus_dir / 'val_code.txt'
        self.test_file = self.corpus_dir / 'test_code.txt'
    
    def extract_codes_raw(self):
    # Извлечение данных поля code
        print("="*60)
        print("ЭТАП 2: Подготовка датасета (RAW режим)")
        print("="*60)
        codes = []
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            # читаем весь файл
            content = f.read()
        # разбиваем по строкам, игнорируя пустые и комментарии
        lines = content.strip().split('\n')
        for line in lines[1:]:
            line = line.strip()
            # пропускаем пустые строки и комментарии
            if not line or line.startswith('#'):
                continue
            # находим второе поле code
            parts = []
            current = ''
            in_quotes = False
            escape_next = False
            for char in line:
                if escape_next:
                    current += char
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                    current += char
                elif char == '"':
                    in_quotes = not in_quotes
                    current += char
                elif char == ',' and not in_quotes:
                    parts.append(current)
                    current = ''
                else:
                    current += char
            parts.append(current)  # последнее поле
            # берем второе поле (code)
            if len(parts) >= 2:
                code = parts[1]
                # убираем кавычки в начале и конце
                if code.startswith('"') and code.endswith('"'):
                    code = code[1:-1]
                if code:  # не пустой
                    codes.append(code)
        total = len(codes)
        print(f"\nВсего примеров кода: {total}")
        print(f"Первый пример: {codes[0][:50]}...")
        # сохраняем содержимое поле code в corpus.txt
        print(f"\n Создаем {self.corpus_file}...")
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            for code in codes:
                f.write(code + '\n')
        print(f"   Готов: {total} строк")
        # проверяем, что \n на месте
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            print(f"\nПроверка corpus.txt:")
            print(f"{first_line[:100]}")
            if '\\n' in first_line:
                print(f"\\n присутствует!")
            else:
                print(f"\\n потерян!")
        # разделение содержимого corpus.txt на train, val и test
        print(f"\nРазделение на train, val и test...")
        train_val, test = train_test_split(
            codes, test_size=500, random_state=42, shuffle=True
        )
        train, val = train_test_split(
            train_val, test_size=800, random_state=42, shuffle=True
        )
        print(f"Train: {len(train)} примеров")
        print(f"Val:   {len(val)} примеров")
        print(f"Test:  {len(test)} примеров")
        # сохраняем train
        with open(self.train_file, 'w', encoding='utf-8') as f:
            for code in train:
                f.write(code + '\n')
        # сохраняем val
        with open(self.val_file, 'w', encoding='utf-8') as f:
            for code in val:
                f.write(code + '\n')
        # сохраняем test
        with open(self.test_file, 'w', encoding='utf-8') as f:
            for code in test:
                f.write(code + '\n')
        print(f"\nФайлы сохранены в {self.corpus_dir}")
        print("="*60)
        return codes, train, val, test

def main():
    preparer = CppDatasetPreparer(
        project_root=Path(__file__).parent.parent
    )
    preparer.extract_codes_raw()


if __name__ == "__main__":
    main()