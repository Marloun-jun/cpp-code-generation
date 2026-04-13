import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import re
import csv
import os
from sklearn.model_selection import train_test_split

# Прямой импорт через tf.keras
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

class CodeDataPreprocessor:
    def __init__(self, max_seq_length=200, vocab_size=10000):
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.description_tokenizer = None
        self.code_tokenizer = None
        
    def load_data(self, csv_path):
        # Загрузка данных из CSV с обработкой ошибок
        print("Загрузка данных...")
        try:
            # Пробуем разные разделители и обработку кавычек
            df = pd.read_csv(csv_path, quotechar='"', escapechar='\\', on_bad_lines='skip')
            print(f"Загружено {len(df)} примеров")
            return df['description'].tolist(), df['code'].tolist()
        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            print("Пробуем альтернативный метод загрузки...")
            return self._load_data_manual(csv_path)
    
    def _load_data_manual(self, csv_path):
        # Ручная загрузка CSV для проблемных файлов
        descriptions = []
        codes = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)  # Пропускаем заголовки
            for i, row in enumerate(reader):
                try:
                    if len(row) >= 2:
                        description = row[0] if len(row) > 0 else ""
                        code = row[1] if len(row) > 1 else ""
                        descriptions.append(description)
                        codes.append(code)
                    else:
                        print(f"Пропущена строка {i+1}: недостаточно полей")
                except Exception as e:
                    print(f"Ошибка в строке {i+1}: {e}")
                    continue
        print(f"Успешно загружено {len(descriptions)} примеров")
        return descriptions, codes
    
    def clean_text(self, text):
        # Очистка текста
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        # Экранируем запятые в коде чтобы они не ломали CSV
        text = text.replace(',', ' [COMMA] ')
        text = text.replace('\n', ' [NEWLINE] ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def prepare_tokenizers(self, descriptions, codes):
        # Подготовка токенизаторов
        print("Создание токенизаторов...")
        # Токенизатор для описаний
        self.description_tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            filters=''
        )
        self.description_tokenizer.fit_on_texts(descriptions)
        # Токенизатор для кода
        self.code_tokenizer = Tokenizer(
            num_words=self.vocab_size,
            oov_token='<OOV>',
            filters=''
        )
        self.code_tokenizer.fit_on_texts(codes)
        # Добавляем специальные токены
        if '<START>' not in self.code_tokenizer.word_index:
            self.code_tokenizer.word_index['<START>'] = len(self.code_tokenizer.word_index) + 1
        if '<END>' not in self.code_tokenizer.word_index:
            self.code_tokenizer.word_index['<END>'] = len(self.code_tokenizer.word_index) + 1
        print(f"Размер словаря описаний: {len(self.description_tokenizer.word_index)}")
        print(f"Размер словаря кода: {len(self.code_tokenizer.word_index)}")
    
    def texts_to_sequences(self, descriptions, codes):
        # Преобразование текстов в последовательности
        print("Преобразование в последовательности...")
        # Описания
        desc_sequences = self.description_tokenizer.texts_to_sequences(descriptions)
        desc_padded = pad_sequences(desc_sequences, maxlen=self.max_seq_length, padding='post')
        # Код с START/END токенами
        code_sequences = self.code_tokenizer.texts_to_sequences(codes)
        code_input = []
        code_target = []
        for seq in code_sequences:
            # Пропускаем пустые последовательности
            if len(seq) == 0:
                continue
            start_token = self.code_tokenizer.word_index.get('<START>', 1)
            end_token = self.code_tokenizer.word_index.get('<END>', 2)
            input_seq = [start_token] + seq
            target_seq = seq + [end_token]
            code_input.append(input_seq)
            code_target.append(target_seq)
        if not code_input:  # Если все последовательности пустые
            raise ValueError("Нет валидных последовательностей кода после обработки")
        code_input_padded = pad_sequences(code_input, maxlen=self.max_seq_length, padding='post')
        code_target_padded = pad_sequences(code_target, maxlen=self.max_seq_length, padding='post')
        return desc_padded, code_input_padded, code_target_padded
    
    def preprocess(self, csv_path, test_size=0.2):
        # Полный процесс предобработки
        descriptions, codes = self.load_data(csv_path)
        if not descriptions or not codes:
            raise ValueError("Не удалось загрузить данные из CSV файла")
        print("Очистка данных...")
        descriptions = [self.clean_text(desc) for desc in descriptions]
        codes = [self.clean_text(code) for code in codes]
        # Фильтруем пустые примеры
        filtered_desc = []
        filtered_codes = []
        for desc, code in zip(descriptions, codes):
            if desc and code and desc.strip() and code.strip():
                filtered_desc.append(desc)
                filtered_codes.append(code)
        print(f"После фильтрации: {len(filtered_desc)} примеров")
        # Разделение на train/test
        train_desc, test_desc, train_code, test_code = train_test_split(
            filtered_desc, filtered_codes, test_size=test_size, random_state=42
        )
        # Подготовка токенизаторов
        self.prepare_tokenizers(train_desc, train_code)
        # Преобразование в последовательности
        train_desc_seq, train_code_in, train_code_out = self.texts_to_sequences(train_desc, train_code)
        test_desc_seq, test_code_in, test_code_out = self.texts_to_sequences(test_desc, test_code)
        print(f"Тренировочные данные: {len(train_desc_seq)} примеров")
        print(f"Тестовые данные: {len(test_desc_seq)} примеров")
        return (train_desc_seq, train_code_in, train_code_out), \
               (test_desc_seq, test_code_in, test_code_out)
    
    def save_tokenizers(self, path):
        # Сохранение токенизаторов в папку 1_my_cpp_nn_project
        # Создаем путь к папке 1_my_cpp_nn_project/tokenizers
        target_path = '1_my_cpp_nn_project/tokenizers'
        os.makedirs(target_path, exist_ok=True)
        # Сохраняем токенизаторы
        with open(f'{target_path}/description_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.description_tokenizer, f)
        with open(f'{target_path}/code_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.code_tokenizer, f)
        print(f"✅ Токенизаторы сохранены в: {target_path}/")

# Дополнительная функция для проверки CSV файла
def inspect_csv_file(csv_path):
    # Проверка структуры CSV файла
    print(f"Проверка файла: {csv_path}")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                print(f"Строка {i+1}: {len(line.split(','))} полей")
                if i >= 15:  # Покажем первые 15 строк
                    break
    except Exception as e:
        print(f"Ошибка при проверке файла: {e}")

# Использование
if __name__ == "__main__":
    # Сначала проверим файл
    inspect_csv_file('1_my_cpp_nn_project/1_cpp_code_generation_dataset.csv')
    # Затем запустим предобработку
    preprocessor = CodeDataPreprocessor()
    try:
        train_data, test_data = preprocessor.preprocess('1_my_cpp_nn_project/1_cpp_code_generation_dataset.csv')
        preprocessor.save_tokenizers('tokenizers')
        print("Предобработка завершена успешно!")
    except Exception as e:
        print(f"Ошибка при предобработке: {e}")