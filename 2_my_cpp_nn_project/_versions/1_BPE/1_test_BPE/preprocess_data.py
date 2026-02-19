import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import re
import csv
from sklearn.model_selection import train_test_split

# Прямой импорт через tf.keras
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# Импорт для BPE токенизации
from tokenizers import Tokenizer as BPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class CodeDataPreprocessor:
    def __init__(self, max_seq_length=200, vocab_size=10000, use_bpe=True):
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.use_bpe = use_bpe  # Флаг для использования BPE
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
    
    def _create_bpe_tokenizer(self, texts):
        """Создание BPE токенизатора"""
        tokenizer = BPETokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<PAD>", "<OOV>", "<START>", "<END>"]
        )
        
        # Обучаем на текстах
        tokenizer.train_from_iterator(texts, trainer)
        return tokenizer
    
    def _bpe_texts_to_sequences(self, tokenizer, texts):
        """Преобразование текстов в последовательности с помощью BPE"""
        sequences = []
        for text in texts:
            encoding = tokenizer.encode(text)
            sequences.append(encoding.ids)
        return sequences
    
    def prepare_tokenizers(self, descriptions, codes):
        # Подготовка токенизаторов
        print("Создание токенизаторов...")
        
        if self.use_bpe:
            print("Используем BPE токенизацию...")
            # BPE токенизатор для описаний
            self.description_bpe_tokenizer = self._create_bpe_tokenizer(descriptions)
            # BPE токенизатор для кода
            self.code_bpe_tokenizer = self._create_bpe_tokenizer(codes)
            
            print(f"Размер словаря описаний (BPE): {self.description_bpe_tokenizer.get_vocab_size()}")
            print(f"Размер словаря кода (BPE): {self.code_bpe_tokenizer.get_vocab_size()}")
            
        else:
            # Стандартный токенизатор для описаний
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
        
        if self.use_bpe:
            # Описания с BPE
            desc_sequences = self._bpe_texts_to_sequences(self.description_bpe_tokenizer, descriptions)
            desc_padded = pad_sequences(desc_sequences, maxlen=self.max_seq_length, padding='post')
            
            # Код с BPE и START/END токенами
            code_sequences = self._bpe_texts_to_sequences(self.code_bpe_tokenizer, codes)
            code_input = []
            code_target = []
            
            start_token = self.code_bpe_tokenizer.token_to_id("<START>") or 2
            end_token = self.code_bpe_tokenizer.token_to_id("<END>") or 3
            
            for seq in code_sequences:
                # Пропускаем пустые последовательности
                if len(seq) == 0:
                    continue
                input_seq = [start_token] + seq
                target_seq = seq + [end_token]
                code_input.append(input_seq)
                code_target.append(target_seq)
                
        else:
            # Стандартная обработка
            desc_sequences = self.description_tokenizer.texts_to_sequences(descriptions)
            desc_padded = pad_sequences(desc_sequences, maxlen=self.max_seq_length, padding='post')
            
            code_sequences = self.code_tokenizer.texts_to_sequences(codes)
            code_input = []
            code_target = []
            
            for seq in code_sequences:
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
        # Сохранение токенизаторов
        import os
        os.makedirs(path, exist_ok=True)
        
        if self.use_bpe:
            # Сохраняем BPE токенизаторы
            self.description_bpe_tokenizer.save(f'{path}/description_bpe_tokenizer.json')
            self.code_bpe_tokenizer.save(f'{path}/code_bpe_tokenizer.json')
        else:
            # Сохраняем стандартные токенизаторы
            with open(f'{path}/description_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.description_tokenizer, f)
            with open(f'{path}/code_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.code_tokenizer, f)
                
        print("Токенизаторы сохранены")

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

# Функция для сравнения токенизаций
def compare_tokenization_examples(preprocessor, texts, num_examples=3):
    """Сравнение обычной и BPE токенизации на примерах"""
    print("\n" + "="*50)
    print("СРАВНЕНИЕ ТОКЕНИЗАЦИИ")
    print("="*50)
    
    for i, text in enumerate(texts[:num_examples]):
        print(f"\nПример {i+1}:")
        print(f"Исходный текст: {text}")
        
        if preprocessor.use_bpe:
            # BPE токенизация
            encoding = preprocessor.description_bpe_tokenizer.encode(text)
            tokens = preprocessor.description_bpe_tokenizer.decode(encoding.ids)
            print(f"BPE токены: {encoding.tokens}")
            print(f"BPE IDs: {encoding.ids}")
            print(f"BPE декодирование: {tokens}")
        else:
            # Стандартная токенизация
            sequence = preprocessor.description_tokenizer.texts_to_sequences([text])[0]
            tokens = [preprocessor.description_tokenizer.index_word.get(idx, '?') for idx in sequence]
            print(f"Стандартные токены: {tokens}")
            print(f"Стандартные IDs: {sequence}")
        print("-" * 30)

# Использование
if __name__ == "__main__":
    # Указываем правильный путь к файлу
    csv_file_path = '2_my_cpp_nn_project/1_cpp_code_generation_dataset.csv'
    # Сначала проверим файл
    inspect_csv_file(csv_file_path)
    # Тестируем оба варианта токенизации
    print("\n1. ТЕСТИРУЕМ BPE ТОКЕНИЗАЦИЮ:")
    preprocessor_bpe = CodeDataPreprocessor(use_bpe=True)
    try:
        train_data_bpe, test_data_bpe = preprocessor_bpe.preprocess(csv_file_path)
        preprocessor_bpe.save_tokenizers('tokenizers_bpe')
        # Покажем примеры токенизации
        sample_texts = ["function to add two numbers", "calculate factorial recursively"]
        compare_tokenization_examples(preprocessor_bpe, sample_texts)
        print("BPE предобработка завершена успешно!")
    except Exception as e:
        print(f"Ошибка при BPE предобработке: {e}")
    print("\n2. ТЕСТИРУЕМ СТАНДАРТНУЮ ТОКЕНИЗАЦИЮ:")
    preprocessor_std = CodeDataPreprocessor(use_bpe=False)
    try:
        train_data_std, test_data_std = preprocessor_std.preprocess(csv_file_path)
        preprocessor_std.save_tokenizers('tokenizers_std')
        # Покажем примеры токенизации
        compare_tokenization_examples(preprocessor_std, sample_texts)
        print("Стандартная предобработка завершена успешно!")
    except Exception as e:
        print(f"Ошибка при стандартной предобработке: {e}")