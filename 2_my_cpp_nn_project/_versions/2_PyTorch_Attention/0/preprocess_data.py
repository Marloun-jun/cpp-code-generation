import pandas as pd
import numpy as np
import pickle
import re
import csv
import os
from sklearn.model_selection import train_test_split

# Импорт для BPE токенизации
from tokenizers import Tokenizer as BPETokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class CodeDataPreprocessor:
    def __init__(self, max_seq_length=200, vocab_size=10000):
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        self.description_tokenizer = None
        self.code_tokenizer = None
        
    def load_data(self, csv_path):
        """Загрузка данных из CSV с обработкой ошибок"""
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
        """Ручная загрузка CSV для проблемных файлов"""
        descriptions = []
        codes = []
        try:
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
        except Exception as e:
            print(f"Ошибка при открытии файла: {e}")
            
        print(f"Успешно загружено {len(descriptions)} примеров")
        return descriptions, codes
    
    def clean_text(self, text):
        """Очистка текста"""
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
    
    def _pad_sequences(self, sequences, maxlen, padding='post', value=0):
        """Аналог pad_sequences из Keras для PyTorch"""
        padded_sequences = []
        for seq in sequences:
            if len(seq) < maxlen:
                if padding == 'post':
                    padded_seq = seq + [value] * (maxlen - len(seq))
                else:  # pre-padding
                    padded_seq = [value] * (maxlen - len(seq)) + seq
            else:
                padded_seq = seq[:maxlen]
            padded_sequences.append(padded_seq)
        return np.array(padded_sequences)
    
    def prepare_tokenizers(self, descriptions, codes):
        """Подготовка BPE токенизаторов"""
        print("Создание BPE токенизаторов...")
        
        # BPE токенизатор для описаний
        self.description_tokenizer = self._create_bpe_tokenizer(descriptions)
        # BPE токенизатор для кода
        self.code_tokenizer = self._create_bpe_tokenizer(codes)
        
        print(f"Размер словаря описаний: {self.description_tokenizer.get_vocab_size()}")
        print(f"Размер словаря кода: {self.code_tokenizer.get_vocab_size()}")
    
    def texts_to_sequences(self, descriptions, codes):
        """Преобразование текстов в последовательности"""
        print("Преобразование в последовательности...")
        
        # Описания с BPE
        desc_sequences = self._bpe_texts_to_sequences(self.description_tokenizer, descriptions)
        desc_padded = self._pad_sequences(desc_sequences, maxlen=self.max_seq_length, padding='post')
        
        # Код с BPE и START/END токенами
        code_sequences = self._bpe_texts_to_sequences(self.code_tokenizer, codes)
        code_input = []
        code_target = []
        
        start_token = self.code_tokenizer.token_to_id("<START>") or 2
        end_token = self.code_tokenizer.token_to_id("<END>") or 3
        
        for seq in code_sequences:
            # Пропускаем пустые последовательности
            if len(seq) == 0:
                continue
            input_seq = [start_token] + seq
            target_seq = seq + [end_token]
            code_input.append(input_seq)
            code_target.append(target_seq)
        
        if not code_input:
            raise ValueError("Нет валидных последовательностей кода после обработки")
            
        code_input_padded = self._pad_sequences(code_input, maxlen=self.max_seq_length, padding='post')
        code_target_padded = self._pad_sequences(code_target, maxlen=self.max_seq_length, padding='post')
        
        return desc_padded, code_input_padded, code_target_padded
    
    def preprocess(self, csv_path, test_size=0.2):
        """Полный процесс предобработки"""
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
        
        # Сохраняем данные в numpy формате для PyTorch
        data_dir = '2_my_cpp_nn_project/data'
        os.makedirs(data_dir, exist_ok=True)
        
        np.save(f'{data_dir}/train_desc.npy', train_desc_seq)
        np.save(f'{data_dir}/train_code_in.npy', train_code_in)
        np.save(f'{data_dir}/train_code_out.npy', train_code_out)
        np.save(f'{data_dir}/test_desc.npy', test_desc_seq)
        np.save(f'{data_dir}/test_code_in.npy', test_code_in)
        np.save(f'{data_dir}/test_code_out.npy', test_code_out)
        
        print(f"✅ Данные сохранены в: {data_dir}")
        
        return (train_desc_seq, train_code_in, train_code_out), \
               (test_desc_seq, test_code_in, test_code_out)
    
    def save_tokenizers(self, path):
        """Сохранение BPE токенизаторов в указанную папку"""
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем BPE токенизаторы в JSON формате
        self.description_tokenizer.save(f'{path}/description_tokenizer.json')
        self.code_tokenizer.save(f'{path}/code_tokenizer.json')
                
        print(f"Токенизаторы сохранены в папку: {path}")

    def show_tokenization_examples(self, texts, num_examples=3):
        """Показать примеры BPE токенизации"""
        print("\n" + "="*50)
        print("ПРИМЕРЫ BPE ТОКЕНИЗАЦИИ")
        print("="*50)
        
        for i, text in enumerate(texts[:num_examples]):
            print(f"\nПример {i+1}:")
            print(f"Исходный текст: {text}")
            
            # BPE токенизация
            encoding = self.description_tokenizer.encode(text)
            tokens = self.description_tokenizer.decode(encoding.ids)
            print(f"BPE токены: {encoding.tokens}")
            print(f"BPE IDs: {encoding.ids}")
            print(f"BPE декодирование: {tokens}")
            print("-" * 30)

    def get_vocab_sizes(self):
        """Получение размеров словарей"""
        return (self.description_tokenizer.get_vocab_size(), 
                self.code_tokenizer.get_vocab_size())

# Функции для загрузки данных в PyTorch
def load_training_data():
    """Загрузка данных для PyTorch"""
    data_dir = '2_my_cpp_nn_project/data'
    
    train_desc = np.load(f'{data_dir}/train_desc.npy')
    train_code_in = np.load(f'{data_dir}/train_code_in.npy')
    train_code_out = np.load(f'{data_dir}/train_code_out.npy')
    test_desc = np.load(f'{data_dir}/test_desc.npy')
    test_code_in = np.load(f'{data_dir}/test_code_in.npy')
    test_code_out = np.load(f'{data_dir}/test_code_out.npy')
    
    print(f"✅ Данные загружены:")
    print(f"   Тренировочные: {len(train_desc)} примеров")
    print(f"   Тестовые: {len(test_desc)} примеров")
    
    return (train_desc, train_code_in, train_code_out), \
           (test_desc, test_code_in, test_code_out)

def load_tokenizers():
    """Загрузка токенизаторов для PyTorch"""
    from tokenizers import Tokenizer
    
    desc_tokenizer = Tokenizer.from_file('2_my_cpp_nn_project/tokenizers/description_tokenizer.json')
    code_tokenizer = Tokenizer.from_file('2_my_cpp_nn_project/tokenizers/code_tokenizer.json')
    
    return desc_tokenizer, code_tokenizer

# Использование
if __name__ == "__main__":
    # Указываем правильный путь к файлу данных
    csv_file_path = '2_my_cpp_nn_project/1_cpp_code_generation_dataset.csv'
    
    # Указываем путь для сохранения токенизаторов
    tokenizers_path = '2_my_cpp_nn_project/tokenizers'
    
    # Создаем препроцессор с BPE
    preprocessor = CodeDataPreprocessor()
    
    try:
        # Запускаем предобработку
        train_data, test_data = preprocessor.preprocess(csv_file_path)
        
        # Сохраняем токенизаторы в папку 2_my_cpp_nn_project/tokenizers
        preprocessor.save_tokenizers(tokenizers_path)
        
        # Покажем примеры токенизации
        sample_texts = ["function to add two numbers", "calculate factorial recursively", "print hello world"]
        preprocessor.show_tokenization_examples(sample_texts)
        
        # Получаем размеры словарей
        desc_vocab_size, code_vocab_size = preprocessor.get_vocab_sizes()
        print(f"\n📊 Размеры словарей для PyTorch:")
        print(f"   Описания: {desc_vocab_size}")
        print(f"   Код: {code_vocab_size}")
        
        print("\n🎉 PyTorch предобработка завершена успешно!")
        
        # Выведем информацию о путях
        print(f"\n📁 Файлы сохранены в:")
        print(f"   - Данные: {csv_file_path}")
        print(f"   - Токенизаторы: {tokenizers_path}/")
        print(f"   - NumPy данные: 2_my_cpp_nn_project/data/")
        
    except Exception as e:
        print(f"❌ Ошибка при предобработке: {e}")
        import traceback
        traceback.print_exc()