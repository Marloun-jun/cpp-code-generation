import tensorflow as tf
import numpy as np
import pickle
import os
import sys
from sklearn.model_selection import train_test_split

# Альтернативные импорты для совместимости
try:
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Concatenate
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Attention
except ImportError:
    from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Concatenate
    from tensorflow.keras.models import Model
    # Если Attention не импортируется, определим позже

class CodeGeneratorModel:
    def __init__(self, desc_vocab_size, code_vocab_size, embedding_dim=256, lstm_units=512):
        self.desc_vocab_size = desc_vocab_size
        self.code_vocab_size = code_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        
    def build_model(self):
        """Построение модели Seq2Seq с Attention"""
        
        # Энкодер
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(
            input_dim=self.desc_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        
        encoder_lstm = LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            name='encoder_lstm'
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Декодер
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(
            input_dim=self.code_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)
        
        decoder_lstm = LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )
        
        # Механизм внимания
        try:
            attention = Attention(name='attention')([decoder_outputs, encoder_outputs])
        except:
            # Альтернативная реализация внимания
            print("Используем альтернативную реализацию Attention")
            from tensorflow.keras.layers import Dot, Activation
            attention_scores = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
            attention_weights = Activation('softmax')(attention_scores)
            attention = Dot(axes=[2, 1])([attention_weights, encoder_outputs])
        
        decoder_combined_context = Concatenate()([decoder_outputs, attention])
        
        # Выходной слой
        decoder_dense = Dense(self.code_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_combined_context)
        
        # Модель для обучения
        self.model = Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            name='code_generator'
        )
        
        # Компиляция
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, train_data, test_data, epochs=50, batch_size=32):
        """Обучение модели"""
        train_desc, train_code_in, train_code_out = train_data
        test_desc, test_code_in, test_code_out = test_data
        
        # Подготовка данных для sparse_categorical_crossentropy
        train_code_out = train_code_out[:, :-1]
        test_code_out = test_code_out[:, :-1]
        
        # Создаем папку для моделей
        os.makedirs('models', exist_ok=True)
        
        history = self.model.fit(
            [train_desc, train_code_in[:, :-1]],
            train_code_out,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [test_desc, test_code_in[:, :-1]],
                test_code_out
            ),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/code_generator_best.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]
        )
        
        return history
    
    def save_model(self, path):
        """Сохранение модели"""
        self.model.save(f'{path}/code_generator_final.h5')
        print("Модель сохранена")

def run_preprocessing():
    """Запуск предобработки данных"""
    print("Запуск предобработки данных...")
    
    # Добавляем текущую директорию в путь для импорта
    sys.path.append('.')
    
    try:
        # Импортируем модуль предобработки
        from preprocess_data import CodeDataPreprocessor
        
        preprocessor = CodeDataPreprocessor()
        train_data, test_data = preprocessor.preprocess('1_cpp_code_generation_dataset.csv')
        
        # Сохраняем данные в numpy
        os.makedirs('data', exist_ok=True)
        train_desc, train_code_in, train_code_out = train_data
        test_desc, test_code_in, test_code_out = test_data
        
        np.save('data/train_desc.npy', train_desc)
        np.save('data/train_code_in.npy', train_code_in)
        np.save('data/train_code_out.npy', train_code_out)
        np.save('data/test_desc.npy', test_desc)
        np.save('data/test_code_in.npy', test_code_in)
        np.save('data/test_code_out.npy', test_code_out)
        
        print("Данные успешно предобработаны и сохранены!")
        return train_data, test_data
        
    except Exception as e:
        print(f"Ошибка при предобработке: {e}")
        return None, None

def load_preprocessed_data():
    """Загрузка предобработанных данных"""
    try:
        # Загрузка токенизаторов
        with open('tokenizers/description_tokenizer.pkl', 'rb') as f:
            desc_tokenizer = pickle.load(f)
        with open('tokenizers/code_tokenizer.pkl', 'rb') as f:
            code_tokenizer = pickle.load(f)
        
        print(f"Размер словаря описаний: {len(desc_tokenizer.word_index)}")
        print(f"Размер словаря кода: {len(code_tokenizer.word_index)}")
        
        # Пробуем загрузить numpy данные
        try:
            train_desc = np.load('data/train_desc.npy')
            train_code_in = np.load('data/train_code_in.npy')
            train_code_out = np.load('data/train_code_out.npy')
            test_desc = np.load('data/test_desc.npy')
            test_code_in = np.load('data/test_code_in.npy')
            test_code_out = np.load('data/test_code_out.npy')
            
            train_data = (train_desc, train_code_in, train_code_out)
            test_data = (test_desc, test_code_in, test_code_out)
            
            print("Numpy файлы успешно загружены!")
            
        except FileNotFoundError:
            print("Numpy файлы не найдены. Запускаем предобработку...")
            train_data, test_data = run_preprocessing()
            if train_data is None:
                raise Exception("Не удалось выполнить предобработку данных")
        
        return train_data, test_data, desc_tokenizer, code_tokenizer
               
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

# Загрузка данных и создание модели
if __name__ == "__main__":
    print("Загрузка данных...")
    train_data, test_data, desc_tokenizer, code_tokenizer = load_preprocessed_data()
    
    if train_data is None:
        print("Не удалось загрузить данные. Завершение работы.")
        exit(1)
    
    train_desc, train_code_in, train_code_out = train_data
    test_desc, test_code_in, test_code_out = test_data
    
    print(f"Тренировочные данные: {train_desc.shape}")
    print(f"Тестовые данные: {test_desc.shape}")
    
    # Создание модели
    desc_vocab_size = len(desc_tokenizer.word_index) + 1
    code_vocab_size = len(code_tokenizer.word_index) + 1
    
    print(f"Размер словаря описаний: {desc_vocab_size}")
    print(f"Размер словаря кода: {code_vocab_size}")
    
    model_builder = CodeGeneratorModel(
        desc_vocab_size=desc_vocab_size,
        code_vocab_size=code_vocab_size,
        embedding_dim=256,  # Уменьшил для меньшего датасета
        lstm_units=512      # Уменьшил для меньшего датасета
    )
    
    print("Построение модели...")
    model = model_builder.build_model()
    model.summary()
    
    # Обучение
    print("Начало обучения...")
    history = model_builder.train(
        train_data,
        test_data,
        epochs=100,      # Уменьшил количество эпох для теста
        batch_size=4    # Уменьшил batch_size для стабильности
    )
    
    # Сохранение финальной модели
    model_builder.save_model('models')
    
    print("Обучение завершено!")