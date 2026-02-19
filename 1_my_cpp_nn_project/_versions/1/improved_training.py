import tensorflow as tf
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

# Упрощенная архитектура без Attention для стабильности
class ImprovedCodeGenerator:
    def __init__(self, desc_vocab_size, code_vocab_size, embedding_dim=128, lstm_units=256):
        self.desc_vocab_size = desc_vocab_size
        self.code_vocab_size = code_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        
    def build_simple_model(self):
        """Упрощенная модель без Attention"""
        # Энкодер
        encoder_inputs = tf.keras.layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = tf.keras.layers.Embedding(
            input_dim=self.desc_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        
        encoder_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            return_state=True,
            name='encoder_lstm'
        )
        _, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Декодер
        decoder_inputs = tf.keras.layers.Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = tf.keras.layers.Embedding(
            input_dim=self.code_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)
        
        decoder_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            name='decoder_lstm'
        )
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=encoder_states
        )
        
        # Выходной слой
        decoder_dense = tf.keras.layers.Dense(self.code_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Модель
        self.model = tf.keras.models.Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            name='improved_code_generator'
        )
        
        # Компиляция с другим оптимизатором
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train_with_validation(self, train_data, test_data, epochs=100, batch_size=8):
        """Обучение с улучшенными callback'ами"""
        train_desc, train_code_in, train_code_out = train_data
        test_desc, test_code_in, test_code_out = test_data
        
        # Подготовка данных
        train_code_out = train_code_out[:, :-1]
        test_code_out = test_code_out[:, :-1]
        
        # Создаем папку для моделей
        os.makedirs('improved_models', exist_ok=True)
        
        # Улучшенные callback'ы
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5,
                min_lr=0.0001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'improved_models/best_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            tf.keras.callbacks.CSVLogger('improved_models/training_log.csv')
        ]
        
        print("Начало улучшенного обучения...")
        print(f"Размер батча: {batch_size}")
        print(f"Тренировочные данные: {len(train_desc)}")
        print(f"Тестовые данные: {len(test_desc)}")
        
        history = self.model.fit(
            [train_desc, train_code_in[:, :-1]],
            train_code_out,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(
                [test_desc, test_code_in[:, :-1]],
                test_code_out
            ),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, path):
        """Сохранение модели"""
        self.model.save(f'{path}/improved_code_generator.h5')
        print("Улучшенная модель сохранена")

def analyze_training_data():
    """Анализ данных для понимания проблемы"""
    print("Анализ тренировочных данных...")
    
    # Загрузка данных
    train_desc = np.load('data/train_desc.npy')
    train_code_in = np.load('data/train_code_in.npy')
    train_code_out = np.load('data/train_code_out.npy')
    
    # Загрузка токенизаторов
    with open('tokenizers/code_tokenizer.pkl', 'rb') as f:
        code_tokenizer = pickle.load(f)
    
    # Анализ частоты токенов
    all_tokens = train_code_out.flatten()
    unique, counts = np.unique(all_tokens, return_counts=True)
    
    print("Топ-10 самых частых токенов:")
    token_freq = list(zip(unique, counts))
    token_freq.sort(key=lambda x: x[1], reverse=True)
    
    for token_id, freq in token_freq[:10]:
        token = code_tokenizer.index_word.get(token_id, f'<UNK:{token_id}>')
        print(f"  '{token}': {freq} раз ({freq/len(all_tokens)*100:.1f}%)")
    
    # Анализ длины последовательностей
    seq_lengths = np.sum(train_code_out != 0, axis=1)
    print(f"\nСредняя длина последовательности: {np.mean(seq_lengths):.1f}")
    print(f"Максимальная длина: {np.max(seq_lengths)}")
    print(f"Минимальная длина: {np.min(seq_lengths)}")

# Основной процесс
if __name__ == "__main__":
    print("🔄 ЗАПУСК УЛУЧШЕННОГО ОБУЧЕНИЯ")
    print("=" * 50)
    
    # Анализ данных
    analyze_training_data()
    
    # Загрузка данных
    train_desc = np.load('data/train_desc.npy')
    train_code_in = np.load('data/train_code_in.npy')
    train_code_out = np.load('data/train_code_out.npy')
    test_desc = np.load('data/test_desc.npy')
    test_code_in = np.load('data/test_code_in.npy')
    test_code_out = np.load('data/test_code_out.npy')
    
    # Загрузка токенизаторов
    with open('tokenizers/description_tokenizer.pkl', 'rb') as f:
        desc_tokenizer = pickle.load(f)
    with open('tokenizers/code_tokenizer.pkl', 'rb') as f:
        code_tokenizer = pickle.load(f)
    
    desc_vocab_size = len(desc_tokenizer.word_index) + 1
    code_vocab_size = len(code_tokenizer.word_index) + 1
    
    print(f"\nПараметры модели:")
    print(f"Словарь описаний: {desc_vocab_size}")
    print(f"Словарь кода: {code_vocab_size}")
    
    # Создание и обучение улучшенной модели
    improved_model = ImprovedCodeGenerator(
        desc_vocab_size=desc_vocab_size,
        code_vocab_size=code_vocab_size,
        embedding_dim=128,
        lstm_units=256
    )
    
    model = improved_model.build_simple_model()
    model.summary()
    
    # Обучение
    train_data = (train_desc, train_code_in, train_code_out)
    test_data = (test_desc, test_code_in, test_code_out)
    
    history = improved_model.train_with_validation(
        train_data, 
        test_data,
        epochs=150,      # Больше эпох
        batch_size=4     # Меньше батч для стабильности
    )
    
    # Сохранение финальной модели
    improved_model.save_model('improved_models')

    # После сохранения модели добавьте:
    def save_model_safe(self, path):
        """Безопасное сохранение модели"""
        # Сохраняем в формате .keras (рекомендуемый)
        self.model.save(f'{path}/improved_code_generator.keras')
        print("✅ Модель сохранена в формате .keras")
        
        # Также сохраняем веса отдельно
        self.model.save_weights(f'{path}/improved_weights.h5')
        print("✅ Веса сохранены отдельно")
    
    print("✅ УЛУЧШЕННОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    
    # Анализ результатов
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nФинальные метрики:")
    print(f"Точность на обучении: {final_train_acc:.4f}")
    print(f"Точность на валидации: {final_val_acc:.4f}")

    improved_model.save_model_safe('improved_models')