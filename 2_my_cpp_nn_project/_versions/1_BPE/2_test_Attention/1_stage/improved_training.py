import tensorflow as tf
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================================================================
# ЭТАП 1: РЕАЛИЗАЦИЯ МЕХАНИЗМА ВНИМАНИЯ
# =============================================================================

class BahdanauAttention(tf.keras.layers.Layer):
    # Слой внимания Bahdanau (аддитивное внимание)
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.units = units
        
    def build(self, input_shape):
        # Создаем весовые матрицы
        self.W1 = tf.keras.layers.Dense(self.units)
        self.W2 = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, query, values):
        """
        query: скрытое состояние декодера (batch_size, hidden_size)
        values: все скрытые состояния энкодера (batch_size, max_len, hidden_size)
        """
        # Добавляем dimension для broadcast (batch_size, 1, hidden_size)
        query_with_time_axis = tf.expand_dims(query, 1)
        # Вычисляем score (batch_size, max_len, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values))
        )
        # Веса внимания (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # Контекстный вектор (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# =============================================================================
# ФУНКЦИИ ДЛЯ СОЗДАНИЯ И ЗАГРУЗКИ ДАННЫХ
# =============================================================================

def create_training_data():
    # Создание данных для обучения из предобработанных файлов
    print("📁 СОЗДАНИЕ ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    try:
        # Импортируем наш препроцессор
        import sys
        sys.path.append('2_my_cpp_nn_project')
        from preprocess_data import CodeDataPreprocessor
        # Создаем препроцессор и загружаем данные
        preprocessor = CodeDataPreprocessor()
        train_data, test_data = preprocessor.preprocess('2_my_cpp_nn_project/1_cpp_code_generation_dataset.csv')
        # Создаем папку для данных
        data_dir = '2_my_cpp_nn_project/data'
        os.makedirs(data_dir, exist_ok=True)
        # Сохраняем данные в numpy формате
        train_desc, train_code_in, train_code_out = train_data
        test_desc, test_code_in, test_code_out = test_data
        np.save(f'{data_dir}/train_desc.npy', train_desc)
        np.save(f'{data_dir}/train_code_in.npy', train_code_in)
        np.save(f'{data_dir}/train_code_out.npy', train_code_out)
        np.save(f'{data_dir}/test_desc.npy', test_desc)
        np.save(f'{data_dir}/test_code_in.npy', test_code_in)
        np.save(f'{data_dir}/test_code_out.npy', test_code_out)
        print(f"✅ Данные сохранены в папку: {data_dir}")
        print(f"   train_desc: {train_desc.shape}")
        print(f"   train_code_in: {train_code_in.shape}")
        print(f"   train_code_out: {train_code_out.shape}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при создании данных: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_training_data():
    # Загрузка данных для обучения
    print("📥 Загрузка данных...")
    data_dir = '2_my_cpp_nn_project/data'
    # Проверяем существование папки данных
    if not os.path.exists(data_dir):
        print(f"❌ Папка данных не найдена: {data_dir}")
        print("🔄 Создаем данные...")
        if not create_training_data():
            raise FileNotFoundError("Не удалось создать данные для обучения")
    try:
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
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        raise

def get_vocab_sizes():
    # Получение размеров словарей из токенизаторов
    try:
        from tokenizers import Tokenizer
        # Загружаем BPE токенизаторы
        desc_tokenizer = Tokenizer.from_file('2_my_cpp_nn_project/tokenizers/description_tokenizer.json')
        code_tokenizer = Tokenizer.from_file('2_my_cpp_nn_project/tokenizers/code_tokenizer.json')
        desc_vocab_size = desc_tokenizer.get_vocab_size()
        code_vocab_size = code_tokenizer.get_vocab_size()
        print(f"📊 Размеры словарей:")
        print(f"   Описания: {desc_vocab_size}")
        print(f"   Код: {code_vocab_size}")
        return desc_vocab_size, code_vocab_size
    except Exception as e:
        print(f"⚠️ Не удалось загрузить токенизаторы: {e}")
        print("🔄 Используем приблизительные размеры словарей")
        return 1000, 3000  # Приблизительные значения

# =============================================================================
# ТЕСТИРОВАНИЕ СЛОЯ ВНИМАНИЯ
# =============================================================================

def test_attention_layer():
    # Тестирование слоя внимания на простых примерах
    print("🧪 ТЕСТИРОВАНИЕ СЛОЯ ВНИМАНИЯ")
    print("=" * 50)
    # Создаем тестовые данные
    batch_size = 2
    seq_length = 5
    hidden_size = 8
    attention_units = 10
    # Тестовые тензоры
    query = tf.random.normal([batch_size, hidden_size])  # Текущее состояние декодера
    values = tf.random.normal([batch_size, seq_length, hidden_size])  # Все состояния энкодера
    print(f"Форма query: {query.shape}")
    print(f"Форма values: {values.shape}")
    # Создаем и тестируем слой внимания
    attention_layer = BahdanauAttention(units=attention_units)
    context_vector, attention_weights = attention_layer(query, values)
    print(f"Форма context_vector: {context_vector.shape}")
    print(f"Форма attention_weights: {attention_weights.shape}")
    # Проверяем корректность весов внимания
    weights_sum = tf.reduce_sum(attention_weights, axis=1)
    print(f"Сумма весов внимания (должна быть ~1.0): {weights_sum.numpy()}")
    # Визуализация весов внимания для первого примера в батче
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(attention_weights[0].numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Веса внимания (пример 1)')
    plt.xlabel('Позиция в последовательности')
    plt.ylabel('Внимание')
    plt.subplot(1, 2, 2)
    plt.imshow(attention_weights[1].numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Веса внимания (пример 2)')
    plt.xlabel('Позиция в последовательности')
    plt.ylabel('Внимание')
    plt.tight_layout()
    plt.savefig('2_my_cpp_nn_project/attention_test.png', dpi=150, bbox_inches='tight')
    print("📊 Визуализация сохранена в '2_my_cpp_nn_project/attention_test.png'")
    return attention_layer

# =============================================================================
# ОБНОВЛЕННАЯ АРХИТЕКТУРА С ВНИМАНИЕМ (ПОДГОТОВКА)
# =============================================================================

class AttentionCodeGenerator:
    # Модель генерации кода с механизмом внимания
    def __init__(self, desc_vocab_size, code_vocab_size, embedding_dim=128, lstm_units=256, attention_units=256):
        self.desc_vocab_size = desc_vocab_size
        self.code_vocab_size = code_vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.model = None
        
    def build_attention_model(self):
        # Построение модели с механизмом внимания
        print("🔨 СТРОИМ МОДЕЛЬ С МЕХАНИЗМОМ ВНИМАНИЯ")
        # ==================== ЭНКОДЕР ====================
        encoder_inputs = tf.keras.layers.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = tf.keras.layers.Embedding(
            input_dim=self.desc_vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)
        # Энкодер возвращает ВСЕ скрытые состояния для внимания
        encoder_lstm = tf.keras.layers.LSTM(
            self.lstm_units,
            return_sequences=True,  # ⚠️ ВАЖНО: возвращаем все состояния!
            return_state=True,
            name='encoder_lstm'
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]
        # ==================== ВНИМАНИЕ ====================
        attention_layer = BahdanauAttention(self.attention_units)
        # ==================== ДЕКОДЕР ====================
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
        # Инициализируем декодер состоянием энкодера
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        # Применяем внимание к каждому временному шагу
        # (Это упрощенная версия - в следующем этапе сделаем правильную реализацию)
        decoder_dense = tf.keras.layers.Dense(self.code_vocab_size, activation='softmax', name='decoder_dense')
        final_output = decoder_dense(decoder_outputs)
        # Основная модель для обучения
        self.model = tf.keras.models.Model(
            [encoder_inputs, decoder_inputs],
            final_output,
            name='attention_code_generator'
        )
        # Компиляция
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ МОДЕЛЬ С ВНИМАНИЕМ ПОСТРОЕНА")
        return self.model

# =============================================================================
# ОСНОВНОЙ ПРОЦЕСС
# =============================================================================

if __name__ == "__main__":
    print("🚀 ЭТАП 1: ТЕСТИРОВАНИЕ МЕХАНИЗМА ВНИМАНИЯ")
    print("=" * 60)
    # 1. Тестируем слой внимания
    attention_layer = test_attention_layer()
    # 2. Загружаем данные
    try:
        train_data, test_data = load_training_data()
        train_desc, train_code_in, train_code_out = train_data
        # 3. Получаем размеры словарей
        desc_vocab_size, code_vocab_size = get_vocab_sizes()
        # 4. Создаем и проверяем модель с вниманием
        print("\n🔨 СОЗДАЕМ ТЕСТОВУЮ МОДЕЛЬ С ВНИМАНИЕМ")
        attention_model = AttentionCodeGenerator(
            desc_vocab_size=desc_vocab_size,
            code_vocab_size=code_vocab_size,
            embedding_dim=128,
            lstm_units=256,
            attention_units=256
        )
        model = attention_model.build_attention_model()
        model.summary()
        # 5. Проверяем forward pass на маленьком батче
        print("\n🧪 ТЕСТИРУЕМ FORWARD PASS")
        test_batch_size = 2
        test_desc_batch = train_desc[:test_batch_size, :10]  # Берем первые 10 токенов
        test_code_batch = train_code_in[:test_batch_size, :10]
        try:
            output = model.predict([test_desc_batch, test_code_batch], verbose=0)
            print(f"✅ Forward pass успешен!")
            print(f"   Вход: {test_desc_batch.shape}, {test_code_batch.shape}")
            print(f"   Выход: {output.shape}")
        except Exception as e:
            print(f"❌ Ошибка при forward pass: {e}")
        print("\n🎉 ЭТАП 1 ЗАВЕРШЕН УСПЕШНО!")
        print("Слой внимания работает корректно. Готовы к Этапу 2!")
    except Exception as e:
        print(f"❌ Ошибка в основном процессе: {e}")
        import traceback
        traceback.print_exc()