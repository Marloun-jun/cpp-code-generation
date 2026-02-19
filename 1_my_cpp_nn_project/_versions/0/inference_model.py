import tensorflow as tf
import numpy as np
import pickle
import re
import os

# Все возможные кастомные объекты которые могут быть в модели
custom_objects = {
    'NotEqual': tf.keras.layers.Lambda(lambda x: tf.not_equal(x, 0)),
    'Equal': tf.keras.layers.Lambda(lambda x: tf.equal(x, 0)),
    'Attention': tf.keras.layers.Attention,
    'Add': tf.keras.layers.Add,
    'Concatenate': tf.keras.layers.Concatenate,
}

def load_model_safe(model_path):
    """Безопасная загрузка модели с разными методами"""
    methods = [
        # Метод 1: С кастомными объектами
        lambda: tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects),
        # Метод 2: Без кастомных объектов
        lambda: tf.keras.models.load_model(model_path, compile=False),
        # Метод 3: Попробовать загрузить веса отдельно
        lambda: load_model_weights_only(model_path),
    ]
    
    for i, method in enumerate(methods):
        try:
            print(f"Попытка загрузки модели (метод {i+1})...")
            model = method()
            print(f"✅ Модель успешно загружена методом {i+1}!")
            return model
        except Exception as e:
            print(f"❌ Метод {i+1} не сработал: {e}")
            continue
    
    raise Exception("Все методы загрузки не сработали")

def load_model_weights_only(model_path):
    """Попытка загрузить только архитектуру + веса"""
    # Создаем новую модель с той же архитектурой
    from preprocess_data import CodeDataPreprocessor
    import tensorflow as tf
    
    # Загружаем токенизаторы чтобы узнать размеры словарей
    with open('tokenizers/description_tokenizer.pkl', 'rb') as f:
        desc_tokenizer = pickle.load(f)
    with open('tokenizers/code_tokenizer.pkl', 'rb') as f:
        code_tokenizer = pickle.load(f)
    
    desc_vocab_size = len(desc_tokenizer.word_index) + 1
    code_vocab_size = len(code_tokenizer.word_index) + 1
    
    # Создаем модель с той же архитектурой
    from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
    from tensorflow.keras.models import Model
    
    # Энкодер
    encoder_inputs = Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = Embedding(
        input_dim=desc_vocab_size,
        output_dim=128,
        mask_zero=True,
        name='encoder_embedding'
    )(encoder_inputs)
    
    encoder_lstm = LSTM(
        256,
        return_sequences=True,
        return_state=True,
        name='encoder_lstm'
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    
    # Декодер
    decoder_inputs = Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = Embedding(
        input_dim=code_vocab_size,
        output_dim=128,
        mask_zero=True,
        name='decoder_embedding'
    )(decoder_inputs)
    
    decoder_lstm = LSTM(
        256,
        return_sequences=True,
        return_state=True,
        name='decoder_lstm'
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    
    # Упрощенное внимание
    from tensorflow.keras.layers import Dot, Activation
    attention_scores = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention_weights = Activation('softmax', name='attention')(attention_scores)
    attention_output = Dot(axes=[2, 1])([attention_weights, encoder_outputs])
    
    decoder_combined_context = Concatenate()([decoder_outputs, attention_output])
    
    # Выходной слой
    decoder_dense = Dense(code_vocab_size, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_combined_context)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='code_generator')
    
    # Загружаем веса
    model.load_weights(model_path)
    return model

class SimpleCodeGenerator:
    def __init__(self, model_path, tokenizers_path):
        """Простой генератор кода"""
        try:
            # Загружаем модель
            self.model = load_model_safe(model_path)
            
            # Загрузка токенизаторов
            with open(f'{tokenizers_path}/description_tokenizer.pkl', 'rb') as f:
                self.desc_tokenizer = pickle.load(f)
            with open(f'{tokenizers_path}/code_tokenizer.pkl', 'rb') as f:
                self.code_tokenizer = pickle.load(f)
            
            print("✅ Все компоненты загружены успешно!")
            print(f"Размер словаря описаний: {len(self.desc_tokenizer.word_index)}")
            print(f"Размер словаря кода: {len(self.code_tokenizer.word_index)}")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            raise
    
    def generate_code(self, description, max_length=15):
        """Генерация кода"""
        print(f"\n🎯 Генерация кода для: '{description}'")
        
        # Препроцессинг
        description = str(description).replace('\n', ' [NEWLINE] ')
        description = re.sub(r'\s+', ' ', description).strip()
        
        # Токенизация
        desc_seq = self.desc_tokenizer.texts_to_sequences([description])
        if not desc_seq or not desc_seq[0]:
            return "❌ Не удалось обработать описание"
            
        desc_padded = tf.keras.preprocessing.sequence.pad_sequences(
            desc_seq, maxlen=200, padding='post'
        )
        
        # Токены
        start_token = self.code_tokenizer.word_index.get('<START>', 1)
        end_token = self.code_tokenizer.word_index.get('<END>', 2)
        
        # Генерация
        generated_tokens = []
        current_sequence = np.array([[start_token]])
        
        print("🔍 Процесс генерации: ", end="")
        
        for i in range(max_length):
            try:
                # Предсказание
                predictions = self.model.predict(
                    [desc_padded, current_sequence], 
                    verbose=0, batch_size=1
                )
                
                # Следующий токен
                next_token_probs = predictions[0, -1, :]
                next_token_id = np.argmax(next_token_probs)
                
                # Проверка окончания
                if next_token_id == end_token or next_token_id == 0:
                    print("<END>")
                    break
                    
                # Получаем слово
                next_token = self.code_tokenizer.index_word.get(next_token_id, '<OOV>')
                generated_tokens.append(next_token)
                print(f"{next_token} ", end="")
                
                # Обновляем последовательность
                current_sequence = np.append(current_sequence, [[next_token_id]], axis=1)
                    
            except Exception as e:
                print(f"\n⚠️ Ошибка на шаге {i}: {e}")
                break
        
        print()  # Новая строка
        
        # Постобработка
        if not generated_tokens:
            return "❌ Модель не смогла сгенерировать код"
            
        code = ' '.join(generated_tokens)
        code = code.replace(' [NEWLINE] ', '\n')
        code = code.replace(' [COMMA] ', ',')
        code = re.sub(r'\s+', ' ', code)
        
        return f"📝 Результат:\n{code}"

# Тестирование
if __name__ == "__main__":
    print("🚀 Инициализация генератора кода...")
    print("=" * 60)
    
    # Проверяем существование файлов
    model_files = [
        'models/code_generator_final.h5',
        'models/code_generator_best.h5'
    ]
    
    existing_models = [f for f in model_files if os.path.exists(f)]
    
    if not existing_models:
        print("❌ Модели не найдены! Сначала обучите модель.")
        exit(1)
    
    print(f"📁 Найдены модели: {existing_models}")
    
    try:
        # Используем первую найденную модель
        model_path = existing_models[0]
        generator = SimpleCodeGenerator(model_path, 'tokenizers/')
        
        # Тестовые примеры
        test_descriptions = [
            "function to add two numbers",
            "print hello world",
            "calculate factorial",
            "simple function",
            "return value"
        ]
        
        print("\n" + "=" * 60)
        print("🧪 ТЕСТИРОВАНИЕ ГЕНЕРАЦИИ КОДА")
        print("=" * 60)
        
        for i, desc in enumerate(test_descriptions, 1):
            print(f"\n📋 Пример {i}:")
            result = generator.generate_code(desc)
            print(result)
            print("-" * 40)
        
        print("\n" + "=" * 60)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        print("\n💡 Советы по устранению:")
        print("1. Проверьте что модель была обучена корректно")
        print("2. Попробуйте переобучить модель")
        print("3. Убедитесь что версии TensorFlow совместимы")