import tensorflow as tf
import numpy as np
import pickle
import re

# Добавляем кастомные объекты для загрузки модели
custom_objects = {
    'NotEqual': tf.keras.layers.Lambda(lambda x: tf.not_equal(x, 0)),
    'Equal': tf.keras.layers.Lambda(lambda x: tf.equal(x, 0)),
    'Add': tf.keras.layers.Add,
    'Concatenate': tf.keras.layers.Concatenate,
}

class ImprovedCodeGeneratorInference:
    def __init__(self, model_path, tokenizers_path):
        try:
            self.model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=custom_objects
            )
            print("✅ Улучшенная модель загружена!")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            print("Пробуем альтернативный метод...")
            self.model = self._load_model_alternative(model_path)
        
        with open(f'{tokenizers_path}/description_tokenizer.pkl', 'rb') as f:
            self.desc_tokenizer = pickle.load(f)
        with open(f'{tokenizers_path}/code_tokenizer.pkl', 'rb') as f:
            self.code_tokenizer = pickle.load(f)
    
    def _load_model_alternative(self, model_path):
        # Альтернативный метод загрузки модели
        print("Создаем новую модель и загружаем веса...")
        # Загружаем токенизаторы чтобы узнать размеры словарей
        with open('1_my_cpp_nn_project/tokenizers/description_tokenizer.pkl', 'rb') as f:
            desc_tokenizer = pickle.load(f)
        with open('1_my_cpp_nn_project/tokenizers/code_tokenizer.pkl', 'rb') as f:
            code_tokenizer = pickle.load(f)
        desc_vocab_size = len(desc_tokenizer.word_index) + 1
        code_vocab_size = len(code_tokenizer.word_index) + 1
        # Создаем упрощенную модель
        from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
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
            return_state=True,
            name='encoder_lstm'
        )
        _, state_h, state_c = encoder_lstm(encoder_embedding)
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
        decoder_outputs, _, _ = decoder_lstm(
            decoder_embedding,
            initial_state=[state_h, state_c]
        )
        # Выходной слой
        decoder_dense = Dense(code_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.load_weights(model_path)
        return model
    
    def generate_code(self, description, max_length=25):
        print(f"\n🎯 Генерация для: '{description}'")
        # Препроцессинг
        desc_seq = self.desc_tokenizer.texts_to_sequences([description])
        if not desc_seq or not desc_seq[0]:
            return "❌ Не удалось обработать описание"
        desc_padded = tf.keras.preprocessing.sequence.pad_sequences(
            desc_seq, maxlen=200, padding='post'
        )
        start_token = self.code_tokenizer.word_index.get('<START>', 1)
        end_token = self.code_tokenizer.word_index.get('<END>', 2)
        generated_tokens = []
        current_sequence = np.array([[start_token]])
        print("🔍 Генерация: ", end="")
        for i in range(max_length):
            try:
                predictions = self.model.predict(
                    [desc_padded, current_sequence], 
                    verbose=0,
                    batch_size=1
                )
                next_token_probs = predictions[0, -1, :]
                next_token_id = np.argmax(next_token_probs)
                # Показываем топ-3 варианта для отладки (первые 3 итерации)
                if i < 3:
                    top_3_indices = np.argsort(next_token_probs)[-3:][::-1]
                    top_3_tokens = [
                        self.code_tokenizer.index_word.get(idx, f'<{idx}>') 
                        for idx in top_3_indices
                    ]
                    top_3_probs = [next_token_probs[idx] for idx in top_3_indices]
                    print(f"\n[Шаг {i}] Топ-3: {list(zip(top_3_tokens, top_3_probs))}")
                if next_token_id == end_token or next_token_id == 0:
                    print("<END>")
                    break
                next_token = self.code_tokenizer.index_word.get(next_token_id, '<OOV>')
                generated_tokens.append(next_token)
                print(f"{next_token} ", end="")
                current_sequence = np.append(current_sequence, [[next_token_id]], axis=1)
            except Exception as e:
                print(f"\n⚠️ Ошибка на шаге {i}: {e}")
                break
        print()
        if not generated_tokens:
            return "❌ Не сгенерировано"
        code = ' '.join(generated_tokens)
        code = code.replace(' [NEWLINE] ', '\n')
        code = code.replace(' [COMMA] ', ',')
        code = re.sub(r'\s+', ' ', code)
        return f"📝 Результат:\n{code}"

    def test_model_output(self, description):
        # Тестирование выходов модели
        print(f"\n🧪 ТЕСТИРОВАНИЕ МОДЕЛИ: '{description}'")
        desc_seq = self.desc_tokenizer.texts_to_sequences([description])
        desc_padded = tf.keras.preprocessing.sequence.pad_sequences(
            desc_seq, maxlen=200, padding='post'
        )
        start_token = self.code_tokenizer.word_index.get('<START>', 1)
        current_sequence = np.array([[start_token]])
        # Первое предсказание
        predictions = self.model.predict([desc_padded, current_sequence], verbose=0)
        first_token_probs = predictions[0, -1, :]
        # Топ-10 наиболее вероятных токенов
        top_10_indices = np.argsort(first_token_probs)[-10:][::-1]
        print("Топ-10 наиболее вероятных первых токенов:")
        for i, idx in enumerate(top_10_indices):
            token = self.code_tokenizer.index_word.get(idx, f'<UNK:{idx}>')
            prob = first_token_probs[idx]
            print(f"  {i+1:2d}. '{token}' ({prob:.4f})")

# Тестирование
if __name__ == "__main__":
    print("🚀 ТЕСТИРОВАНИЕ УЛУЧШЕННОЙ МОДЕЛИ")
    print("=" * 50)
    try:
        generator = ImprovedCodeGeneratorInference(
            '1_my_cpp_nn_project/improved_models/improved_code_generator.h5',
            '1_my_cpp_nn_project/tokenizers/'
        )
        # Сначала протестируем выходы модели
        test_description = "function to add two numbers"
        generator.test_model_output(test_description)
        print("\n" + "=" * 50)
        print("🧪 ГЕНЕРАЦИЯ КОДА")
        print("=" * 50)
        test_cases = [
            "function to add two numbers",
            "print hello world", 
            "calculate factorial",
            "simple function that returns value",
            "create vector of integers"
        ]
        for i, desc in enumerate(test_cases, 1):
            print(f"\n📋 Пример {i}:")
            result = generator.generate_code(desc)
            print(result)
            print("-" * 50)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")