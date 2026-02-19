import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys

# Добавляем путь для импорта функций предобработки
sys.path.append('2_my_cpp_nn_project')
from preprocess_data import load_training_data, load_tokenizers

# =============================================================================
# ЭТАП 1: СОЗДАНИЕ СЛОЯ ВНИМАНИЯ
# =============================================================================

class BahdanauAttention(nn.Module):
    # Слой внимания Bahdanau (аддитивное внимание)
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        # Learnable parameters
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_values = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, query, values):
        """
        query: tensor (batch_size, hidden_size) - текущее состояние декодера
        values: tensor (batch_size, seq_len, hidden_size) - все выходы энкодера
        returns: context_vector, attention_weights
        """
        # Добавляем dimension для broadcast
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        # Вычисляем score (energy)
        score = self.V(torch.tanh(self.W_query(query) + self.W_values(values)))
        # Веса внимания
        attention_weights = torch.softmax(score, dim=1)
        # Контекстный вектор
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

def test_attention_layer():
    # Тестирование слоя внимания
    print("🧪 ЭТАП 1: ТЕСТИРУЕМ СЛОЙ ВНИМАНИЯ")
    print("=" * 50)
    # Параметры теста
    batch_size = 2
    seq_len = 5
    hidden_size = 8
    # Создаем тестовые данные
    query = torch.randn(batch_size, hidden_size)
    values = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Форма query: {query.shape}")
    print(f"Форма values: {values.shape}")
    # Создаем и тестируем слой внимания
    attention = BahdanauAttention(hidden_size)
    context_vector, attention_weights = attention(query, values)
    print(f"Форма context_vector: {context_vector.shape}")
    print(f"Форма attention_weights: {attention_weights.shape}")
    # Проверяем корректность весов
    weights_sum = torch.sum(attention_weights, dim=1)
    print(f"Сумма весов внимания: {weights_sum.detach().numpy()}")
    # Визуализация
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(attention_weights[0].detach().numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Веса внимания (пример 1)')
    plt.xlabel('Позиция')
    plt.ylabel('Внимание')
    plt.subplot(1, 2, 2)
    plt.imshow(attention_weights[1].detach().numpy().T, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Веса внимания (пример 2)')
    plt.xlabel('Позиция')
    plt.ylabel('Внимание')
    plt.tight_layout()
    plt.savefig('2_my_cpp_nn_project/attention_test.png', dpi=150, bbox_inches='tight')
    print("📊 Визуализация сохранена: '2_my_cpp_nn_project/attention_test.png'")
    return attention

# =============================================================================
# ЭТАП 2: МОДИФИКАЦИЯ ЭНКОДЕРА
# =============================================================================

class Encoder(nn.Module):
    # Энкодер с возвратом всех скрытых состояний для внимания
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Слои
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len)
        returns: outputs, (hidden, cell)
        """
        # Встраивание
        embedded = self.dropout(self.embedding(x))
        # LSTM возвращает все скрытые состояния
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

def test_encoder():
    # Тестирование энкодера
    print("\n🧪 ЭТАП 2: ТЕСТИРУЕМ ЭНКОДЕР")
    print("=" * 50)
    # Параметры теста
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    embedding_dim = 128
    hidden_size = 256
    # Тестовые данные
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Создаем энкодер
    encoder = Encoder(vocab_size, embedding_dim, hidden_size)
    outputs, (hidden, cell) = encoder(test_input)
    print(f"Вход: {test_input.shape}")
    print(f"Выходы энкодера: {outputs.shape}")  # Должны быть все состояния!
    print(f"Скрытое состояние: {hidden.shape}")
    print(f"Состояние ячейки: {cell.shape}")
    return encoder

# =============================================================================
# ЭТАП 3: ИНТЕГРАЦИЯ ВНИМАНИЯ В ДЕКОДЕР
# =============================================================================

class DecoderWithAttention(nn.Module):
    # Декодер с механизмом внимания
    def __init__(self, vocab_size, embedding_dim, hidden_size, attention_layer, num_layers=1, dropout=0.1):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # Слои
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = attention_layer
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_outputs, hidden, cell):
        """
        x: (batch_size, 1) - вход декодера (один токен)
        encoder_outputs: (batch_size, seq_len, hidden_size)
        hidden, cell: состояния LSTM
        """
        # Встраивание
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(x))
        # Внимание
        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        # Объединяем вход с контекстным вектором
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)
        # LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # Объединяем выход с контекстом для предсказания
        output = torch.cat([output, context_vector.unsqueeze(1)], dim=2)
        # Предсказание
        prediction = self.fc_out(output)
        prediction = prediction.squeeze(1)
        return prediction, (hidden, cell), attention_weights

class Seq2SeqWithAttention(nn.Module):
    # Полная модель Seq2Seq с механизмом внимания
    def __init__(self, encoder, decoder):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch_size, src_len) - исходные последовательности
        trg: (batch_size, trg_len) - целевые последовательности
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        # Тензор для хранения выходов
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        # Энкодирование
        encoder_outputs, (hidden, cell) = self.encoder(src)
        # Первый вход декодера - <START> токен
        input = trg[:, 0]
        # Декодирование пошагово
        for t in range(1, trg_len):
            output, (hidden, cell), _ = self.decoder(input, encoder_outputs, hidden, cell)
            outputs[:, t] = output
            # Teacher forcing
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

def test_full_model():
    # Тестирование полной модели
    print("\n🧪 ЭТАП 3: ТЕСТИРУЕМ ПОЛНУЮ МОДЕЛЬ")
    print("=" * 50)
    # Параметры
    src_vocab_size = 1000
    trg_vocab_size = 2000
    embedding_dim = 128
    hidden_size = 256
    batch_size = 2
    src_len = 10
    trg_len = 12
    # Тестовые данные
    src = torch.randint(0, src_vocab_size, (batch_size, src_len))
    trg = torch.randint(0, trg_vocab_size, (batch_size, trg_len))
    # Создаем компоненты
    attention = BahdanauAttention(hidden_size)
    encoder = Encoder(src_vocab_size, embedding_dim, hidden_size)
    decoder = DecoderWithAttention(trg_vocab_size, embedding_dim, hidden_size, attention)
    model = Seq2SeqWithAttention(encoder, decoder)
    # Forward pass
    outputs = model(src, trg)
    print(f"Вход (src): {src.shape}")
    print(f"Цель (trg): {trg.shape}")
    print(f"Выход модели: {outputs.shape}")
    return model

# =============================================================================
# ЭТАП 4: ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ
# =============================================================================

class AttentionVisualizer:
    # Класс для визуализации механизма внимания
    def __init__(self, model, desc_tokenizer, code_tokenizer):
        self.model = model
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer
        
    def visualize_attention(self, src_sequence, trg_sequence, save_path=None):
        # Визуализация весов внимания для конкретной последовательности
        self.model.eval()
        with torch.no_grad():
            # Подготовка данных
            src_tensor = torch.LongTensor(src_sequence).unsqueeze(0)
            trg_tensor = torch.LongTensor(trg_sequence).unsqueeze(0)
            # Энкодирование
            encoder_outputs, (hidden, cell) = self.model.encoder(src_tensor)
            # Декодирование с сохранением весов внимания
            trg_len = trg_tensor.shape[1]
            attention_weights = []
            input = trg_tensor[:, 0]
            for t in range(1, trg_len):
                output, (hidden, cell), attn_weights = self.model.decoder(
                    input, encoder_outputs, hidden, cell
                )
                attention_weights.append(attn_weights.squeeze().numpy())
                input = trg_tensor[:, t]
            attention_weights = np.array(attention_weights)
            # Декодируем токены
            src_tokens = self._decode_tokens(src_sequence, self.desc_tokenizer)
            trg_tokens = self._decode_tokens(trg_sequence[1:], self.code_tokenizer)  # Пропускаем <START>
            # Визуализация
            plt.figure(figsize=(12, 8))
            plt.imshow(attention_weights.T, cmap='viridis', aspect='auto')
            plt.colorbar(label='Вес внимания')
            plt.xticks(range(len(trg_tokens)), trg_tokens, rotation=45, ha='right')
            plt.yticks(range(len(src_tokens)), src_tokens)
            plt.xlabel('Сгенерированный код')
            plt.ylabel('Описание')
            plt.title('Матрица внимания')
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📊 Визуализация сохранена: {save_path}")
            else:
                plt.show()
            return attention_weights
    
    def _decode_tokens(self, sequence, tokenizer):
        # Декодирование последовательности токенов
        tokens = []
        for token_id in sequence:
            if token_id == 0:  # PAD
                continue
            token = tokenizer.decode([token_id])
            tokens.append(token)
        return tokens

def test_visualization():
    # Тестирование визуализации
    print("\n🧪 ЭТАП 4: ТЕСТИРУЕМ ВИЗУАЛИЗАЦИЮ")
    print("=" * 50)
    # Создаем тестовые данные для визуализации
    src_seq = [26, 294, 23, 38, 192, 461, 285]  # "function to add"
    trg_seq = [2, 34, 36, 29, 32, 38, 3]  # "<START> print <END>"
    # Создаем простую модель для теста
    attention = BahdanauAttention(64)
    encoder = Encoder(1000, 64, 64)
    decoder = DecoderWithAttention(2000, 64, 64, attention)
    model = Seq2SeqWithAttention(encoder, decoder)
    # Mock токенизаторы
    class MockTokenizer:
        def decode(self, ids):
            return f"token_{ids[0]}"
    desc_tokenizer = MockTokenizer()
    code_tokenizer = MockTokenizer()
    # Визуализация
    visualizer = AttentionVisualizer(model, desc_tokenizer, code_tokenizer)
    visualizer.visualize_attention(
        src_seq, trg_seq, 
        save_path='2_my_cpp_nn_project/attention_visualization_test.png'
    )
    return visualizer

# =============================================================================
# ОСНОВНОЙ ПРОЦЕСС ОБУЧЕНИЯ
# =============================================================================

def main():
    # Основной процесс обучения
    print("🚀 ЗАПУСК ПОЭТАПНОЙ РЕАЛИЗАЦИИ ВНИМАНИЯ")
    print("=" * 60)
    try:
        # Этап 1: Тестируем слой внимания
        attention_layer = test_attention_layer()
        # Этап 2: Тестируем энкодер
        encoder = test_encoder()
        # Этап 3: Тестируем полную модель
        model = test_full_model()
        # Этап 4: Тестируем визуализацию
        visualizer = test_visualization()
        print("\n🎉 ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("✅ Слой внимания работает")
        print("✅ Энкодер возвращает все состояния")
        print("✅ Модель с вниманием собрана")
        print("✅ Визуализация готова")
        print("\n📝 Следующий шаг: обучение на реальных данных!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()