import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys
import time
from datetime import datetime

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
# ЭТАП 5: ОБУЧЕНИЕ И СРАВНЕНИЕ
# =============================================================================

class SimpleSeq2Seq(nn.Module):
    # Простая модель Seq2Seq без внимания для сравнения
    def __init__(self, src_vocab_size, trg_vocab_size, embedding_dim, hidden_size):
        super(SimpleSeq2Seq, self).__init__()
        
        self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_dim)
        self.fc = nn.Linear(hidden_size, trg_vocab_size)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        
        # Энкодирование
        src_embedded = self.src_embedding(src)
        _, (hidden, cell) = self.encoder(src_embedded)
        
        # Декодирование
        trg_embedded = self.trg_embedding(trg)
        outputs, _ = self.decoder(trg_embedded, (hidden, cell))
        predictions = self.fc(outputs)
        
        return predictions

class TrainingManager:
    # Менеджер для обучения и сравнения моделей
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {}
        
    def train_model(self, model, train_loader, val_loader, model_name, epochs=10, lr=0.001):
        # Обучение одной модели
        print(f"\n🎯 ОБУЧАЕМ МОДЕЛЬ: {model_name}")
        print("=" * 50)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем PAD токены
        train_losses = []
        val_losses = []
        val_accuracies = []
        for epoch in range(epochs):
            # Обучение
            model.train()
            train_loss = 0
            for batch_idx, (src, trg_in, trg_out) in enumerate(train_loader):
                src, trg_in, trg_out = src.to(self.device), trg_in.to(self.device), trg_out.to(self.device)
                optimizer.zero_grad()
                if isinstance(model, Seq2SeqWithAttention):
                    output = model(src, trg_in)
                else:
                    output = model(src, trg_in)
                loss = criterion(output.reshape(-1, output.shape[-1]), trg_out.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            # Валидация
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for src, trg_in, trg_out in val_loader:
                    src, trg_in, trg_out = src.to(self.device), trg_in.to(self.device), trg_out.to(self.device)
                    if isinstance(model, Seq2SeqWithAttention):
                        output = model(src, trg_in, teacher_forcing_ratio=0)
                    else:
                        output = model(src, trg_in)
                    loss = criterion(output.reshape(-1, output.shape[-1]), trg_out.reshape(-1))
                    val_loss += loss.item()
                    # Точность (игнорируем PAD токены)
                    predictions = output.argmax(-1)
                    mask = trg_out != 0
                    correct += ((predictions == trg_out) & mask).sum().item()
                    total += mask.sum().item()
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            accuracy = correct / total if total > 0 else 0
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(accuracy)
            print(f'Эпоха {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')
        # Сохраняем результаты
        self.results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': val_accuracies[-1] if val_accuracies else 0
        }
        return model
    
    def compare_models(self):
        # Сравнение результатов моделей
        print("\n📊 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 50)
        # Таблица результатов
        print(f"{'Модель':<20} {'Финальная точность':<20} {'Лучшая точность':<20}")
        print("-" * 60)
        for model_name, result in self.results.items():
            final_acc = result['final_accuracy']
            best_acc = max(result['val_accuracies'])
            print(f"{model_name:<20} {final_acc:.4f}{'':<15} {best_acc:.4f}{'':<15}")
        # Графики
        plt.figure(figsize=(15, 5))
        # График потерь
        plt.subplot(1, 3, 1)
        for model_name, result in self.results.items():
            plt.plot(result['train_losses'], label=f'{model_name} (train)')
            plt.plot(result['val_losses'], label=f'{model_name} (val)', linestyle='--')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.title('Сравнение потерь')
        plt.legend()
        plt.grid(True)
        # График точности
        plt.subplot(1, 3, 2)
        for model_name, result in self.results.items():
            plt.plot(result['val_accuracies'], label=model_name)
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.title('Сравнение точности')
        plt.legend()
        plt.grid(True)
        # Bar chart финальной точности
        plt.subplot(1, 3, 3)
        model_names = list(self.results.keys())
        final_accuracies = [self.results[name]['final_accuracy'] for name in model_names]
        bars = plt.bar(model_names, final_accuracies, color=['skyblue', 'lightcoral'])
        plt.ylabel('Финальная точность')
        plt.title('Финальная точность моделей')
        plt.xticks(rotation=45)
        # Добавляем значения на столбцы
        for bar, accuracy in zip(bars, final_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{accuracy:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('2_my_cpp_nn_project/model_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Графики сохранены: '2_my_cpp_nn_project/model_comparison.png'")

def prepare_data_for_training():
    # Подготовка данных для обучения
    print("📥 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    # Загружаем данные
    (train_desc, train_code_in, train_code_out), (test_desc, test_code_in, test_code_out) = load_training_data()
    # Загружаем токенизаторы для получения размеров словарей
    desc_tokenizer, code_tokenizer = load_tokenizers()
    desc_vocab_size = desc_tokenizer.get_vocab_size()
    code_vocab_size = code_tokenizer.get_vocab_size()
    print(f"Размеры словарей: описания={desc_vocab_size}, код={code_vocab_size}")
    print(f"Данные: train={len(train_desc)}, test={len(test_desc)}")
    # Конвертируем в PyTorch тензоры
    train_desc_tensor = torch.LongTensor(train_desc)
    train_code_in_tensor = torch.LongTensor(train_code_in)
    train_code_out_tensor = torch.LongTensor(train_code_out)
    test_desc_tensor = torch.LongTensor(test_desc)
    test_code_in_tensor = torch.LongTensor(test_code_in)
    test_code_out_tensor = torch.LongTensor(test_code_out)
    # Создаем DataLoader'ы
    train_dataset = TensorDataset(train_desc_tensor, train_code_in_tensor, train_code_out_tensor)
    test_dataset = TensorDataset(test_desc_tensor, test_code_in_tensor, test_code_out_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return (train_loader, test_loader, desc_vocab_size, code_vocab_size, 
            desc_tokenizer, code_tokenizer)

def demonstrate_attention_visualization(model, desc_tokenizer, code_tokenizer):
    # Демонстрация визуализации внимания на реальных данных
    print("\n👁️ ДЕМОНСТРАЦИЯ ВНИМАНИЯ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 50)
    # Загружаем тестовые данные
    (train_desc, train_code_in, train_code_out), (test_desc, test_code_in, test_code_out) = load_training_data()
    # Берем первый пример из тестовой выборки
    src_sequence = test_desc[0]
    trg_sequence = test_code_in[0]
    # Создаем визуализатор
    visualizer = AttentionVisualizer(model, desc_tokenizer, code_tokenizer)
    # Визуализируем внимание
    attention_weights = visualizer.visualize_attention(
        src_sequence, 
        trg_sequence,
        save_path='2_my_cpp_nn_project/real_attention_visualization.png'
    )
    print("✅ Визуализация внимания на реальных данных завершена!")
    return attention_weights

def main_training():
    # Основной процесс обучения и сравнения
    print("🚀 ЭТАП 5: ОБУЧЕНИЕ И СРАВНЕНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    try:
        # Подготовка данных
        train_loader, test_loader, desc_vocab_size, code_vocab_size, desc_tokenizer, code_tokenizer = prepare_data_for_training()
        # Параметры моделей
        embedding_dim = 128
        hidden_size = 256
        # Создаем менеджер обучения
        trainer = TrainingManager(device=device)
        # МОДЕЛЬ 1: С ВНИМАНИЕМ
        attention_layer = BahdanauAttention(hidden_size)
        encoder_attention = Encoder(desc_vocab_size, embedding_dim, hidden_size)
        decoder_attention = DecoderWithAttention(code_vocab_size, embedding_dim, hidden_size, attention_layer)
        model_with_attention = Seq2SeqWithAttention(encoder_attention, decoder_attention)
        # Обучаем модель с вниманием
        trained_attention_model = trainer.train_model(
            model_with_attention, train_loader, test_loader, 
            "С вниманием", epochs=10, lr=0.001
        )
        # МОДЕЛЬ 2: БЕЗ ВНИМАНИЯ (для сравнения)
        model_without_attention = SimpleSeq2Seq(desc_vocab_size, code_vocab_size, embedding_dim, hidden_size)
        # Обучаем модель без внимания
        trained_simple_model = trainer.train_model(
            model_without_attention, train_loader, test_loader,
            "Без внимания", epochs=10, lr=0.001
        )
        # Сравниваем модели
        trainer.compare_models()
        # Демонстрируем работу внимания
        demonstrate_attention_visualization(trained_attention_model, desc_tokenizer, code_tokenizer)
        # Сохраняем лучшую модель
        os.makedirs('2_my_cpp_nn_project/models', exist_ok=True)
        torch.save(trained_attention_model.state_dict(), '2_my_cpp_nn_project/models/best_attention_model.pth')
        print(f"\n💾 Модель сохранена: '2_my_cpp_nn_project/models/best_attention_model.pth'")
        # Анализ результатов
        print("\n📈 АНАЛИЗ РЕЗУЛЬТАТОВ:")
        attention_result = trainer.results["С вниманием"]
        simple_result = trainer.results["Без внимания"]
        improvement = attention_result['final_accuracy'] - simple_result['final_accuracy']
        print(f"Улучшение от механизма внимания: {improvement:.4f} ({improvement*100:.2f}%)")
        if improvement > 0:
            print("✅ Механизм внимания УЛУЧШИЛ качество модели!")
        else:
            print("⚠️ Механизм внимания не показал улучшения на этом наборе данных")
        print("\n🎉 ЭТАП 5 ЗАВЕРШЕН! Полный цикл реализации внимания выполнен!")
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_training()