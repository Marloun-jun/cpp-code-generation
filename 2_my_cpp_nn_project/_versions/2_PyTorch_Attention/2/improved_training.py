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
# МОДЕЛЬ С МЕХАНИЗМОМ ВНИМАНИЯ
# =============================================================================

class BahdanauAttention(nn.Module):
    # Слой внимания Bahdanau
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_values = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, query, values):
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        score = self.V(torch.tanh(self.W_query(query) + self.W_values(values)))
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class Encoder(nn.Module):
    # Энкодер с возвратом всех скрытых состояний
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)

class DecoderWithAttention(nn.Module):
    # Декодер с механизмом внимания
    def __init__(self, vocab_size, embedding_dim, hidden_size, attention_layer, num_layers=1, dropout=0.1):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = attention_layer
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_outputs, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        context_vector, attention_weights = self.attention(hidden[-1], encoder_outputs)
        lstm_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = torch.cat([output, context_vector.unsqueeze(1)], dim=2)
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
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, (hidden, cell), _ = self.decoder(input, encoder_outputs, hidden, cell)
            outputs[:, t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

# =============================================================================
# ВИЗУАЛИЗАЦИЯ ВНИМАНИЯ
# =============================================================================

class AttentionVisualizer:
    # Класс для визуализации механизма внимания
    def __init__(self, model, desc_tokenizer, code_tokenizer):
        self.model = model
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer
        
    def visualize_attention(self, src_sequence, trg_sequence, save_path=None):
        self.model.eval()
        with torch.no_grad():
            src_tensor = torch.LongTensor(src_sequence).unsqueeze(0)
            trg_tensor = torch.LongTensor(trg_sequence).unsqueeze(0)
            encoder_outputs, (hidden, cell) = self.model.encoder(src_tensor)
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
            src_tokens = self._decode_tokens(src_sequence, self.desc_tokenizer)
            trg_tokens = self._decode_tokens(trg_sequence[1:], self.code_tokenizer)
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
        tokens = []
        for token_id in sequence:
            if token_id == 0:
                continue
            token = tokenizer.decode([token_id])
            tokens.append(token)
        return tokens

# =============================================================================
# ОБУЧЕНИЕ И ТРЕНИРОВКА
# =============================================================================

class CodeGeneratorTrainer:
    # Тренер для модели генерации кода с вниманием"""
    def __init__(self, device='cpu'):
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def prepare_data(self):
        # Подготовка данных для обучения
        print("📥 ПОДГОТОВКА ДАННЫХ")
        (train_desc, train_code_in, train_code_out), (test_desc, test_code_in, test_code_out) = load_training_data()
        desc_tokenizer, code_tokenizer = load_tokenizers()
        desc_vocab_size = desc_tokenizer.get_vocab_size()
        code_vocab_size = code_tokenizer.get_vocab_size()
        print(f"Размеры словарей: описания={desc_vocab_size}, код={code_vocab_size}")
        print(f"Данные: train={len(train_desc)}, test={len(test_desc)}")
        train_desc_tensor = torch.LongTensor(train_desc)
        train_code_in_tensor = torch.LongTensor(train_code_in)
        train_code_out_tensor = torch.LongTensor(train_code_out)
        test_desc_tensor = torch.LongTensor(test_desc)
        test_code_in_tensor = torch.LongTensor(test_code_in)
        test_code_out_tensor = torch.LongTensor(test_code_out)
        train_dataset = TensorDataset(train_desc_tensor, train_code_in_tensor, train_code_out_tensor)
        test_dataset = TensorDataset(test_desc_tensor, test_code_in_tensor, test_code_out_tensor)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        return (train_loader, test_loader, desc_vocab_size, code_vocab_size, 
                desc_tokenizer, code_tokenizer)
    
    def create_model(self, desc_vocab_size, code_vocab_size, embedding_dim=128, hidden_size=256):
        # Создание модели с вниманием
        print("🔨 СОЗДАНИЕ МОДЕЛИ С ВНИМАНИЕМ")
        attention_layer = BahdanauAttention(hidden_size)
        encoder = Encoder(desc_vocab_size, embedding_dim, hidden_size)
        decoder = DecoderWithAttention(code_vocab_size, embedding_dim, hidden_size, attention_layer)
        model = Seq2SeqWithAttention(encoder, decoder)
        print(f"✅ Модель создана:")
        print(f"   - Энкодер: {desc_vocab_size} -> {embedding_dim} -> {hidden_size}")
        print(f"   - Декодер: {code_vocab_size} -> {embedding_dim} -> {hidden_size}")
        print(f"   - Внимание: {hidden_size} units")
        return model
    
    def train(self, model, train_loader, val_loader, epochs=20, lr=0.001):
        # Обучение модели
        print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 50)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        best_accuracy = 0
        best_model_state = None
        for epoch in range(epochs):
            start_time = time.time()
            # Обучение
            model.train()
            train_loss = 0
            for batch_idx, (src, trg_in, trg_out) in enumerate(train_loader):
                src, trg_in, trg_out = src.to(self.device), trg_in.to(self.device), trg_out.to(self.device)
                optimizer.zero_grad()
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
                    output = model(src, trg_in, teacher_forcing_ratio=0)
                    loss = criterion(output.reshape(-1, output.shape[-1]), trg_out.reshape(-1))
                    val_loss += loss.item()
                    predictions = output.argmax(-1)
                    mask = trg_out != 0
                    correct += ((predictions == trg_out) & mask).sum().item()
                    total += mask.sum().item()
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            accuracy = correct / total if total > 0 else 0
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)
            epoch_time = time.time() - start_time
            print(f'Эпоха {epoch+1}/{epochs} ({epoch_time:.1f}с):')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {accuracy:.4f}')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                print(f'  🏆 Новый лучший результат!')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f'\n🎉 Лучшая точность: {best_accuracy:.4f}')
        return model
    
    def plot_training_progress(self):
        # Визуализация прогресса обучения
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss', linestyle='--')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.title('Прогресс обучения')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy', color='green')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.title('Точность на валидации')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('2_my_cpp_nn_project/training_progress.png', dpi=150, bbox_inches='tight')
        print("📈 График обучения сохранен: '2_my_cpp_nn_project/training_progress.png'")
    
    def save_model(self, model, path):
        # Сохранение модели
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"💾 Модель сохранена: {path}")

# =============================================================================
# ОСНОВНОЙ ПРОЦЕСС
# =============================================================================

def main():
    # Основной процесс обучения
    print("🚀 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ С МЕХАНИЗМОМ ВНИМАНИЯ")
    print("=" * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    try:
        # Создаем тренер
        trainer = CodeGeneratorTrainer(device=device)
        # Подготавливаем данные
        train_loader, test_loader, desc_vocab_size, code_vocab_size, desc_tokenizer, code_tokenizer = trainer.prepare_data()
        # Создаем модель
        model = trainer.create_model(desc_vocab_size, code_vocab_size)
        # Обучаем модель
        trained_model = trainer.train(model, train_loader, test_loader, epochs=15)
        # Визуализируем прогресс
        trainer.plot_training_progress()
        # Сохраняем модель
        trainer.save_model(trained_model, '2_my_cpp_nn_project/models/attention_code_generator.pth')
        # Демонстрация внимания
        print("\n👁️ ДЕМОНСТРАЦИЯ МЕХАНИЗМА ВНИМАНИЯ")
        print("=" * 50)
        (train_desc, train_code_in, train_code_out), (test_desc, test_code_in, test_code_out) = load_training_data()
        # Берем пример для визуализации
        src_sequence = test_desc[0]
        trg_sequence = test_code_in[0]
        visualizer = AttentionVisualizer(trained_model, desc_tokenizer, code_tokenizer)
        attention_weights = visualizer.visualize_attention(
            src_sequence, 
            trg_sequence,
            save_path='2_my_cpp_nn_project/attention_demonstration.png'
        )
        print("✅ Визуализация внимания завершена!")
        # Финальная статистика
        final_accuracy = trainer.val_accuracies[-1] if trainer.val_accuracies else 0
        best_accuracy = max(trainer.val_accuracies) if trainer.val_accuracies else 0
        print(f"\n📊 ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"   Финальная точность: {final_accuracy:.4f}")
        print(f"   Лучшая точность: {best_accuracy:.4f}")
        print(f"   Всего эпох: {len(trainer.train_losses)}")
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("Модель с механизмом внимания готова к использованию!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()