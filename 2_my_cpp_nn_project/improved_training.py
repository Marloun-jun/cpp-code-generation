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
from collections import deque
import math

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
        # query: (batch_size, hidden_size) - последнее скрытое состояние декодера
        # values: (batch_size, seq_len, hidden_size) - выходы энкодера
        
        # Расширяем query для совместимости с values
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Вычисляем scores
        scores = self.V(torch.tanh(self.W_query(query) + self.W_values(values)))  # (batch_size, seq_len, 1)
        
        # Softmax по dimension seq_len
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        
        # Вычисляем context vector
        context_vector = torch.sum(attention_weights * values, dim=1)  # (batch_size, hidden_size)
        
        return context_vector, attention_weights.squeeze(-1)  # (batch_size, seq_len)

class Encoder(nn.Module):
    # Энкодер с возвратом всех скрытых состояний
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.6):
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
    def __init__(self, vocab_size, embedding_dim, hidden_size, attention_layer, num_layers=1, dropout=0.7):
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
        # x: (batch_size,)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        # hidden, cell: (num_layers, batch_size, hidden_size)
        
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(x))  # (batch_size, 1, embedding_dim)
        
        # Получаем последнее скрытое состояние для attention
        query = hidden[-1]  # (batch_size, hidden_size) - берем последний слой
        
        # Получаем контекстный вектор и веса внимания
        context_vector, attention_weights = self.attention(query, encoder_outputs)  # (batch_size, hidden_size), (batch_size, seq_len)
        
        # Добавляем dimension для конкатенации
        context_vector = context_vector.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Конкатенируем embedded и context_vector
        lstm_input = torch.cat([embedded, context_vector], dim=2)  # (batch_size, 1, embedding_dim + hidden_size)
        
        # Пропускаем через LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))  # output: (batch_size, 1, hidden_size)
        
        # Конкатенируем output и context_vector для финального слоя
        output = torch.cat([output, context_vector], dim=2)  # (batch_size, 1, hidden_size * 2)
        
        # Финальный линейный слой
        prediction = self.fc_out(output)  # (batch_size, 1, vocab_size)
        prediction = prediction.squeeze(1)  # (batch_size, vocab_size)
        
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
        
        input = trg[:, 0]  # Первый токен - start token
        
        for t in range(1, trg_len):
            output, (hidden, cell), _ = self.decoder(input, encoder_outputs, hidden, cell)
            outputs[:, t] = output
            
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

    def generate_with_beam_search(self, description, beam_width=3, max_length=50, 
                                 length_penalty=0.6, early_stopping=True):
        """
        Генерация последовательности с использованием beam search
        """
        self.eval()
        with torch.no_grad():
            # Подготовка входных данных
            if isinstance(description, (list, np.ndarray)):
                src_tensor = torch.LongTensor(description).unsqueeze(0)
            else:
                src_tensor = description.unsqueeze(0) if description.dim() == 1 else description
            
            # Кодирование входной последовательности
            encoder_outputs, (hidden, cell) = self.encoder(src_tensor)
            
            # Начальный луч с токеном начала последовательности
            start_token = 1
            beams = [{
                'sequence': [start_token],
                'score': 0.0,
                'hidden': hidden,
                'cell': cell,
                'finished': False
            }]
            
            for step in range(max_length):
                candidates = []
                
                if early_stopping and all(beam['finished'] for beam in beams):
                    break
                
                for beam in beams:
                    if beam['finished']:
                        candidates.append(beam)
                        continue
                    
                    last_token = beam['sequence'][-1]
                    input_token = torch.LongTensor([last_token])  # (1,)
                    
                    # Декодируем следующий токен
                    output, (new_hidden, new_cell), _ = self.decoder(
                        input_token, encoder_outputs, 
                        beam['hidden'], beam['cell']
                    )
                    
                    probabilities = torch.softmax(output.squeeze(), dim=0)
                    topk_probs, topk_indices = torch.topk(probabilities, beam_width)
                    
                    for i in range(beam_width):
                        token = topk_indices[i].item()
                        token_prob = topk_probs[i].item()
                        
                        log_prob = math.log(token_prob) if token_prob > 0 else -float('inf')
                        new_score = beam['score'] + log_prob
                        
                        finished = (token == 2)  # end token
                        
                        new_beam = {
                            'sequence': beam['sequence'] + [token],
                            'score': new_score,
                            'hidden': new_hidden,
                            'cell': new_cell,
                            'finished': finished
                        }
                        
                        if length_penalty != 0:
                            length = len(new_beam['sequence'])
                            new_beam['score'] = new_beam['score'] / (length ** length_penalty)
                        
                        candidates.append(new_beam)
                
                # Сортируем кандидатов по score и выбираем top-k
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_width]
            
            # Возвращаем лучшую последовательность
            best_beam = max(beams, key=lambda x: x['score'])
            return best_beam['sequence']

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
        self.learning_rates = []  # Для отслеживания learning rate
        
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
    
    def create_model(self, desc_vocab_size, code_vocab_size, embedding_dim=64, hidden_size=128):
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
        print(f"   - Dropout: 0.4")
        return model
    
    def train(self, model, train_loader, val_loader, epochs=30, lr=0.0001):
        # Обучение модели
        print(f"\n🎯 НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 50)
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        
        best_accuracy = 0
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 5
        
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
                # Gradient clipping для стабильности
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            
            # Обновление learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)
            epoch_time = time.time() - start_time
            
            print(f'Эпоха {epoch+1}/{epochs} ({epoch_time:.1f}с):')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Val Accuracy: {accuracy:.4f}')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Ранняя остановка
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f'  🏆 Новый лучший результат!')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'  ⏹️  Ранняя остановка после {epoch+1} эпох')
                    break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f'\n🎉 Лучшая точность: {best_accuracy:.4f}')
        return model
    
    def plot_training_progress(self):
        # Визуализация прогресса обучения
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss', linestyle='--')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.title('Прогресс обучения')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy', color='green')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.title('Точность на валидации')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates, label='Learning Rate', color='red')
        plt.xlabel('Эпоха')
        plt.ylabel('Learning Rate')
        plt.title('Изменение Learning Rate')
        plt.yscale('log')
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
        trained_model = trainer.train(model, train_loader, test_loader, epochs=30)
        # Визуализируем прогресс
        trainer.plot_training_progress()
        # Сохраняем модель
        trainer.save_model(trained_model, '2_my_cpp_nn_project/models/attention_code_generator.pth')
        
        # Демонстрация beam search
        print("\n🔍 ДЕМОНСТРАЦИЯ BEAM SEARCH")
        print("=" * 50)
        (train_desc, train_code_in, train_code_out), (test_desc, test_code_in, test_code_out) = load_training_data()
        
        # Тестируем beam search на примере
        sample_description = test_desc[0]
        print("Входное описание:", sample_description[:10], "...")  # Показываем первые 10 токенов
        
        # Генерация с beam search
        generated_sequence = trained_model.generate_with_beam_search(
            sample_description, 
            beam_width=3, 
            max_length=30,
            length_penalty=0.6
        )
        
        print("Сгенерированная последовательность:", generated_sequence)
        
        # Демонстрация внимания
        print("\n👁️ ДЕМОНСТРАЦИЯ МЕХАНИЗМА ВНИМАНИЯ")
        print("=" * 50)
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
        print(f"   Final Learning Rate: {trainer.learning_rates[-1]:.6f}")
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("Модель с механизмом внимания и beam search готова к использованию!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()