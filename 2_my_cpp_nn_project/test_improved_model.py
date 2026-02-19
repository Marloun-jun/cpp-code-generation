import torch
import torch.nn as nn
import numpy as np
import sys
import os
import re
import math
from pathlib import Path

# Добавляем путь для импорта модулей
sys.path.append('2_my_cpp_nn_project')

# Импортируем улучшенное декодирование
from preprocess_data import decode_bpe_sequence, load_tokenizers

# Импортируем ТОЧНО ТЕ ЖЕ классы модели, что использовались при обучении
class BahdanauAttention(nn.Module):
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
                    input_token = torch.LongTensor([last_token])
                    
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

class ImprovedCodeGeneratorInference:
    def __init__(self, model_path, tokenizers_path):
        try:
            # Загружаем токенизаторы
            self.desc_tokenizer, self.code_tokenizer = load_tokenizers()
            desc_vocab_size = self.desc_tokenizer.get_vocab_size()
            code_vocab_size = self.code_tokenizer.get_vocab_size()
            
            print(f"📊 Размеры словарей: desc={desc_vocab_size}, code={code_vocab_size}")
            
            # Получаем ID специальных токенов
            self.start_token = self.code_tokenizer.token_to_id("<START>")
            self.end_token = self.code_tokenizer.token_to_id("<END>") 
            self.pad_token = self.code_tokenizer.token_to_id("<PAD>")
            
            print(f"🔤 Специальные токены: START={self.start_token}, END={self.end_token}, PAD={self.pad_token}")
            
            # Создаем модель с ТОЧНО ТЕМИ ЖЕ параметрами, что и при обучении
            attention_layer = BahdanauAttention(hidden_size=128)
            encoder = Encoder(desc_vocab_size, embedding_dim=64, hidden_size=128, dropout=0.6)
            decoder = DecoderWithAttention(code_vocab_size, embedding_dim=64, hidden_size=128, 
                                         attention_layer=attention_layer, dropout=0.7)
            self.model = Seq2SeqWithAttention(encoder, decoder)
            
            # Загружаем веса
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint)
                print("✅ Веса модели успешно загружены")
            else:
                print(f"❌ Файл модели не найден: {model_path}")
                return
                
            self.model.eval()
            print("🎯 Модель с механизмом внимания загружена и готова к работе!")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def preprocess_description(self, description, max_length=50):
        """Препроцессинг описания в последовательность токенов"""
        # Токенизируем описание
        encoded = self.desc_tokenizer.encode(description)
        tokens = encoded.ids
        
        # Обрезаем или дополняем до max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + [self.pad_token] * (max_length - len(tokens))
            
        return torch.LongTensor([tokens])
    
    def generate_greedy(self, description, max_length=50):
        """Жадная генерация кода"""
        print(f"\n🎯 Генерация (жадная): '{description}'")
        
        try:
            # Препроцессинг
            src_tensor = self.preprocess_description(description)
            
            self.model.eval()
            generated_tokens = []
            
            with torch.no_grad():
                # Прямой проход через энкодер
                encoder_outputs, (hidden, cell) = self.model.encoder(src_tensor)
                
                # Начальный токен
                input_token = torch.LongTensor([self.start_token])
                
                print("🔍 Генерация токенов: ", end="")
                
                for i in range(max_length):
                    # Прямой проход через декодер
                    output, (hidden, cell), _ = self.model.decoder(
                        input_token, encoder_outputs, hidden, cell
                    )
                    
                    # Жадное декодирование - берем самый вероятный токен
                    next_token_id = torch.argmax(output, dim=-1).item()
                    
                    # Проверка на конец последовательности
                    if next_token_id == self.end_token:
                        print("<END>")
                        break
                        
                    if next_token_id == self.pad_token:
                        continue
                    
                    # Добавляем токен в последовательность
                    generated_tokens.append(next_token_id)
                    token_text = decode_bpe_sequence(self.code_tokenizer, [next_token_id], remove_special_tokens=False)
                    print(f"{token_text} ", end="", flush=True)
                    
                    # Обновляем вход для следующей итерации
                    input_token = torch.LongTensor([next_token_id])
                
                print()
            
            if not generated_tokens:
                return "❌ Не сгенерировано токенов"
            
            # Декодируем полную последовательность
            code = decode_bpe_sequence(self.code_tokenizer, generated_tokens)
            return f"📝 Результат:\n{code}"
            
        except Exception as e:
            return f"❌ Ошибка генерации: {e}"
    
    def generate_with_beam_search(self, description, beam_width=3, max_length=50):
        """Генерация с использованием beam search"""
        print(f"\n🎯 Генерация (beam search {beam_width}): '{description}'")
        
        try:
            # Препроцессинг
            src_tensor = self.preprocess_description(description)
            
            self.model.eval()
            with torch.no_grad():
                # Генерация с beam search
                generated_tokens = self.model.generate_with_beam_search(
                    src_tensor[0],  # Берем первую последовательность из батча
                    beam_width=beam_width,
                    max_length=max_length,
                    length_penalty=0.6
                )
            
            # Убираем start token (первый токен)
            if generated_tokens and generated_tokens[0] == self.start_token:
                generated_tokens = generated_tokens[1:]
            
            # Убираем end token если есть
            if generated_tokens and generated_tokens[-1] == self.end_token:
                generated_tokens = generated_tokens[:-1]
            
            if not generated_tokens:
                return "❌ Не сгенерировано токенов"
            
            print(f"🔍 Сгенерированные токены: {generated_tokens}")
            
            # Декодируем полную последовательность
            code = decode_bpe_sequence(self.code_tokenizer, generated_tokens)
            return f"📝 Результат:\n{code}"
            
        except Exception as e:
            return f"❌ Ошибка beam search: {e}"
    
    def test_model_capabilities(self, description):
        """Тестирование различных возможностей модели"""
        print(f"\n🧪 ТЕСТИРОВАНИЕ МОДЕЛИ: '{description}'")
        
        try:
            src_tensor = self.preprocess_description(description)
            
            self.model.eval()
            with torch.no_grad():
                # Прямой проход через энкодер
                encoder_outputs, (hidden, cell) = self.model.encoder(src_tensor)
                
                # Начальный токен
                input_token = torch.LongTensor([self.start_token])
                
                # Первое предсказание
                output, _, _ = self.model.decoder(input_token, encoder_outputs, hidden, cell)
                probs = torch.softmax(output, dim=-1)
                
                # Топ-10 наиболее вероятных токенов
                top_probs, top_indices = torch.topk(probs, 10)
                
                print("Топ-10 наиболее вероятных первых токенов:")
                for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                    token_text = decode_bpe_sequence(self.code_tokenizer, [idx.item()], remove_special_tokens=False)
                    print(f"  {i+1:2d}. '{token_text}' (id={idx.item():3d}, p={prob:.4f})")
                    
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")

# Тестирование
if __name__ == "__main__":
    print("🚀 ТЕСТИРОВАНИЕ МОДЕЛИ С МЕХАНИЗМОМ ВНИМАНИЯ")
    print("=" * 60)
    
    try:
        # Создаем генератор
        model_path = '2_my_cpp_nn_project/models/attention_code_generator.pth'
        
        if not os.path.exists(model_path):
            print(f"⚠️ Файл модели не найден: {model_path}")
            print("💡 Нужно сначала обучить модель!")
            print("   Запусти: python improved_training.py")
            sys.exit(1)
            
        generator = ImprovedCodeGeneratorInference(
            model_path,
            '2_my_cpp_nn_project/tokenizers/'
        )
        
        # Сначала протестируем выходы модели
        test_description = "print hello world"
        generator.test_model_capabilities(test_description)
        
        print("\n" + "=" * 60)
        print("🧪 СРАВНЕНИЕ МЕТОДОВ ГЕНЕРАЦИИ")
        print("=" * 60)
        
        # Тестовые случаи
        test_cases = [
            "print hello world",
            "add two numbers", 
            "calculate sum",
            "create variable",
            "simple function",
            "loop through array"
        ]
        
        for i, desc in enumerate(test_cases, 1):
            print(f"\n📋 ТЕСТ {i}: '{desc}'")
            print("-" * 40)
            
            # Жадная генерация
            result_greedy = generator.generate_greedy(desc, max_length=30)
            print(result_greedy)
            
            print("-" * 20)
            
            # Beam search генерация
            result_beam = generator.generate_with_beam_search(desc, beam_width=3, max_length=30)
            print(result_beam)
            
            print("=" * 40)
            
        print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print("💡 Анализ: Обратите внимание на разницу между жадной генерацией и beam search")
        print("   - Жадная: быстро, но может зацикливаться")
        print("   - Beam search: медленнее, но дает более качественные результаты")
            
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()