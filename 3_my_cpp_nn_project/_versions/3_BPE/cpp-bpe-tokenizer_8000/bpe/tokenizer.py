# bpe/tokenizer.py
import json
import re
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict, Counter

class BPETokenizer:
    """
    Byte-Pair Encoding токенизатор
    Поддерживает byte-level режим и специальные токены
    """
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 byte_level: bool = True,
                 special_tokens: List[str] = None):
        """
        Args:
            vocab_size: Размер словаря (включая специальные токены)
            byte_level: Использовать byte-level представление UTF-8
            special_tokens: Список специальных токенов
        """
        self.vocab_size = vocab_size
        self.byte_level = byte_level
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        # Основные структуры данных
        self.vocab = {}  # id -> токен
        self.inverse_vocab = {}  # токен -> id
        self.merges = {}  # пара -> ранг
        self.byte_encoder = None
        self.byte_decoder = None
        
        if self.byte_level:
            self._init_byte_encoder()
    
    def _init_byte_encoder(self):
        """Инициализация byte-level encoder/decoder как в GPT-4"""
        # Базовые 256 байтов
        self.byte_encoder = {}
        self.byte_decoder = {}
        
        # Печатные ASCII символы оставляем как есть
        for i in range(ord('!'), ord('~') + 1):
            self.byte_encoder[i] = chr(i)
        for i in range(ord('¡'), ord('¬') + 1):
            self.byte_encoder[i] = chr(i)
        for i in range(ord('®'), ord('ÿ') + 1):
            self.byte_encoder[i] = chr(i)
        
        # Остальные байты маппим на юникод в private use area
        n = 0
        for i in range(256):
            if i not in self.byte_encoder:
                self.byte_encoder[i] = chr(0xE000 + n)
                n += 1
        
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def _byte_encode(self, text: str) -> str:
        """Конвертация UTF-8 текста в byte-level строку"""
        if not self.byte_level:
            return text
        
        bytes_data = text.encode('utf-8')
        return ''.join(self.byte_encoder[b] for b in bytes_data)
    
    def _byte_decode(self, text: str) -> str:
        """Декодирование byte-level строки обратно в UTF-8"""
        if not self.byte_level:
            return text
    
        # Проверяем, инициализирован ли декодер
        if not hasattr(self, 'byte_decoder') or not self.byte_decoder:
            self._init_byte_encoder()
    
        bytes_data = bytearray()
        for ch in text:
            if ch in self.byte_decoder:
                bytes_data.append(self.byte_decoder[ch])
            else:
                # Если символ не найден, пропускаем
                continue
    
        try:
            return bytes_data.decode('utf-8')
        except:
            return bytes_data.decode('utf-8', errors='replace')
    
    def _get_stats(self, words: List[str]) -> Dict[Tuple[str, str], int]:
        """Подсчет частот пар символов"""
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], words: List[str]) -> List[str]:
        """Слияние пары токенов во всех словах"""
        new_words = []
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words.append(new_word)
        
        return new_words
    
    def train(self, corpus: List[str], verbose: bool = True):
        """
        Обучение BPE токенизатора
        
        Args:
            corpus: Список строк для обучения
            verbose: Выводить прогресс обучения
        """
        print("Препроцессинг корпуса...")
        # 1. Препроцессинг
        processed_corpus = []
        for text in corpus:
            if self.byte_level:
                text = self._byte_encode(text)
            # Разбиваем на символы и добавляем маркер конца
            chars = list(text)
            text = ' '.join(chars) + ' </w>'
            processed_corpus.append(text)
        
        print("Сбор уникальных символов...")
        # 2. Инициализация базового словаря символами
        symbols = set()
        for text in processed_corpus:
            symbols.update(text.split())
        
        print(f"Найдено {len(symbols)} уникальных символов/байтов")
        
        # 3. Начинаем с базовых символов + специальные токены
        current_vocab = {}
        self.inverse_vocab = {}

        # Сначала добавляем специальные токены
        for i, token in enumerate(self.special_tokens):
            current_vocab[i] = token
            self.inverse_vocab[token] = i

        # ПОТОМ добавляем базовые символы
        for i, sym in enumerate(sorted(symbols), start=len(self.special_tokens)):
            current_vocab[i] = sym
            self.inverse_vocab[sym] = i

        self.vocab = current_vocab
        print(f"Начальный словарь: {len(self.vocab)} токенов")
        
        # 4. Основной цикл обучения
        num_merges = self.vocab_size - len(self.vocab)
        merges = {}
        
        print(f"Начинаем {num_merges} мердж-операций...")
        
        for i in range(num_merges):
            # Подсчитываем частоты пар
            pairs = self._get_stats(processed_corpus)
            
            if not pairs:
                print(f"Нет больше пар для слияния на шаге {i}")
                break
            
            # Находим самую частую пару
            most_frequent = max(pairs.items(), key=lambda x: x[1])
            best_pair, freq = most_frequent
            
            # Сливаем пару
            processed_corpus = self._merge_vocab(best_pair, processed_corpus)
            
            # Добавляем новую мерж-операцию
            new_token = ''.join(best_pair)
            merges[best_pair] = i
            
            # Добавляем новый токен в словарь
            new_id = len(self.vocab)
            self.vocab[new_id] = new_token
            self.inverse_vocab[new_token] = new_id
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_token} (freq: {freq})")
        
        self.merges = merges
        print(f"Обучение завершено! Итоговый словарь: {len(self.vocab)} токенов")
        return self
    
    def encode(self, text: str) -> List[int]:
        """
        Кодирование текста в последовательность ID
        
        Args:
            text: Входной текст
            
        Returns:
            Список ID токенов
        """
        # 1. Byte-level кодирование
        if self.byte_level:
            text = self._byte_encode(text)
        
        # 2. Разбиваем на символы
        tokens = list(text) + ['</w>']
        current_text = ' '.join(tokens)
        
        # 3. Применяем все возможные слияния
        while True:
            pairs = self._get_stats([current_text])
            mergeable = {}
            
            for pair in pairs:
                if pair in self.merges:
                    mergeable[pair] = self.merges[pair]
            
            if not mergeable:
                break
            
            # Находим пару с наименьшим рангом (раннее слияние)
            best_pair = min(mergeable, key=mergeable.get)
            current_text = self._merge_vocab(best_pair, [current_text])[0]
        
        # 4. Конвертируем в ID
        tokens = current_text.split()
        ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                ids.append(self.inverse_vocab[token])
            else:
                ids.append(self.inverse_vocab['<UNK>'])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Декодирование ID обратно в текст
        """
        # 1. Конвертируем ID в токены
        tokens = []
        for idx in ids:
            if idx in self.vocab:
                token = self.vocab[idx]
                if token not in self.special_tokens:
                    tokens.append(token)
        
        # 2. Объединяем токены
        text = ''.join(tokens)
        
        # 3. Убираем маркер конца слова
        text = text.replace('</w>', ' ')
        
        # 4. Byte-level декодирование
        if self.byte_level:
            text = self._byte_decode(text)
        
        # 5. Очистка
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def save(self, vocab_path: str, merges_path: str):
        """
        Сохранение словаря и мердж-операций
        """
        # Сохраняем словарь
        vocab_dict = {str(k): v for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        # Сохраняем мердж-операции
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for pair, rank in sorted(self.merges.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\n")
    
    def load(self, vocab_path: str, merges_path: str):
        """
        Загрузка обученного токенизатора
        """
        # Загружаем словарь
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
    
        self.vocab = {int(k): v for k, v in vocab_dict.items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
        # Загружаем мердж-операции
        self.merges = {}
        with open(merges_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('#version'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = i - 1
    
        # ВАЖНО: Инициализируем byte-level encoder/decoder
        if self.byte_level:
            self._init_byte_encoder()