#!/usr/bin/env python3
# ======================================================================
# tokenizer.py - Реализация Byte-Pair Encoding (BPE) токенизатора
# ======================================================================
#
# @file tokenizer.py
# @brief Реализация Byte-Pair Encoding (BPE) токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Модуль предоставляет BPE токенизатор с поддержкой byte-level кодирования,
#          специальных токенов и совместимостью с трансформер моделями.
#          Поддерживает:
#          - Обучение на корпусе текстов
#          - Кодирование текста в токены
#          - Декодирование токенов обратно в текст
#          - Сохранение и загрузку модели
#          - Byte-level обработку UTF-8 (аналог GPT-4)
#
# @usage from tokenizer import BPETokenizer
#
# @example
#   tokenizer = BPETokenizer(vocab_size=5000)
#   tokenizer.train(["пример текста", "другой пример"])
#   ids = tokenizer.encode("int main() { return 0; }")
#   text = tokenizer.decode(ids)
#   tokenizer.save("vocab.json", "merges.txt")
#
# ======================================================================

import json
import logging
import re

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Union

# ======================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ======================================================================
# ОСНОВНОЙ КЛАСС ТОКЕНИЗАТОРА
# ======================================================================

class BPETokenizer:
    """
    Byte-Pair Encoding токенизатор с поддержкой byte-level режима.
    
    Реализует алгоритм BPE как описано в статье "Neural Machine Translation of
    Rare Words with Subword Units" (Sennrich et al., 2016). Поддерживает
    byte-level предобработку аналогично GPT-4 для обработки произвольного UTF-8 текста.
    
    Attributes:
        vocab_size: Максимальный размер словаря (включая специальные токены)
        byte_level: Использовать ли byte-level предобработку
        special_tokens: Список специальных токенов (например, <PAD>, <UNK>)
        vocab: Словарь, отображающий ID токенов в сами токены
        inverse_vocab: Обратный словарь, отображающий токены в ID
        merges: Словарь, отображающий пары токенов в ранги слияния
    """
    
    # Константы для byte-level кодирования (начало приватной области Unicode)
    PRIVATE_USE_AREA_START = 0xE000
    
    # ======================================================================
    # КОНСТРУКТОР
    # ======================================================================
    
    def __init__(
        self,
        vocab_size: int = 30000,
        byte_level: bool = True,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Инициализация BPE токенизатора.
        
        Args:
            vocab_size: Максимальный размер словаря (включая специальные токены)
            byte_level: Использовать byte-level предобработку для UTF-8 текста
            special_tokens: Список специальных токенов. По умолчанию:
                ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        Example:
            >>> tokenizer = BPETokenizer(vocab_size=5000)
            >>> tokenizer.train(["привет мир", "тестовая строка"])
        """
        self.vocab_size = vocab_size
        self.byte_level = byte_level
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        # Основные структуры данных
        self.vocab: Dict[int, str] = {}
        self.inverse_vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        
        # Byte-level кодировщики/декодировщики (ленивая инициализация)
        self._byte_encoder: Optional[Dict[int, str]] = None
        self._byte_decoder: Optional[Dict[str, int]] = None
        
        if self.byte_level:
            self._init_byte_encoder()
    
    # ======================================================================
    # BYTE-LEVEL КОДИРОВАНИЕ
    # ======================================================================
    
    def _init_byte_encoder(self) -> None:
        """
        Инициализация byte-level кодировщика/декодировщика аналогично GPT-4.
        """
        self._byte_encoder = {}
        self._byte_decoder = {}
        
        # Оставляем печатные ASCII символы без изменений
        # Диапазон: '!' (33) до '~' (126)
        for i in range(ord('!'), ord('~') + 1):
            self._byte_encoder[i] = chr(i)
        
        # Оставляем расширенные Latin-1 символы без изменений
        # Диапазон: '¡' (161) до '¬' (172) и '®' (174) до 'ÿ' (255)
        for i in range(ord('¡'), ord('¬') + 1):
            self._byte_encoder[i] = chr(i)
        for i in range(ord('®'), ord('ÿ') + 1):
            self._byte_encoder[i] = chr(i)
        
        # Оставшиеся байты отображаем на приватную область Unicode
        n = 0
        for i in range(256):
            if i not in self._byte_encoder:
                self._byte_encoder[i] = chr(self.PRIVATE_USE_AREA_START + n)
                n += 1
        
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
    
    def _byte_encode(self, text: str) -> str:
        """
        Конвертация UTF-8 текста в byte-level строковое представление.
        
        Args:
            text: Входной UTF-8 текст
            
        Returns:
            str: Строка, где каждый символ представляет один байт
            
        Raises:
            RuntimeError: Если byte-level кодировщик не инициализирован
        """
        if not self.byte_level:
            return text
        
        if self._byte_encoder is None:
            raise RuntimeError("Byte-level кодировщик не инициализирован")
        
        bytes_data = text.encode('utf-8')
        return ''.join(self._byte_encoder[b] for b in bytes_data)
    
    def _byte_decode(self, text: str) -> str:
        """
        Декодирование byte-level строки обратно в UTF-8.
        
        Args:
            text: Byte-level закодированная строка
            
        Returns:
            str: Декодированный UTF-8 текст
        """
        if not self.byte_level:
            return text
        
        # Ленивая инициализация если нужно
        if self._byte_decoder is None:
            self._init_byte_encoder()
        
        bytes_data = bytearray()
        for ch in text:
            if ch in self._byte_decoder:
                bytes_data.append(self._byte_decoder[ch])
            else:
                # Пропускаем неизвестные символы
                continue
        
        try:
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            return bytes_data.decode('utf-8', errors='replace')
    
    # ======================================================================
    # СТАТИЧЕСКИЕ МЕТОДЫ ДЛЯ BPE
    # ======================================================================
    
    @staticmethod
    def _get_stats(words: List[str]) -> Dict[Tuple[str, str], int]:
        """
        Подсчет частот соседних пар символов.
        
        Args:
            words: Список токенизированных слов (символы через пробел)
            
        Returns:
            Dict[Tuple[str, str], int]: Словарь, отображающий пары символов в их частоты
        """
        pairs = defaultdict(int)
        for word in words:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs
    
    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], words: List[str]) -> List[str]:
        """
        Слияние пары символов во всех словах.
        
        Args:
            pair: Кортеж из двух символов для слияния
            words: Список токенизированных слов
            
        Returns:
            List[str]: Список слов с выполненным слиянием
        """
        new_words = []
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words.append(new_word)
        
        return new_words
    
    # ======================================================================
    # ОБУЧЕНИЕ
    # ======================================================================
    
    def train(self, corpus: List[str], verbose: bool = True) -> 'BPETokenizer':
        """
        Обучение BPE токенизатора на входном корпусе текстов.
        
        Args:
            corpus: Список текстовых строк для обучения
            verbose: Выводить ли прогресс обучения в лог
            
        Returns:
            BPETokenizer: Self для возможности цепочечных вызовов
            
        Raises:
            ValueError: Если корпус пуст или vocab_size слишком маленький
        """
        if not corpus:
            raise ValueError("Корпус для обучения не может быть пустым")
        
        if self.vocab_size < len(self.special_tokens) + 1:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) слишком мал для специальных токенов"
            )
        
        logger.info("Предобработка корпуса...")
        processed_corpus = self._preprocess_corpus(corpus)
        
        logger.info("Сбор уникальных символов...")
        symbols = self._collect_symbols(processed_corpus)
        logger.info(f"   Найдено {len(symbols)} уникальных символов/байтов")
        
        logger.info("Инициализация словаря...")
        self._initialize_vocabulary(symbols)
        
        logger.info(f"Начало слияний (цель: {self.vocab_size} токенов)...")
        self._perform_merges(processed_corpus, verbose)
        
        logger.info(f"Обучение завершено! Итоговый словарь: {len(self.vocab)} токенов")
        return self
    
    def _preprocess_corpus(self, corpus: List[str]) -> List[str]:
        """
        Предобработка корпуса для обучения.
        
        Args:
            corpus: Список текстов для обработки
            
        Returns:
            List[str]: Обработанный корпус
        """
        processed = []
        for text in corpus:
            if self.byte_level:
                text = self._byte_encode(text)
            # Разбиваем на символы и добавляем маркер конца слова
            chars = list(text)
            processed.append(' '.join(chars) + ' </w>')
        return processed
    
    def _collect_symbols(self, processed_corpus: List[str]) -> Set[str]:
        """
        Сбор всех уникальных символов из обработанного корпуса.
        
        Args:
            processed_corpus: Обработанный корпус
            
        Returns:
            Set[str]: Множество уникальных символов
        """
        symbols = set()
        for text in processed_corpus:
            symbols.update(text.split())
        return symbols
    
    def _initialize_vocabulary(self, symbols: Set[str]) -> None:
        """
        Инициализация словаря специальными токенами и базовыми символами.
        
        Args:
            symbols: Множество базовых символов
        """
        self.vocab.clear()
        self.inverse_vocab.clear()
        
        # Сначала добавляем специальные токены
        for i, token in enumerate(self.special_tokens):
            self.vocab[i] = token
            self.inverse_vocab[token] = i
        
        # Затем добавляем базовые символы
        next_id = len(self.special_tokens)
        for sym in sorted(symbols):
            self.vocab[next_id] = sym
            self.inverse_vocab[sym] = next_id
            next_id += 1
    
    def _perform_merges(self, processed_corpus: List[str], verbose: bool) -> None:
        """
        Выполнение операций слияния до достижения целевого размера словаря.
        
        Args:
            processed_corpus: Обработанный корпус
            verbose: Выводить прогресс
        """
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            # Подсчитываем частоты пар
            pairs = self._get_stats(processed_corpus)
            
            if not pairs:
                if verbose:
                    logger.info(f" !!! Нет больше пар для слияния на шаге {i}")
                break
            
            # Находим самую частую пару
            most_frequent = max(pairs.items(), key=lambda x: x[1])
            best_pair, freq = most_frequent
            
            # Выполняем слияние
            processed_corpus = self._merge_vocab(best_pair, processed_corpus)
            
            # Добавляем новую операцию слияния
            new_token = ''.join(best_pair)
            self.merges[best_pair] = i
            
            # Добавляем новый токен в словарь
            new_id = len(self.vocab)
            self.vocab[new_id] = new_token
            self.inverse_vocab[new_token] = new_id
            
            if verbose and (i + 1) % 100 == 0:
                logger.info(f"   Слияние {i + 1}/{num_merges}: {best_pair} -> {new_token} "
                          f"(частота: {freq})")
    
    # ======================================================================
    # КОДИРОВАНИЕ И ДЕКОДИРОВАНИЕ
    # ======================================================================
    
    def encode(self, text: str) -> List[int]:
        """
        Кодирование текста в последовательность ID токенов.
        
        Args:
            text: Входной текст для кодирования
            
        Returns:
            List[int]: Список ID токенов
            
        Example:
            >>> tokenizer = BPETokenizer.load('vocab.json', 'merges.txt')
            >>> ids = tokenizer.encode("int main() { return 0; }")
            >>> print(ids)
            [45, 67, 89, 123, 45]
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
            
            # Находим пару с наименьшим рангом (самое раннее слияние)
            best_pair = min(mergeable, key=mergeable.get)
            current_text = self._merge_vocab(best_pair, [current_text])[0]
        
        # 4. Конвертируем в ID
        tokens = current_text.split()
        ids = []
        unknown_id = self.inverse_vocab.get('<UNK>', -1)
        
        for token in tokens:
            if token in self.inverse_vocab:
                ids.append(self.inverse_vocab[token])
            else:
                ids.append(unknown_id)
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Декодирование ID обратно в текст.
        
        Args:
            ids: Список ID токенов для декодирования
            
        Returns:
            str: Восстановленный текст
            
        Example:
            >>> tokenizer = BPETokenizer.load('vocab.json', 'merges.txt')
            >>> text = tokenizer.decode([45, 67, 89, 123, 45])
            >>> print(text)
            "int main() { return 0; }"
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
        
        # 3. Убираем маркер конца слова и заменяем на пробел
        text = text.replace('</w>', ' ')
        
        # 4. Byte-level декодирование
        if self.byte_level:
            text = self._byte_decode(text)
        
        # 5. Очистка от лишних пробелов
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    # ======================================================================
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    # ======================================================================
    
    def save(self, vocab_path: str, merges_path: str) -> None:
        """
        Сохранение словаря и операций слияния в файлы.
        
        Args:
            vocab_path: Путь для сохранения словаря (JSON формат)
            merges_path: Путь для сохранения операций слияния (текстовый формат)
            
        Example:
            >>> tokenizer.save('vocab.json', 'merges.txt')
        """
        # Сохраняем словарь
        vocab_dict = {str(k): v for k, v in self.vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
        
        # Сохраняем операции слияния
        with open(merges_path, 'w', encoding='utf-8') as f:
            f.write("#version: 0.2\n")
            for pair, rank in sorted(self.merges.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\n")
        
        logger.info(f"Словарь сохранен в {vocab_path}")
        logger.info(f"Операции слияния сохранены в {merges_path}")
    
    def save_binary(self, path: str) -> None:
        """
        Сохранение модели в единый бинарный файл.
        
        Args:
            path: Путь для сохранения
            
        Example:
            >>> tokenizer.save_binary('model.bin')
        """
        import pickle
        
        data = {
            'vocab_size': self.vocab_size,
            'byte_level': self.byte_level,
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
            'merges': self.merges,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Бинарная модель сохранена в {path}")
    
    @classmethod
    def load(cls, vocab_path: str, merges_path: str, byte_level: bool = True) -> 'BPETokenizer':
        """
        Загрузка обученного токенизатора из файлов.
        
        Args:
            vocab_path: Путь к файлу словаря (JSON формат)
            merges_path: Путь к файлу операций слияния (текстовый формат)
            byte_level: Использовать ли byte-level режим
            
        Returns:
            BPETokenizer: Загруженный экземпляр токенизатора
            
        Example:
            >>> tokenizer = BPETokenizer.load('vocab.json', 'merges.txt')
        """
        tokenizer = cls(byte_level=byte_level)
        
        # Загружаем словарь
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        
        tokenizer.vocab = {int(k): v for k, v in vocab_dict.items()}
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        # Определяем специальные токены из словаря
        special_tokens = [t for t in tokenizer.vocab.values() 
                         if t.startswith('<') and t.endswith('>')]
        if special_tokens:
            tokenizer.special_tokens = special_tokens
        
        # Загружаем операции слияния
        tokenizer.merges = {}
        with open(merges_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.startswith('#version'):
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    tokenizer.merges[(parts[0], parts[1])] = i - 1
        
        # Инициализируем byte-level кодировщик если нужно
        if tokenizer.byte_level:
            tokenizer._init_byte_encoder()
        
        logger.info(f"Токенизатор загружен из {vocab_path} и {merges_path}")
        return tokenizer
    
    @classmethod
    def load_binary(cls, path: str) -> 'BPETokenizer':
        """
        Загрузка модели из бинарного файла.
        
        Args:
            path: Путь к бинарному файлу
            
        Returns:
            BPETokenizer: Загруженный экземпляр токенизатора
        """
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls(
            vocab_size=data['vocab_size'],
            byte_level=data['byte_level'],
            special_tokens=data['special_tokens']
        )
        
        tokenizer.vocab = data['vocab']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = data['merges']
        
        if tokenizer.byte_level:
            tokenizer._init_byte_encoder()
        
        logger.info(f"Токенизатор загружен из {path}")
        return tokenizer
    
    # ======================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ======================================================================
    
    def vocab_size(self) -> int:
        """
        Получить текущий размер словаря.
        
        Returns:
            int: Количество токенов в словаре
        """
        return len(self.vocab)
    
    def merges_count(self) -> int:
        """
        Получить количество выполненных слияний.
        
        Returns:
            int: Количество правил слияния
        """
        return len(self.merges)
    
    def unknown_token_id(self) -> int:
        """
        Получить ID токена <UNK>.
        
        Returns:
            int: ID токена <UNK> или -1 если не найден
        """
        return self.inverse_vocab.get('<UNK>', -1)
    
    def __repr__(self) -> str:
        """
        Строковое представление токенизатора.
        
        Returns:
            str: Информация о токенизаторе
        """
        return (f"BPETokenizer(vocab_size={len(self.vocab)}, "
                f"byte_level={self.byte_level}, "
                f"special_tokens={len(self.special_tokens)})")


# ======================================================================
# ТЕСТИРОВАНИЕ ПРИ ЗАПУСКЕ
# ======================================================================

if __name__ == '__main__':
    # Простой тест при запуске модуля
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ BPE ТОКЕНИЗАТОРА")
    print("=" * 60)
    
    # Создаем маленький корпус
    corpus = [
        "int main() { return 0; }",
        "std::cout << \"Hello\" << std::endl;",
        "class Test { public: void method(); };",
        "template<typename T> T max(T a, T b);",
    ]
    
    print(f"\nКорпус: {len(corpus)} примеров")
    
    # Обучаем токенизатор
    tokenizer = BPETokenizer(vocab_size=50, byte_level=True)
    tokenizer.train(corpus, verbose=True)
    
    # Тестируем на примере
    test_text = corpus[0]
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nТест encode/decode:")
    print(f"   Оригинал: {test_text}")
    print(f"   Токены: {encoded}")
    print(f"   Декодировано: {decoded}")
    print(f"   Совпадение: {'[OK]' if test_text == decoded else '[BAD]'}")
    
    print(f"\nТест завершен!")