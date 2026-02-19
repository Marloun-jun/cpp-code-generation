# bpe/trainer.py
import os
import sys
from typing import List, Optional
from pathlib import Path
from tokenizer import BPETokenizer

def train_from_corpus(
    corpus_path: str,
    vocab_size: int = 8000,
    byte_level: bool = True,
    special_tokens: List[str] = None,
    output_dir: str = './bpe'
):
    """
    Обучение BPE токенизатора из файла корпуса
    
    Args:
        corpus_path: Путь к файлу с корпусом (построчно)
        vocab_size: Размер словаря
        byte_level: Использовать byte-level
        special_tokens: Специальные токены
        output_dir: Директория для сохранения
    """
    print(f"Loading corpus from {corpus_path}...")
    
    # Загружаем корпус      
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(line)
    
    print(f"Loaded {len(corpus)} examples")
    
    # Инициализируем токенизатор
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        byte_level=byte_level,
        special_tokens=special_tokens
    )
    
    # Обучаем
    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tokenizer.train(corpus, verbose=True)
    
    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    
    vocab_path = os.path.join(output_dir, 'vocab.json')
    merges_path = os.path.join(output_dir, 'merges.txt')
    
    tokenizer.save(vocab_path, merges_path)
    print(f"Tokenizer saved to {output_dir}")
    
    return tokenizer

def test_tokenizer(tokenizer: BPETokenizer, test_texts: List[str]):
    """
    Тестирование токенизатора на примерах
    """
    print("\n" + "="*50)
    print("TESTING TOKENIZER")
    print("="*50)
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        print(f"\nOriginal: {text}")
        print(f"Encoded: {encoded[:20]}... (length: {len(encoded)})")
        print(f"Decoded: {decoded}")
        print(f"Match: {text == decoded}")
        
        # Показываем первые 10 токенов
        tokens = []
        for idx in encoded[:10]:
            tokens.append(tokenizer.vocab[idx])
        print(f"Tokens: {tokens}")

if __name__ == '__main__':
    # ================ НАСТРОЙКИ ================
    CORPUS_PATH = '3_my_cpp_nn_project/cpp-bpe-tokenizer/data/corpus/train_code.txt'  # <-- ИЗМЕНИТЬ ПУТЬ ПРИ НЕОБХОДИМОСТИ
    VOCAB_SIZE = 10000
    OUTPUT_DIR = './bpe'
    # ============================================
    
    # Специальные токены для кода на C++
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<CPP>', '<CODE>']
    
    # Проверяем существует ли файл
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus file not found at {CORPUS_PATH}")
        print("Please update CORPUS_PATH variable with correct path to train_code.txt")
        sys.exit(1)
    
    # Обучение
    tokenizer = train_from_corpus(
        corpus_path=CORPUS_PATH,
        vocab_size=VOCAB_SIZE,
        byte_level=True,
        special_tokens=special_tokens,
        output_dir=OUTPUT_DIR
    )
    
    # Тестирование
    test_texts = [
        "#include <iostream>",
        "int main() { return 0; }",
        'std::cout << "Hello, мир!" << std::endl;',
        "// Это комментарий на русском языке"
    ]
    
    test_tokenizer(tokenizer, test_texts)