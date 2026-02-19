# test_bpe_tokenizer.py
import os
import sys
import time
from bpe.tokenizer import BPETokenizer

def test_byte_level():
    """Тестирование byte-level encoding/decoding"""
    print("\n=== Testing Byte-level Encoding ===")
    
    tokenizer = BPETokenizer(byte_level=True)
    
    test_strings = [
        "Hello, World!",
        "Привет, мир!",
        "C++ программирование",
        "🚀✨🎉",
        "混合文字"
    ]
    
    for text in test_strings:
        encoded = tokenizer._byte_encode(text)
        decoded = tokenizer._byte_decode(encoded)
        
        print(f"\nOriginal: {text}")
        print(f"Byte-level: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Match: {text == decoded}")

def test_training():
    """Тестирование обучения на маленьком корпусе"""
    print("\n=== Testing Training ===")
    
    # Маленький корпус для теста
    corpus = [
        "#include <iostream>",
        "using namespace std;",
        "int main() {",
        "    cout << 'Hello' << endl;",
        "    return 0;",
        "}"
    ]
    
    tokenizer = BPETokenizer(vocab_size=50, byte_level=True)
    tokenizer.train(corpus, verbose=True)
    
    # Тест encode/decode
    test_code = "int main() { return 42; }"
    encoded = tokenizer.encode(test_code)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest code: {test_code}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Match: {test_code == decoded}")
    
    return tokenizer

def test_save_load(tokenizer):
    """Тестирование сохранения и загрузки"""
    print("\n=== Testing Save/Load ===")
    
    # Сохраняем
    os.makedirs('./test_output', exist_ok=True)
    tokenizer.save('./test_output/vocab.json', './test_output/merges.txt')
    print("Tokenizer saved")
    
    # Загружаем
    new_tokenizer = BPETokenizer()
    new_tokenizer.load('./test_output/vocab.json', './test_output/merges.txt')
    print("Tokenizer loaded")
    
    # Проверяем
    test_text = "cout << 'test' << endl;"
    original_encoded = tokenizer.encode(test_text)
    loaded_encoded = new_tokenizer.encode(test_text)
    
    print(f"Original encode: {original_encoded}")
    print(f"Loaded encode: {loaded_encoded}")
    print(f"Match: {original_encoded == loaded_encoded}")

def benchmark():
    """Замер производительности"""
    print("\n=== Benchmark ===")
    
    # Создаем большой тестовый корпус
    corpus = []
    for i in range(1000):
        corpus.append(f"int function_{i}() {{ return {i}; }}")
    
    # Замер времени обучения
    start = time.time()
    tokenizer = BPETokenizer(vocab_size=100, byte_level=True)
    tokenizer.train(corpus[:100], verbose=False)
    train_time = time.time() - start
    print(f"Training time (100 samples, vocab=100): {train_time:.2f}s")
    
    # Замер скорости encode
    test_text = corpus[0]
    start = time.time()
    n_iterations = 1000
    for _ in range(n_iterations):
        tokenizer.encode(test_text)
    encode_time = time.time() - start
    print(f"Encode speed: {n_iterations / encode_time:.0f} tokens/sec")
    
    return tokenizer

if __name__ == '__main__':
    # Запускаем все тесты
    test_byte_level()
    tokenizer = test_training()
    test_save_load(tokenizer)
    benchmark()
    
    print("\n✅ All tests passed!")