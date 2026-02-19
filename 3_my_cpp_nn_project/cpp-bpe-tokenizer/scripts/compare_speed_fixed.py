#!/usr/bin/env python3
import sys
import os
import time
import json

# Добавляем путь к bpe
bpe_path = os.path.join(os.path.dirname(__file__), '..', 'bpe')
sys.path.insert(0, bpe_path)

# Импортируем токенизатор
from tokenizer import BPETokenizer

def test_python_tokenizer(text, vocab_path, merges_path, iterations=50):
    """Тестирование Python токенизатора"""
    print(f"  Загрузка Python токенизатора...")
    tokenizer = BPETokenizer(32000, byte_level=True)
    
    vocab_full = os.path.join(bpe_path, vocab_path)
    merges_full = os.path.join(bpe_path, merges_path)
    
    print(f"  Vocab: {vocab_full}")
    print(f"  Merges: {merges_full}")
    
    tokenizer.load(vocab_full, merges_full)
    
    # Прогрев
    for _ in range(3):
        tokenizer.encode(text[:100])
    
    # Измерение
    start = time.time()
    total_tokens = 0
    for _ in range(iterations):
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)
    py_time = (time.time() - start) * 1000  # ms
    
    return total_tokens // iterations, py_time / iterations

def main():
    print("=== Сравнение Python и C++ токенизаторов ===\n")
    
    # Тестовый текст
    test_text = """#include <iostream>
#include <vector>

class Test {
public:
    Test(const std::string& name) : name_(name) {}
    void process() {
        for (int i = 0; i < 10; ++i) {
            data_.push_back(i);
        }
    }
private:
    std::string name_;
    std::vector<int> data_;
};

int main() {
    Test t("example");
    t.process();
    return 0;
}
"""
    
    print(f"Размер текста: {len(test_text)} байт\n")
    
    # Тестируем Python
    print("Тестирование Python...")
    py_tokens, py_time = test_python_tokenizer(
        test_text,
        "vocab.json",
        "merges.txt"
    )
    print(f"  Среднее время: {py_time:.3f} ms")
    print(f"  Среднее токенов: {py_tokens}")
    print(f"  Скорость: {len(test_text) / py_time * 1000:.0f} байт/сек")
    
    print("\nДля сравнения с C++ запустите:")
    print("cd ~/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/cpp/build")
    print("./benchmarks/bench_fast_tokenizer")

if __name__ == "__main__":
    main()