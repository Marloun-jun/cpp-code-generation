import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bpe.tokenizer import BPETokenizer

# Загружаем токенизатор
tokenizer = BPETokenizer(byte_level=True)
tokenizer._init_byte_encoder()  # <-- ЯВНО инициализируем
tokenizer.load(
    vocab_path='3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe/vocab.json',
    merges_path='3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe/merges.txt'
)

test_cases = [
    "#include <iostream>",
    "int main() { return 0; }",
    'std::cout << "Hello, мир!" << std::endl;',
    "// Это комментарий на русском языке"
]

for text in test_cases:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {text}")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {text == decoded}")
    print()

    # '3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe/vocab.json', '3_my_cpp_nn_project/cpp-bpe-tokenizer/bpe/merges.txt'