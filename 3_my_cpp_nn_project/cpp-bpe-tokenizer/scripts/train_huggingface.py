from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os

# Загружаем датасет
train_file = '/home/john/Projects/NS/3_my_cpp_nn_project/cpp-bpe-tokenizer/data/corpus/train_code.txt'
if not os.path.exists(train_file):
    print(f"Файл не найден: {train_file}")
    # Создаем тестовые данные
    with open(train_file, 'w') as f:
        for _ in range(1000):
            f.write("int main() { return 0; }\n")

with open(train_file, 'r') as f:
    lines = f.readlines()

# Создаем BPE токенизатор
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# Тренируем
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    min_frequency=2,
    show_progress=True
)

print("Обучение HuggingFace токенизатора...")
tokenizer.train_from_iterator(lines, trainer=trainer)
tokenizer.save("hf_tokenizer.json")
print("✅ HuggingFace токенизатор сохранен в hf_tokenizer.json")