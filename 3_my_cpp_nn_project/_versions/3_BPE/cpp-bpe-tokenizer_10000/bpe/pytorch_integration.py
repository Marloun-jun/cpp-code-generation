# bpe/pytorch_integration.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Optional
import json

class BPETokenizerWrapper:
    """
    Обертка для использования BPE токенизатора с PyTorch
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        """
        Args:
            tokenizer: Обученный BPETokenizer
            max_length: Максимальная длина последовательности
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # ID специальных токенов
        self.pad_id = tokenizer.inverse_vocab.get('<PAD>', 0)
        self.unk_id = tokenizer.inverse_vocab.get('<UNK>', 1)
        self.bos_id = tokenizer.inverse_vocab.get('<BOS>', 2)
        self.eos_id = tokenizer.inverse_vocab.get('<EOS>', 3)
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Кодирование текста в тензоры PyTorch
        
        Returns:
            Dict с input_ids, attention_mask
        """
        tokens = self.tokenizer.encode(text)
        
        # Добавляем специальные токены
        if add_special_tokens:
            tokens = [self.bos_id] + tokens + [self.eos_id]
        
        # Обрезаем при необходимости
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Создаем attention_mask
        attention_mask = [1] * len(tokens)
        
        # Паддинг
        if padding:
            pad_len = self.max_length - len(tokens)
            if pad_len > 0:
                tokens += [self.pad_id] * pad_len
                attention_mask += [0] * pad_len
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
    
    def decode(self, tokens: Union[List[int], torch.Tensor]) -> str:
        """Декодирование тензора обратно в текст"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # Фильтруем специальные токены
        filtered_tokens = []
        for token in tokens:
            if token not in [self.pad_id, self.bos_id, self.eos_id]:
                filtered_tokens.append(token)
        
        return self.tokenizer.decode(filtered_tokens)

class CodeDataset(Dataset):
    """Dataset для кода на C++"""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer_wrapper: BPETokenizerWrapper,
                 max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer_wrapper
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        return encoded

def create_dataloader(
    texts: List[str],
    tokenizer: BPETokenizerWrapper,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True
) -> DataLoader:
    """
    Создание DataLoader для PyTorch
    """
    dataset = CodeDataset(texts, tokenizer, max_length)
    
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

# Пример использования с PyTorch
def example_usage():
    """
    Пример интеграции с PyTorch
    """
    from tokenizer import BPETokenizer
    
    # Загружаем обученный токенизатор
    tokenizer = BPETokenizer()
    tokenizer.load('./bpe/vocab.json', './bpe/merges.txt')
    
    # Создаем обертку
    wrapper = BPETokenizerWrapper(tokenizer, max_length=128)
    
    # Пример текстов
    texts = [
        "#include <iostream>",
        "int main() { return 0; }",
        "std::vector<int> vec;"
    ]
    
    # Кодируем один пример
    encoded = wrapper.encode(texts[0], padding=True)
    print(f"Input shape: {encoded['input_ids'].shape}")
    print(f"Input IDs: {encoded['input_ids']}")
    print(f"Attention mask: {encoded['attention_mask']}")
    
    # Декодируем обратно
    decoded = wrapper.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")
    
    # Создаем DataLoader
    dataloader = create_dataloader(texts, wrapper, batch_size=2)
    
    for batch in dataloader:
        print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

if __name__ == '__main__':
    example_usage()