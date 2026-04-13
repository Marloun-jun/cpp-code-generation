#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# model.py - Полная модель трансформера для генерации C++ кода
# ======================================================================
#
# @file model.py
# @brief Реализация полной архитектуры Transformer для генерации C++ кода
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Этот модуль реализует полную модель трансформера для авторегрессионной
#          генерации C++ кода. Модель наследуется от PyTorch Lightning для
#          удобного обучения и валидации.
#
#          **Основные возможности:**
#
#          1. **Полная архитектура Transformer**
#             - Token Embedding (обучаемые)
#             - Positional Encoding (sin/cos)
#             - N блоков Transformer (self-attention + FFN)
#             - LayerNorm и Residual connections
#             - LM Head для генерации токенов
#
#          2. **PyTorch Lightning интеграция**
#             - training_step с логированием
#             - validation_step с perplexity
#             - Автоматическая настройка оптимизатора с warmup
#             - Checkpointing и логирование
#
#          3. **Механизмы регуляризации**
#             - Dropout на всех уровнях
#             - Weight decay в AdamW
#             - Xavier инициализация весов
#
#          4. **Генерация кода**
#             - Авторегрессионная генерация
#             - Сэмплирование с температурой
#             - Поддержка BOS/EOS токенов (считывание из токенизатора)
#             - Защита от зацикливания
#             - Обрезка слишком длинных промптов
#             - Top-k и Top-p sampling (optional)
#
#          5. **Гибкость архитектуры**
#             - Настраиваемые размеры (d_model, nhead)
#             - Конфигурируемое количество слоев
#             - Параметризуемый max_len и dropout
#             - Настраиваемый scheduler с warmup
#
# @usage
#     model = CppCodeModel(
#         vocab_size=10000,
#         d_model=256,
#         nhead=4,
#         num_layers=4
#     )
#
# @example
#     # Обучение
#     trainer = pl.Trainer(max_epochs=10)
#     trainer.fit(model, datamodule)
#
#     # Генерация кода
#     code = model.generate(tokenizer, "#include <iostream>")
#
# ======================================================================

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from .positional import PositionalEncoding
from .transformer import TransformerBlock

# ======================================================================
# КЛАСС CppCodeModel
# ======================================================================

class CppCodeModel(pl.LightningModule):
    """
    Модель для генерации C++ кода на основе трансформера.
    
    **Архитектура:**
        Token Embedding -> Positional Encoding -> N Transformer Blocks -> LayerNorm -> LM Head
    
    **Компоненты:**
        1. **Token Embedding**     - преобразует ID токенов в векторы d_model
        2. **Positional Encoding** - добавляет информацию о позиции (sin/cos)
        3. **Transformer Blocks**  - N слоев с self-attention и FFN
        4. **LayerNorm**           - финальная нормализация
        5. **LM Head**             - проекция на словарь для предсказания токенов
    
    Args:
        vocab_size (int):      Размер словаря (количество уникальных токенов)
        d_model (int):         Размерность эмбеддингов (по умолчанию: 256)
        nhead (int):           Количество голов внимания (по умолчанию: 4)
        num_layers (int):      Количество слоев трансформера (по умолчанию: 4)
        max_len (int):         Максимальная длина последовательности (по умолчанию: 1024)
        dropout (float):       Вероятность dropout (по умолчанию: 0.1)
        learning_rate (float): Скорость обучения для AdamW (по умолчанию: 3e-4)
        weight_decay (float):  Коэффициент L2 регуляризации (по умолчанию: 0.01)
        scheduler_t_max (int): Количество эпох для CosineAnnealing (по умолчанию: 10)
        warmup_steps (int):    Количество шагов для линейного warmup (по умолчанию: 0)
    
    @note
        Модель использует causal masking внутри Transformer блоков для
        авторегрессионной генерации (не подглядывает в будущее).
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_layers: int = 4,
                 max_len: int = 1024,
                 dropout: float = 0.1,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 0.01,
                 scheduler_t_max: int = 10,
                 warmup_steps: int = 0):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Embedding слои
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer блоки
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # Выходные слои
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """
        Инициализация весов Xavier/Glorot.
        
        **Метод инициализации:**
        - Для линейных слоев: Xavier uniform
        - Для эмбеддингов: стандартная PyTorch инициализация
        - Для слоев нормализации: стандартная (weight=1, bias=0)
        
        Xavier инициализация помогает избежать проблем с затухающими/взрывающимися
        градиентами на начальных этапах обучения.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            x (torch.Tensor):              Входные токены, форма (batch_size, seq_len)
            mask (torch.Tensor, optional): Маска внимания для паддинга,
                                           форма (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Логиты предсказаний, форма (batch_size, seq_len, vocab_size)
        
        **Процесс:**
        1. Преобразование токенов в эмбеддинги
        2. Добавление позиционного кодирования
        3. Последовательное применение Transformer блоков
        4. Финальная LayerNorm
        5. Проекция на размер словаря (LM Head)
        """
        # Embedding
        x = self.token_embedding(x)    # (batch, seq_len, d_model)
        x = self.pos_encoding(x)       # (batch, seq_len, d_model)
        
        # Transformer блоки
        for block in self.blocks:
            x = block(x, mask)
        
        # Выходной слой
        x = self.ln_f(x)
        logits = self.lm_head(x)    # (batch, seq_len, vocab_size)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        """
        Один шаг обучения.
        
        Args:
            batch (dict):    Словарь с данными батча
                             - 'input_ids':      Входные токены
                             - 'labels':         Целевые токены
                             - 'attention_mask': Маска внимания
            batch_idx (int): Индекс батча
            
        Returns:
            torch.Tensor: Значение функции потерь (cross-entropy)
        
        **Особенности:**
        - ignore_index=0 игнорирует <PAD> токены при расчете loss
        - Логирует train_loss в прогресс-бар
        """
        input_ids = batch['input_ids']
        labels = batch['labels']
        mask = batch['attention_mask']
        
        # Forward pass
        logits = self(input_ids, mask)
        
        # Вычисляем loss (игнорируем паддинг)
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.hparams.vocab_size),
            labels.view(-1),
            ignore_index=0    # Игнорирование <PAD>
        )
        
        # Логирование
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Шаг валидации.
        
        Args:
            batch (dict):    Словарь с данными батча
            batch_idx (int): Индекс батча
            
        Returns:
            torch.Tensor: Значение функции потерь
        
        **Метрики:**
        - val_loss:       cross-entropy loss на валидации
        - val_perplexity: exp(val_loss), мера уверенности модели
        
        Perplexity (перплексия) интерпретируется как "среднее количество
        равновероятных вариантов следующего токена". Чем ниже - тем лучше.
        """
        input_ids = batch['input_ids']
        labels = batch['labels']
        mask = batch['attention_mask']
        
        logits = self(input_ids, mask)
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, self.hparams.vocab_size),
            labels.view(-1),
            ignore_index=0
        )
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', perplexity, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Настройка оптимизатора и планировщика скорости обучения.
        
        Returns:
            dict: Словарь с оптимизатором и scheduler'ом
        
        **Оптимизатор: AdamW**
        - betas=(0.9, 0.95): Стандартные для Transformer
        - weight_decay:      L2 регуляризация (из параметров)
        
        **Scheduler: LinearWarmup + CosineAnnealing**
        - Warmup: линейное увеличение lr от 0 до target
        - Затем:  cosine decay до eta_min
        - Стабилизирует обучение в начале
        
        @note
            Warmup scheduler критически важен для стабильного обучения
            больших Transformer моделей.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay
        )
        
        # Получаем warmup steps (по умолчанию 0 - без warmup)
        warmup_steps = self.hparams.warmup_steps
        
        if warmup_steps > 0:
            # Создаем cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler_t_max,
                eta_min=1e-6
            )
            
            # Функция для линейного warmup
            def warmup_lambda(step):
                """Линейное увеличение learning rate от 0 до 1 за warmup_steps шагов"""
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return 1.0
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            
            # Комбинируем warmup и cosine annealing
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'step'    # Обновляем каждый шаг во время warmup
                }
            }
        else:
            # Без warmup используем только cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.scheduler_t_max,
                eta_min=1e-6
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
    
    def generate(self, tokenizer, prompt: str, max_tokens: int = 100, 
                 temperature: float = 0.8, repetition_penalty: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0) -> str:
        """
        Генерация кода по промпту (авторегрессионная генерация).
        
        Args:
            tokenizer:                  FastBPETokenizer - токенизатор для кодирования/декодирования
            prompt (str):               Начальный текст (промпт) для генерации
            max_tokens (int):           Максимальное количество генерируемых токенов
            temperature (float):        Температура сэмплирования (чем выше - тем разнообразнее)
            repetition_penalty (float): Штраф за повторения (>1 уменьшает повторы)
            top_k (int):                Top-k sampling (0 - отключено, >0 оставляет только k лучших токенов)
            top_p (float):              Top-p (nucleus) sampling (1.0 - отключено, <1.0 фильтрует по кумулятивной вероятности)
        
        Returns:
            str: Сгенерированный код
        
        **Алгоритм генерации:**
        1. Токенизация промпта и добавление <BOS> токена
        2. Обрезка слишком длинного промпта (оставляем последние max_len токенов)
        3. Итеративное предсказание следующего токена:
           - Forward pass текущей последовательности
           - Взятие логитов последнего токена
           - Применение repetition penalty (если задан)
           - Применение temperature
           - Top-k фильтрация (если задана)
           - Top-p (nucleus) фильтрация (если задана)
           - Softmax для получения вероятностей
           - Сэмплирование следующего токена
        4. Остановка при генерации <EOS> или достижении max_tokens
        5. Защита от зацикливания (5+ одинаковых токенов подряд)
        6. Декодирование (убираем <BOS>)
        
        **Temperature эффекты:**
        - Низкая (0.1-0.5):  Детерминированная, предсказуемая генерация
        - Средняя (0.7-1.0): Баланс между разнообразием и качеством
        - Высокая (1.0+):    Креативная, но может быть бессмысленной
        
        **Repetition penalty:**
        - 1.0:     Без штрафа
        - 1.1-1.2: Умеренное подавление повторений
        - 1.5+:    Сильное подавление (может нарушить грамматику)
        
        **Top-k sampling:**
        - 0:      Отключено
        - 40-100: Стандартные значения, отсекает маловероятные токены
        
        **Top-p (nucleus) sampling:**
        - 1.0:      Отключено
        - 0.9-0.95: Стандартные значения, выбирает токены с кумулятивной вероятностью p
        
        @note
            Модель переводится в режим eval() на время генерации
            (отключает dropout, batch norm и т.д.)
        """
        self.eval()
        
        # Получаем ID специальных токенов из токенизатора
        bos_token_id = getattr(tokenizer, 'bos_token_id', 2)
        eos_token_id = getattr(tokenizer, 'eos_token_id', 3)
        pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
        
        # Токенизация промпта
        input_ids = tokenizer.encode(prompt)
        
        # Обрезка слишком длинного промпта (оставляем последние max_len-1 токенов)
        max_context_len = self.hparams.max_len - 1    # -1 для BOS
        if len(input_ids) > max_context_len:
            input_ids = input_ids[-max_context_len:]
        
        input_ids = [bos_token_id] + input_ids    # <BOS>
        
        generated = input_ids.copy()
        
        # Для отслеживания повторений
        last_tokens = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Преобразуем в тензор
                x = torch.tensor([generated], device=self.device)
                
                # Forward pass
                logits = self(x)
                
                # Берем последний токен
                next_token_logits = logits[0, -1, :].clone()
                
                # Применяем repetition penalty (штраф за повторяющиеся токены)
                if repetition_penalty != 1.0 and last_tokens:
                    for token_id in set(last_tokens):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty
                
                # Применяем temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                else:
                    # Аргументированный выбор: берем argmax
                    next_token = torch.argmax(next_token_logits).item()
                    generated.append(next_token)
                    continue
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Сэмплируем следующий токен
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Останавливаемся на EOS
                if next_token == eos_token_id:
                    break
                
                # Защита от зацикливания (5 одинаковых токенов подряд)
                last_tokens.append(next_token)
                if len(last_tokens) > 5:
                    last_tokens.pop(0)
                    if len(set(last_tokens)) == 1:    # Все токены одинаковые
                        break
                
                generated.append(next_token)
        
        # Декодируем (убираем <BOS>)
        return tokenizer.decode(generated[1:])
    
    def on_fit_start(self):
        """
        Метод, вызываемый в начале обучения.
        
        Полезен для:
        - Проверки совместимости с токенизатором
        - Вывода информации о модели
        - Инициализации дополнительных компонентов
        """
        print(f"\n{'='*60}")
        print(f"Модель CppCodeModel инициализирована:")
        print(f"- vocab_size: {self.hparams.vocab_size}")
        print(f"- d_model: {self.hparams.d_model}")
        print(f"- nhead: {self.hparams.nhead}")
        print(f"- num_layers: {self.hparams.num_layers}")
        print(f"- max_len: {self.hparams.max_len}")
        print(f"- learning_rate: {self.hparams.learning_rate}")
        print(f"- weight_decay: {self.hparams.weight_decay}")
        print(f"- warmup_steps: {self.hparams.warmup_steps}")
        print(f"- Всего параметров: {sum(p.numel() for p in self.parameters()):,}")
        print(f"{'='*60}\n")