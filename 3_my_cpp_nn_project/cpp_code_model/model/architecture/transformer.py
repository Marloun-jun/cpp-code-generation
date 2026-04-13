#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# transformer.py - Блок трансформера с вниманием и residual связями
# ======================================================================
#
# @file transformer.py
# @brief Реализация одного слоя (блока) трансформера для языковой модели
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль реализует стандартный блок трансформера (Transformer Block),
#          который является основным строительным элементом современных языковых
#          моделей. Каждый блок состоит из механизма внимания и полносвязной сети
#          с residual связями и нормализацией.
#
#          **Основные возможности:**
#
#          1. **Multi-Head Self-Attention**
#             - Механизм внимания для захвата зависимостей между токенами
#             - Встроенное causal маскирование для авторегрессии
#             - Поддержка внешней маски для паддинга
#
#          2. **Feed-Forward Network**
#             - Двухслойная сеть с GELU активацией
#             - Увеличивает выразительную способность модели
#             - Позиционно-независимое применение
#
#          3. **Residual Connections (Skip Connections)**
#             - Помогают обучать глубокие сети
#             - Предотвращают затухание градиентов
#             - Улучшают сходимость
#
#          4. **Layer Normalization (Pre-LN)**
#             - Стабилизирует обучение
#             - Ускоряет сходимость
#             - Расположена ДО residual сложения (современная практика)
#
#          5. **Dropout для регуляризации**
#             - Применяется после каждого подслоя
#             - Предотвращает переобучение
#
# @usage
#     block = TransformerBlock(d_model=512, nhead=8, dropout=0.1)
#     output = block(x, mask=attention_mask)
#
# @example
#     # Базовое использование
#     x = torch.randn(32, 128, 512)    # (batch, seq_len, d_model)
#     mask = torch.ones(32, 128)       # маска для паддинга
#     out = block(x, mask)
#
#     # Стек из N блоков
#     blocks = nn.ModuleList([
#         TransformerBlock(512, 8) for _ in range(6)
#     ])
#     for block in blocks:
#         x = block(x, mask)
#
# ======================================================================

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward

# ======================================================================
# КЛАСС TransformerBlock
# ======================================================================

class TransformerBlock(nn.Module):
    """
    Базовый блок трансформера (Transformer Block / Transformer Layer).
    
    **Архитектура (Pre-LN, современный вариант):**
    x -> Attention -> dropout -> + -> LayerNorm -> FFN -> dropout -> + -> LayerNorm -> output
    |
    residual
    
    **Структура:**
    1. Multi-Head Self-Attention
    2. Residual connection (x + attention_output)
    3. Layer Normalization
    4. Feed-Forward Network
    5. Residual connection (x + ffn_output)
    6. Layer Normalization

    **Pre-LN vs Post-LN:**
    - Pre-LN (эта реализация):            LayerNorm перед residual сложением
    - Post-LN (оригинальный Transformer): LayerNorm после residual сложения
    - Pre-LN обеспечивает лучшую стабильность градиентов для глубоких сетей

    Args:
        d_model (int):   Размерность эмбеддингов
        nhead (int):     Количество голов внимания
        dropout (float): Вероятность dropout (по умолчанию: 0.1)

    **Современные модификации:**
    - Pre-LN:      Более стабильное обучение (используется в GPT, LLaMA)
    - Post-LN:     Оригинальная версия (сложнее в обучении)
    - Sandwich-LN: Дополнительная нормализация внутри FFN

    @note
    Порядок операций критически важен для стабильности обучения.
    Данная реализация следует современной практике Pre-LN.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход через блок трансформера.
        
        Args:
            x (torch.Tensor):              Входной тензор, форма (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Маска внимания для паддинга, форма (batch_size, seq_len)
        
        Returns:
            torch.Tensor: Выходной тензор, форма (batch_size, seq_len, d_model)
        
        **Процесс:**
        
        1. **Self-Attention с residual:**
        attn_out = attention(x, x, x, mask)
        x = norm1(x + dropout(attn_out))
        - x используется как query, key, value (self-attention)
        - Dropout применяется к выходу внимания
        - Residual сложение: x + attn_out
        - LayerNorm стабилизирует активации

        2. **Feed-Forward с residual:**
        ffn_out = ffn(x)
        x = norm2(x + dropout(ffn_out))
        - FFN применяется позиционно-независимо
        - Снова residual связь и нормализация

        @note
        Causal маска добавляется внутри MultiHeadAttention,
        поэтому здесь передается только маска для паддинга.

        **Важные детали:**
        - Dropout применяется ДО residual сложения (стандартная практика)
        - LayerNorm применяется ПОСЛЕ residual сложения (Pre-LN)
        - Порядок: residual -> norm (не norm -> residual)
        """
        # Self-attention с residual
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward с residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x