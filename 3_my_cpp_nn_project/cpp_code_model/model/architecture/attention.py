#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# attention.py - Многоголовое внимание с маскированием для авторегрессии
# ======================================================================
#
# @file attention.py
# @brief Реализация многоголового внимания (Multi-Head Attention) с causal маской
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль реализует механизм многоголового внимания, ключевой
#          компонент архитектуры Transformer. Предназначен для авторегрессионных
#          задач (causal language modeling), где важно не "подглядывать"
#          в будущие токены.
#
#          **Основные возможности:**
#
#          1. **Многоголовое внимание**
#             - Разделение d_model на nhead голов
#             - Каждая голова изучает разные паттерны
#             - Scaled dot-product attention
#
#          2. **Causal маскирование**
#             - Верхнетреугольная матрица для запрета будущих токенов
#             - Автоматическое создание и кэширование маски
#             - Поддержка различных длин последовательностей
#
#          3. **Внешние маски**
#             - Поддержка маски для паддинга (<PAD> токенов)
#             - Совместимость с attention mask из DataModule
#
#          4. **Проекции Q, K, V**
#             - Три независимых линейных слоя
#             - Выходная проекция после объединения голов
#             - Dropout для регуляризации
#
#          5. **Эффективность**
#             - Кэширование causal mask для повторного использования
#             - Оптимизированные операции с тензорами
#             - Поддержка batch-обработки
#
# @usage
#     attention = MultiHeadAttention(d_model=512, nhead=8, dropout=0.1)
#     output = attention(query, key, value, mask=attention_mask)
#
# @example
#     # Базовое использование (self-attention)
#     x = torch.randn(32, 128, 512)    # (batch, seq_len, d_model)
#     out = attention(x, x, x)
#
#     # С внешней маской для паддинга
#     mask = torch.ones(32, 128)  # 1 - реальные токены, 0 - паддинг
#     out = attention(x, x, x, mask=mask)
#
#     # Cross-attention (например, encoder-decoder)
#     encoder_out = torch.randn(32, 128, 512)
#     decoder_in = torch.randn(32, 64, 512)
#     out = attention(decoder_in, encoder_out, encoder_out)
#
# ======================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# КЛАСС MultiHeadAttention
# ======================================================================

class MultiHeadAttention(nn.Module):
    """
    Многоголовое внимание с causal masking.
    
    Позволяет модели фокусироваться на разных аспектах последовательности.
    Causal mask предотвращает "подглядывание" в будущие токены.
    
    **Архитектура:**
    1. Три линейных проекции для Q, K, V
    2. Разделение на nhead голов (каждая размерности d_k = d_model / nhead)
    3. Scaled dot-product attention
    4. Объединение голов
    5. Выходная линейная проекция
    
    Args:
        d_model (int):   Размерность эмбеддингов (должна делиться на nhead)
        nhead (int):     Количество голов внимания
        dropout (float): Вероятность dropout (по умолчанию: 0.1)
    
    **Теоретическая справка:**
    - Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    - Causal mask обнуляет attention к будущим позициям
    - Многоголовость позволяет модели захватывать разные типы зависимостей
    
    @note
        Требует, чтобы d_model был кратен nhead. Обычно используются:
        - d_model = 512, nhead = 8  (d_k = 64)
        - d_model = 768, nhead = 12 (d_k = 64)
        - d_model = 1024, nhead = 16 (d_k = 64)
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead    # Размерность каждой головы
        
        # Проекции Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Выходная проекция
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Регистрируем causal mask как буфер (не обучаемый параметр)
        self.register_buffer("causal_mask", None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Создает causal mask (верхнетреугольную матрицу) для авторегрессии.
        
        Args:
            seq_len (int):         Длина последовательности
            device (torch.device): Устройство для тензора
            
        Returns:
            torch.Tensor: Булева маска размера (seq_len, seq_len),
                          где True означает запрещенные позиции
        
        **Алгоритм:**
        1. Создается матрица единиц размера (seq_len, seq_len)
        2. torch.triu() оставляет элементы выше главной диагонали
        3. diagonal=1 означает, что главная диагональ не включается
        4. Сравнение с 1 дает булеву маску (True для запрещенных)
        
        @note
            Маска кэшируется для повторного использования с разными seq_len.
            При увеличении seq_len маска пересоздается.
        """
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask == 1    # True для запрещенных позиций
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Прямой проход многоголового внимания.
        
        Args:
            query (torch.Tensor):          Тензор запросов, форма (batch_size, seq_len, d_model)
            key (torch.Tensor):            Тензор ключей, форма (batch_size, seq_len, d_model)
            value (torch.Tensor):          Тензор значений, форма (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Внешняя маска (для паддинга),
                                           форма (batch_size, seq_len) или (seq_len, seq_len)
        
        Returns:
            torch.Tensor: Выходной тензор, форма (batch_size, seq_len, d_model)
        
        **Процесс вычислений:**
        1. Линейные проекции Q, K, V
        2. Reshape для разделения на головы: (batch, nhead, seq_len, d_k)
        3. Вычисление scores = Q @ K^T / sqrt(d_k)
        4. Применение causal mask (запрет будущих позиций)
        5. Применение внешней маски (для паддинга)
        6. Softmax и dropout
        7. Взвешенная сумма значений: out = scores @ V
        8. Объединение голов: (batch, seq_len, d_model)
        9. Выходная проекция
        
        @note
            Для self-attention передавайте query=key=value=x
            Для cross-attention: query - декодер, key/value - энкодер
        """
        batch_size, seq_len, _ = query.shape
        
        # Линейные проекции и reshape для голов
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: (batch, nhead, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Применяем causal mask (для авторегрессии)
        causal_mask = self._get_causal_mask(seq_len, query.device)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Применяем внешнюю маску (для паддинга)
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax и dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Применяем внимание к значениям
        out = torch.matmul(attn, V)    # (batch, nhead, seq_len, d_k)
        
        # Объединяем головы
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Выходная проекция
        return self.w_o(out)