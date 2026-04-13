#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# feedforward.py - Feed Forward Network с GELU активацией для Transformer
# ======================================================================
#
# @file feedforward.py
# @brief Реализация полносвязной сети прямого распространения (FFN) с GELU
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль реализует компонент Feed Forward Network (FFN),
#          который является частью блока Transformer. FFN применяется после
#          механизма внимания и обеспечивает нелинейные преобразования.
#
#          **Основные возможности:**
#
#          1. **Двухслойная архитектура**
#             - Расширение размерности (d_model → d_ff)
#             - Нелинейная активация (GELU)
#             - Сжатие обратно (d_ff → d_model)
#
#          2. **GELU активация**
#             - Gaussian Error Linear Unit
#             - Более гладкая, чем ReLU
#             - Лучшие результаты для Transformer моделей
#
#          3. **Dropout регуляризация**
#             - После активации
#             - После выходного слоя
#             - Предотвращает переобучение
#
#          4. **Гибкость размерности**
#             - Автоматическое d_ff = d_model * 4
#             - Возможность явного указания d_ff
#             - Поддержка различных архитектур
#
#          5. **Совместимость**
#             - Сохраняет размерность (batch, seq_len, d_model)
#             - Работает с батчами любой длины
#             - Совместим с остальными компонентами
#
# @usage
#     ff = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
#     output = ff(x)    # x.shape = (batch, seq_len, 512)
#
# @example
#     # Стандартное использование (d_ff = 4 * d_model)
#     ff = FeedForward(d_model=768)
#
#     # Явное указание размерности скрытого слоя
#     ff = FeedForward(d_model=512, d_ff=1024)
#
#     # Без dropout (не рекомендуется)
#     ff = FeedForward(d_model=256, dropout=0.0)
#
# ======================================================================

import torch
import torch.nn as nn

# ======================================================================
# КЛАСС FeedForward
# ======================================================================

class FeedForward(nn.Module):
    """
    Двухслойная сеть прямого распространения.
    
    Архитектура: Linear -> GELU -> Dropout -> Linear -> Dropout
    
    **Теоретическая справка:**
    В оригинальном Transformer (Vaswani et al., 2017) FFN применяется
    позиционно-независимо к каждому элементу последовательности.
    
    **Почему GELU?**
    - GELU (Gaussian Error Linear Unit) = x * Φ(x), где Φ - функция распределения Гаусса
    - Плавная аппроксимация ReLU с ненулевыми градиентами для отрицательных значений
    - Показывает лучшие результаты в Transformer моделях (BERT, GPT и др.)
    - GELU(x) ≈ x * sigmoid(1.702 * x)
    
    Args:
        d_model (int):        Размерность эмбеддингов (входная и выходная)
        d_ff (int, optional): Размерность скрытого слоя.
                              По умолчанию: d_model * 4
        dropout (float):      Вероятность dropout (по умолчанию: 0.1)
    
    **Размерность скрытого слоя:**
    - В оригинальном Transformer: d_ff = 2048 при d_model = 512 (коэффициент 4)
    - Для малых моделей: d_ff = d_model * 2
    - Для больших моделей: d_ff = d_model * 4 (стандарт)
    - Для очень больших: d_ff = d_model * 8
    
    @note
        FFN - самый ресурсоемкий компонент по памяти (из-за расширения размерности).
        При d_ff = 4 * d_model, FFN потребляет ~80% параметров модели!
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()    # GELU лучше чем ReLU для трансформеров
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через Feed Forward Network.
        
        Args:
            x (torch.Tensor): Входной тензор, форма (batch_size, seq_len, d_model)
        
        Returns:
            torch.Tensor: Выходной тензор, форма (batch_size, seq_len, d_model)
        
        **Процесс вычислений:**
        1. Проекция в скрытое пространство (d_model → d_ff)
        2. Применение GELU активации (нелинейность)
        3. Dropout для регуляризации
        4. Проекция обратно (d_ff → d_model)
        5. Dropout перед выходом
        
        @note
            FFN применяется одинаково к каждой позиции независимо.
            Это позволяет моделировать сложные нелинейные зависимости.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x