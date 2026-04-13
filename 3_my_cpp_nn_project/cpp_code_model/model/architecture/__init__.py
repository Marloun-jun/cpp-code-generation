#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# __init__.py - Модуль архитектуры трансформера для C++ кода
# ======================================================================
#
# @file __init__.py
# @brief Пакет с компонентами архитектуры трансформера для модели C++ кода
#
# @author Евгений П.
# @date 2026
# @version 3.1.0
#
# @details Этот модуль объединяет все компоненты архитектуры трансформера
#          в единый пакет для удобного импорта.
#
#          **Основные компоненты:**
#
#          1. **PositionalEncoding** - Позиционное кодирование
#             - Sin/cos кодирование для Transformer
#             - Поддержка различных максимальных длин
#
#          2. **MultiHeadAttention** - Многоголовое внимание
#             - Scaled dot-product attention
#             - Множество голов для захвата разных паттернов
#
#          3. **FeedForward** - Полносвязная сеть
#             - Два линейных слоя с активацией
#             - Расширение и сжатие размерности
#
#          4. **TransformerBlock** - Блок трансформера
#             - Attention + FeedForward
#             - LayerNorm и residual connections
#
#          5. **CppCodeModel** - Основная модель
#             - Полная архитектура для C++ кода
#             - Embedding + Transformer + LM head
#
# @usage
#     from architecture import CppCodeModel, TransformerBlock
#     from architecture import PositionalEncoding, MultiHeadAttention
#
# @example
#     # Создание модели
#     model = CppCodeModel(
#         vocab_size=10000,
#         d_model=512,
#         n_heads=8,
#         n_layers=6
#     )
#
#     # Использование отдельных компонентов
#     pos_enc = PositionalEncoding(d_model=512, max_len=1024)
#     attention = MultiHeadAttention(d_model=512, n_heads=8)
#
# ======================================================================

# ======================================================================
# ИМПОРТ КОМПОНЕНТОВ АРХИТЕКТУРЫ
# ======================================================================

from .positional import PositionalEncoding
from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .transformer import TransformerBlock
from .model import CppCodeModel

# ======================================================================
# ПУБЛИЧНЫЙ ИНТЕРФЕЙС
# ======================================================================

__all__ = [
    'PositionalEncoding',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerBlock',
    'CppCodeModel'
]