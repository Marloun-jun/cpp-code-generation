#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np

# Загружаем результаты
with open('benchmark_results.json', 'r') as f:
    results = json.load(f)

# Подготовка данных
names = ['HuggingFace', 'Python', 'C++']
encode_speed = [results['huggingface']['encode_speed'] / 1000,  # в K токен/сек
                results['python']['encode_speed'] / 1000,
                results['cpp']['encode_speed'] / 1000]

encode_time = [results['huggingface']['encode_time_ms'],
               results['python']['encode_time_ms'],
               results['cpp']['encode_time_ms']]

memory = [results['huggingface']['memory_mb'],
          results['python']['memory_mb'],
          results['cpp']['memory_mb']]

oov = [results['huggingface']['oov_rate'] * 100,
       results['python']['oov_rate'] * 100,
       results['cpp']['oov_rate'] * 100]

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Сравнение BPE токенизаторов', fontsize=16)

# График 1: Скорость encode
ax1 = axes[0, 0]
bars = ax1.bar(names, encode_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.set_ylabel('Скорость (K токенов/сек)')
ax1.set_title('Скорость encode')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.0f}K', ha='center', va='bottom')

# График 2: Время encode
ax2 = axes[0, 1]
bars = ax2.bar(names, encode_time, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax2.set_ylabel('Время (ms)')
ax2.set_title('Время encode (на текст)')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}ms', ha='center', va='bottom')

# График 3: Память
ax3 = axes[1, 0]
bars = ax3.bar(names, memory, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax3.set_ylabel('Память (MB)')
ax3.set_title('Использование памяти')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}MB', ha='center', va='bottom')

# График 4: OOV частота
ax4 = axes[1, 1]
bars = ax4.bar(names, oov, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax4.set_ylabel('OOV частота (%)')
ax4.set_title('Неизвестные токены')
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../reports/figures/comparison.png', dpi=150)
plt.show()

print("✅ Графики сохранены в reports/figures/comparison.png")