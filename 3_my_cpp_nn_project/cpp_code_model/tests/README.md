# 🧪 Тесты для проверки моделей

Эта папка содержит тесты для проверки метрик и качества генерации моделей. Тесты позволяют оценить perplexity, производительность, использование памяти и качество генерируемого кода.

## 📋 Содержание

- [🧪 Тесты для проверки моделей](#-тесты-для-проверки-моделей)
  - [📋 Содержание](#-содержание)
  - [📊 Тесты метрик](#-тесты-метрик)
    - [**`test_base_model.py`** — метрики базовой модели](#test_base_modelpy--метрики-базовой-модели)
    - [**`test_lora_model.py`** — метрики LoRA модели](#test_lora_modelpy--метрики-lora-модели)
  - [🎯 Тесты генерации](#-тесты-генерации)
    - [**`test_work_base_model.py`** — генерация из затравки](#test_work_base_modelpy--генерация-из-затравки)
    - [**`test_work_lora_model.py`** — генерация по инструкции](#test_work_lora_modelpy--генерация-по-инструкции)
  - [📈 Сравнение моделей](#-сравнение-моделей)
    - [**`compare_base_models.py`** — сравнение Tiny/Small/Medium](#compare_base_modelspy--сравнение-tinysmallmedium)
  - [📊 Сводная таблица тестов](#-сводная-таблица-тестов)
  - [🚀 Типичные сценарии использования](#-типичные-сценарии-использования)
    - [**Проверка качества базовой модели**](#проверка-качества-базовой-модели)
    - [**Проверка качества LoRA модели**](#проверка-качества-lora-модели)
    - [**Сравнение всех моделей**](#сравнение-всех-моделей)
  - [📁 Структура каталога](#-структура-каталога)

## 📊 Тесты метрик

### **`test_base_model.py`** — метрики базовой модели

**Назначение:** Комплексное тестирование метрик базовой модели (Tiny/Small/Medium).

**Измеряемые метрики:**
| Тест | Метрики |
|------|---------|
| Базовые метрики | Параметры, размер модели, размер чекпоинта |
| Численная стабильность | NaN, Inf, max logit |
| Perplexity | Perplexity, avg loss, std loss |
| Производительность | Время инференса, пропускная способность |
| Использование памяти | Память для batch_size 1/2/4/8 |

**Запуск:**
```bash
python tests/test_base_model.py
```

**Выходные файлы (пример для модели medium):**
- `reports/test_medium/memory_usage.png` — график памяти
- `reports/test_medium/metrics_report.json` — JSON с метриками
- `reports/test_medium/METRICS_REPORT.txt` — текстовый отчёт
- `reports/test_medium/performance_plot.png` — график производительности

**Примечание:** тестируемая модель указывается в ручную в коде 

### **`test_lora_model.py`** — метрики LoRA модели

**Назначение:** Комплексное тестирование метрик LoRA модели (Medium + LoRA).

**Измеряемые метрики:**
| Тест | Метрики |
|------|---------|
| Базовые метрики | Всего параметров, обучаемых (LoRA), размер LoRA весов |
| Численная стабильность | NaN, Inf, max logit |
| Perplexity | Perplexity на тестовых данных |
| Производительность | Время инференса с LoRA слоями |
| Использование памяти | Память с учётом LoRA |

**Запуск:**
```bash
# Автоматический поиск лучшей эпохи
python tests/test_lora_model.py

# Тестирование конкретной эпохи
python tests/test_lora_model.py --epoch 5

# На GPU
python tests/test_lora_model.py --device cuda
```

**Выходные файлы:**
- `reports/test_lora_medium/memory_usage.png`
- `reports/test_lora_medium/metrics_report.json`
- `reports/test_lora_medium/METRICS_REPORT.txt`
- `reports/test_lora_medium/performance_plot.png`

## 🎯 Тесты генерации

### **`test_work_base_model.py`** — генерация из затравки

**Назначение:** Тестирование генерации C++ кода базовой моделью из затравки (code prompt).

**Особенности:**
- Использует только C++ токенизатор
- Генерация из незаконченного кода
- Три температуры (0.7, 0.8, 0.9)

**Тестовые промпты:**
| Имя промпта | Промпт | Описание |
|-------------|--------|----------|
| main_function | `int main() {` | Простая функция main() |
| hello_world | `#include <iostream>\n\nint main() {\n    std::cout << "Hello, World!" << std::endl;` | Hello World программа |
| function_definition | `int add(int a, int b) {\n    return a + b;` | Простая функция сложения |
| class_definition | `class MyClass {\npublic:\n    MyClass() {}\n    void print() {` | Определение класса |
| template_function | `template<typename T>\nT max(T a, T b) {\n    return (a > b) ? a : b;` | Шаблонная функция |
| vector_usage | `#include <vector>\n#include <algorithm>\n\nvoid sortVector(std::vector<int>& vec) {\n    std::sort(vec.begin(), vec.end());` | Использование STL вектора |
| recursive_fibonacci | `int fibonacci(int n) {\n    if (n <= 1) return n;\n    return fibonacci(n-1) + fibonacci(n-2);` | Рекурсивная функция Фибоначчи |

**Запуск:**
```bash
python tests/test_work_base_model.py
```

**Выходные файлы (пример для модели medium):**
- `reports/test_work_medium/generated_samples.json` — все сгенерированные примеры
- `reports/test_work_medium/GENERATION_REPORT.txt` — текстовый отчёт

### **`test_work_lora_model.py`** — генерация по инструкции

**Назначение:** Тестирование генерации C++ кода LoRA моделью по русским инструкциям.

**Особенности:**
- Использует два токенизатора (русский + C++)
- "Понимает" русские инструкции
- Расширенный словарь (14000 токенов)
- Защита от раннего EOS (min_tokens=45)

**Тестовые промпты:**
| Имя промпта | Промпт | Описание |
|-------------|--------|----------|
| main_function | `Напиши программу с функцией main` | Простая функция main() |
| hello_world | `Напиши программу которая выводит Hello World` | Hello World программа |
| sum_function | `Напиши функцию которая складывает два числа` | Простая функция сложения |
| class_person | `Создай класс Person с полями имя и возраст` | Класс Person |
| vector_sort | `Напиши функцию которая сортирует вектор целых чисел` | Сортировка вектора |
| fibonacci | `Напиши рекурсивную функцию вычисления чисел Фибоначчи` | Числа Фибоначчи |

**Запуск:**
```bash
python tests/test_work_lora_model.py
```

**Выходные файлы:**
- `reports/test_work_lora_medium/generated_samples.json`
- `reports/test_work_lora_medium/GENERATION_REPORT.txt`

## 📈 Сравнение моделей

### **`compare_base_models.py`** — сравнение Tiny/Small/Medium

**Назначение:** Комплексное сравнение трёх базовых моделей по всем метрикам.

**Сравниваемые метрики:**
- Perplexity и Loss
- Время инференса
- Пропускная способность
- Использование памяти
- Размер модели и чекпоинта
- Количество параметров

**Запуск:**
```bash
python tests/compare_base_models.py
```

**Выходные файлы:**
- reports/model_comparison/comparison_report.html — HTML отчёт
- reports/model_comparison/comparison_results.json — все метрики
- reports/model_comparison/inference_time_comparison.png
- reports/model_comparison/memory_comparison.png
- reports/model_comparison/model_size_comparison.png
- reports/model_comparison/perplexity_comparison.png

## 📊 Сводная таблица тестов

| Тест | Назначение | Модель | Выходные данные |
|------|------------|--------|-----------------|
| `test_base_model.py` | Метрики базовой модели | Tiny/Small/Medium | `reports/test_.../` |
| `test_lora_model.py` | Метрики LoRA модели | Medium + LoRA | `reports/test_lora_medium/` |
| `test_work_base_model.py` | Генерация из затравки | Tiny/Small/Medium | `reports/test_work_.../` |
| `test_work_lora_model.py` | Генерация по инструкции | Medium + LoRA | `reports/test_work_lora_medium/` |
| `compare_base_models.py` | Сравнение моделей | Tiny/Small/Medium | `reports/model_comparison/` |

## 🚀 Типичные сценарии использования

### Проверка качества базовой модели

```bash
# Метрики
python tests/test_base_model.py

# Генерация кода
python tests/test_work_base_model.py
```

### Проверка качества LoRA модели

```bash
# Метрики
python tests/test_lora_model.py --epoch 5

# Генерация по инструкциям
python tests/test_work_lora_model.py
```

### Сравнение всех моделей

```bash
python tests/compare_base_models.py
# Открыть reports/model_comparison/comparison_report.html
```

## 📁 Структура каталога

```text
tests/    # тесты
├── compare_base_models.py     # Сравнение трех базовых моделей Tiny/Small/Medium
├── README.md                  # Этот файл
├── test_base_model.py         # Тест проверки метрик базовой модели
├── test_lora_model.py         # Тест проверки метрик lore модели
├── test_work_base_model.py    # Тест работы базовой модели
└── test_work_lora_model.py    # Тест работы LoRA модели
```
---

**Автор:** Евгений П.  
**Лицензия:** MIT  
**Дата:** 2026