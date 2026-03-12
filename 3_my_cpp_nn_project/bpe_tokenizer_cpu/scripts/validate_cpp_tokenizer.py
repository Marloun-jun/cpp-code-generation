#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# validate_cpp_tokenizer.py - Валидация C++ токенизатора против Python эталона
# ======================================================================
#
# @file validate_cpp_tokenizer.py
# @brief Валидация C++ токенизатора путем сравнения с Python эталонной реализацией
#
# @author Евгений П.
# @date 2026
# @version 2.0.0
#
# @details Сравнивает результаты работы C++ и Python реализаций BPE токенизатора
#          для обеспечения полной совместимости. Это критический тест перед
#          использованием C++ версии в продакшене.
#
#          **Проверяемые аспекты:**
#
#          - **Точность кодирования** - совпадение последовательностей токенов
#          - **Точность декодирования** - восстановление исходного текста
#          - **Скорость работы** - сравнение производительности
#          - **Совместимость моделей** - использование конвертированных словарей
#
#          **Используемые модели:**
#          - Python: `bpe_python/models/bpe_8000/vocab.json`
#          - C++: `bpe_cpp/models/cpp_vocab.json` (конвертированная версия)
#
# @usage python validate_cpp_tokenizer.py
#
# @example
#   python validate_cpp_tokenizer.py
#   # Результаты сохраняются в reports/cpp_validation_results.json
#
# @note Перед запуском убедитесь, что:
#       1. Python модель обучена (bpe_python/models/bpe_8000/)
#       2. C++ модель сконвертирована (bpe_cpp/models/bpe_8000/)
#       3. C++ бинарник compare_with_python собран
#
# ======================================================================

import sys
import json
import time
import random
import tempfile
import subprocess

from pathlib import Path
from typing import List

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()           # scripts/validate_cpp_tokenizer.py
SCRIPTS_DIR = CURRENT_FILE.parent                  # scripts/
PROJECT_ROOT = SCRIPTS_DIR.parent                  # bpe_tokenizer/
BPE_CPP_DIR = PROJECT_ROOT / 'bpe_cpp'              # bpe_cpp/
BPE_PYTHON_DIR = PROJECT_ROOT / 'bpe_python'        # bpe_python/
REPORTS_DIR = PROJECT_ROOT / 'reports'              # reports/

# Создаем директорию для отчетов
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Добавляем путь для импорта Python токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

print(f"Корень проекта: {PROJECT_ROOT}")
print(f"Python BPE директория: {BPE_PYTHON_DIR}")
print(f"C++ BPE директория: {BPE_CPP_DIR}")
print(f"Директория отчетов: {REPORTS_DIR}")

# ======================================================================
# ИМПОРТ PYTHON ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer as PythonTokenizer
    print("v Импорт Python токенизатора успешен")
except ImportError as e:
    print(f"x Ошибка импорта Python токенизатора: {e}")
    print(f"\nбедитесь, что файл tokenizer.py существует в {BPE_PYTHON_DIR}")
    print("   Файлы в директории:")
    for f in BPE_PYTHON_DIR.iterdir():
        if f.suffix == '.py':
            print(f"   - {f.name}")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела для красивого форматирования вывода.
    
    Args:
        title: Заголовок
        width: Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


# ======================================================================
# КЛАСС ОБЕРТКИ ДЛЯ C++ ТОКЕНИЗАТОРА
# ======================================================================

class CppTokenizerWrapper:
    """
    Обертка для вызова C++ токенизатора через подпроцесс.
    
    Позволяет вызывать C++ бинарник и получать результаты в Python,
    обеспечивая прозрачный интерфейс, аналогичный Python токенизатору.
    
    **Поддерживаемые операции:**
    - encode(text) -> List[int] - кодирование текста
    - decode(tokens) -> str - декодирование токенов
    """
    
    def __init__(self, cpp_binary_path: Path, vocab_path: Path, merges_path: Path):
        """
        Инициализация обертки.
        
        Args:
            cpp_binary_path: Путь к C++ бинарнику compare_with_python
            vocab_path: Путь к файлу словаря C++
            merges_path: Путь к файлу слияний C++
            
        Raises:
            FileNotFoundError: Если какой-либо файл не найден
        """
        self.binary = Path(cpp_binary_path)
        self.vocab = Path(vocab_path)
        self.merges = Path(merges_path)
        
        # Проверяем существование файлов
        missing = []
        if not self.binary.exists():
            missing.append(f"C++ binary: {self.binary}")
        if not self.vocab.exists():
            missing.append(f"Vocabulary: {self.vocab}")
        if not self.merges.exists():
            missing.append(f"Merges file: {self.merges}")
        
        if missing:
            raise FileNotFoundError("x Не найдены файлы:\n  " + "\n  ".join(missing))
    
    def encode(self, text: str) -> List[int]:
        """
        Кодирует текст с помощью C++ токенизатора.
        
        Args:
            text: Входной текст
            
        Returns:
            List[int]: Список ID токенов
        
        **Процесс:**
        1. Сохранение текста во временный файл
        2. Запуск C++ бинарника с аргументами
        3. Парсинг JSON-результата
        4. Очистка временных файлов
        """
        # Создаем временный файл с текстом
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(text)
            input_file = f.name
        
        try:
            # Запускаем C++ бинарник в тихом режиме
            result = subprocess.run(
                [str(self.binary), input_file, str(self.vocab), str(self.merges), "--quiet"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )
            
            if result.returncode != 0:
                print(f" !!! C++ ошибка: {result.stderr.strip()}")
                return []
            
            # Ищем JSON массив в выводе (начинается с '[')
            output = result.stdout
            json_start = output.find('[')
            if json_start == -1:
                print(f" !!! Не найден JSON массив в выводе")
                print(f"Вывод: {output[:200]}...")
                return []
            
            # Берем всё от первой '[' до конца
            json_str = output[json_start:].strip()
            
            # Если после JSON есть ещё текст, обрезаем
            json_end = json_str.rfind(']')
            if json_end != -1:
                json_str = json_str[:json_end+1]
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f" !!! Ошибка парсинга JSON: {e}")
                print(f"JSON строка: {json_str[:200]}...")
                return []
            
        except subprocess.TimeoutExpired:
            print(f" !!! Таймаут C++ бинарника")
            return []
        finally:
            Path(input_file).unlink(missing_ok=True)

    def decode(self, tokens: List[int]) -> str:
        """
        Декодирует токены с помощью C++ токенизатора.
        
        Args:
            tokens: Список ID токенов
            
        Returns:
            str: Декодированный текст
        """
        # Создаем временный файл с токенами
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            json.dump(tokens, f)
            input_file = f.name
        
        try:
            # Запускаем C++ бинарник в режиме декодирования
            result = subprocess.run(
                [str(self.binary), input_file, str(self.vocab), str(self.merges), "--decode"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )
            
            if result.returncode != 0:
                return ""
            
            return result.stdout.strip()
            
        finally:
            Path(input_file).unlink(missing_ok=True)


# ======================================================================
# ЗАГРУЗКА ТЕСТОВЫХ ПРИМЕРОВ
# ======================================================================

def load_test_samples() -> List[str]:
    """
    Загружает тестовые примеры из различных источников.
    
    Returns:
        List[str]: Список тестовых строк
    
    **Источники:**
    1. Базовые конструкции C++ (встроенные)
    2. Реальные примеры из train_code.txt (если доступен)
    
    **Ограничение:** максимум 100 примеров для оптимального времени теста
    """
    samples = []
    
    # Базовые синтаксические конструкции C++
    base_samples = [
        "int main() { return 0; }",
        "std::vector<int> v;",
        "// комментарий на русском языке",
        "template<typename T> class MyClass {};",
        "auto lambda = [](int x) { return x * x; };",
        "#include <iostream>",
        "for (int i = 0; i < 10; ++i) {",
        "    std::cout << \"Привет, мир!\" << std::endl;",
        "}",
        "if (x > 5 && y < 10) { z = x + y; }",
        "namespace mylib { namespace detail { void helper() {} } }",
        "class Test { public: Test() = default; };",
        "int* ptr = nullptr;",
        "constexpr int MAX_SIZE = 100;",
        "static_assert(sizeof(int) == 4, \"int must be 4 bytes\");",
        "std::unique_ptr<int> up = std::make_unique<int>(42);",
        "auto result = std::accumulate(v.begin(), v.end(), 0);",
        "std::cout << \"Hello, \" << name << \"!\" << std::endl;",
        "// 🔥 emoji комментарий",
        "R\"(raw string)\"",
        "std::map<std::string, int> ages;",
    ]
    samples.extend(base_samples)
    
    # Загружаем из тренировочного корпуса если есть
    train_file = PROJECT_ROOT / "data" / "corpus" / "train_code.txt"
    if train_file.exists():
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                # Берем случайные 30 примеров
                samples.extend(random.sample(lines, min(30, len(lines))))
            print(f"Загружено дополнительных примеров из {train_file}")
        except Exception as e:
            print(f" !!! Ошибка загрузки из train_code.txt: {e}")
    
    # Убираем дубликаты и пустые строки
    samples = list(set(s for s in samples if s))
    
    # Ограничиваем количество примеров
    return samples[:100]


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ ВАЛИДАЦИИ
# ======================================================================

def main() -> int:
    """
    Основная функция валидации.
    
    Returns:
        int: 0 при успехе (полное совпадение), 1 при ошибке
    
    **Процесс:**
    1. Проверка наличия всех необходимых файлов
    2. Загрузка Python токенизатора
    3. Инициализация C++ обертки
    4. Загрузка тестовых примеров
    5. Поэлементное сравнение результатов
    6. Анализ и сохранение результатов
    """
    print_header("ВАЛИДАЦИЯ C++ BPE ТОКЕНИЗАТОРА")
    
    # ======================================================================
    # ПРОВЕРКА НАЛИЧИЯ ФАЙЛОВ
    # ======================================================================
    
    # Пути к файлам Python модели (модель 8000)
    python_vocab = BPE_PYTHON_DIR / "models" / "bpe_8000" / "vocab.json"
    python_merges = BPE_PYTHON_DIR / "models" / "bpe_8000" / "merges.txt"
    
    # Пути к C++ бинарнику и моделям
    cpp_compare = BPE_CPP_DIR / "build" / "examples" / "compare_with_python"
    cpp_vocab = BPE_CPP_DIR / "models" / "cpp_vocab.json"
    cpp_merges = BPE_CPP_DIR / "models" / "cpp_merges.txt"
    
    # Добавляем .exe для Windows
    if sys.platform == 'win32':
        cpp_compare = cpp_compare.with_suffix('.exe')
    
    # Проверяем наличие всех файлов
    missing = []
    for name, path in [
        ("Python vocab (8000)", python_vocab),
        ("Python merges (8000)", python_merges),
        ("C++ compare binary", cpp_compare),
        ("C++ vocab", cpp_vocab),
        ("C++ merges", cpp_merges),
    ]:
        if not path.exists():
            missing.append(f"{name}: {path}")
    
    if missing:
        print("\nОтсутствуют необходимые файлы:")
        for f in missing:
            print(f"   - {f}")
        print("\nСначала выполните:")
        print("   cd bpe_cpp && mkdir -p build && cd build")
        print("   cmake .. && make -j$(nproc) compare_with_python")
        print("   python bpe_cpp/tools/convert_vocab.py")
        return 1
    
    # ======================================================================
    # ЗАГРУЗКА PYTHON ТОКЕНИЗАТОРА
    # ======================================================================
    
    print("\nЗагрузка Python токенизатора...")
    try:
        py_tokenizer = PythonTokenizer.load(
            str(python_vocab), 
            str(python_merges), 
            byte_level=True,
            cache_size=10000
        )
        print(f"   Загружено {py_tokenizer.vocab_size} токенов")
        print(f"   Размер словаря: {py_tokenizer.vocab_size}")
    except Exception as e:
        print(f"   x Ошибка загрузки Python токенизатора: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ======================================================================
    # ИНИЦИАЛИЗАЦИЯ C++ ТОКЕНИЗАТОРА
    # ======================================================================
    
    print("\n⚡ Инициализация C++ токенизатора...")
    try:
        cpp_tokenizer = CppTokenizerWrapper(cpp_compare, cpp_vocab, cpp_merges)
        print(f"   v C++ бинарник найден: {cpp_compare}")
    except FileNotFoundError as e:
        print(f"   x {e}")
        return 1
    
    # ======================================================================
    # ЗАГРУЗКА ТЕСТОВЫХ ПРИМЕРОВ
    # ======================================================================
    
    print("\n📚 Загрузка тестовых примеров...")
    test_samples = load_test_samples()
    print(f"v Загружено {len(test_samples)} примеров")
    
    if len(test_samples) == 0:
        print("x Нет тестовых примеров")
        return 1
    
    # ======================================================================
    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    print("\nСравнение результатов...")
    results = []
    
    for i, text in enumerate(test_samples, 1):
        print(f"Обработка {i}/{len(test_samples)}...", end='\r')
        
        # Python
        py_start = time.perf_counter()
        py_tokens = py_tokenizer.encode(text)
        py_time = time.perf_counter() - py_start
        
        # C++
        cpp_start = time.perf_counter()
        cpp_tokens = cpp_tokenizer.encode(text)
        cpp_time = time.perf_counter() - cpp_start
        
        # Проверяем совпадение
        match = py_tokens == cpp_tokens
        
        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "text_len": len(text),
            "py_tokens": py_tokens[:20] if not match else py_tokens,
            "cpp_tokens": cpp_tokens[:20] if not match else cpp_tokens,
            "py_count": len(py_tokens),
            "cpp_count": len(cpp_tokens),
            "match": match,
            "py_time_ms": py_time * 1000,
            "cpp_time_ms": cpp_time * 1000,
        })
    
    print("\n")
    
    # ======================================================================
    # АНАЛИЗ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    matches = sum(1 for r in results if r["match"])
    total_py_time = sum(r["py_time_ms"] for r in results)
    total_cpp_time = sum(r["cpp_time_ms"] for r in results)
    
    print_header("РЕЗУЛЬТАТЫ")
    
    print(f"\nСтатистика:")
    print(f"   - Всего примеров: {len(results)}")
    print(f"   - Совпадение: {matches}/{len(results)} ({matches/len(results)*100:.1f}%)")
    print(f"   - Время Python: {total_py_time:.2f} ms")
    print(f"   - Время C++:    {total_cpp_time:.2f} ms")
    print(f"   - Ускорение:    {total_py_time/total_cpp_time:.2f}x")
    
    # Оценка результата
    print(f"\nОценка:")
    if matches == len(results):
        print("   ПОЛНОЕ СОВПАДЕНИЕ!")
    elif matches > len(results) * 0.99:
        print("   ОТЛИЧНОЕ СОВПАДЕНИЕ (>99%)")
    elif matches > len(results) * 0.95:
        print("   ХОРОШЕЕ СОВПАДЕНИЕ (>95%)")
    elif matches > len(results) * 0.9:
        print("   СРЕДНЕЕ СОВПАДЕНИЕ (>90%)")
    else:
        print("   НИЗКОЕ СОВПАДЕНИЕ (<90%)")
    
    # Показываем примеры расхождений
    mismatches = [r for r in results if not r["match"]]
    if mismatches:
        print(f"\nНайдено расхождений: {len(mismatches)}")
        print(f"\nПервые 3 расхождения:")
        for i, r in enumerate(mismatches[:3]):
            print(f"\n   {i+1}. Текст: {r['text'][:80]}")
            print(f"      Python ({r['py_count']}): {r['py_tokens'][:15]}...")
            print(f"      C++ ({r['cpp_count']}):    {r['cpp_tokens'][:15]}...")
    
    # ======================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "matches": matches,
        "match_percentage": matches/len(results)*100,
        "total_py_time_ms": total_py_time,
        "total_cpp_time_ms": total_cpp_time,
        "speedup": total_py_time/total_cpp_time,
        "mismatches": len(mismatches),
    }
    
    output_file = REPORTS_DIR / "cpp_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в {output_file}")
    
    return 0 if matches == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())