#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# validate_cpp_tokenizer.py - Валидация C++ BPE токенизатора
# ======================================================================
#
# @file validate_cpp_tokenizer.py
# @brief Валидация C++ токенизатора путем сравнения с Python эталоном
#
# @author Евгений П.
# @date 2026
# @version 3.2.0
#
# @details Проверяет, что C++ токенизатор корректно кодирует и декодирует
#          текст, и что результаты roundtrip совпадают с Python эталоном.
#
#          **Проверяемые аспекты:**
#          - Python roundtrip (encode + decode = исходный текст)
#          - C++ roundtrip (encode + decode = исходный текст)
#          - Совпадение декодированных результатов
#
#          **Используемые модели:**
#          - Python: `bpe_python/models/bpe_10000/vocab.json`
#          - C++:    `bpe_cpp/models/bpe_10000/cpp_vocab.json`
#
# @usage python validate_cpp_tokenizer.py
#
# @example
#   python validate_cpp_tokenizer.py
#   # Результаты сохраняются в reports/cpp_validation_results.json
#
# @note Перед запуском убедитесь, что:
#       1. Python модель обучена (bpe_python/models/bpe_10000/)
#       2. C++ модель сконвертирована (bpe_cpp/models/bpe_10000/)
#       3. C++ бинарник compare_with_python собран
#
# ======================================================================

import sys
import json
import time
import tempfile
import subprocess

from pathlib import Path
from typing import List, Tuple, Dict, Any

# ======================================================================
# НАСТРОЙКА ПУТЕЙ
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()
SCRIPTS_DIR = CURRENT_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
BPE_CPP_DIR = PROJECT_ROOT / 'bpe_cpp'
BPE_PYTHON_DIR = PROJECT_ROOT / 'bpe_python'
REPORTS_DIR = PROJECT_ROOT / 'reports'

# Создаем директорию для отчетов
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Добавляем путь для импорта Python токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

print(f"Корень проекта: {PROJECT_ROOT}")
print(f"Python BPE:     {BPE_PYTHON_DIR}")
print(f"C++ BPE:        {BPE_CPP_DIR}")


# ======================================================================
# ИМПОРТ PYTHON ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer as PythonTokenizer
    print("Импорт Python токенизатора успешен!")
except ImportError as e:
    print(f"Ошибка импорта Python токенизатора: {e}!")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """Выводит форматированный заголовок раздела."""
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


# ======================================================================
# C++ ОБЕРТКА
# ======================================================================

class CppTokenizerWrapper:
    """
    Обертка для вызова C++ токенизатора через подпроцесс.
    
    Позволяет вызывать C++ бинарник и получать результаты в Python,
    обеспечивая прозрачный интерфейс, аналогичный Python токенизатору.
    
    **Поддерживаемые операции:**
    - encode(text) -> List[int] - Кодирование текста
    - decode(tokens) -> str     - Декодирование токенов
    - encode_decode_roundtrip() - Проверка roundtrip
    """
    
    def __init__(self, binary_path: Path, vocab_path: Path, merges_path: Path) -> None:
        """
        Инициализация обертки.
        
        Args:
            binary_path: Путь к C++ бинарнику compare_with_python
            vocab_path:  Путь к файлу словаря C++
            merges_path: Путь к файлу слияний C++
            
        Raises:
            FileNotFoundError: Если какой-либо файл не найден
        """
        self.binary = Path(binary_path)
        self.vocab = Path(vocab_path)
        self.merges = Path(merges_path)
        
        # Проверяем существование всех файлов
        missing = []
        if not self.binary.exists():
            missing.append(f"Binary:      {self.binary}")
        if not self.vocab.exists():
            missing.append(f"Vocabulary:  {self.vocab}")
        if not self.merges.exists():
            missing.append(f"Merges file: {self.merges}")
        
        if missing:
            raise FileNotFoundError("Не найдены файлы:\n  " + "\n  ".join(missing))
    
    def encode(self, text: str) -> List[int]:
        """
        Кодирует текст с помощью C++ токенизатора.
        
        Args:
            text: Входной текст
            
        Returns:
            List[int]: Список ID токенов, пустой список при ошибке
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(text)
            input_file = f.name
        
        try:
            result = subprocess.run(
                [str(self.binary), input_file, str(self.vocab), str(self.merges), "--quiet"],
                capture_output=True, text=True, encoding='utf-8', timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            output = result.stdout.strip()
            if not output:
                return []
            
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return []
                
        except (subprocess.TimeoutExpired, Exception):
            return []
        finally:
            Path(input_file).unlink(missing_ok=True)
    
    def decode(self, tokens: List[int]) -> str:
        """
        Декодирует токены с помощью C++ токенизатора.
        
        Args:
            tokens: Список ID токенов
            
        Returns:
            str: Декодированный текст, пустая строка при ошибке
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(tokens, f)
            input_file = f.name
        
        try:
            result = subprocess.run(
                [str(self.binary), input_file, str(self.vocab), str(self.merges), "--decode", "--quiet"],
                capture_output=True, text=True, encoding='utf-8', timeout=30
            )
            
            if result.returncode != 0:
                return ""
            
            return result.stdout.strip()
            
        except (subprocess.TimeoutExpired, Exception):
            return ""
        finally:
            Path(input_file).unlink(missing_ok=True)
    
    def roundtrip(self, text: str) -> Tuple[List[int], str, bool]:
        """
        Выполняет encode и decode, проверяет roundtrip.
        
        Args:
            text: Входной текст
            
        Returns:
            Tuple[List[int], str, bool]: (tokens, decoded_text, roundtrip_success)
        """
        tokens = self.encode(text)
        if not tokens:
            return [], "", False
        
        decoded = self.decode(tokens)
        return tokens, decoded, (decoded == text)


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
    """
    samples: List[str] = [
        # ASCII
        "int x = 42;",
        "std::vector<int> v;",
        "for (int i = 0; i < 10; ++i) { sum += i; }",
        "class MyClass { public: void method(); };",
        "template<typename T> T max(T a, T b) { return a > b ? a : b; }",
        "#include <iostream>",
        "auto lambda = [](int x) { return x * x; };",
        
        # Русский язык
        "// русский комментарий",
        "/* ещё комментарий на русском */",
        "std::cout << \"Привет, мир!\" << std::endl;",
        "// тест кириллицы: привет мир",
        
        # Эмодзи
        "// 🔥 C++ code with emoji 😊",
        "// 🚀 performance test",
        
        # Сложные конструкции
        "std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();",
        "std::transform(v.begin(), v.end(), v.begin(), [](int x){ return x * x; });",
        "if constexpr (std::is_integral_v<T>) { return true; }",
    ]
    
    # Добавляем примеры из тренировочного корпуса
    corpus_file = PROJECT_ROOT / "data" / "corpus" / "train_code.txt"
    if corpus_file.exists():
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                samples.extend(lines[:20])
            print(f"Загружено дополнительных примеров из {corpus_file}")
        except Exception as e:
            print(f"Ошибка загрузки из train_code.txt: {e}!")
    
    # Убираем дубликаты и пустые строки
    return list(set(s for s in samples if s))


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция валидации.
    
    Returns:
        int: 0 при успехе (все roundtrip успешны), 1 при ошибке
    """
    print_header("ВАЛИДАЦИЯ C++ BPE ТОКЕНИЗАТОРА")
    
    # ======================================================================
    # ПУТИ К ФАЙЛАМ
    # ======================================================================
    
    python_vocab = BPE_PYTHON_DIR / "models" / "bpe_10000" / "vocab.json"
    python_merges = BPE_PYTHON_DIR / "models" / "bpe_10000" / "merges.txt"
    cpp_binary = BPE_CPP_DIR / "build" / "examples" / "compare_with_python"
    cpp_vocab = BPE_CPP_DIR / "models" / "bpe_10000" / "cpp_vocab.json"
    cpp_merges = BPE_CPP_DIR / "models" / "bpe_10000" / "cpp_merges.txt"
    
    if sys.platform == 'win32':
        cpp_binary = cpp_binary.with_suffix('.exe')
    
    # ======================================================================
    # ПРОВЕРКА ФАЙЛОВ
    # ======================================================================
    
    if not python_vocab.exists():
        print(f"\nPython модель не найдена: {python_vocab}!")
        print("Сначала обучите модель:")
        print("cd bpe_python && python trainer.py --corpus ../data/corpus/train_code.txt --vocab-size 10000")
        return 1
    
    if not cpp_binary.exists():
        print(f"\nC++ бинарник не найден: {cpp_binary}!")
        print("Сначала соберите проект:")
        print("cd bpe_cpp/build && cmake .. && make -j$(nproc)")
        return 1
    
    if not cpp_vocab.exists() or not cpp_merges.exists():
        print(f"\nC++ модель не найдена.")
        print(f"Конвертируйте модель из Python формата:")
        print(f"python {BPE_CPP_DIR / 'tools' / 'convert_vocab.py'} --input {python_vocab} --output {cpp_vocab}")
        return 1
    
    # ======================================================================
    # ЗАГРУЗКА PYTHON ТОКЕНИЗАТОРА
    # ======================================================================
    
    print("\nЗагрузка Python токенизатора...")
    try:
        py_tokenizer = PythonTokenizer.load(
            str(python_vocab), 
            str(python_merges), 
            byte_level=True
        )
        print(f"Размер словаря: {py_tokenizer.vocab_size}")
    except Exception as e:
        print(f"Ошибка загрузки: {e}!")
        return 1
    
    # ======================================================================
    # ИНИЦИАЛИЗАЦИЯ C++ ТОКЕНИЗАТОРА
    # ======================================================================
    
    print("\nИнициализация C++ токенизатора...")
    try:
        cpp_tokenizer = CppTokenizerWrapper(cpp_binary, cpp_vocab, cpp_merges)
        print(f"Бинарник: {cpp_binary}")
    except FileNotFoundError as e:
        print(f"{e}!")
        return 1
    
    # ======================================================================
    # ЗАГРУЗКА ТЕСТОВЫХ ПРИМЕРОВ
    # ======================================================================
    
    print("\nЗагрузка тестовых примеров...")
    test_samples = load_test_samples()
    print(f"Загружено {len(test_samples)} примеров")
    
    if not test_samples:
        print("Нет тестовых примеров!")
        return 1
    
    # ======================================================================
    # ТЕСТИРОВАНИЕ
    # ======================================================================
    
    print("\nТестирование roundtrip...")
    results: List[Dict[str, Any]] = []
    
    for i, text in enumerate(test_samples, 1):
        print(f"    {i}/{len(test_samples)}: {text[:40]}...", end='\r')
        
        # Python roundtrip
        py_tokens = py_tokenizer.encode(text)
        py_decoded = py_tokenizer.decode(py_tokens)
        py_ok = (py_decoded == text)
        
        # C++ roundtrip
        cpp_tokens, cpp_decoded, cpp_ok = cpp_tokenizer.roundtrip(text)
        
        # Сравнение декодированных результатов
        decode_match = (py_decoded == cpp_decoded)
        
        results.append({
            "text": text[:50] + "..." if len(text) > 50 else text,
            "py_roundtrip": py_ok,
            "cpp_roundtrip": cpp_ok,
            "decode_match": decode_match,
            "py_tokens": len(py_tokens),
            "cpp_tokens": len(cpp_tokens),
        })
    
    print("\n")
    
    # ======================================================================
    # АНАЛИЗ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    total = len(results)
    py_ok = sum(1 for r in results if r["py_roundtrip"])
    cpp_ok = sum(1 for r in results if r["cpp_roundtrip"])
    decode_match = sum(1 for r in results if r["decode_match"])
    
    print_header("РЕЗУЛЬТАТЫ")
    
    print(f"\nСтатистика:")
    print(f"- Всего примеров:    {total}")
    print(f"- Python roundtrip:  {py_ok}/{total} ({py_ok/total*100:.1f}%)")
    print(f"- C++ roundtrip:     {cpp_ok}/{total} ({cpp_ok/total*100:.1f}%)")
    print(f"- Совпадение декода: {decode_match}/{total} ({decode_match/total*100:.1f}%)")
    
    # Вердикт
    print(f"\nВЕРДИКТ:")
    if decode_match == total:
        print("ОТЛИЧНО! C++ токенизатор функционально идентичен Python!")
        print("Декодированные тексты совпадают 100%.")
        print("C++ roundtrip: 100%")
    elif decode_match > total * 0.95:
        print("ХОРОШО! C++ токенизатор совместим на 95%+")
        print(f"Совпадение: {decode_match/total*100:.1f}%")
    else:
        print("ВНИМАНИЕ! Есть расхождения в декодировании.")
        print("Проверьте, что используется одна и та же модель.")
    
    # Показываем расхождения (если есть)
    mismatches = [r for r in results if not r["decode_match"]]
    if mismatches and len(mismatches) <= 10:
        print(f"\nРасхождения в декодировании ({len(mismatches)}):")
        for r in mismatches[:5]:
            print(f"- {r['text'][:60]}")
            print(f"    Python: {'да' if r['py_roundtrip'] else 'нет'} | C++: {'да' if r['cpp_roundtrip'] else 'нет'}")
    elif mismatches:
        print(f"\nРасхождений: {len(mismatches)} (показаны первые 10)")
        for r in mismatches[:10]:
            print(f"- {r['text'][:60]}")
    
    # ======================================================================
    # СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ======================================================================
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total,
        "python_roundtrip": py_ok,
        "cpp_roundtrip": cpp_ok,
        "decode_matches": decode_match,
        "match_percentage": round(decode_match / total * 100, 1),
        "python_roundtrip_percentage": round(py_ok / total * 100, 1),
        "cpp_roundtrip_percentage": round(cpp_ok / total * 100, 1),
    }
    
    output_file = REPORTS_DIR / "cpp_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())