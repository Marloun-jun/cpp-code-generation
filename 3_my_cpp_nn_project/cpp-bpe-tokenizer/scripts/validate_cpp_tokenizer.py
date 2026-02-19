#!/usr/bin/env python3
# ======================================================================
# validate_cpp_tokenizer.py - Валидация C++ токенизатора против Python эталона
# ======================================================================
#
# @file validate_cpp_tokenizer.py
# @brief Валидация C++ токенизатора против Python эталона
#
# @author Евгений П.
# @date 2026
# @version 2.0.0
#
# @details Сравнивает результаты работы C++ и Python реализаций BPE токенизатора:
#          - Точность кодирования/декодирования
#          - Скорость работы
#          - Процент совпадения токенов
#
# @usage python validate_cpp_tokenizer.py
#
# @example
#   python validate_cpp_tokenizer.py
#   # Результаты сохраняются в reports/cpp_validation_results.json
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
PROJECT_ROOT = SCRIPTS_DIR.parent                  # cpp-bpe-tokenizer/
BPE_PYTHON_DIR = PROJECT_ROOT / 'bpe'              # bpe/
CPP_DIR = PROJECT_ROOT / 'cpp'                      # cpp/
REPORTS_DIR = PROJECT_ROOT / 'reports'              # reports/

# Создаем директорию для отчетов
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Добавляем путь для импорта Python токенизатора
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ PYTHON ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer as PythonTokenizer
    print("Импорт Python токенизатора успешен")
except ImportError as e:
    print(f"Ошибка импорта Python токенизатора: {e}")
    print(f"Убедитесь, что файл tokenizer.py существует в {BPE_PYTHON_DIR}")
    sys.exit(1)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def print_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок раздела.
    
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
    
    Позволяет вызывать C++ бинарник и получать результаты в Python.
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
            raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))
    
    def encode(self, text: str) -> List[int]:
        """
        Кодирует текст с помощью C++ токенизатора.
        
        Args:
            text: Входной текст
            
        Returns:
            List[int]: Список ID токенов
        """
        # Создаем временный файл с текстом
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8'
        ) as f:
            f.write(text)
            input_file = f.name
        
        try:
            # Запускаем C++ бинарник
            result = subprocess.run(
                [str(self.binary), input_file, str(self.vocab), str(self.merges)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )
            
            if result.returncode != 0:
                print(f" !!! C++ ошибка: {result.stderr.strip()}")
                return []
            
            # Парсим JSON результат
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f" !!! Ошибка парсинга JSON: {result.stdout[:100]}...")
                return []
                
        finally:
            # Удаляем временный файл
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
        except Exception as e:
            print(f" !!! Ошибка загрузки из train_code.txt: {e}")
    
    # Убираем дубликаты и пустые строки
    samples = list(set(s for s in samples if s))
    
    # Ограничиваем количество примеров
    return samples[:100]


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция валидации.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    print_header("ВАЛИДАЦИЯ C++ BPE ТОКЕНИЗАТОРА")
    
    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Python BPE: {BPE_PYTHON_DIR}")
    print(f"C++ директория: {CPP_DIR}")
    print(f"Отчеты: {REPORTS_DIR}")
    
    # Пути к файлам
    python_vocab = BPE_PYTHON_DIR / "vocab.json"
    python_merges = BPE_PYTHON_DIR / "merges.txt"
    
    cpp_compare = CPP_DIR / "build" / "examples" / "compare_with_python"
    cpp_vocab = CPP_DIR / "models" / "cpp_vocab.json"
    cpp_merges = CPP_DIR / "models" / "cpp_merges.txt"
    
    # Проверяем наличие всех файлов
    missing = []
    for name, path in [
        ("Python vocab", python_vocab),
        ("Python merges", python_merges),
        ("C++ compare", cpp_compare),
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
        print("   cd bpe_cpp && ./scripts/build.sh")
        print("   python bpe_cpp/tools/convert_vocab.py")
        return 1
    
    # Загружаем Python токенизатор
    print("\nЗагрузка Python токенизатора...")
    try:
        py_tokenizer = PythonTokenizer(byte_level=True)
        py_tokenizer.load(str(python_vocab), str(python_merges))
        print(f"Загружено {py_tokenizer.vocab_size()} токенов")
    except Exception as e:
        print(f"Ошибка загрузки Python токенизатора: {e}")
        return 1
    
    # Создаем обертку для C++
    print("\nИнициализация C++ токенизатора...")
    try:
        cpp_tokenizer = CppTokenizerWrapper(cpp_compare, cpp_vocab, cpp_merges)
        print(f"   ✓ C++ бинарник найден: {cpp_compare}")
    except FileNotFoundError as e:
        print(f"{e}")
        return 1
    
    # Загружаем тестовые примеры
    print("\nЗагрузка тестовых примеров...")
    test_samples = load_test_samples()
    print(f"   ✓ Загружено {len(test_samples)} примеров")
    
    if len(test_samples) == 0:
        print("Нет тестовых примеров")
        return 1
    
    # Сравниваем
    print("\nСравнение результатов...")
    results = []
    
    for i, text in enumerate(test_samples, 1):
        print(f"   Обработка {i}/{len(test_samples)}...", end='\r')
        
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
    
    # Анализ результатов
    matches = sum(1 for r in results if r["match"])
    total_py_time = sum(r["py_time_ms"] for r in results)
    total_cpp_time = sum(r["cpp_time_ms"] for r in results)
    
    print_header("РЕЗУЛЬТАТЫ")
    
    print(f"\nСтатистика:")
    print(f"   • Всего примеров: {len(results)}")
    print(f"   • Совпадение: {matches}/{len(results)} ({matches/len(results)*100:.1f}%)")
    print(f"   • Время Python: {total_py_time:.2f} ms")
    print(f"   • Время C++:    {total_cpp_time:.2f} ms")
    print(f"   • Ускорение:    {total_py_time/total_cpp_time:.2f}x")
    
    # Оценка результата
    print(f"\nОценка:")
    if matches == len(results):
        print("   ПОЛНОЕ СОВПАДЕНИЕ!")
    elif matches > len(results) * 0.99:
        print("   ОТЛИЧНОЕ СОВПАДЕНИЕ (>99%)")
    elif matches > len(results) * 0.95:
        print("   ! ХОРОШЕЕ СОВПАДЕНИЕ (>95%)")
    elif matches > len(results) * 0.9:
        print("   !! СРЕДНЕЕ СОВПАДЕНИЕ (>90%)")
    else:
        print("   !!! НИЗКОЕ СОВПАДЕНИЕ (<90%)")
    
    # Показываем примеры расхождений
    mismatches = [r for r in results if not r["match"]]
    if mismatches:
        print(f"\nНайдено расхождений: {len(mismatches)}")
        print(f"\nПервые 3 расхождения:")
        for i, r in enumerate(mismatches[:3]):
            print(f"\n   {i+1}. Текст: {r['text'][:80]}")
            print(f"      Python ({r['py_count']}): {r['py_tokens'][:15]}...")
            print(f"      C++ ({r['cpp_count']}):    {r['cpp_tokens'][:15]}...")
    
    # Сохраняем результаты
    output_file = REPORTS_DIR / "cpp_validation_results.json"
    
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
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Результаты сохранены в {output_file}")
    
    return 0 if matches == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())