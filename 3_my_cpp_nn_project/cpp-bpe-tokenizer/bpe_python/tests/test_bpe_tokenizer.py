#!/usr/bin/env python3
# ======================================================================
# test_bpe_tokenizer.py - Модуль тестирования BPE токенизатора
# ======================================================================
#
# @file test_bpe_tokenizer.py
# @brief Модуль тестирования BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Содержит набор тестов для проверки функциональности:
#          - Byte-level кодирование/декодирование UTF-8 текста
#          - Обучение на маленьком корпусе C++ кода
#          - Сохранение и загрузка модели в различных форматах
#          - Замер производительности при разных размерах словаря
#          - Обработка граничных случаев (пустой текст, спецсимволы)
#
# @usage python test_bpe_tokenizer.py [--verbose]
#
# @example
#   python test_bpe_tokenizer.py
#   python test_bpe_tokenizer.py --verbose
#
# ======================================================================

import sys
import time
import logging
import tempfile
import argparse

from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

# Получаем абсолютные пути
CURRENT_FILE = Path(__file__).resolve()           # tests/test_bpe_tokenizer.py
TESTS_DIR = CURRENT_FILE.parent                    # tests/
BPE_PYTHON_DIR = TESTS_DIR.parent                  # bpe_python/
PROJECT_ROOT = BPE_PYTHON_DIR.parent               # cpp-bpe-tokenizer/

# Добавляем пути для импорта токенизатора
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}")
    print(f"Убедитесь, что файл tokenizer.py существует в {BPE_PYTHON_DIR}")
    sys.exit(1)

# ======================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def get_project_paths() -> Dict[str, Path]:
    """
    Получить пути проекта.
    
    Returns:
        Dict[str, Path]: Словарь с путями:
            - project_root: корень проекта
            - bpe_python_dir: директория с Python реализацией
            - tests_dir: директория с тестами
            - test_output_dir: директория для выходных файлов
    """
    paths = {
        "project_root": PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "tests_dir": TESTS_DIR,
        "test_output_dir": TESTS_DIR / 'test_output',
    }
    
    # Создаем выходную директорию если её нет
    paths["test_output_dir"].mkdir(exist_ok=True)
    
    return paths


def print_test_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок теста.
    
    Args:
        title: Заголовок теста
        width: Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")


def print_test_result(name: str, passed: bool) -> None:
    """
    Вывести результат теста.
    
    Args:
        name: Название теста
        passed: Результат (True/False)
    """
    status = "ПРОЙДЕН" if passed else "ПРОВАЛЕН"
    print(f"  {name}: {status}")


# ======================================================================
# КЛАСС ДЛЯ ТЕСТИРОВАНИЯ
# ======================================================================

class BPETokenizerTest:
    """
    Класс для тестирования BPE токенизатора.
    
    Содержит набор тестов для проверки различных аспектов работы
    токенизатора: кодирование, обучение, сохранение/загрузка,
    производительность и граничные случаи.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Инициализация тестов.
        
        Args:
            verbose: Подробный вывод
        """
        self.verbose = verbose
        self.paths = get_project_paths()
        self.test_output_dir = self.paths["test_output_dir"]
        
        logger.info(f"Директория для результатов тестов: {self.test_output_dir}")
    
    # ======================================================================
    # ТЕСТ 1: BYTE-LEVEL КОДИРОВАНИЕ
    # ======================================================================
    
    def test_byte_level(self) -> bool:
        """
        Тестирование byte-level encoding/decoding.
        
        Проверяет корректность преобразования UTF-8 текста
        в byte-level представление и обратно.
        
        Returns:
            bool: True если все тесты пройдены
        """
        print_test_header("ТЕСТ 1: BYTE-LEVEL КОДИРОВАНИЕ")
        
        tokenizer = BPETokenizer(byte_level=True)
        
        test_strings = [
            "Hello, World!",
            "Привет, мир!",
            "C++ программирование",
            "🚀✨🎉",
            "混合文字",
            "int main() { return 42; }",
            "std::cout << \"тест\" << std::endl;",
            "// комментарий на русском",
            "\n\t\r\b\f\v",  # Спецсимволы
            "a" * 1000,       # Длинная строка
        ]
        
        all_passed = True
        passed_count = 0
        
        for i, text in enumerate(test_strings, 1):
            try:
                encoded = tokenizer._byte_encode(text)
                decoded = tokenizer._byte_decode(encoded)
                
                is_match = (text == decoded)
                all_passed = all_passed and is_match
                
                if is_match:
                    passed_count += 1
                
                if self.verbose or not is_match:
                    print(f"\n  {i}. {text[:50]}{'...' if len(text) > 50 else ''}")
                    print(f"     Оригинал: {len(text)} символов")
                    print(f"     Байтов: {len(encoded)}")
                    print(f"     Результат: {'✓' if is_match else '✗'}")
                    
            except Exception as e:
                print(f"\n  {i}. Ошибка: {e}")
                all_passed = False
        
        success_rate = 100.0 * passed_count / len(test_strings)
        print(f"\n  Результат: {passed_count}/{len(test_strings)} ({success_rate:.1f}%)")
        print_test_result("Byte-level тест", all_passed)
        
        return all_passed
    
    # ======================================================================
    # ТЕСТ 2: ОБУЧЕНИЕ НА МАЛЕНЬКОМ КОРПУСЕ
    # ======================================================================
    
    def _create_test_corpus(self) -> List[str]:
        """
        Создать тестовый корпус C++ кода.
        
        Returns:
            List[str]: Список строк с примерами кода
        """
        return [
            "#include <iostream>",
            "#include <vector>",
            "#include <string>",
            "using namespace std;",
            "",
            "int main() {",
            "    cout << \"Hello\" << endl;",
            "    return 0;",
            "}",
            "",
            "class Test {",
            "public:",
            "    Test() = default;",
            "    void process(int x) {",
            "        data_.push_back(x);",
            "    }",
            "private:",
            "    vector<int> data_;",
            "};",
            "",
            "template<typename T>",
            "T square(T x) {",
            "    return x * x;",
            "}",
            "",
            "int main() {",
            "    vector<int> numbers = {1, 2, 3, 4, 5};",
            "    for (auto n : numbers) {",
            "        cout << n << endl;",
            "    }",
            "    return 0;",
            "}",
        ]
    
    def test_training_small(self) -> Optional[BPETokenizer]:
        """
        Тестирование обучения на маленьком корпусе.
        
        Обучает токенизатор на небольшом наборе C++ кода
        и проверяет базовую функциональность.
        
        Returns:
            Optional[BPETokenizer]: Обученный токенизатор или None при ошибке
        """
        print_test_header("ТЕСТ 2: ОБУЧЕНИЕ НА МАЛЕНЬКОМ КОРПУСЕ")
        
        corpus = self._create_test_corpus()
        vocab_size = 50
        
        print(f"  Размер корпуса: {len(corpus)} строк")
        print(f"  Целевой размер словаря: {vocab_size}")
        
        try:
            tokenizer = BPETokenizer(vocab_size=vocab_size, byte_level=True)
            
            start_time = time.time()
            tokenizer.train(corpus, verbose=self.verbose)
            train_time = time.time() - start_time
            
            actual_vocab_size = len(tokenizer.vocab)
            merges_count = tokenizer.merges_count()
            
            print(f"\n  Время обучения: {train_time:.3f} сек")
            print(f"  Итоговый словарь: {actual_vocab_size} токенов")
            print(f"  Выполнено слияний: {merges_count}")
            
            # Проверяем encode/decode на нескольких примерах
            test_cases = [
                "int main() { return 42; }",
                "cout << \"test\" << endl;",
                "vector<int> data;",
                "template<typename T>",
            ]
            
            print("\n  Проверка encode/decode:")
            all_correct = True
            
            for test in test_cases:
                encoded = tokenizer.encode(test)
                decoded = tokenizer.decode(encoded)
                is_correct = (test == decoded)
                all_correct = all_correct and is_correct
                
                if self.verbose or not is_correct:
                    print(f"\n    Текст: {test}")
                    print(f"    Токенов: {len(encoded)}")
                    print(f"    Результат: {'✓' if is_correct else '✗'}")
            
            print_test_result("Обучение", all_correct)
            return tokenizer if all_correct else None
            
        except Exception as e:
            print(f"\nОшибка при обучении: {e}")
            import traceback
            if self.verbose:
                traceback.print_exc()
            return None
    
    # ======================================================================
    # ТЕСТ 3: СОХРАНЕНИЕ И ЗАГРУЗКА
    # ======================================================================
    
    def test_save_load(self, tokenizer: BPETokenizer) -> bool:
        """
        Тестирование сохранения и загрузки модели.
        
        Args:
            tokenizer: Обученный токенизатор
            
        Returns:
            bool: True если тест пройден
        """
        print_test_header("ТЕСТ 3: СОХРАНЕНИЕ И ЗАГРУЗКА")
        
        if tokenizer is None:
            print("Токенизатор не обучен, тест пропущен")
            return False
        
        # Создаем временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vocab_path = tmp_path / 'vocab.json'
            merges_path = tmp_path / 'merges.txt'
            binary_path = tmp_path / 'model.bin'
            
            print(f"  Временная директория: {tmp_path}")
            
            # ===== ТЕКСТОВЫЙ ФОРМАТ =====
            print("\nТекстовый формат (JSON):")
            
            try:
                # Сохраняем
                tokenizer.save(str(vocab_path), str(merges_path))
                
                vocab_size_kb = vocab_path.stat().st_size / 1024
                merges_size_kb = merges_path.stat().st_size / 1024
                
                print(f"    vocab.json: {vocab_size_kb:.2f} KB")
                print(f"    merges.txt: {merges_size_kb:.2f} KB")
                
                # Загружаем
                loaded_tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
                print(f"    Загружено: {len(loaded_tokenizer.vocab)} токенов")
                
                # Проверяем идентичность
                test_text = "int main() { return 42; }"
                original_encoded = tokenizer.encode(test_text)
                loaded_encoded = loaded_tokenizer.encode(test_text)
                
                text_match = (original_encoded == loaded_encoded)
                print(f"    Совпадение: {'✓' if text_match else '✗'}")
                
            except Exception as e:
                print(f"Ошибка: {e}")
                text_match = False
            
            # ===== БИНАРНЫЙ ФОРМАТ =====
            print("\nБинарный формат:")
            
            try:
                # Сохраняем
                tokenizer.save_binary(str(binary_path))
                binary_size_kb = binary_path.stat().st_size / 1024
                print(f"    model.bin: {binary_size_kb:.2f} KB")
                
                # Загружаем
                loaded_binary = BPETokenizer.load_binary(str(binary_path))
                print(f"    Загружено: {len(loaded_binary.vocab)} токенов")
                
                # Проверяем идентичность
                test_text = "int main() { return 42; }"
                original_encoded = tokenizer.encode(test_text)
                binary_encoded = loaded_binary.encode(test_text)
                
                binary_match = (original_encoded == binary_encoded)
                print(f"    Совпадение: {'✓' if binary_match else '✗'}")
                
            except Exception as e:
                print(f"    Ошибка: {e}")
                binary_match = False
            
            # ===== ИТОГ =====
            all_match = text_match and binary_match
            print_test_result("Сохранение/загрузка", all_match)
            
            return all_match
    
    # ======================================================================
    # ТЕСТ 4: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ
    # ======================================================================
    
    def benchmark(self, sizes: List[int] = None) -> Dict[int, Dict]:
        """
        Замер производительности при разных размерах словаря.
        
        Args:
            sizes: Список размеров словаря для тестирования
            
        Returns:
            Dict[int, Dict]: Словарь с результатами бенчмарка
        """
        if sizes is None:
            sizes = [50, 100, 200]
        
        print_test_header("ТЕСТ 4: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
        
        # Создаем тестовый корпус
        corpus = []
        for i in range(500):
            corpus.append(f"int function_{i}() {{ return {i}; }}")
            corpus.append(f"class Class_{i} {{ int value_{i}; }};")
            corpus.append(f"template<typename T_{i}> T_{i} process(T_{i} x) {{ return x; }}")
        
        print(f"  Размер корпуса: {len(corpus)} строк")
        print(f"  Используем {len(corpus[:100])} примеров для обучения")
        
        results = {}
        
        for vocab_size in sizes:
            print(f"\nТест vocab_size = {vocab_size}")
            
            try:
                # Обучение
                start = time.time()
                tokenizer = BPETokenizer(vocab_size=vocab_size, byte_level=True)
                tokenizer.train(corpus[:100], verbose=False)
                train_time = time.time() - start
                
                # Тест encode
                test_text = corpus[0]
                n_iterations = 1000
                
                start = time.time()
                for _ in range(n_iterations):
                    tokenizer.encode(test_text)
                encode_time = time.time() - start
                
                encode_speed = n_iterations / encode_time  # операций/сек
                encode_ms = encode_time * 1000 / n_iterations  # мс/операцию
                
                # Тест decode
                encoded = tokenizer.encode(test_text)
                
                start = time.time()
                for _ in range(n_iterations):
                    tokenizer.decode(encoded)
                decode_time = time.time() - start
                
                decode_speed = n_iterations / decode_time  # операций/сек
                decode_ms = decode_time * 1000 / n_iterations  # мс/операцию
                
                results[vocab_size] = {
                    'train_time': train_time,
                    'encode_time': encode_time,
                    'decode_time': decode_time,
                    'encode_speed': encode_speed,
                    'decode_speed': decode_speed,
                    'encode_ms': encode_ms,
                    'decode_ms': decode_ms,
                    'vocab_size_actual': len(tokenizer.vocab)
                }
                
                print(f"    Обучение: {train_time:.3f} сек")
                print(f"    Encode: {encode_speed:.0f} оп/сек ({encode_ms:.3f} мс/оп)")
                print(f"    Decode: {decode_speed:.0f} оп/сек ({decode_ms:.3f} мс/оп)")
                print(f"    Реальный размер словаря: {len(tokenizer.vocab)}")
                
            except Exception as e:
                print(f"    Ошибка: {e}")
                results[vocab_size] = {'error': str(e)}
        
        # Сводная таблица
        print("\n  " + "=" * 60)
        print("СВОДКА РЕЗУЛЬТАТОВ")
        print("  " + "=" * 60)
        print(f"  {'Vocab':>6} {'Train(s)':>10} {'Encode/s':>10} {'Decode/s':>10} {'E/D ratio':>10}")
        print("  " + "-" * 60)
        
        for vocab_size, res in results.items():
            if 'error' not in res:
                ed_ratio = res['encode_speed'] / res['decode_speed']
                print(f"  {vocab_size:>6} {res['train_time']:>10.3f} "
                      f"{res['encode_speed']:>10.0f} {res['decode_speed']:>10.0f} "
                      f"{ed_ratio:>10.2f}")
        
        return results
    
    # ======================================================================
    # ТЕСТ 5: ГРАНИЧНЫЕ СЛУЧАИ
    # ======================================================================
    
    def test_edge_cases(self) -> bool:
        """
        Тестирование граничных случаев.
        
        Returns:
            bool: True если все тесты пройдены
        """
        print_test_header("ТЕСТ 5: ГРАНИЧНЫЕ СЛУЧАИ")
        
        tokenizer = BPETokenizer(byte_level=True)
        all_passed = True
        results = []
        
        # Тест 1: Пустой текст
        try:
            empty_text = ""
            encoded = tokenizer.encode(empty_text)
            decoded = tokenizer.decode(encoded)
            passed = (decoded == "")
            results.append(("Пустой текст", passed))
            if self.verbose:
                print(f"\n  1. Пустой текст: {'✓' if passed else '✗'}")
        except Exception as e:
            results.append(("Пустой текст", False))
            print(f"\n  1. Пустой текст: {e}")
        
        # Тест 2: Очень длинный текст
        try:
            long_text = "x" * 10000
            encoded = tokenizer.encode(long_text)
            decoded = tokenizer.decode(encoded)
            passed = (decoded == long_text)
            results.append(("Длинный текст (10000 символов)", passed))
            if self.verbose:
                compression = len(long_text) / len(encoded)
                print(f"  2. Длинный текст: {len(encoded)} токенов, "
                      f"сжатие {compression:.2f}x - {'✓' if passed else '✗'}")
        except Exception as e:
            results.append(("Длинный текст", False))
            print(f"\n  2. Длинный текст: {e}")
        
        # Тест 3: Специальные символы
        try:
            special_text = "\n\t\r\b\f\v"
            encoded = tokenizer.encode(special_text)
            decoded = tokenizer.decode(encoded)
            passed = (special_text == decoded)
            results.append(("Спецсимволы", passed))
            if self.verbose:
                print(f"  3. Спецсимволы: {repr(special_text)} -> {repr(decoded)} - "
                      f"{'✓' if passed else '✗'}")
        except Exception as e:
            results.append(("Спецсимволы", False))
            print(f"\n  3. Спецсимволы: {e}")
        
        # Тест 4: Очень маленький словарь
        try:
            tiny_tokenizer = BPETokenizer(vocab_size=10, byte_level=True)
            tiny_tokenizer.train(["a", "b", "c"], verbose=False)
            encoded = tiny_tokenizer.encode("abc")
            decoded = tiny_tokenizer.decode(encoded)
            passed = (decoded == "abc")
            results.append(("Очень маленький словарь (10)", passed))
            if self.verbose:
                print(f"  4. Словарь из 10 токенов: {len(tiny_tokenizer.vocab)} токенов - "
                      f"{'✓' if passed else '✗'}")
        except Exception as e:
            results.append(("Очень маленький словарь", False))
            print(f"\n  4. Очень маленький словарь: {e}")
        
        # Тест 5: UTF-8 граничные символы
        try:
            utf8_text = "𐍈𐍉𐍊"  # Готические буквы (4 байта)
            encoded = tokenizer.encode(utf8_text)
            decoded = tokenizer.decode(encoded)
            passed = (utf8_text == decoded)
            results.append(("UTF-8 4-байтовые символы", passed))
            if self.verbose:
                print(f"  5. UTF-8 4-байтовые: {len(utf8_text)} символов, "
                      f"{len(encoded)} байт - {'✓' if passed else '✗'}")
        except Exception as e:
            results.append(("UTF-8 граничные", False))
            print(f"\n  5. UTF-8 граничные: {e}")
        
        # Итог
        passed_count = sum(1 for _, p in results if p)
        print(f"\n  Результат: {passed_count}/{len(results)} тестов пройдено")
        
        for name, passed in results:
            print_test_result(name, passed)
            all_passed = all_passed and passed
        
        return all_passed
    
    # ======================================================================
    # ЗАПУСК ВСЕХ ТЕСТОВ
    # ======================================================================
    
    def run_all_tests(self) -> int:
        """
        Запустить все тесты.
        
        Returns:
            int: 0 если все тесты пройдены, иначе 1
        """
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ BPE ТОКЕНИЗАТОРА".center(60))
        print("=" * 60)
        
        # Счетчик пройденных тестов
        passed = 0
        total = 5
        
        # Тест 1: Byte-level
        if self.test_byte_level():
            passed += 1
        
        # Тест 2: Обучение
        tokenizer = self.test_training_small()
        if tokenizer is not None:
            passed += 1
        
        # Тест 3: Сохранение/загрузка
        if self.test_save_load(tokenizer):
            passed += 1
        
        # Тест 4: Бенчмарк
        try:
            self.benchmark(sizes=[50, 100, 200])
            passed += 1
        except Exception as e:
            print(f"\nБенчмарк провален: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        
        # Тест 5: Граничные случаи
        if self.test_edge_cases():
            passed += 1
        
        # Итог
        print("\n" + "=" * 60)
        print(f"ИТОГ: {passed}/{total} ТЕСТОВ ПРОЙДЕНО".center(60))
        print("=" * 60)
        
        if passed == total:
            print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            return 0
        else:
            print(f"\n !!! ПРОЙДЕНО {passed} ИЗ {total} ТЕСТОВ")
            return 1


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция запуска тестов.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(description='Тестирование BPE токенизатора')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    try:
        tester = BPETokenizerTest(verbose=args.verbose)
        return tester.run_all_tests()
        
    except KeyboardInterrupt:
        print("\n\n !!! Тестирование прервано пользователем")
        return 1
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())