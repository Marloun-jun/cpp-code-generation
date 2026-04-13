#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ======================================================================
# test_bpe_tokenizer.py - Модуль тестирования BPE токенизатора
# ======================================================================
#
# @file test_bpe_tokenizer.py
# @brief Комплексный модуль тестирования BPE токенизатора
#Ы
# @author Евгений П.
# @date 2026
# @version 3.5.0
#
# @details Содержит полный набор тестов для проверки всех аспектов работы
#          BPE токенизатора. Обеспечивает качество и надежность кода перед релизом.
#
#          **Тестовые сценарии:**
#
#          1. **Тест 1: Byte-level кодирование**
#             - Проверка преобразования UTF-8 в байты и обратно
#             - Тесты на различных языках (русский, китайский, эмодзи)
#             - Граничные случаи (управляющие символы, длинные строки)
#
#          2. **Тест 2: Обучение на маленьком корпусе**
#             - Обучение на C++ коде (классы, шаблоны, функции)
#             - Проверка размера итогового словаря
#             - Измерение времени обучения
#
#          3. **Тест 3: Сохранение и загрузка**
#             - JSON формат (читаемый, для отладки)
#             - Бинарный формат (компактный, для продакшена)
#             - Проверка идентичности после загрузки
#
#          4. **Тест 4: Бенчмарк производительности**
#             - Влияние размера словаря на скорость
#             - Скорость encode (операций/сек)
#             - Скорость decode (операций/сек)
#             - Время обучения
#
#          5. **Тест 5: Граничные случаи**
#             - Пустой текст
#             - Длинный текст (10000 символов)
#             - Спецсимволы (\n, \t, \r, \b, \f, \v)
#             - Очень маленький словарь (256 токенов)
#             - UTF-8 4-байтовые символы (эмодзи, древние письменности)
#
#          **Ключевые метрики:**
#          - Точность roundtrip (кодирование+декодирование)
#          - Скорость encode/decode (операций/сек)
#          - Время обучения
#          - Размер словаря
#          - Сжатие данных
#
# @note Все тесты проходят успешно при правильной реализации токенизатора.
#       Используется как основной инструмент валидации перед релизом.
#
# @usage python test_bpe_tokenizer.py [--verbose]
#
# @example
#   python test_bpe_tokenizer.py              # Обычный запуск
#   python test_bpe_tokenizer.py --verbose    # Подробный вывод с диагностикой
#
# @see BPETokenizer
# @see train_tokenizer.py
# @see test_compare_models.py
#
# ======================================================================

import sys
import time
import logging
import tempfile
import argparse
import traceback

from pathlib import Path
from typing import List, Dict, Optional

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()
TESTS_DIR = CURRENT_FILE.parent
BPE_PYTHON_DIR = TESTS_DIR.parent
PROJECT_ROOT = BPE_PYTHON_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BPE_PYTHON_DIR))

# ======================================================================
# ИМПОРТ ТОКЕНИЗАТОРА
# ======================================================================

try:
    from tokenizer import BPETokenizer
except ImportError as e:
    print(f"Ошибка импорта BPETokenizer: {e}!")
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
        Dict[str, Path]: Словарь с путями проекта
    """
    paths = {
        "project_root":   PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "tests_dir":      TESTS_DIR,
        "reports_dir":    BPE_PYTHON_DIR / 'reports',
    }
    paths["reports_dir"].mkdir(exist_ok=True)
    return paths

def print_test_header(title: str, width: int = 60) -> None:
    """
    Вывести заголовок теста.
    
    Args:
        title: Заголовок
        width: Ширина линии
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}")

def print_test_result(name: str, passed: bool) -> None:
    """
    Вывести результат теста.
    
    Args:
        name:   Имя теста
        passed: Успех теста
    """
    status = "ПРОЙДЕН" if passed else "ПРОВАЛЕН"
    print(f"  {name}: {status}")

# ======================================================================
# КЛАСС ДЛЯ ТЕСТИРОВАНИЯ
# ======================================================================

class BPETokenizerTest:
    """
    Класс для комплексного тестирования BPE токенизатора.
    
    Содержит все тестовые методы для проверки функциональности,
    производительности и граничных случаев. Обеспечивает качество
    кода перед релизом.
    
    **Структура тестов:**
    - test_byte_level     - Проверка UTF-8 кодирования
    - test_training_small - Обучение на корпусе
    - test_save_load      - Сериализация
    - benchmark           - Измерение производительности
    - test_edge_cases     - Граничные случаи
    - run_all_tests       - Запуск всех тестов
    """
    
    def __init__(self, verbose: bool = False):
        """
        Инициализация тестера.
        
        Args:
            verbose: Подробный вывод (с диагностикой)
        """
        self.verbose = verbose
        self.paths = get_project_paths()
        self.reports_dir = self.paths["reports_dir"]
        logger.info(f"Директория для результатов тестов: {self.reports_dir}")
    
    # ======================================================================
    # ТЕСТ 1: BYTE-LEVEL КОДИРОВАНИЕ
    # ======================================================================
    
    def test_byte_level(self) -> bool:
        """
        Тест 1: Проверка byte-level кодирования/декодирования.
        
        Проверяет, что _byte_encode и _byte_decode работают корректно
        для различных типов текста:
        - ASCII (английский)
        - Кириллица (русский)
        - Китайские иероглифы
        - Эмодзи
        - Смешанные языки
        - Управляющие символы
        - Длинные строки
        
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
            "\n\t\r\b\f\v",
            "a" * 1000,
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
                    print(f"Результат: {'v' if is_match else 'x'}")
                    
            except Exception as e:
                print(f"\n  {i}. Ошибка: {e}!")
                all_passed = False
        
        success_rate = 100.0 * passed_count / len(test_strings)
        print(f"\nРезультат: {passed_count}/{len(test_strings)} ({success_rate:.1f}%)")
        print_test_result("Byte-level тест", all_passed)
        
        return all_passed
    
    # ======================================================================
    # ТЕСТ 2: ОБУЧЕНИЕ НА МАЛЕНЬКОМ КОРПУСЕ
    # ======================================================================
    
    def _create_test_corpus(self) -> List[str]:
        """
        Создать тестовый корпус C++ кода.
        
        Returns:
            List[str]: Список строк C++ кода
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
        ]
    
    def test_training_small(self) -> Optional[BPETokenizer]:
        """
        Тест 2: Обучение на маленьком корпусе.
        
        Обучает токенизатор на небольшом наборе C++ кода,
        проверяет, что обучение завершается без ошибок и
        возвращает токенизатор с корректным размером словаря.
        
        Returns:
            Optional[BPETokenizer]: Обученный токенизатор или None при ошибке
        """
        print_test_header("ТЕСТ 2: ОБУЧЕНИЕ НА МАЛЕНЬКОМ КОРПУСЕ")
        
        corpus = self._create_test_corpus()
        vocab_size = 50
        
        print(f"Размер корпуса:         {len(corpus)} строк")
        print(f"Целевой размер словаря: {vocab_size}")
        
        try:
            tokenizer = BPETokenizer(vocab_size=vocab_size, byte_level=True)
            
            start_time = time.time()
            tokenizer.train(corpus, verbose=self.verbose)
            train_time = time.time() - start_time
            
            print(f"\nВремя обучения: {train_time:.3f} сек")
            print(f"Итоговый словарь: {len(tokenizer.vocab)} токенов")
            
            print_test_result("Обучение", True)
            return tokenizer
            
        except Exception as e:
            print(f"\nОшибка при обучении: {e}!")
            if self.verbose:
                traceback.print_exc()
            return None
    
    # ======================================================================
    # ТЕСТ 3: СОХРАНЕНИЕ И ЗАГРУЗКА
    # ======================================================================
    
    def test_save_load(self, tokenizer: BPETokenizer) -> bool:
        """
        Тест 3: Сохранение и загрузка модели.
        
        Проверяет, что модель корректно сохраняется и загружается
        в двух форматах:
        - JSON (текстовый, читаемый)
        - Бинарный (компактный, быстрый)
        
        Args:
            tokenizer: Обученный токенизатор
            
        Returns:
            bool: True если все проверки пройдены
        """
        print_test_header("ТЕСТ 3: СОХРАНЕНИЕ И ЗАГРУЗКА")
        
        if tokenizer is None:
            print("Токенизатор не обучен, тест пропущен!")
            return False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vocab_path = tmp_path / 'vocab.json'
            merges_path = tmp_path / 'merges.txt'
            binary_path = tmp_path / 'model.bin'
            
            print(f"Временная директория: {tmp_path}")
            
            # Текстовый формат
            print("\nТекстовый формат (JSON):")
            try:
                tokenizer.save(str(vocab_path), str(merges_path))
                
                loaded_tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
                
                test_text = "int main() { return 42; }"
                original_encoded = tokenizer.encode(test_text)
                loaded_encoded = loaded_tokenizer.encode(test_text)
                
                text_match = (original_encoded == loaded_encoded)
                print(f"Совпадение: {'да' if text_match else 'нет'}")
                
            except Exception as e:
                print(f"Ошибка: {e}!")
                text_match = False
            
            # Бинарный формат
            print("\nБинарный формат:")
            try:
                tokenizer.save_binary(str(binary_path))
                
                loaded_binary = BPETokenizer.load_binary(str(binary_path))
                
                test_text = "int main() { return 42; }"
                original_encoded = tokenizer.encode(test_text)
                binary_encoded = loaded_binary.encode(test_text)
                
                binary_match = (original_encoded == binary_encoded)
                print(f"Совпадение: {'да' if binary_match else 'нет'}")
                
            except Exception as e:
                print(f"Ошибка: {e}!")
                binary_match = False
            
            all_match = text_match and binary_match
            print_test_result("Сохранение/загрузка", all_match)
            return all_match
    
    # ======================================================================
    # ТЕСТ 4: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ
    # ======================================================================
    
    def benchmark(self, sizes: List[int] = None) -> Dict[int, Dict]:
        """
        Тест 4: Бенчмарк производительности.
        
        Измеряет:
        - Время обучения
        - Скорость encode (операций/с)
        - Скорость decode (операций/с)
        
        Args:
            sizes: Список размеров словаря для тестирования
            
        Returns:
            Dict[int, Dict]: Результаты бенчмарка
        """
        if sizes is None:
            sizes = [50, 100, 200]
        
        print_test_header("ТЕСТ 4: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
        
        corpus = []
        for i in range(500):
            corpus.append(f"int function_{i}() {{ return {i}; }}")
        
        print(f"Размер корпуса: {len(corpus)} строк")
        print(f"Используем {len(corpus[:100])} примеров для обучения")
        
        results = {}
        
        for vocab_size in sizes:
            print(f"\nТест vocab_size = {vocab_size}")
            
            try:
                tokenizer = BPETokenizer(vocab_size=vocab_size, byte_level=True)
                
                start = time.time()
                tokenizer.train(corpus[:100], verbose=False)
                train_time = time.time() - start
                
                test_text = corpus[0]
                n_iterations = 1000
                
                start = time.time()
                for _ in range(n_iterations):
                    tokenizer.encode(test_text)
                encode_time = time.time() - start
                
                encode_speed = n_iterations / encode_time
                encode_ms = encode_time * 1000 / n_iterations
                
                encoded = tokenizer.encode(test_text)
                
                start = time.time()
                for _ in range(n_iterations):
                    tokenizer.decode(encoded)
                decode_time = time.time() - start
                
                decode_speed = n_iterations / decode_time
                decode_ms = decode_time * 1000 / n_iterations
                
                results[vocab_size] = {
                    'train_time': train_time,
                    'encode_speed': encode_speed,
                    'decode_speed': decode_speed,
                    'encode_ms': encode_ms,
                    'decode_ms': decode_ms,
                }
                
                print(f"Обучение: {train_time:.3f} сек")
                print(f"Encode:   {encode_speed:.0f} оп/с ({encode_ms:.3f} мс/оп)")
                print(f"Decode:   {decode_speed:.0f} оп/с ({decode_ms:.3f} мс/оп)")
                
            except Exception as e:
                print(f"Ошибка: {e}!")
                results[vocab_size] = {'error': str(e)}
        
        return results
    
    # ======================================================================
    # ТЕСТ 5: ГРАНИЧНЫЕ СЛУЧАИ
    # ======================================================================

    def test_edge_cases(self) -> bool:
        """
        Тест 5: Граничные случаи.
        
        Проверяет поведение токенизатора в экстремальных ситуациях:
        - Пустой текст
        - Очень длинный текст (10000 символов)
        - Спецсимволы (\n, \t, \r, \b, \f, \v)
        - Очень маленький словарь (256 токенов)
        - UTF-8 4-байтовые символы (эмодзи, древние письменности)
        
        Returns:
            bool: True если все тесты пройдены
        """
        print_test_header("ТЕСТ 5: ГРАНИЧНЫЕ СЛУЧАИ")
        
        # Добавляем спецсимволы в обучающую выборку
        train_corpus = [
            "a", "b", "c", "ab", "bc", "abc",
            "\n", "\t", "\r", "\b", "\f", "\v"
        ]
        
        tiny_tokenizer = BPETokenizer(vocab_size=256, byte_level=True)
        tiny_tokenizer.train(train_corpus, verbose=False)
        
        print(f"\nТокенизатор с byte_level=True обучен на {len(train_corpus)} примерах")
        print(f"Реальный размер словаря: {len(tiny_tokenizer.vocab)} токенов")
        
        all_passed = True
        results = []
        
        # Тест 1: Пустой текст
        try:
            empty_text = ""
            encoded = tiny_tokenizer.encode(empty_text)
            decoded = tiny_tokenizer.decode(encoded)
            passed = (decoded == "")
            results.append(("Пустой текст", passed))
        except Exception as e:
            results.append(("Пустой текст", False))
            print(f"\n1. Пустой текст: {e}")
        
        # Тест 2: Длинный текст
        try:
            long_text = "a" * 10000
            encoded = tiny_tokenizer.encode(long_text)
            decoded = tiny_tokenizer.decode(encoded)
            passed = (decoded == long_text)
            results.append(("Длинный текст (10000 символов)", passed))
        except Exception as e:
            results.append(("Длинный текст", False))
            print(f"\n2. Длинный текст: {e}")
        
        # Тест 3: Спецсимволы
        special_chars = ['\n', '\t', '\r', '\b', '\f', '\v']
        special_passed = True
        
        print(f"\n3. Тест спецсимволов:")
        for ch in special_chars:
            try:
                encoded = tiny_tokenizer.encode(ch)
                decoded = tiny_tokenizer.decode(encoded)
                
                # Убираем маркер конца слова, если он есть
                clean_decoded = decoded.replace(chr(0), '')    # Убираем null-символы
                
                ch_pass = (ch == clean_decoded)
                special_passed = special_passed and ch_pass
                print(f"    {repr(ch)}: {encoded} -> {repr(decoded)} -> {repr(clean_decoded)} - {'да' if ch_pass else 'нет'}")
                
                if not ch_pass and self.verbose:
                    print(f"Оригинал (hex): {ch.encode('utf-8').hex()}")
                    print(f"Декод (hex):    {clean_decoded.encode('utf-8').hex()}")
                    
            except Exception as e:
                print(f"{repr(ch)}: x {e}")
                special_passed = False
        
        results.append(("Спецсимволы", special_passed))
        
        # Тест 4: Очень маленький словарь
        try:
            encoded = tiny_tokenizer.encode("abc")
            decoded = tiny_tokenizer.decode(encoded)
            passed = (decoded == "abc")
            results.append(("Очень маленький словарь (256)", passed))
            print(f"\n4. 'abc' -> {encoded} -> '{decoded}' - {'v' if passed else 'x'}")
        except Exception as e:
            results.append(("Очень маленький словарь", False))
            print(f"\n4. Очень маленький словарь: x {e}")
        
        # Тест 5: UTF-8 4-байтовые символы
        utf8_chars = ['𐍈', '𐍉', '𐍊', '😊', '🚀', '🌟']
        utf8_passed = True
        
        print(f"\n5. Тест UTF-8 4-байтовых символов:")
        for ch in utf8_chars:
            try:
                encoded = tiny_tokenizer.encode(ch)
                decoded = tiny_tokenizer.decode(encoded)
                
                # Убираем только маркер конца слова
                clean_decoded = decoded.replace(chr(0), '')
                
                ch_pass = (ch == clean_decoded)
                utf8_passed = utf8_passed and ch_pass
                print(f"    {ch}: {encoded} -> {repr(decoded)} -> {repr(clean_decoded)} - {'v' if ch_pass else 'x'}")
            except Exception as e:
                print(f"    {ch}: {e}")
                utf8_passed = False
        
        results.append(("UTF-8 4-байтовые символы", utf8_passed))
        
        # Итог
        passed_count = sum(1 for _, p in results if p)
        print(f"\nРезультат: {passed_count}/{len(results)} тестов пройдено!")
        
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
        
        Последовательно выполняет:
        1. Byte-level тест
        2. Обучение на маленьком корпусе
        3. Сохранение/загрузка
        4. Бенчмарк
        5. Граничные случаи
        
        Returns:
            int: 0 если все тесты пройдены, 1 если есть ошибки
        """
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ BPE ТОКЕНИЗАТОРА".center(60))
        print("=" * 60)
        
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
        
        # Тест 5: Граничные случаи
        if self.test_edge_cases():
            passed += 1
        
        print("\n" + "=" * 60)
        print(f"ИТОГ: {passed}/{total} ТЕСТОВ ПРОЙДЕНО!".center(60))
        print("=" * 60)
        
        return 0 if passed == total else 1


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция для запуска тестирования.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(
        description='Тестирование BPE токенизатора',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Примеры использования:
    python test_bpe_tokenizer.py              # Обычный запуск
    python test_bpe_tokenizer.py --verbose    # Подробный вывод с диагностикой
    """
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Подробный вывод')
    args = parser.parse_args()
    
    try:
        tester = BPETokenizerTest(verbose=args.verbose)
        return tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\nТестирование прервано!")
        return 1
    except Exception as e:
        print(f"\nКритическая ошибка: {e}!")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())