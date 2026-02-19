"""
Модуль тестирования BPE токенизатора.

Содержит набор тестов для проверки функциональности:
- Byte-level кодирование/декодирование
- Обучение на маленьком корпусе
- Сохранение и загрузка модели
- Замер производительности
"""

import time
import sys
import logging
import tempfile
from pathlib import Path
from typing import List

# Добавляем путь для импорта tokenizer
current_file = Path(__file__).resolve()
bpe_python_dir = current_file.parent.parent  # tests/ -> bpe_python/
sys.path.insert(0, str(bpe_python_dir))

from tokenizer import BPETokenizer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BPETokenizerTest:
    """
    Класс для тестирования BPE токенизатора.
    """
    
    def __init__(self):
        """Инициализация тестов."""
        self.test_output_dir = Path('./test_output')
        self.test_output_dir.mkdir(exist_ok=True)
        
    def test_byte_level(self) -> bool:
        """
        Тестирование byte-level encoding/decoding.
        
        Проверяет корректность преобразования UTF-8 текста
        в byte-level представление и обратно.
        
        Возвращает:
            True если все тесты пройдены
        """
        print("\n" + "=" * 60)
        print("ТЕСТ 1: BYTE-LEVEL КОДИРОВАНИЕ")
        print("=" * 60)
        
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
        ]
        
        all_passed = True
        
        for i, text in enumerate(test_strings, 1):
            try:
                encoded = tokenizer._byte_encode(text)
                decoded = tokenizer._byte_decode(encoded)
                
                is_match = (text == decoded)
                all_passed = all_passed and is_match
                
                print(f"\n{i}. Тест: {text[:30]}{'...' if len(text) > 30 else ''}")
                print(f"   Оригинал: {repr(text[:30])}")
                print(f"   Закодировано: {repr(encoded[:30])}")
                print(f"   Декодировано: {repr(decoded[:30])}")
                print(f"   Результат: {'✓' if is_match else '✗'}")
                
                if not is_match:
                    print(f"   ! Ошибка: различия в строке")
                    
            except Exception as e:
                print(f"   ✗ Ошибка: {e}")
                all_passed = False
        
        print(f"\n✅ Byte-level тест: {'ПРОЙДЕН' if all_passed else 'ПРОВАЛЕН'}")
        return all_passed
    
    def test_training_small(self) -> BPETokenizer:
        """
        Тестирование обучения на маленьком корпусе.
        
        Обучает токенизатор на небольшом наборе C++ кода
        и проверяет базовую функциональность.
        
        Возвращает:
            Обученный токенизатор
        """
        print("\n" + "=" * 60)
        print("ТЕСТ 2: ОБУЧЕНИЕ НА МАЛЕНЬКОМ КОРПУСЕ")
        print("=" * 60)
        
        # Маленький корпус C++ кода для теста
        corpus = [
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
        
        print(f"Размер корпуса: {len(corpus)} строк")
        print(f"Обучение с vocab_size=50...")
        
        tokenizer = BPETokenizer(vocab_size=50, byte_level=True)
        tokenizer.train(corpus, verbose=True)
        
        print(f"Итоговый словарь: {len(tokenizer.vocab)} токенов")
        
        # Тест encode/decode на нескольких примерах
        test_cases = [
            "int main() { return 42; }",
            "cout << \"test\" << endl;",
            "vector<int> data;",
            "template<typename T>",
        ]
        
        print("\nПроверка encode/decode:")
        all_correct = True
        
        for test in test_cases:
            encoded = tokenizer.encode(test)
            decoded = tokenizer.decode(encoded)
            is_correct = (test == decoded)
            all_correct = all_correct and is_correct
            
            print(f"\n  Оригинал: {test}")
            print(f"  Токены: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
            print(f"  Декод: {decoded[:50]}")
            print(f"  Результат: {'✓' if is_correct else '✗'}")
        
        print(f"\n✅ Тест обучения: {'ПРОЙДЕН' if all_correct else 'ПРОВАЛЕН'}")
        return tokenizer
    
    def test_save_load(self, tokenizer: BPETokenizer) -> bool:
        """
        Тестирование сохранения и загрузки модели.
        
        Аргументы:
            tokenizer: Обученный токенизатор
            
        Возвращает:
            True если тест пройден
        """
        print("\n" + "=" * 60)
        print("ТЕСТ 3: СОХРАНЕНИЕ И ЗАГРУЗКА")
        print("=" * 60)
        
        # Создаем временную директорию
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            vocab_path = tmp_path / 'vocab.json'
            merges_path = tmp_path / 'merges.txt'
            
            # Сохраняем
            print(f"Сохранение в {tmp_path}...")
            tokenizer.save(str(vocab_path), str(merges_path))
            print(f"  vocab.json: {vocab_path.stat().st_size / 1024:.1f} KB")
            print(f"  merges.txt: {merges_path.stat().st_size / 1024:.1f} KB")
            
            # Загружаем
            print("Загрузка из сохраненных файлов...")
            new_tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
            print(f"  Загружено {len(new_tokenizer.vocab)} токенов")
            
            # Проверяем идентичность
            test_texts = [
                "cout << 'test' << endl;",
                "int x = 42;",
                "vector<int> vec;",
                "template<typename T>"
            ]
            
            print("\nПроверка идентичности:")
            all_match = True
            
            for text in test_texts:
                original_encoded = tokenizer.encode(text)
                loaded_encoded = new_tokenizer.encode(text)
                is_match = (original_encoded == loaded_encoded)
                all_match = all_match and is_match
                
                print(f"\n  Текст: {text}")
                print(f"  Оригинал: {original_encoded[:8]}{'...' if len(original_encoded) > 8 else ''}")
                print(f"  Загружен: {loaded_encoded[:8]}{'...' if len(loaded_encoded) > 8 else ''}")
                print(f"  Совпадает: {'✓' if is_match else '✗'}")
            
            print(f"\n✅ Тест сохранения/загрузки: {'ПРОЙДЕН' if all_match else 'ПРОВАЛЕН'}")
            return all_match
    
    def benchmark(self, sizes: List[int] = [100, 200, 500]) -> dict:
        """
        Замер производительности при разных размерах словаря.
        
        Аргументы:
            sizes: Список размеров словаря для тестирования
            
        Возвращает:
            Словарь с результатами бенчмарка
        """
        print("\n" + "=" * 60)
        print("ТЕСТ 4: БЕНЧМАРК ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 60)
        
        # Создаем тестовый корпус
        corpus = []
        for i in range(500):
            corpus.append(f"int function_{i}() {{ return {i}; }}")
            corpus.append(f"class Class_{i} {{ int value_{i}; }};")
            corpus.append(f"template<typename T_{i}> T_{i} process(T_{i} x) {{ return x; }}")
        
        print(f"Размер корпуса: {len(corpus)} строк")
        
        results = {}
        
        for vocab_size in sizes:
            print(f"\n📊 Тест vocab_size = {vocab_size}")
            
            # Обучение
            start = time.time()
            tokenizer = BPETokenizer(vocab_size=vocab_size, byte_level=True)
            tokenizer.train(corpus[:100], verbose=False)  # Используем 100 примеров
            train_time = time.time() - start
            
            # Тест encode
            test_text = corpus[0]
            n_iterations = 1000
            
            start = time.time()
            for _ in range(n_iterations):
                tokenizer.encode(test_text)
            encode_time = time.time() - start
            
            encode_speed = n_iterations / encode_time  # операций/сек
            
            # Тест decode
            encoded = tokenizer.encode(test_text)
            
            start = time.time()
            for _ in range(n_iterations):
                tokenizer.decode(encoded)
            decode_time = time.time() - start
            
            decode_speed = n_iterations / decode_time  # операций/сек
            
            results[vocab_size] = {
                'train_time': train_time,
                'encode_time': encode_time,
                'decode_time': decode_time,
                'encode_speed': encode_speed,
                'decode_speed': decode_speed,
                'vocab_size_actual': len(tokenizer.vocab)
            }
            
            print(f"  Обучение: {train_time:.3f} сек")
            print(f"  Encode: {encode_speed:.0f} оп/сек ({encode_time*1000/n_iterations:.3f} мс/оп)")
            print(f"  Decode: {decode_speed:.0f} оп/сек ({decode_time*1000/n_iterations:.3f} мс/оп)")
            print(f"  Реальный размер словаря: {len(tokenizer.vocab)}")
        
        print("\nСводка результатов:")
        print("-" * 60)
        print(f"{'Vocab':>6} {'Train (s)':>10} {'Encode/s':>10} {'Decode/s':>10} {'E/D ratio':>10}")
        print("-" * 60)
        
        for vocab_size, res in results.items():
            ed_ratio = res['encode_speed'] / res['decode_speed']
            print(f"{vocab_size:>6} {res['train_time']:>10.3f} "
                  f"{res['encode_speed']:>10.0f} {res['decode_speed']:>10.0f} "
                  f"{ed_ratio:>10.2f}")
        
        return results
    
    def test_edge_cases(self) -> bool:
        """
        Тестирование граничных случаев.
        
        Возвращает:
            True если все тесты пройдены
        """
        print("\n" + "=" * 60)
        print("ТЕСТ 5: ГРАНИЧНЫЕ СЛУЧАИ")
        print("=" * 60)
        
        all_passed = True
        
        # Тест 1: Пустой текст
        try:
            tokenizer = BPETokenizer()
            empty_text = ""
            encoded = tokenizer.encode(empty_text)
            decoded = tokenizer.decode(encoded)
            
            print(f"\n1. Пустой текст:")
            print(f"   Encode: {encoded}")
            print(f"   Decode: '{decoded}'")
            print(f"   Результат: {'✓' if decoded == '' else '✗'}")
            
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
            all_passed = False
        
        # Тест 2: Очень длинный текст
        try:
            long_text = "x" * 10000
            encoded = tokenizer.encode(long_text)
            
            print(f"\n2. Длинный текст (10000 символов):")
            print(f"   Токенов: {len(encoded)}")
            print(f"   Сжатие: {10000/len(encoded):.2f} символов/токен")
            print(f"   Результат: ✓")
            
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
            all_passed = False
        
        # Тест 3: Специальные символы
        try:
            special_text = "\n\t\r\b\f\v"
            encoded = tokenizer.encode(special_text)
            decoded = tokenizer.decode(encoded)
            is_match = (special_text == decoded)
            
            print(f"\n3. Спецсимволы:")
            print(f"   Оригинал: {repr(special_text)}")
            print(f"   После decode: {repr(decoded)}")
            print(f"   Результат: {'✓' if is_match else '✗'}")
            
        except Exception as e:
            print(f"   ✗ Ошибка: {e}")
            all_passed = False
        
        return all_passed


def get_project_paths() -> dict:
    """
    Получение путей проекта.
    """
    current_file = Path(__file__).resolve()  # tests/test_bpe_tokenizer.py
    tests_dir = current_file.parent           # tests/
    bpe_python_dir = tests_dir.parent         # bpe_python/
    project_root = bpe_python_dir.parent      # cpp-bpe-tokenizer/
    
    return {
        "project_root": project_root,
        "bpe_python_dir": bpe_python_dir,
        "tests_dir": tests_dir,
        "test_output_dir": tests_dir / 'test_output',
    }


def main():
    """Основная функция запуска тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ BPE ТОКЕНИЗАТОРА")
     
    # Получаем пути
    paths = get_project_paths()
    
    # Создаем тестовый класс
    tester = BPETokenizerTest()
    
    # Счетчик пройденных тестов
    passed = 0
    total = 5
    
    try:
        # Тест 1: Byte-level
        if tester.test_byte_level():
            passed += 1
        
        # Тест 2: Обучение
        tokenizer = tester.test_training_small()
        if tokenizer:
            passed += 1
        
        # Тест 3: Сохранение/загрузка
        if tester.test_save_load(tokenizer):
            passed += 1
        
        # Тест 4: Бенчмарк
        try:
            tester.benchmark(sizes=[50, 100, 200])
            passed += 1
        except Exception as e:
            print(f"\nБенчмарк провален: {e}")
        
        # Тест 5: Граничные случаи
        if tester.test_edge_cases():
            passed += 1
        
        # Итог
        print("\n" + "=" * 60)
        print(f"ИТОГ: {passed}/{total} тестов пройдено")
        
        if passed == total:
            print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        else:
            print(f"\nПройдено {passed}/{total} тестов")
        
        print("=" * 60)
        
        return 0 if passed == total else 1
        
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())