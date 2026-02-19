#!/usr/bin/env python3
# ======================================================================
# test_compare_speed.py - Тест производительности BPE токенизатора
# ======================================================================
#
# @file test_compare_speed.py
# @brief Тест производительности BPE токенизатора
#
# @author Евгений П.
# @date 2026
# @version 3.3.0
#
# @details Измеряет скорость encode/decode операций на большом объеме C++ кода.
#          Проводит множественные итерации для получения статистически значимых
#          результатов. Включает:
#          - Тест скорости encode
#          - Тест скорости decode
#          - Тест точности roundtrip
#          - Детальная статистика по времени и объему
#
# @usage python test_compare_speed.py [--model-size SIZE] [--iterations N] [--verbose]
#
# @example
#   python test_compare_speed.py
#   python test_compare_speed.py --model-size 10000 --iterations 100
#   python test_compare_speed.py --verbose
#
# ======================================================================

import sys
import time
import logging
import argparse

from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

# ======================================================================
# НАСТРОЙКА ПУТЕЙ ДЛЯ ИМПОРТА
# ======================================================================

CURRENT_FILE = Path(__file__).resolve()           # tests/test_compare_speed.py
TESTS_DIR = CURRENT_FILE.parent                    # tests/
BPE_PYTHON_DIR = TESTS_DIR.parent                  # bpe_python/
PROJECT_ROOT = BPE_PYTHON_DIR.parent               # cpp-bpe-tokenizer/

# Добавляем путь для импорта токенизатора
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
# БОЛЬШОЙ ТЕСТОВЫЙ C++ КОД ДЛЯ БЕНЧМАРКА
# ======================================================================

TEST_CODE = '''#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>

class BenchmarkTest {
public:
    BenchmarkTest(const std::string& name) : name_(name) {
        logger_ = std::make_unique<std::stringstream>();
    }
    
    void process_data(int count) {
        data_.reserve(count);
        for (int i = 0; i < count; ++i) {
            data_.push_back(process_item(i));
        }
    }
    
    double process_item(int value) {
        // Симуляция сложной обработки
        double result = static_cast<double>(value);
        for (int j = 0; j < 5; ++j) {
            result = result * 1.5 + j;
        }
        return result;
    }
    
    void print_stats() const {
        if (!logger_) return;
        
        *logger_ << "=== Stats for " << name_ << " ===\\n";
        *logger_ << "Items: " << data_.size() << "\\n";
        
        if (!data_.empty()) {
            auto [min_it, max_it] = std::minmax_element(data_.begin(), data_.end());
            double sum = 0.0;
            for (double val : data_) {
                sum += val;
            }
            
            *logger_ << "Min: " << *min_it << "\\n";
            *logger_ << "Max: " << *max_it << "\\n";
            *logger_ << "Avg: " << sum / data_.size() << "\\n";
        }
        
        std::cout << logger_->str();
    }

private:
    std::string name_;
    std::vector<double> data_;
    std::unique_ptr<std::stringstream> logger_;
};

template<typename T>
class TemplateProcessor {
public:
    TemplateProcessor() = default;
    
    template<typename... Args>
    void process_all(Args&&... args) {
        (process_single(std::forward<Args>(args)), ...);
    }
    
    void process_single(const T& item) {
        if constexpr (std::is_arithmetic_v<T>) {
            processed_.push_back(item * item);
        } else {
            processed_.push_back(item);
        }
    }
    
    const auto& get_results() const { return processed_; }

private:
    std::vector<T> processed_;
};

int main() {
    // Тестовый код для бенчмарка
    BenchmarkTest test("benchmark");
    test.process_data(1000);
    test.print_stats();
    
    TemplateProcessor<int> int_processor;
    int_processor.process_all(1, 2, 3, 4, 5);
    
    auto results = int_processor.get_results();
    for (auto val : results) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
'''


# ======================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================================================================

def get_project_paths() -> Dict[str, Path]:
    """
    Получить пути проекта.
    
    Returns:
        Dict[str, Path]: Словарь с путями проекта
    """
    return {
        "project_root": PROJECT_ROOT,
        "bpe_python_dir": BPE_PYTHON_DIR,
        "tests_dir": TESTS_DIR,
        "models_dir": BPE_PYTHON_DIR / 'models',
    }


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


def load_test_text(multiplier: int = 20) -> str:
    """
    Загрузить тестовый C++ код с повторением.
    
    Args:
        multiplier: Количество повторений базового кода
        
    Returns:
        str: Строка с тестовым кодом
    """
    return TEST_CODE * multiplier


def find_model_files(model_size: int = 8000) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Найти файлы модели по стандартным путям.
    
    Args:
        model_size: Размер модели (8000, 10000, 12000)
        
    Returns:
        Tuple[Optional[Path], Optional[Path]]: (путь к vocab.json, путь к merges.txt)
    """
    paths = get_project_paths()
    
    # Стандартный путь в директории models
    vocab_path = paths['models_dir'] / f'bpe_{model_size}' / 'vocab.json'
    merges_path = paths['models_dir'] / f'bpe_{model_size}' / 'merges.txt'
    
    if vocab_path.exists() and merges_path.exists():
        return vocab_path, merges_path
    
    # Альтернативные пути
    alt_paths = [
        (Path.cwd() / 'vocab.json', Path.cwd() / 'merges.txt'),
        (Path.cwd().parent / 'vocab.json', Path.cwd().parent / 'merges.txt'),
        (paths['bpe_python_dir'] / 'vocab.json', paths['bpe_python_dir'] / 'merges.txt'),
    ]
    
    for v_path, m_path in alt_paths:
        if v_path.exists() and m_path.exists():
            return v_path, m_path
    
    return None, None


# ======================================================================
# КЛАСС ДЛЯ ТЕСТИРОВАНИЯ СКОРОСТИ
# ======================================================================

class SpeedTest:
    """
    Класс для тестирования скорости работы токенизатора.
    
    Проводит комплексное тестирование производительности:
    - Скорость encode операций
    - Скорость decode операций
    - Точность roundtrip преобразований
    """
    
    def __init__(self, tokenizer: BPETokenizer, verbose: bool = False):
        """
        Инициализация теста скорости.
        
        Args:
            tokenizer: Загруженный токенизатор
            verbose: Подробный вывод
        """
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    # ======================================================================
    # ТЕСТЫ
    # ======================================================================
    
    def warmup(self, text: str, iterations: int = 5) -> None:
        """
        Прогрев токенизатора перед замерами.
        
        Args:
            text: Тестовый текст
            iterations: Количество итераций прогрева
        """
        print(f"\nПрогрев токенизатора ({iterations} итераций)...")
        for i in range(iterations):
            _ = self.tokenizer.encode(text)
            if self.verbose and (i + 1) % 10 == 0:
                print(f"   Прогрев: {i + 1}/{iterations}")
        print("   ✓ Прогрев завершен")
    
    def test_encode_speed(self, text: str, iterations: int = 50) -> Dict[str, Any]:
        """
        Тестирование скорости encode.
        
        Args:
            text: Тестовый текст
            iterations: Количество итераций
            
        Returns:
            Dict[str, Any]: Словарь с результатами
        """
        print(f"\n⚡ Тест encode ({iterations} итераций)...")
        
        start_time = time.perf_counter()
        total_tokens = 0
        token_counts = []
        
        for i in range(iterations):
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)
            total_tokens += token_count
            token_counts.append(token_count)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"   Прогресс: {i + 1}/{iterations}")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Статистика
        results = {
            'total_time': duration,
            'total_tokens': total_tokens,
            'iterations': iterations,
            'avg_time_per_encode': duration / iterations * 1000,  # мс
            'tokens_per_second': total_tokens / duration,
            'bytes_per_second': len(text) * iterations / duration,
            'mb_per_second': (len(text) * iterations / duration) / (1024 * 1024),
            'avg_tokens_per_text': total_tokens / iterations,
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'std_tokens': self._std_dev(token_counts),
        }
        
        print(f"   ✓ Среднее время: {results['avg_time_per_encode']:.3f} мс")
        print(f"   ✓ Скорость: {results['mb_per_second']:.2f} MB/сек")
        
        return results
    
    def test_decode_speed(self, encoded_texts: List[List[int]], iterations: int = 50) -> Dict[str, Any]:
        """
        Тестирование скорости decode.
        
        Args:
            encoded_texts: Список закодированных текстов
            iterations: Количество итераций
            
        Returns:
            Dict[str, Any]: Словарь с результатами
        """
        print(f"\nТест decode ({iterations} итераций)...")
        
        start_time = time.perf_counter()
        total_chars = 0
        
        for i in range(iterations):
            for encoded in encoded_texts:
                decoded = self.tokenizer.decode(encoded)
                total_chars += len(decoded)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"   Прогресс: {i + 1}/{iterations}")
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        total_ops = iterations * len(encoded_texts)
        
        results = {
            'total_time': duration,
            'total_chars': total_chars,
            'iterations': iterations,
            'operations': total_ops,
            'avg_time_per_decode': duration / total_ops * 1000,  # мс
            'chars_per_second': total_chars / duration,
            'mb_per_second': (total_chars / duration) / (1024 * 1024),
        }
        
        print(f"   ✓ Среднее время: {results['avg_time_per_decode']:.3f} мс")
        print(f"   ✓ Скорость: {results['mb_per_second']:.2f} MB/сек")
        
        return results
    
    def test_roundtrip(self, test_strings: List[str]) -> Dict[str, Any]:
        """
        Тест точности encode-decode (roundtrip).
        
        Args:
            test_strings: Список тестовых строк
            
        Returns:
            Dict[str, Any]: Словарь с результатами
        """
        print(f"\nТест roundtrip точности ({len(test_strings)} примеров)...")
        
        results = {
            'total': len(test_strings),
            'perfect': 0,
            'failed': []
        }
        
        for i, text in enumerate(test_strings):
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            
            if text == decoded:
                results['perfect'] += 1
            else:
                results['failed'].append({
                    'index': i,
                    'original': text[:50] + '...' if len(text) > 50 else text,
                    'tokens': encoded[:10]
                })
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"   Прогресс: {i + 1}/{len(test_strings)}")
        
        results['accuracy'] = (results['perfect'] / results['total']) * 100
        
        print(f"   ✓ Точность: {results['accuracy']:.1f}% "
              f"({results['perfect']}/{results['total']})")
        
        return results
    
    # ======================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ======================================================================
    
    def _std_dev(self, values: List[float]) -> float:
        """
        Вычислить стандартное отклонение.
        
        Args:
            values: Список значений
            
        Returns:
            float: Стандартное отклонение
        """
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    # ======================================================================
    # ОТЧЕТ
    # ======================================================================
    
    def print_report(self, encode_results: Dict, decode_results: Dict, 
                    roundtrip_results: Dict, text: str) -> None:
        """
        Вывести отчет о тестировании.
        
        Args:
            encode_results: Результаты encode теста
            decode_results: Результаты decode теста
            roundtrip_results: Результаты roundtrip теста
            text: Тестовый текст
        """
        print_header("ОТЧЕТ О ТЕСТИРОВАНИИ СКОРОСТИ")
        
        # Информация о модели
        print(f"\nМодель:")
        print(f"  Размер словаря: {len(self.tokenizer.vocab)} токенов")
        print(f"  Byte-level: {self.tokenizer.byte_level}")
        print(f"  Спец. токены: {', '.join(self.tokenizer.special_tokens)}")
        
        # Информация о тесте
        text_bytes = len(text.encode('utf-8'))
        print(f"\nПараметры теста:")
        print(f"  Размер текста: {len(text)} символов")
        print(f"  Размер в байтах: {text_bytes / 1024:.2f} KB")
        print(f"  Итераций encode: {encode_results['iterations']}")
        print(f"  Итераций decode: {decode_results['iterations']}")
        
        # Результаты encode
        print(f"\nENCODE:")
        print(f"  Общее время: {encode_results['total_time']*1000:.2f} мс")
        print(f"  Среднее время: {encode_results['avg_time_per_encode']:.3f} мс")
        print(f"  Скорость: {encode_results['mb_per_second']:.2f} MB/сек")
        print(f"  Токенов/сек: {encode_results['tokens_per_second']:.0f}")
        print(f"  Токенов/текст: {encode_results['avg_tokens_per_text']:.1f} "
              f"(min: {encode_results['min_tokens']}, max: {encode_results['max_tokens']})")
        if encode_results['std_tokens'] > 0:
            print(f"  Станд. отклонение: ±{encode_results['std_tokens']:.1f}")
        
        # Результаты decode
        print(f"\nDECODE:")
        print(f"  Общее время: {decode_results['total_time']*1000:.2f} мс")
        print(f"  Среднее время: {decode_results['avg_time_per_decode']:.3f} мс")
        print(f"  Скорость: {decode_results['mb_per_second']:.2f} MB/сек")
        print(f"  Символов/сек: {decode_results['chars_per_second']:.0f}")
        
        # Roundtrip точность
        print(f"\nТОЧНОСТЬ ROUNDTRIP:")
        print(f"  Точность: {roundtrip_results['accuracy']:.1f}% "
              f"({roundtrip_results['perfect']}/{roundtrip_results['total']})")
        
        if roundtrip_results['failed']:
            print(f"  Ошибки ({len(roundtrip_results['failed'])}):")
            for fail in roundtrip_results['failed'][:3]:  # Показываем первые 3
                print(f"    • Пример {fail['index']}: {fail['original']}")
        
        print(f"\n{'=' * 60}")


# ======================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ======================================================================

def main() -> int:
    """
    Основная функция тестирования.
    
    Returns:
        int: 0 при успехе, 1 при ошибке
    """
    parser = argparse.ArgumentParser(description='Тест производительности BPE токенизатора')
    parser.add_argument('--model-size', type=int, default=8000, choices=[8000, 10000, 12000],
                       help='Размер модели (8000, 10000, 12000)')
    parser.add_argument('--iterations', '-n', type=int, default=50,
                       help='Количество итераций для усреднения')
    parser.add_argument('--multiplier', '-m', type=int, default=20,
                       help='Множитель тестового текста')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    args = parser.parse_args()
    
    print_header("ТЕСТ СКОРОСТИ BPE ТОКЕНИЗАТОРА")
    
    # Получаем пути
    paths = get_project_paths()
    
    # Ищем файлы модели
    vocab_path, merges_path = find_model_files(args.model_size)
    
    if not vocab_path or not merges_path:
        print(f"\nМодель bpe_{args.model_size} не найдена!")
        print(f"   Искали в: {paths['models_dir']}")
        print(f"\nДоступные модели:")
        if paths['models_dir'].exists():
            for model_dir in sorted(paths['models_dir'].iterdir()):
                if model_dir.is_dir() and model_dir.name.startswith('bpe_'):
                    size = model_dir.name.replace('bpe_', '')
                    vocab = model_dir / 'vocab.json'
                    merges = model_dir / 'merges.txt'
                    status = '✓' if vocab.exists() and merges.exists() else '✗'
                    print(f"   {status} {model_dir.name}")
        else:
            print(f"   Директория {paths['models_dir']} не существует")
        return 1
    
    try:
        # Загружаем токенизатор
        print(f"\nЗагрузка модели bpe_{args.model_size}...")
        tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
        print(f"   ✓ Загружено {len(tokenizer.vocab)} токенов")
        
        # Создаем тест
        speed_test = SpeedTest(tokenizer, verbose=args.verbose)
        
        # Загружаем тестовый текст
        print(f"\nПодготовка тестовых данных...")
        text = load_test_text(multiplier=args.multiplier)
        text_bytes = len(text.encode('utf-8'))
        print(f"   Размер текста: {len(text)} символов")
        print(f"   Размер в байтах: {text_bytes / 1024:.2f} KB")
        print(f"   Размер в MB: {text_bytes / (1024 * 1024):.2f} MB")
        
        # Прогрев
        speed_test.warmup(text, iterations=5)
        
        # Тест encode
        encode_results = speed_test.test_encode_speed(text, iterations=args.iterations)
        
        # Подготовка данных для decode теста
        print(f"\nПодготовка encode данных для decode теста...")
        encoded_texts = [tokenizer.encode(text) for _ in range(10)]
        total_tokens = sum(len(et) for et in encoded_texts)
        print(f"   Подготовлено {len(encoded_texts)} закодированных текстов")
        print(f"   Всего токенов: {total_tokens}")
        
        # Тест decode
        decode_results = speed_test.test_decode_speed(encoded_texts, iterations=args.iterations)
        
        # Тест roundtrip
        test_strings = [
            "#include <iostream>",
            "int main() { return 0; }",
            "std::vector<int> vec = {1, 2, 3};",
            "template<typename T> class Vector {",
            "// Это комментарий на русском языке",
            "auto ptr = std::make_unique<int>(42);",
            "std::cout << \"Привет, мир!\" << std::endl;",
            "class MyClass { public: void method(); };",
        ]
        roundtrip_results = speed_test.test_roundtrip(test_strings)
        
        # Вывод отчета
        speed_test.print_report(encode_results, decode_results, roundtrip_results, text)
        
        print("\nТестирование завершено успешно!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n !!! Тестирование прервано пользователем")
        return 1
    except Exception as e:
        print(f"\nОшибка при тестировании: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())