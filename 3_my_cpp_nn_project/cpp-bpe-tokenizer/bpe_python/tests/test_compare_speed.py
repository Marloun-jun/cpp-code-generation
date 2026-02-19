"""
Тест производительности BPE токенизатора.

Измеряет скорость encode/decode операций на большом объеме C++ кода.
Проводит множественные итерации для получения статистически значимых результатов.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional

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


# Большой тестовый C++ код для бенчмарка
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


def load_test_text(multiplier: int = 20) -> str:
    """
    Загрузка тестового C++ кода с повторением.
    
    Аргументы:
        multiplier: Количество повторений базового кода
        
    Возвращает:
        Строка с тестовым кодом
    """
    return TEST_CODE * multiplier


def find_model_files(model_size: int = 8000) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Поиск файлов модели по стандартным путям.
    
    Аргументы:
        model_size: Размер модели (8000, 10000, 12000)
        
    Возвращает:
        Кортеж (путь к vocab.json, путь к merges.txt)
    """
    possible_paths = [
        # В директории models относительно bpe_python
        bpe_python_dir / 'models' / f'bpe_{model_size}' / 'vocab.json',
        bpe_python_dir / 'models' / f'bpe_{model_size}' / 'merges.txt',
        
        # В текущей директории
        Path.cwd() / 'vocab.json',
        Path.cwd() / 'merges.txt',
        
        # В родительской директории
        Path.cwd().parent / 'vocab.json',
        Path.cwd().parent / 'merges.txt',
    ]
    
    vocab_path = None
    merges_path = None
    
    # Ищем vocab.json
    for path in possible_paths[::2]:  # Четные индексы - vocab
        if path.exists():
            vocab_path = path
            break
    
    # Ищем merges.txt
    for path in possible_paths[1::2]:  # Нечетные индексы - merges
        if path.exists():
            merges_path = path
            break
    
    return vocab_path, merges_path


class SpeedTest:
    """
    Класс для тестирования скорости работы токенизатора.
    """
    
    def __init__(self, tokenizer: BPETokenizer):
        """
        Инициализация теста скорости.
        
        Аргументы:
            tokenizer: Загруженный токенизатор
        """
        self.tokenizer = tokenizer
        self.results = {}
        
    def warmup(self, text: str, iterations: int = 5) -> None:
        """
        Прогрев токенизатора перед замерами.
        
        Аргументы:
            text: Тестовый текст
            iterations: Количество итераций прогрева
        """
        logger.info("Прогрев токенизатора...")
        for i in range(iterations):
            _ = self.tokenizer.encode(text)
            if (i + 1) % 10 == 0:
                logger.debug(f"Прогрев: {i + 1}/{iterations}")
        logger.info("Прогрев завершен")
    
    def test_encode_speed(self, text: str, iterations: int = 50) -> dict:
        """
        Тестирование скорости encode.
        
        Аргументы:
            text: Тестовый текст
            iterations: Количество итераций
            
        Возвращает:
            Словарь с результатами
        """
        logger.info(f"Запуск encode теста ({iterations} итераций)...")
        
        start_time = time.perf_counter()
        total_tokens = 0
        token_counts = []
        
        for i in range(iterations):
            tokens = self.tokenizer.encode(text)
            token_count = len(tokens)
            total_tokens += token_count
            token_counts.append(token_count)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Encode: {i + 1}/{iterations}")
        
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
        }
        
        return results
    
    def test_decode_speed(self, encoded_texts: list, iterations: int = 50) -> dict:
        """
        Тестирование скорости decode.
        
        Аргументы:
            encoded_texts: Список закодированных текстов
            iterations: Количество итераций
            
        Возвращает:
            Словарь с результатами
        """
        logger.info(f"Запуск decode теста ({iterations} итераций)...")
        
        start_time = time.perf_counter()
        total_chars = 0
        
        for i in range(iterations):
            for encoded in encoded_texts:
                decoded = self.tokenizer.decode(encoded)
                total_chars += len(decoded)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"Decode: {i + 1}/{iterations}")
        
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
        
        return results
    
    def test_roundtrip(self, test_strings: list) -> dict:
        """
        Тест точности encode-decode (roundtrip).
        
        Аргументы:
            test_strings: Список тестовых строк
            
        Возвращает:
            Словарь с результатами
        """
        logger.info("Тестирование roundtrip точности...")
        
        results = {
            'total': 0,
            'perfect': 0,
            'failed': []
        }
        
        for text in test_strings:
            results['total'] += 1
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            
            if text == decoded:
                results['perfect'] += 1
            else:
                results['failed'].append({
                    'original': text,
                    'decoded': decoded,
                    'tokens': [self.tokenizer.vocab.get(idx, '<UNK>') 
                              for idx in encoded[:10]]
                })
        
        results['accuracy'] = (results['perfect'] / results['total']) * 100
        
        return results
    
    def print_report(self, encode_results: dict, decode_results: dict, 
                    roundtrip_results: dict, text: str) -> None:
        """
        Вывод отчета о тестировании.
        
        Аргументы:
            encode_results: Результаты encode теста
            decode_results: Результаты decode теста
            roundtrip_results: Результаты roundtrip теста
            text: Тестовый текст
        """
        separator = "=" * 60
        print(f"\n{separator}")
        print("ОТЧЕТ О ТЕСТИРОВАНИИ СКОРОСТИ BPE ТОКЕНИЗАТОРА")
        print(separator)
        
        # Информация о модели
        print(f"\nМодель:")
        print(f"  Размер словаря: {len(self.tokenizer.vocab)} токенов")
        print(f"  Byte-level: {self.tokenizer.byte_level}")
        print(f"  Спец. токены: {self.tokenizer.special_tokens}")
        
        # Информация о тесте
        print(f"\nПараметры теста:")
        print(f"  Размер текста: {len(text)} символов")
        print(f"  Размер текста: {len(text.encode('utf-8')) / 1024:.2f} KB")
        print(f"  Итераций encode: {encode_results['iterations']}")
        print(f"  Итераций decode: {decode_results['iterations']}")
        
        # Результаты encode
        print(f"\n⚡ encode:")
        print(f"  Общее время: {encode_results['total_time']*1000:.2f} мс")
        print(f"  Среднее время: {encode_results['avg_time_per_encode']:.3f} мс")
        print(f"  Скорость: {encode_results['mb_per_second']:.2f} MB/сек")
        print(f"  Токенов/сек: {encode_results['tokens_per_second']:.0f}")
        print(f"  Токенов/текст: {encode_results['avg_tokens_per_text']:.1f} "
              f"(min: {encode_results['min_tokens']}, max: {encode_results['max_tokens']})")
        
        # Результаты decode
        print(f"\ndecode:")
        print(f"  Общее время: {decode_results['total_time']*1000:.2f} мс")
        print(f"  Среднее время: {decode_results['avg_time_per_decode']:.3f} мс")
        print(f"  Скорость: {decode_results['mb_per_second']:.2f} MB/сек")
        print(f"  Символов/сек: {decode_results['chars_per_second']:.0f}")
        
        # Roundtrip точность
        print(f"\nТочность roundtrip:")
        print(f"  Точность: {roundtrip_results['accuracy']:.1f}% "
              f"({roundtrip_results['perfect']}/{roundtrip_results['total']})")
        
        if roundtrip_results['failed']:
            print(f"  Ошибки ({len(roundtrip_results['failed'])}):")
            for fail in roundtrip_results['failed'][:3]:  # Показываем первые 3
                print(f"    • {fail['original'][:50]}...")
        
        print(f"\n{separator}")


def get_project_paths() -> dict:
    """
    Получение путей проекта.
    """
    current_file = Path(__file__).resolve()  # tests/test_compare_speed.py
    tests_dir = current_file.parent  # tests/
    bpe_python_dir = tests_dir.parent  # bpe_python/
    project_root = bpe_python_dir.parent  # cpp-bpe-tokenizer/
    
    return {
        "project_root": project_root,
        "bpe_python_dir": bpe_python_dir,
        "tests_dir": tests_dir,
        "models_dir": bpe_python_dir / 'models',
    }


def main():
    """Основная функция тестирования."""
    print("=" * 60)
    print("ТЕСТ СКОРОСТИ BPE ТОКЕНИЗАТОРА")
    print("=" * 60)
    
    # Получаем пути
    paths = get_project_paths()
    
    # Параметры
    model_size = 8000  # Можно изменить на 10000 или 12000
    iterations = 50
    
    # Ищем файлы модели
    vocab_path = paths['models_dir'] / f'bpe_{model_size}' / 'vocab.json'
    merges_path = paths['models_dir'] / f'bpe_{model_size}' / 'merges.txt'
    
    if not vocab_path.exists() or not merges_path.exists():
        print(f"\nМодель bpe_{model_size} не найдена!")
        print(f"  Искали vocab: {vocab_path}")
        print(f"  Искали merges: {merges_path}")
        print(f"\nДоступные модели:")
        for model_dir in paths['models_dir'].iterdir():
            if model_dir.is_dir() and model_dir.name.startswith('bpe_'):
                print(f"  • {model_dir.name}")
        return 1
    
    try:
        # Загружаем токенизатор
        print(f"\nЗагрузка модели bpe_{model_size}...")
        tokenizer = BPETokenizer.load(str(vocab_path), str(merges_path))
        print(f"  ✓ Загружено {len(tokenizer.vocab)} токенов")
        
        # Создаем тест
        speed_test = SpeedTest(tokenizer)
        
        # Загружаем тестовый текст
        print(f"\nПодготовка тестовых данных...")
        text = load_test_text(multiplier=20)
        print(f"  Размер текста: {len(text)} символов")
        print(f"  Размер в байтах: {len(text.encode('utf-8')) / 1024:.2f} KB")
        
        # Прогрев
        speed_test.warmup(text, iterations=5)
        
        # Тест encode
        encode_results = speed_test.test_encode_speed(text, iterations=iterations)
        
        # Подготовка данных для decode теста
        print(f"\nодготовка encode данных для decode теста...")
        encoded_texts = [tokenizer.encode(text) for _ in range(10)]
        
        # Тест decode
        decode_results = speed_test.test_decode_speed(encoded_texts, iterations=iterations)
        
        # Тест roundtrip
        test_strings = [
            "#include <iostream>",
            "int main() { return 0; }",
            "std::vector<int> vec = {1, 2, 3};",
            "template<typename T> class Vector {",
            "// Это комментарий на русском языке",
            "auto ptr = std::make_unique<int>(42);",
        ]
        roundtrip_results = speed_test.test_roundtrip(test_strings)
        
        # Вывод отчета
        speed_test.print_report(encode_results, decode_results, roundtrip_results, text)
        
        print("\nТестирование завершено успешно!")
        return 0
        
    except Exception as e:
        print(f"\nОшибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())