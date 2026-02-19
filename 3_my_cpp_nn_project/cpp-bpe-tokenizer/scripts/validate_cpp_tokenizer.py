"""
Валидация C++ токенизатора против Python эталона.
"""

import json
import subprocess
import sys
from pathlib import Path
import tempfile
import random
import time

# Добавляем путь к Python модулям bpe
sys.path.append(str(Path(__file__).parent.parent / "bpe"))

try:
    from tokenizer import BPETokenizer as PythonTokenizer
except ImportError:
    print("❌ Cannot import Python tokenizer. Make sure bpe/tokenizer.py exists")
    sys.exit(1)

class CppTokenizerWrapper:
    """Обертка для вызова C++ токенизатора"""
    
    def __init__(self, cpp_binary_path, vocab_path, merges_path):
        self.binary = Path(cpp_binary_path)
        self.vocab = Path(vocab_path)
        self.merges = Path(merges_path)
        
        # Проверяем существование файлов
        if not self.binary.exists():
            raise FileNotFoundError(f"C++ binary not found: {self.binary}")
        if not self.vocab.exists():
            raise FileNotFoundError(f"Vocabulary not found: {self.vocab}")
        if not self.merges.exists():
            raise FileNotFoundError(f"Merges file not found: {self.merges}")
    
    def encode(self, text):
        """Кодирует текст с помощью C++ токенизатора"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                       delete=False, encoding='utf-8') as f:
            f.write(text)
            input_file = f.name
        
        try:
            result = subprocess.run(
                [str(self.binary), input_file, str(self.vocab), str(self.merges)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"⚠️  C++ error: {result.stderr}")
                return []
            
            return json.loads(result.stdout)
        finally:
            Path(input_file).unlink(missing_ok=True)
    
    def decode(self, tokens):
        """Декодирует токены с помощью C++ токенизатора"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                       delete=False, encoding='utf-8') as f:
            json.dump(tokens, f)
            input_file = f.name
        
        try:
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


def load_test_samples():
    """Загружает тестовые примеры"""
    project_root = Path(__file__).parent.parent
    samples = []
    
    # Базовые синтаксические конструкции
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
    ]
    samples.extend(base_samples)
    
    # Загружаем из тренировочного корпуса
    train_file = project_root / "data" / "corpus" / "train_code.txt"
    if train_file.exists():
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                samples.extend(random.sample(lines, min(20, len(lines))))
        except Exception:
            pass
    
    return samples[:50]


def main():
    print("🔍 Валидация C++ BPE токенизатора")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # Пути к файлам
    python_vocab = project_root / "bpe" / "vocab.json"
    python_merges = project_root / "bpe" / "merges.txt"
    cpp_compare = project_root / "cpp" / "build" / "examples" / "compare_with_python"
    cpp_vocab = project_root / "cpp" / "models" / "cpp_vocab.json"
    cpp_merges = project_root / "cpp" / "models" / "cpp_merges.txt"
    
    # Проверяем наличие всех файлов
    missing = []
    for f in [python_vocab, python_merges, cpp_compare, cpp_vocab, cpp_merges]:
        if not f.exists():
            missing.append(str(f))
    
    if missing:
        print("❌ Отсутствуют файлы:")
        for f in missing:
            print(f"   - {f}")
        print("\nСначала выполните:")
        print("  cd cpp && ./scripts/build.sh")
        print("  python cpp/tools/convert_vocab.py")
        return 1
    
    # Загружаем Python токенизатор
    print("\n📦 Загрузка Python токенизатора...")
    py_tokenizer = PythonTokenizer()
    py_tokenizer.load_from_files(str(python_vocab), str(python_merges))
    print(f"   Словарь: {py_tokenizer.vocab_size()} токенов")
    
    # Создаем обертку для C++
    cpp_tokenizer = CppTokenizerWrapper(cpp_compare, cpp_vocab, cpp_merges)
    
    # Загружаем тестовые примеры
    print("📋 Загрузка тестовых примеров...")
    test_samples = load_test_samples()
    print(f"   Загружено {len(test_samples)} примеров")
    
    # Сравниваем
    print("\n⚙️ Сравнение результатов...")
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
        
        results.append({
            "text": text,
            "py_tokens": py_tokens,
            "cpp_tokens": cpp_tokens,
            "match": py_tokens == cpp_tokens,
            "py_time": py_time,
            "cpp_time": cpp_time
        })
    
    print("\n")
    
    # Анализ результатов
    matches = sum(1 for r in results if r["match"])
    total_py_time = sum(r["py_time"] for r in results)
    total_cpp_time = sum(r["cpp_time"] for r in results)
    
    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ:")
    print(f"  Всего примеров: {len(results)}")
    print(f"  Совпадение: {matches}/{len(results)} ({matches/len(results)*100:.1f}%)")
    print(f"  Время Python: {total_py_time*1000:.2f} ms")
    print(f"  Время C++:    {total_cpp_time*1000:.2f} ms")
    print(f"  Ускорение:    {total_py_time/total_cpp_time:.2f}x")
    
    if matches == len(results):
        print("\n✅ ПОЛНОЕ СОВПАДЕНИЕ!")
    elif matches > len(results) * 0.95:
        print("\n⚠️ ОТЛИЧНОЕ СОВПАДЕНИЕ (>95%)")
    else:
        print("\n❌ ЕСТЬ РАСХОЖДЕНИЯ")
        
        # Показываем примеры расхождений
        mismatches = [r for r in results if not r["match"]]
        print(f"\nПервые 3 расхождения:")
        for i, r in enumerate(mismatches[:3]):
            print(f"\n{i+1}. Текст: {r['text'][:100]}")
            print(f"   Python ({len(r['py_tokens'])}): {r['py_tokens'][:20]}...")
            print(f"   C++ ({len(r['cpp_tokens'])}):    {r['cpp_tokens'][:20]}...")
    
    # Сохраняем результаты
    output_file = project_root / "reports" / "cpp_validation_results.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "matches": matches,
        "match_percentage": matches/len(results)*100,
        "total_py_time_ms": total_py_time*1000,
        "total_cpp_time_ms": total_cpp_time*1000,
        "speedup": total_py_time/total_cpp_time
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Результаты сохранены в {output_file}")
    
    return 0 if matches == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())