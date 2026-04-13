/**
 * @file compare_with_python.cpp
 * @brief Программа для валидации C++ токенизатора сравнением с Python эталоном
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.1.0
 * 
 * @details Вспомогательная программа, используемая Python скриптом validate_cpp_tokenizer.py
 *          для проверки корректности C++ реализации. Обеспечивает двустороннюю
 *          совместимость между C++ и Python версиями токенизатора.
 * 
 *          **Архитектура валидации:**
 *          ┌───────────────┐     ┌─────────────────┐     ┌───────────────┐
 *          │  Python тест  │ <-> │  compare_with_  │ <-> │      C++      │
 *          │    скрипт     │     │   python.cpp    │     │  токенизатор  │
 *          └───────────────┘     └─────────────────┘     └───────────────┘
 *                  |                      |                      │
 *           JSON с токенами        JSON с токенами          encode(text)
 * 
 *          **Режимы работы:**
 *          ┌─────────┬────────────────────────────────────────────────┐
 *          │ Режим   │ Описание                                       │
 *          ├─────────┼────────────────────────────────────────────────┤
 *          │ encode  │ Кодирует текст из файла -> JSON массив токенов │
 *          │ decode  │ Декодирует JSON массив токенов -> текст        │
 *          │ quiet   │ Минимальный вывод (только результат)           │
 *          └─────────┴────────────────────────────────────────────────┘
 * 
 *          **Форматы обмена:**
 *          @code
 *          // Входной текст (обычный файл)
 *          int main() { return 42; }
 *          
 *          // Выходной JSON (для encode режима)
 *          [42, 17, 35, 98, 3, 105, 32, 12]
 *          
 *          // Входной JSON (для decode режима)
 *          [42, 17, 35, 98, 3, 105, 32, 12]
 *          
 *          // Выходной текст (для decode режима)
 *          int main() { return 42; }
 *          @endcode
 * 
 * @note Используется ТОЛЬКО для тестирования и валидации
 * @see validate_cpp_tokenizer.py
 */

#include "fast_tokenizer.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace bpe;
using json = nlohmann::json;

// ============================================================================
// Вспомогательные функции для работы с файлами
// ============================================================================

/**
 * @brief Читает содержимое файла в строку
 * 
 * @param path Путь к файлу
 * @return std::string Содержимое файла
 * @throws std::runtime_error если файл не удалось открыть
 * 
 * @note Используется для чтения входного текста в encode режиме
 */
std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("[compare] Не удалось открыть файл: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

/**
 * @brief Читает JSON массив токенов из файла
 * 
 * @param path Путь к JSON файлу
 * @return std::vector<uint32_t> Вектор ID токенов
 * @throws std::runtime_error при ошибках чтения или неверном формате
 * 
 * @note Используется в decode режиме для чтения токенов из Python
 */
std::vector<uint32_t> read_tokens(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("[compare] Не удалось открыть файл с токенами: " + path);
    }
    
    json tokens_json;
    file >> tokens_json;
    
    if (!tokens_json.is_array()) {
        throw std::runtime_error("[compare] Ожидался JSON массив токенов!");
    }
    
    return tokens_json.get<std::vector<uint32_t>>();
}

// ============================================================================
// Основная функция
// ============================================================================

/**
 * @brief Точка входа
 * 
 * @param argc Количество аргументов
 * @param argv Аргументы командной строки:
 *             argv[0] - Имя программы
 *             argv[1] - input_file (текст для encode или JSON для decode)
 *             argv[2] - vocab_file (путь к JSON словарю)
 *             argv[3] - merges_file (путь к TXT слияниям)
 *             argv[4] - Опционально: --decode или --quiet
 * @return int 0 при успехе, 1 при ошибке
 * 
 * @code
 * // Примеры использования:
 * 
 * // Режим encode (по умолчанию)
 * ./compare_with_python test.cpp vocab.json merges.txt
 * 
 * // Режим encode с тихим выводом (только JSON)
 * ./compare_with_python test.cpp vocab.json merges.txt --quiet
 * 
 * // Режим decode
 * ./compare_with_python tokens.json vocab.json merges.txt --decode
 * @endcode
 */
int main(int argc, char* argv[]) {
    try {
        // ====================================================================
        // Проверка аргументов командной строки
        // ====================================================================
        
        if (argc < 4) {
            std::cerr << "Использование: " << argv[0] << std::endl;
            std::cerr << "<input_file>  - Текст для encode или JSON для decode" << std::endl;
            std::cerr << "<vocab_file>  - Путь к JSON словарю (vocab.json)" << std::endl;
            std::cerr << "<merges_file> - Путь к TXT слияниям (merges.txt)" << std::endl;
            std::cerr << "[--decode]    - Режим декодирования (по умолчанию encode)" << std::endl;
            std::cerr << "[--quiet]     - Только результат, без логов" << std::endl;
            return 1;
        }
        
        std::string input_path = argv[1];
        std::string vocab_path = argv[2];
        std::string merges_path = argv[3];
        
        bool decode_mode = false;
        bool quiet_mode = false;
        
        // Парсинг опциональных аргументов
        for (int i = 4; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--decode") {
                decode_mode = true;
            } else if (arg == "--quiet") {
                quiet_mode = true;
            }
        }
        
        // ====================================================================
        // Настройка вывода для тихого режима
        // ====================================================================
        
        if (quiet_mode) {
            // Отключаем весь не-результатный вывод
            std::cout.setstate(std::ios_base::failbit);
        }
        
        // ====================================================================
        // Инициализация токенизатора
        // ====================================================================
        
        TokenizerConfig config;
        config.byte_level = true;      // Обязательно для Unicode
        config.enable_cache = true;    // Включаем кэш для скорости
        config.cache_size = 10000;     // 10K записей достаточно
        
        FastBPETokenizer tokenizer(config);
        
        if (!quiet_mode) {
            std::cout << "[compare] Загрузка модели из: " << vocab_path << std::endl;
            std::cout << "[compare]                  и: " << merges_path << std::endl;
        }
        
        // Загрузка модели
        if (!tokenizer.load(vocab_path, merges_path)) {
            std::cerr << "[compare] КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель!" << std::endl;
            std::cerr << "[compare] - vocab_path:  " << vocab_path << std::endl;
            std::cerr << "[compare] - merges_path: " << merges_path << std::endl;
            return 1;
        }
        
        // Проверка, что словарь не пуст
        size_t vocab_size = tokenizer.vocab_size();
        if (vocab_size == 0) {
            std::cerr << "[compare] КРИТИЧЕСКАЯ ОШИБКА: Словарь пуст после загрузки!" << std::endl;
            std::cerr << "[compare] - vocab_path:  " << vocab_path << std::endl;
            std::cerr << "[compare] - merges_path: " << merges_path << std::endl;
            return 1;
        }
        
        if (!quiet_mode) {
            std::cout << "[compare] Модель загружена: " << vocab_size << " токенов!" << std::endl;
        }
        
        // ====================================================================
        // Режим декодирования (токены -> текст)
        // ====================================================================
        
        if (decode_mode) {
            if (!quiet_mode) {
                std::cout << "[compare] Режим DECODE: читаем токены из " << input_path << std::endl;
            }
            
            // Чтение токенов из JSON файла
            auto tokens = read_tokens(input_path);
            
            if (!quiet_mode) {
                std::cout << "[compare] Прочитано токенов: " << tokens.size() << std::endl;
                std::cout << "[compare] Декодируем..." << std::endl;
            }
            
            // Декодирование
            std::string text = tokenizer.decode(tokens);
            
            // В тихом режиме включаем cout обратно для вывода результата
            if (quiet_mode) {
                std::cout.clear();
            }
            
            // Вывод результата (только текст)
            std::cout << text;
            return 0;
        }
        
        // ====================================================================
        // Режим кодирования (текст -> токены) - по умолчанию
        // ====================================================================
        
        if (!quiet_mode) {
            std::cout << "[compare] Режим ENCODE: читаем текст из " << input_path << std::endl;
        }
        
        // Чтение входного текста
        std::string text = read_file(input_path);

        if (!quiet_mode) {
            std::cout << "[compare] Текст прочитан: " << text.size() << " байт" << std::endl;
            std::cout << "[compare] Кодирование..." << std::endl;
        }

        // Кодирование
        auto tokens = tokenizer.encode(text);

        if (!quiet_mode) {
            std::cout << "[compare] Получено токенов: " << tokens.size() << std::endl;
            std::cout << "[compare] Результат (JSON): " << std::endl;
        }

        // В тихом режиме включаем cout обратно для вывода результата
        if (quiet_mode) {
            std::cout.clear();
        }

        // Вывод JSON массива токенов
        json tokens_json = tokens;
        std::cout << tokens_json.dump() << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[compare] ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}