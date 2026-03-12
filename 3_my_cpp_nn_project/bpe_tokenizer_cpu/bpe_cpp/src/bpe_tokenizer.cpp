/**
 * @file bpe_tokenizer.cpp
 * @brief Реализация BPE токенизатора - ядра всего проекта
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Этот файл содержит полную реализацию алгоритма Byte Pair Encoding (BPE)
 *          для токенизации текста. Является основой всех токенизаторов в проекте.
 * 
 *          **Алгоритм BPE (Byte Pair Encoding):**
 *          1. Начинаем со словаря отдельных символов (или байтов)
 *          2. Итеративно находим самую частую пару соседних токенов
 *          3. Сливаем эту пару в новый токен
 *          4. Добавляем новый токен в словарь
 *          5. Повторяем до достижения желаемого размера словаря
 * 
 *          **Режимы работы:**
 *          1) **Обычный режим** (byte_level=false)
 *             - Работает с символами UTF-8 как есть
 *             - Лучшее сжатие для ASCII текстов
 *             - Быстрее для латиницы
 *          
 *          2) **Byte-level режим** (byte_level=true)
 *             - Каждый байт UTF-8 становится отдельным токеном
 *             - Гарантированная работа с любыми символами
 *             - Поддержка эмодзи, кириллицы, иероглифов
 *             - Начальный словарь из 256 байт
 * 
 *          **Потокобезопасность:**
 *          - encode()/decode():    shared_lock для параллельного чтения
 *          - train()/load():       unique_lock для монопольной записи
 * 
 * @note В byte-level режиме словарь инициализируется всеми 256 байтами
 * @warning При обучении требуется достаточно оперативной памяти для хранения корпуса
 * 
 * @see BPETokenizer
 * @see Vocabulary
 * @see MergePair
 */

#include "bpe_tokenizer.hpp"
#include "bpe_export.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <shared_mutex>
#include <sstream>
#include <unordered_set>
#include <vector>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace bpe {

// ======================================================================
// Глобальные переменные для управления выводом
// ======================================================================

/**
 * @brief Глобальная переменная для тихого режима
 * 
 * Когда установлена в true, все отладочные сообщения подавляются.
 * Полезно для автоматизации и CI/CD.
 */
bool g_quiet_mode = false;

/**
 * @brief Вспомогательная функция для условного вывода
 * 
 * Выводит сообщение только если не включен тихий режим.
 * 
 * @param message Сообщение для вывода
 */
void print_if_not_quiet(const std::string& message) {
    if (!g_quiet_mode) {
        std::cout << message;
    }
}

/**
 * @brief Вспомогательная функция для условного вывода с переводом строки
 * 
 * @param message Сообщение для вывода
 * @param newline Добавить ли перевод строки в конце
 */
void print_if_not_quiet(const std::string& message, bool newline) {
    if (!g_quiet_mode) {
        std::cout << message;
        if (newline) std::cout << std::endl;
    }
}

// ======================================================================
// Конструкторы и деструктор
// ======================================================================

/**
 * @brief Конструктор по умолчанию
 * 
 * Создает пустой токенизатор с настройками по умолчанию:
 * - byte_level = false (обычный режим)
 * - unknown_token = "<UNK>"
 * - vocab_size = 8000
 * 
 * @note После создания необходимо загрузить модель через load_from_files()
 *       или обучить через train()
 */
BPETokenizer::BPETokenizer() 
    : vocab_()
    , merges_()
    , byte_level_(false)
    , unknown_token_("<UNK>")
    , max_token_length_(1000)
    , vocab_size_(8000) {
    // Пустой конструктор - словарь инициализируется при загрузке или обучении
}

/**
 * @brief Конструктор с параметрами
 * 
 * @param vocab_size Желаемый размер словаря
 * @param byte_level Использовать byte-level режим
 * 
 * В byte-level режиме автоматически добавляются все 256 байт в словарь.
 * В обычном режиме словарь инициализируется пустым.
 * В обоих режимах добавляются специальные токены (<UNK>, <PAD>, <BOS>, <EOS>).
 */
BPETokenizer::BPETokenizer(size_t vocab_size, bool byte_level)
    : vocab_()
    , merges_()
    , byte_level_(byte_level)
    , unknown_token_("<UNK>")
    , max_token_length_(1000)
    , vocab_size_(vocab_size) {
    
    if (byte_level_) {
        // Инициализация словаря всеми возможными байтами (0-255)
        for (int i = 0; i < 256; ++i) {
            std::string byte_str(1, static_cast<char>(i));
            vocab_.add_token(byte_str);
        }
    }
    
    // Добавление специальных токенов для управления последовательностями
    vocab_.add_token(unknown_token_);
    vocab_.add_token("<PAD>");
    vocab_.add_token("<BOS>");
    vocab_.add_token("<EOS>");
    vocab_.add_token("<MASK>");
    
    print_if_not_quiet("Создан токенизатор с " + std::to_string(vocab_.size()) + " базовыми токенами", true);
}

/**
 * @brief Конструктор перемещения
 */
BPETokenizer::BPETokenizer(BPETokenizer&& other) noexcept
    : vocab_(std::move(other.vocab_))
    , merges_(std::move(other.merges_))
    , byte_level_(other.byte_level_)
    , unknown_token_(std::move(other.unknown_token_))
    , max_token_length_(other.max_token_length_)
    , vocab_size_(other.vocab_size_)
    , metadata_(std::move(other.metadata_)) {
}

/**
 * @brief Оператор присваивания перемещением
 */
BPETokenizer& BPETokenizer::operator=(BPETokenizer&& other) noexcept {
    if (this != &other) {
        std::unique_lock lock(mutex_);
        
        vocab_ = std::move(other.vocab_);
        merges_ = std::move(other.merges_);
        byte_level_ = other.byte_level_;
        unknown_token_ = std::move(other.unknown_token_);
        max_token_length_ = other.max_token_length_;
        vocab_size_ = other.vocab_size_;
        metadata_ = std::move(other.metadata_);
    }
    return *this;
}

BPETokenizer::~BPETokenizer() = default;

// ======================================================================
// Специальные токены
// ======================================================================

/**
 * @brief Получить ID токена для неизвестных символов
 * 
 * @return token_id_t ID токена <UNK> или 0 если не найден
 */
token_id_t BPETokenizer::unknown_token_id() const {
    std::shared_lock lock(mutex_);
    token_id_t id = vocab_.token_to_id(unknown_token_);
    return (id == INVALID_TOKEN) ? 0 : id;
}

// ======================================================================
// Загрузка/сохранение из текстовых файлов
// ======================================================================

/**
 * @brief Загрузить модель из текстовых файлов
 * 
 * @param vocab_path Путь к файлу словаря (JSON)
 * @param merges_path Путь к файлу слияний (TXT)
 * @return true при успешной загрузке, false при ошибке
 * 
 * @note Выводит отладочную информацию о загруженных токенах
 */
bool BPETokenizer::load_from_files(const std::string& vocab_path, const std::string& merges_path) {
    std::unique_lock lock(mutex_);
    
    // Загрузка словаря
    if (!vocab_.load(vocab_path)) {
        std::cerr << "Ошибка: не удалось загрузить словарь из " << vocab_path << std::endl;
        return false;
    }
    
    // Отладка: вывод загруженных токенов
    print_if_not_quiet("Загружен словарь с " + std::to_string(vocab_.size()) + " токенами:", true);
    auto tokens = vocab_.get_all_tokens();
    for (size_t i = 0; i < tokens.size() && !g_quiet_mode && i < 10; ++i) {
        std::cout << "  " << i << ": " << tokens[i] << std::endl;
    }
    if (tokens.size() > 10 && !g_quiet_mode) {
        std::cout << "  ... и еще " << (tokens.size() - 10) << " токенов" << std::endl;
    }
    
    // Загрузка правил слияния
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл слияний: " << merges_path << std::endl;
        return false;
    }
    
    merges_.clear();
    std::string line;
    int rank = 0;
    
    while (std::getline(merges_file, line)) {
        // Пропускаем пустые строки и комментарии
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        std::istringstream iss(line);
        std::string left, right;
        if (iss >> left >> right) {
            merges_[{left, right}] = rank++;
        }
    }
    
    print_if_not_quiet("Загружено слияний: " + std::to_string(merges_.size()), true);
    return true;
}

/**
 * @brief Сохранить модель в текстовые файлы
 * 
 * @param vocab_path Путь для сохранения словаря
 * @param merges_path Путь для сохранения слияний
 * @return true при успешном сохранении, false при ошибке
 * 
 * @note Слияния сортируются по рангу для детерминированного вывода
 */
bool BPETokenizer::save_to_files(const std::string& vocab_path, const std::string& merges_path) const {
    std::shared_lock lock(mutex_);
    
    // Сохранение словаря
    if (!vocab_.save(vocab_path)) {
        std::cerr << "Ошибка: не удалось сохранить словарь в " << vocab_path << std::endl;
        return false;
    }
    
    // Сохранение правил слияния
    std::ofstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Ошибка: не удалось открыть файл слияний для записи: " << merges_path << std::endl;
        return false;
    }
    
    merges_file << "#version: 0.2\n";
    
    // Сортируем слияния по рангу для детерминированного вывода
    std::vector<std::pair<MergePair, int>> sorted_merges(merges_.begin(), merges_.end());
    std::sort(sorted_merges.begin(), sorted_merges.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    for (const auto& [pair, rank] : sorted_merges) {
        merges_file << pair.left << " " << pair.right << "\n";
    }
    
    return true;
}

// ======================================================================
// Основные методы encode/decode
// ======================================================================

/**
 * @brief Закодировать текст в последовательность токенов
 * 
 * @param text Входной текст
 * @return std::vector<token_id_t> Вектор ID токенов
 * 
 * @details Алгоритм:
 * 1. Для byte-level режима:
 *    - Конвертируем UTF-8 в байты
 *    - Каждый байт -> токен (0-255)
 * 2. Для обычного режима:
 *    - Пре-токенизация на слова
 *    - Каждое слово разбивается на символы
 *    - Применяются правила слияния
 * 
 * @note Содержит подробную отладочную информацию при !g_quiet_mode
 */
std::vector<token_id_t> BPETokenizer::encode(const std::string& text) const {
    std::shared_lock lock(mutex_);
    
    std::vector<token_id_t> result;
    token_id_t unk_id = unknown_token_id();
    token_id_t space_id = vocab_.token_to_id(" ");
    
    if (!g_quiet_mode) {
        std::cout << "=== encode ===" << std::endl;
        std::cout << "текст: '" << text << "'" << std::endl;
        std::cout << "режим: " << (byte_level_ ? "byte-level" : "обычный") << std::endl;
    }
    
    if (byte_level_) {
        // Byte-level режим: каждый байт кодируется отдельно
        auto bytes = utf8_to_bytes(text);
        if (!g_quiet_mode) {
            std::cout << "байт: " << bytes.size() << std::endl;
        }
        result.reserve(bytes.size());
        
        for (size_t i = 0; i < bytes.size(); ++i) {
            uint8_t b = bytes[i];
            std::string byte_str(1, static_cast<char>(b));
            token_id_t id = vocab_.token_to_id(byte_str);
            result.push_back((id != INVALID_TOKEN) ? id : unk_id);
            
            if (i < 10 && !g_quiet_mode) {
                std::cout << " байт " << i << ": 0x" << std::hex << (int)b << std::dec 
                          << " -> ID " << id << std::endl;
            }
        }
    } else {
        // Обычный режим: применяем BPE к словам
        auto words = pre_tokenize(text);
        if (!g_quiet_mode) {
            std::cout << "pre_tokenize вернул " << words.size() << " слов:" << std::endl;
            for (size_t i = 0; i < words.size(); ++i) {
                std::cout << "  word[" << i << "] = '" << words[i] << "'" << std::endl;
            }
        }
        
        result.reserve(words.size() * 2);
        
        for (size_t i = 0; i < words.size(); ++i) {
            const auto& word = words[i];
            
            if (word == " ") {
                if (!g_quiet_mode) {
                    std::cout << "  слово " << i << " - ПРОБЕЛ" << std::endl;
                }
                if (space_id != INVALID_TOKEN) {
                    result.push_back(space_id);
                    if (!g_quiet_mode) {
                        std::cout << "    добавлен токен пробела: " << space_id << std::endl;
                    }
                }
                continue;
            }
            
            if (!g_quiet_mode) {
                std::cout << "  слово " << i << " '" << word << "' - разбиваем на символы" << std::endl;
            }
            auto chars = split_into_chars(word);
            if (!g_quiet_mode) {
                std::cout << "    символов: " << chars.size() << std::endl;
            }
            
            auto merged = apply_merges(chars);
            if (!g_quiet_mode) {
                std::cout << "    после слияний: " << merged.size() << " токенов" << std::endl;
            }
            
            for (const auto& token : merged) {
                token_id_t id = vocab_.token_to_id(token);
                result.push_back((id != INVALID_TOKEN) ? id : unk_id);
                if (!g_quiet_mode) {
                    std::cout << "      токен '" << token << "' -> ID " << id << std::endl;
                }
            }
        }
    }
    
    if (!g_quiet_mode) {
        std::cout << "итого токенов: " << result.size() << std::endl;
        std::cout << "=== конец encode ===" << std::endl;
    }
    
    return result;
}

/**
 * @brief Декодировать последовательность токенов обратно в текст
 * 
 * @param tokens Вектор ID токенов
 * @return std::string Восстановленный текст
 * 
 * @details Просто конкатенирует строковые представления токенов.
 *          Для byte-level режима это автоматически восстанавливает UTF-8.
 */
std::string BPETokenizer::decode(const std::vector<token_id_t>& tokens) const {
    std::shared_lock lock(mutex_);
    
    // Используем string напрямую, а не vector<uint8_t>
    std::string result;
    result.reserve(tokens.size() * 4);    // Резервируем место для UTF-8
    
    for (token_id_t id : tokens) {
        if (!vocab_.contains_id(id)) {
            continue;
        }
        
        std::string token = vocab_.id_to_token(id);
        
        // Добавляем токен как есть, без преобразования в байты
        result.append(token);
    }
    
    return result;
}

/**
 * @brief Пакетное кодирование нескольких текстов
 * 
 * @param texts Вектор входных текстов
 * @return std::vector<std::vector<token_id_t>> Вектор результатов
 * 
 * @note Просто последовательно вызывает encode() для каждого текста
 */
std::vector<std::vector<token_id_t>> BPETokenizer::encode_batch(
    const std::vector<std::string>& texts) const {
    
    std::vector<std::vector<token_id_t>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(encode(text));
    }
    
    return results;
}

// ======================================================================
// Обучение
// ======================================================================

/**
 * @brief Обучить токенизатор на корпусе текстов
 * 
 * @param corpus Вектор строк для обучения
 * 
 * @details Алгоритм обучения:
 * 1. Предобработка корпуса (разбиение на символы/байты)
 * 2. Инициализация словаря уникальными символами/байтами
 * 3. Добавление специальных токенов
 * 4. Итеративное слияние самых частых пар
 * 5. Сохранение правил слияния с рангами
 * 
 * @note Выводит прогресс каждые 1000 слияний
 */
void BPETokenizer::train(const std::vector<std::string>& corpus) {
    std::unique_lock lock(mutex_);
    
    print_if_not_quiet("Обучение BPE токенизатора на " + std::to_string(corpus.size()) + " примерах...", true);
    
    // 1. Предобработка корпуса
    std::vector<std::vector<std::string>> processed_corpus;
    processed_corpus.reserve(corpus.size());
    
    for (const auto& text : corpus) {
        if (text.empty()) {
            continue;
        }
        
        if (byte_level_) {
            auto bytes = utf8_to_bytes(text);
            std::vector<std::string> byte_tokens;
            byte_tokens.reserve(bytes.size());
            
            for (uint8_t b : bytes) {
                byte_tokens.push_back(std::string(1, static_cast<char>(b)));
            }
            processed_corpus.push_back(std::move(byte_tokens));
        } else {
            auto chars = split_into_chars(text);
            processed_corpus.push_back(std::move(chars));
        }
    }
    
    // 2. Инициализация словаря символами/байтами
    std::unordered_map<std::string, int> word_freq;
    for (const auto& tokens : processed_corpus) {
        for (const auto& token : tokens) {
            ++word_freq[token];
        }
    }
    
    // Добавляем все уникальные символы в словарь
    for (const auto& [token, _] : word_freq) {
        if (!vocab_.contains(token)) {
            vocab_.add_token(token);
        }
    }
    
    // 3. Добавляем специальные токены, если их нет
    vocab_.add_token(unknown_token_);
    vocab_.add_token("<PAD>");
    vocab_.add_token("<BOS>");
    vocab_.add_token("<EOS>");
    vocab_.add_token("<MASK>");
    
    // 4. Основной цикл обучения BPE
    size_t num_merges = vocab_size_ - vocab_.size();
    print_if_not_quiet("Начинаем " + std::to_string(num_merges) + " операций слияния...", true);
    
    for (size_t merge_step = 0; merge_step < num_merges; ++merge_step) {
        // Подсчет частот пар
        std::unordered_map<MergePair, int, MergePairHash> pair_freqs;
        
        for (const auto& tokens : processed_corpus) {
            if (tokens.size() < 2) {
                continue;
            }
            
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                ++pair_freqs[{tokens[i], tokens[i + 1]}];
            }
        }
        
        if (pair_freqs.empty()) {
            print_if_not_quiet("Нет пар для слияния на шаге " + std::to_string(merge_step), true);
            break;
        }
        
        // Находим самую частую пару
        MergePair best_pair;
        int best_freq = 0;
        
        for (const auto& [pair, freq] : pair_freqs) {
            if (freq > best_freq) {
                best_freq = freq;
                best_pair = pair;
            }
        }
        
        // Создаем новый токен из слияния пары
        std::string new_token = best_pair.left + best_pair.right;
        
        // Применяем слияние ко всему корпусу
        for (auto& tokens : processed_corpus) {
            std::vector<std::string> new_tokens;
            new_tokens.reserve(tokens.size());
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i < tokens.size() - 1 && 
                    tokens[i] == best_pair.left && 
                    tokens[i + 1] == best_pair.right) {
                    new_tokens.push_back(new_token);
                    ++i;    // Пропускаем следующий токен
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            
            tokens = std::move(new_tokens);
        }
        
        // Добавляем новый токен в словарь и сохраняем правило слияния
        vocab_.add_token(new_token);
        merges_[best_pair] = static_cast<int>(merge_step);
        
        // Логирование прогресса
        if ((merge_step + 1) % 1000 == 0 && !g_quiet_mode) {
            std::cout << "Слияние " << (merge_step + 1) << "/" << num_merges 
                      << ": '" << best_pair.left << "' + '" << best_pair.right 
                      << "' -> '" << new_token << "' (частота: " << best_freq << ")" << std::endl;
        }
    }
    
    print_if_not_quiet("Обучение завершено! Размер словаря: " + std::to_string(vocab_.size()), true);
}

// ======================================================================
// Приватные вспомогательные методы
// ======================================================================

/**
 * @brief Преобразовать UTF-8 строку в вектор байтов
 * 
 * @param str Входная строка
 * @return std::vector<uint8_t> Вектор байтов
 */
std::vector<uint8_t> BPETokenizer::utf8_to_bytes(const std::string& str) const {
    std::vector<uint8_t> bytes;
    bytes.reserve(str.size());
    
    for (char c : str) {
        bytes.push_back(static_cast<uint8_t>(c));
    }
    return bytes;
}

/**
 * @brief Преобразовать вектор байтов обратно в UTF-8 строку
 * 
 * @param bytes Вектор байтов
 * @return std::string Восстановленная строка
 */
std::string BPETokenizer::bytes_to_utf8(const std::vector<uint8_t>& bytes) const {
    std::string result;
    result.reserve(bytes.size());
    
    for (uint8_t b : bytes) {
        result.push_back(static_cast<char>(b));
    }
    return result;
}

/**
 * @brief Разбить слово на символы (или байты)
 * 
 * @param word Входное слово
 * @return std::vector<std::string> Вектор символов/байтов
 */
std::vector<std::string> BPETokenizer::split_into_chars(const std::string& word) const {
    std::vector<std::string> chars;
    
    if (byte_level_) {
        auto bytes = utf8_to_bytes(word);
        chars.reserve(bytes.size());
        
        for (uint8_t b : bytes) {
            chars.push_back(std::string(1, static_cast<char>(b)));
        }
    } else {
        chars.reserve(word.size());
        for (char c : word) {
            chars.push_back(std::string(1, c));
        }
    }
    
    return chars;
}

/**
 * @brief Пре-токенизация текста на слова
 * 
 * @param text Входной текст
 * @return std::vector<std::string> Вектор слов
 * 
 * @details В byte-level режиме учитывает пробелы как отдельные токены.
 *          В обычном режиме использует потоковый ввод-вывод.
 */
std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> words;
    
    if (byte_level_) {
        auto bytes = utf8_to_bytes(text);
        std::string current_word;
        current_word.reserve(bytes.size());
        
        for (uint8_t b : bytes) {
            if (b == ' ') {
                if (!current_word.empty()) {
                    words.push_back(std::move(current_word));
                    current_word.clear();
                }
                words.push_back(" ");
            } else {
                current_word += static_cast<char>(b);
            }
        }
        
        if (!current_word.empty()) {
            words.push_back(std::move(current_word));
        }
    } else {
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            words.push_back(std::move(word));
        }
    }
    
    return words;
}

/**
 * @brief Применить правила слияния к слову
 * 
 * @param word_parts Слово, разбитое на части
 * @return std::vector<std::string> Слово после применения слияний
 * 
 * @details Применяет слияния в порядке возрастания ранга
 *          (самые ранние слияния применяются первыми)
 */
std::vector<std::string> BPETokenizer::apply_merges(const std::vector<std::string>& word_parts) const {
    if (word_parts.size() < 2) {
        return word_parts;
    }
    
    std::vector<std::string> current = word_parts;
    
    while (current.size() > 1) {
        int best_rank = -1;
        size_t best_pos = 0;
        MergePair best_pair;
        
        // Поиск пары с наименьшим рангом (самое раннее слияние)
        for (size_t i = 0; i < current.size() - 1; ++i) {
            MergePair pair{current[i], current[i + 1]};
            auto it = merges_.find(pair);
            
            if (it != merges_.end()) {
                if (best_rank == -1 || it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                    best_pair = pair;
                }
            }
        }
        
        // Если не нашли пару для слияния, выходим
        if (best_rank == -1) {
            break;
        }
        
        // Выполняем слияние
        std::vector<std::string> new_parts;
        new_parts.reserve(current.size() - 1);
        
        for (size_t i = 0; i < current.size(); ++i) {
            if (i == best_pos) {
                new_parts.push_back(best_pair.left + best_pair.right);
                ++i;    // Пропускаем следующий элемент
            } else {
                new_parts.push_back(current[i]);
            }
        }
        
        current = std::move(new_parts);
    }
    
    return current;
}

/**
 * @brief Получить частотность пар для обучения
 * 
 * @param corpus Корпус текстов
 * @return std::unordered_map<MergePair, int, MergePairHash> Карта пар с частотами
 */
std::unordered_map<MergePair, int, MergePairHash> BPETokenizer::get_pair_frequencies(
    const std::vector<std::string>& corpus) const {
    
    std::unordered_map<MergePair, int, MergePairHash> frequencies;
    
    for (const auto& word : corpus) {
        auto chars = split_into_chars(word);
        if (chars.size() < 2) {
            continue;
        }
        
        for (size_t i = 0; i < chars.size() - 1; ++i) {
            ++frequencies[{chars[i], chars[i + 1]}];
        }
    }
    
    return frequencies;
}

/**
 * @brief Выполнить слияние пары во всем корпусе
 * 
 * @param pair Пара для слияния
 * @param corpus Текущий корпус
 * @return std::vector<std::string> Обновленный корпус
 */
std::vector<std::string> BPETokenizer::merge_pair(
    const MergePair& pair, 
    const std::vector<std::string>& corpus) const {
    
    std::vector<std::string> new_corpus;
    new_corpus.reserve(corpus.size());
    
    std::string combined = pair.left + pair.right;
    
    for (const auto& word : corpus) {
        auto chars = split_into_chars(word);
        std::vector<std::string> new_chars;
        new_chars.reserve(chars.size());
        
        for (size_t i = 0; i < chars.size(); ++i) {
            if (i < chars.size() - 1 && chars[i] == pair.left && chars[i + 1] == pair.right) {
                new_chars.push_back(combined);
                ++i;    // Пропускаем правый элемент пары
            } else {
                new_chars.push_back(chars[i]);
            }
        }
        
        // Собираем обратно в строку
        std::string new_word;
        new_word.reserve(word.size());    // Примерная оценка размера
        
        for (const auto& c : new_chars) {
            new_word += c;
        }
        new_corpus.push_back(std::move(new_word));
    }
    
    return new_corpus;
}

} // namespace bpe