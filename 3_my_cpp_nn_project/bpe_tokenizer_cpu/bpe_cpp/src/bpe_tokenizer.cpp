/**
 * @file bpe_tokenizer.cpp
 * @brief Реализация базового BPE токенизатора (эталонная версия)
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details Эталонная реализация алгоритма Byte Pair Encoding (BPE).
 *          Служит референсной версией для сравнения с оптимизированной
 *          реализацией FastBPETokenizer.
 * 
 *          **Архитектурные решения:**
 *          - Потокобезопасность через shared_mutex
 *          - Поддержка двух режимов (обычный/byte-level)
 *          - Детальная отладка через условный вывод
 *          - RAII управление ресурсами
 * 
 *          **Процессы:**
 *          1. Обучение      - Итеративное слияние частых пар
 *          2. Кодирование   - Применение правил к тексту
 *          3. Декодирование - Конкатенация токенов
 *          4. Сериализация  - Сохранение/загрузка модели
 * 
 * @note Для продакшена используйте FastBPETokenizer
 * @see BPETokenizer, FastBPETokenizer, Vocabulary
 */

#include "bpe_tokenizer.hpp"
#include "bpe_export.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <shared_mutex>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace bpe {

// ============================================================================
// Глобальное управление выводом (для отладки и CI/CD)
// ============================================================================

bool g_quiet_mode = false;    ///< Глобальный флаг тихого режима (подавляет отладочный вывод)

/**
 * @brief Условный вывод сообщения (без перевода строки)
 * @param message Сообщение для вывода
 */
void print_if_not_quiet(const std::string& message) {
    if (!g_quiet_mode) {
        std::cout << message;
    }
}

/**
 * @brief Условный вывод сообщения с опциональным переводом строки
 * @param message Сообщение для вывода
 * @param newline Добавить ли '\n' в конце
 */
void print_if_not_quiet(const std::string& message, bool newline) {
    if (!g_quiet_mode) {
        std::cout << message;
        if (newline) std::cout << std::endl;
    }
}

// ============================================================================
// Конструкторы и управление ресурсами
// ============================================================================

BPETokenizer::BPETokenizer() 
    : vocab_()
    , merges_()
    , byte_level_(false)
    , unknown_token_("<UNK>")
    , max_token_length_(1000)
    , vocab_size_(10000) {
    // Пустой конструктор - инициализация списком инициализации
}

BPETokenizer::BPETokenizer(size_t vocab_size, bool byte_level)
    : vocab_()
    , merges_()
    , byte_level_(byte_level)
    , unknown_token_("<UNK>")
    , max_token_length_(1000)
    , vocab_size_(vocab_size) {
    
    if (byte_level_) {
        // Byte-level режим: инициализируем всеми 256 возможными байтами
        for (int i = 0; i < 256; ++i) {
            std::string byte_str(1, static_cast<char>(i));
            vocab_.add_token(byte_str);
        }
    }
    
    // Специальные токены для NLP задач (всегда присутствуют)
    vocab_.add_token(unknown_token_);
    vocab_.add_token("<PAD>");
    vocab_.add_token("<BOS>");
    vocab_.add_token("<EOS>");
    vocab_.add_token("<MASK>");
    
    print_if_not_quiet("Создан токенизатор с " + std::to_string(vocab_.size()) + " базовыми токенами", true);
}

// Перемещение
BPETokenizer::BPETokenizer(BPETokenizer&& other) noexcept
    : vocab_(std::move(other.vocab_))
    , merges_(std::move(other.merges_))
    , byte_level_(other.byte_level_)
    , unknown_token_(std::move(other.unknown_token_))
    , max_token_length_(other.max_token_length_)
    , vocab_size_(other.vocab_size_)
    , metadata_(std::move(other.metadata_)) {
}

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

// ============================================================================
// Геттеры для специальных токенов
// ============================================================================

token_id_t BPETokenizer::unknown_token_id() const {
    std::shared_lock lock(mutex_);
    token_id_t id = vocab_.token_to_id(unknown_token_);
    return (id == INVALID_TOKEN) ? 0 : id;    // fallback на 0 если не найден
}

bool BPETokenizer::is_special_token(token_id_t id) const {
    std::shared_lock lock(mutex_);
    if (!vocab_.contains_id(id)) return false;
    
    std::string token = vocab_.id_to_token(id);
    return token == "<UNK>" || token == "<PAD>" || token == "<BOS>" || 
           token == "<EOS>" || token == "<MASK>";
}

// ============================================================================
// Сериализация в базовые форматы (совместимость с Python)
// ============================================================================

bool BPETokenizer::load_from_files(const std::string& vocab_path, const std::string& merges_path) {
    std::unique_lock lock(mutex_);
    
    // ------------------------------------------------------------------------
    // Загрузка словаря
    // ------------------------------------------------------------------------
    if (!vocab_.load(vocab_path)) {
        std::cerr << "[BPE] Ошибка: не удалось загрузить словарь из " << vocab_path << std::endl;
        return false;
    }
    
    // Отладочный вывод первых 10 токенов
    if (!g_quiet_mode) {
        std::cout << "Загружен словарь с " << vocab_.size() << " токенами:" << std::endl;
        auto tokens = vocab_.get_all_tokens();
        for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); ++i) {
            std::cout << "  " << i << ": '" << tokens[i] << "'" << std::endl;
        }
        if (tokens.size() > 10) {
            std::cout << "  ... и еще " << (tokens.size() - 10) << " токенов" << std::endl;
        }
    }
    
    // ------------------------------------------------------------------------
    // Загрузка правил слияния
    // ------------------------------------------------------------------------
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "[BPE] Ошибка: не удалось открыть файл слияний: " << merges_path << std::endl;
        return false;
    }
    
    merges_.clear();
    std::string line;
    int rank = 0;
    
    while (std::getline(merges_file, line)) {
        // Пропускаем комментарии и пустые строки
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

bool BPETokenizer::save_to_files(const std::string& vocab_path, const std::string& merges_path) const {
    std::shared_lock lock(mutex_);
    
    // ------------------------------------------------------------------------
    // Сохранение словаря
    // ------------------------------------------------------------------------
    if (!vocab_.save(vocab_path)) {
        std::cerr << "[BPE] Ошибка: не удалось сохранить словарь в " << vocab_path << std::endl;
        return false;
    }
    
    // ------------------------------------------------------------------------
    // Сохранение правил слияния (с сортировкой по рангу)
    // ------------------------------------------------------------------------
    std::ofstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "[BPE] Ошибка: не удалось открыть файл слияний для записи: " << merges_path << std::endl;
        return false;
    }
    
    merges_file << "#version: 3.5.0\n";    // Заголовок для совместимости
    
    // Сортировка для детерминированного вывода
    std::vector<std::pair<MergePair, int>> sorted_merges(merges_.begin(), merges_.end());
    std::sort(sorted_merges.begin(), sorted_merges.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    for (const auto& [pair, rank] : sorted_merges) {
        merges_file << pair.left << " " << pair.right << "\n";
    }
    
    return true;
}

// ============================================================================
// Основные методы: encode / decode
// ============================================================================

std::vector<token_id_t> BPETokenizer::encode(const std::string& text) const {
    std::shared_lock lock(mutex_);
    
    std::vector<token_id_t> result;
    token_id_t unk_id = unknown_token_id();
    token_id_t space_id = vocab_.token_to_id(" ");
    
    // Отладочная информация
    if (!g_quiet_mode) {
        std::cout << "\n=== encode ===" << std::endl;
        std::cout << "текст: '" << text << "'" << std::endl;
        std::cout << "режим: " << (byte_level_ ? "byte-level" : "обычный") << std::endl;
    }
    
    if (byte_level_) {
        // --------------------------------------------------------------------
        // Режим 1: Byte-level (каждый байт отдельно)
        // --------------------------------------------------------------------
        auto bytes = utf8_to_bytes(text);
        result.reserve(bytes.size());
        
        for (size_t i = 0; i < bytes.size(); ++i) {
            uint8_t b = bytes[i];
            std::string byte_str(1, static_cast<char>(b));
            token_id_t id = vocab_.token_to_id(byte_str);
            result.push_back((id != INVALID_TOKEN) ? id : unk_id);
            
            // Отладка первых 10 байт
            if (i < 10 && !g_quiet_mode) {
                std::cout << " байт " << i << ": 0x" << std::hex << (int)b << std::dec 
                          << " -> ID " << id << std::endl;
            }
        }
    } else {
        // --------------------------------------------------------------------
        // Режим 2: Обычный BPE (с пре-токенизацией)
        // --------------------------------------------------------------------
        auto words = pre_tokenize(text);
        
        if (!g_quiet_mode) {
            std::cout << "pre_tokenize вернул " << words.size() << " слов:" << std::endl;
            for (size_t i = 0; i < words.size(); ++i) {
                std::cout << "  word[" << i << "] = '" << words[i] << "'" << std::endl;
            }
        }
        
        result.reserve(words.size() * 2);    // Эвристика
        
        for (size_t i = 0; i < words.size(); ++i) {
            const auto& word = words[i];
            
            // Обработка пробела как отдельного токена
            if (word == " ") {
                if (space_id != INVALID_TOKEN) {
                    result.push_back(space_id);
                }
                continue;
            }
            
            // Разбиваем слово на символы и применяем слияния
            auto chars = split_into_chars(word);
            auto merged = apply_merges(chars);
            
            for (const auto& token : merged) {
                token_id_t id = vocab_.token_to_id(token);
                result.push_back((id != INVALID_TOKEN) ? id : unk_id);
            }
        }
    }
    
    if (!g_quiet_mode) {
        std::cout << "итого токенов: " << result.size() << std::endl;
        std::cout << "=== конец encode ===" << std::endl;
    }
    
    return result;
}

std::string BPETokenizer::decode(const std::vector<token_id_t>& tokens) const {
    std::shared_lock lock(mutex_);
    
    // Предварительное резервирование памяти (эвристика: 4 байта на токен)
    std::string result;
    result.reserve(tokens.size() * 4);
    
    for (token_id_t id : tokens) {
        if (!vocab_.contains_id(id)) {
            continue;    // Пропускаем неизвестные ID
        }
        result.append(vocab_.id_to_token(id));
    }
    
    return result;
}

std::vector<std::vector<token_id_t>> BPETokenizer::encode_batch(
    const std::vector<std::string>& texts) const {
    
    std::vector<std::vector<token_id_t>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(encode(text));
    }
    
    return results;
}

// ============================================================================
// Обучение модели
// ============================================================================

void BPETokenizer::train(const std::vector<std::string>& corpus) {
    std::unique_lock lock(mutex_);
    
    print_if_not_quiet("Обучение BPE токенизатора на " + std::to_string(corpus.size()) + " примерах...", true);
    
    // ------------------------------------------------------------------------
    // Шаг 1: Предобработка корпуса
    // ------------------------------------------------------------------------
    std::vector<std::vector<std::string>> processed_corpus;
    processed_corpus.reserve(corpus.size());
    
    for (const auto& text : corpus) {
        if (text.empty()) continue;
        
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
    
    // ------------------------------------------------------------------------
    // Шаг 2: Инициализация словаря символами
    // ------------------------------------------------------------------------
    std::unordered_map<std::string, int> word_freq;
    for (const auto& tokens : processed_corpus) {
        for (const auto& token : tokens) {
            ++word_freq[token];
        }
    }
    
    for (const auto& [token, _] : word_freq) {
        if (!vocab_.contains(token)) {
            vocab_.add_token(token);
        }
    }
    
    // Добавляем специальные токены (если ещё не добавлены)
    vocab_.add_token(unknown_token_);
    vocab_.add_token("<PAD>");
    vocab_.add_token("<BOS>");
    vocab_.add_token("<EOS>");
    vocab_.add_token("<MASK>");
    
    // ------------------------------------------------------------------------
    // Шаг 3: Основной цикл обучения BPE
    // ------------------------------------------------------------------------
    size_t num_merges = vocab_size_ - vocab_.size();
    print_if_not_quiet("Начинаем " + std::to_string(num_merges) + " операций слияния...", true);
    
    for (size_t merge_step = 0; merge_step < num_merges; ++merge_step) {
        // Подсчет частот пар
        std::unordered_map<MergePair, int, MergePairHash> pair_freqs;
        
        for (const auto& tokens : processed_corpus) {
            if (tokens.size() < 2) continue;
            
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                ++pair_freqs[{tokens[i], tokens[i + 1]}];
            }
        }
        
        if (pair_freqs.empty()) {
            print_if_not_quiet("Нет пар для слияния на шаге " + std::to_string(merge_step), true);
            break;
        }
        
        // Поиск самой частой пары
        MergePair best_pair;
        int best_freq = 0;
        for (const auto& [pair, freq] : pair_freqs) {
            if (freq > best_freq) {
                best_freq = freq;
                best_pair = pair;
            }
        }
        
        // Создание нового токена
        std::string new_token = best_pair.left + best_pair.right;
        
        // Применение слияния ко всему корпусу
        for (auto& tokens : processed_corpus) {
            std::vector<std::string> new_tokens;
            new_tokens.reserve(tokens.size());
            
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i < tokens.size() - 1 && 
                    tokens[i] == best_pair.left && 
                    tokens[i + 1] == best_pair.right) {
                    new_tokens.push_back(new_token);
                    ++i;    // Пропускаем правый элемент
                } else {
                    new_tokens.push_back(tokens[i]);
                }
            }
            
            tokens = std::move(new_tokens);
        }
        
        // Добавление в словарь и сохранение правила
        vocab_.add_token(new_token);
        merges_[best_pair] = static_cast<int>(merge_step);
        
        // Логирование прогресса (каждые 1000 шагов)
        if ((merge_step + 1) % 1000 == 0 && !g_quiet_mode) {
            std::cout << "Слияние " << (merge_step + 1) << "/" << num_merges 
                      << ": '" << best_pair.left << "' + '" << best_pair.right 
                      << "' -> '" << new_token << "' (частота: " << best_freq << ")" << std::endl;
        }
    }
    
    print_if_not_quiet("Обучение завершено! Размер словаря: " + std::to_string(vocab_.size()), true);
}

void BPETokenizer::train_with_progress(const std::vector<std::string>& corpus, bool verbose) {
    // Для базовой версии просто вызываем train с флагом verbose
    bool old_quiet = g_quiet_mode;
    g_quiet_mode = !verbose;
    train(corpus);
    g_quiet_mode = old_quiet;
}

// ============================================================================
// Геттеры для доступа к внутренним структурам
// ============================================================================

const Vocabulary& BPETokenizer::vocabulary() const {
    std::shared_lock lock(mutex_);
    return vocab_;
}

size_t BPETokenizer::vocab_size() const {
    std::shared_lock lock(mutex_);
    return vocab_.size();
}

size_t BPETokenizer::merges_count() const {
    std::shared_lock lock(mutex_);
    return merges_.size();
}

size_t BPETokenizer::max_token_length() const {
    std::shared_lock lock(mutex_);
    return max_token_length_;
}

token_id_t BPETokenizer::pad_id() const {
    std::shared_lock lock(mutex_);
    if (vocab_.contains("<PAD>")) {
        return vocab_.token_to_id("<PAD>");
    }
    return unknown_token_id();
}

token_id_t BPETokenizer::bos_id() const {
    std::shared_lock lock(mutex_);
    if (vocab_.contains("<BOS>")) {
        return vocab_.token_to_id("<BOS>");
    }
    return unknown_token_id();
}

token_id_t BPETokenizer::eos_id() const {
    std::shared_lock lock(mutex_);
    if (vocab_.contains("<EOS>")) {
        return vocab_.token_to_id("<EOS>");
    }
    return unknown_token_id();
}

token_id_t BPETokenizer::mask_id() const {
    std::shared_lock lock(mutex_);
    if (vocab_.contains("<MASK>")) {
        return vocab_.token_to_id("<MASK>");
    }
    return unknown_token_id();
}

// ============================================================================
// Методы для тестирования
// ============================================================================

token_id_t BPETokenizer::token_to_id(const std::string& token) const {
    std::shared_lock lock(mutex_);
    return vocab_.token_to_id(token);
}

std::string BPETokenizer::id_to_token(token_id_t id) const {
    std::shared_lock lock(mutex_);
    return vocab_.id_to_token(id);
}

bool BPETokenizer::contains_token(const std::string& token) const {
    std::shared_lock lock(mutex_);
    return vocab_.contains(token);
}

token_id_t BPETokenizer::add_token(const std::string& token) {
    std::unique_lock lock(mutex_);
    return vocab_.add_token(token);
}

void BPETokenizer::reset_stats() {
    // Базовая версия не собирает статистику
}

// ============================================================================
// Приватные методы: конфигурация
// ============================================================================

void BPETokenizer::set_byte_level(bool enable) {
    std::unique_lock lock(mutex_);
    byte_level_ = enable;
}

void BPETokenizer::set_unknown_token(const std::string& token) {
    std::unique_lock lock(mutex_);
    unknown_token_ = token;
}

void BPETokenizer::set_vocab_size(size_t size) {
    std::unique_lock lock(mutex_);
    vocab_size_ = size;
}

void BPETokenizer::set_max_token_length(size_t length) {
    std::unique_lock lock(mutex_);
    max_token_length_ = length;
}

// ============================================================================
// Приватные методы: вспомогательные функции токенизации
// ============================================================================

std::vector<uint8_t> BPETokenizer::utf8_to_bytes(const std::string& str) const {
    std::vector<uint8_t> bytes;
    bytes.reserve(str.size());
    
    for (char c : str) {
        bytes.push_back(static_cast<uint8_t>(c));
    }
    return bytes;
}

std::string BPETokenizer::bytes_to_utf8(const std::vector<uint8_t>& bytes) const {
    std::string result;
    result.reserve(bytes.size());
    
    for (uint8_t b : bytes) {
        result.push_back(static_cast<char>(b));
    }
    return result;
}

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

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> words;
    
    if (byte_level_) {
        // Byte-level режим: сохраняем пробелы как отдельные токены
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
        // Обычный режим: просто разбиваем по пробелам
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            words.push_back(std::move(word));
        }
    }
    
    return words;
}

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
            
            if (it != merges_.end() && (best_rank == -1 || it->second < best_rank)) {
                best_rank = it->second;
                best_pos = i;
                best_pair = pair;
            }
        }
        
        if (best_rank == -1) break;    // Нет больше применимых слияний
        
        // Выполнение слияния
        std::vector<std::string> new_parts;
        new_parts.reserve(current.size() - 1);
        
        for (size_t i = 0; i < current.size(); ++i) {
            if (i == best_pos) {
                new_parts.push_back(best_pair.left + best_pair.right);
                ++i;    // Пропускаем правый элемент
            } else {
                new_parts.push_back(current[i]);
            }
        }
        
        current = std::move(new_parts);
    }
    
    return current;
}

// ============================================================================
// Приватные методы: вспомогательные функции обучения
// ============================================================================

std::unordered_map<MergePair, int, MergePairHash> BPETokenizer::get_pair_frequencies(
    const std::vector<std::string>& corpus) const {
    
    std::unordered_map<MergePair, int, MergePairHash> frequencies;
    
    for (const auto& word : corpus) {
        auto chars = split_into_chars(word);
        if (chars.size() < 2) continue;
        
        for (size_t i = 0; i < chars.size() - 1; ++i) {
            ++frequencies[{chars[i], chars[i + 1]}];
        }
    }
    
    return frequencies;
}

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
                ++i;
            } else {
                new_chars.push_back(chars[i]);
            }
        }
        
        // Сборка обратно в строку
        std::string new_word;
        new_word.reserve(word.size());
        for (const auto& c : new_chars) {
            new_word += c;
        }
        new_corpus.push_back(std::move(new_word));
    }
    
    return new_corpus;
}

// ============================================================================
// Приватные методы: метаданные
// ============================================================================

void BPETokenizer::update_metadata() const {
    // Обновление метаданных перед сохранением
    metadata_.model_type = "BPETokenizer";
    metadata_.version = "3.5.0";
    metadata_.vocab_size = vocab_.size();
    metadata_.merges_count = merges_.size();
    metadata_.byte_level = byte_level_;
}

}    // namespace bpe