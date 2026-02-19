/**
 * @file bpe_tokenizer.cpp
 * @brief Реализация BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Реализация алгоритма Byte Pair Encoding (BPE) для токенизации текста.
 *          Поддерживает два режима работы:
 *          - Обычный режим: работа с символами UTF-8
 *          - Byte-level режим: работа с сырыми байтами (256 базовых токенов)
 * 
 *          Ключевые возможности:
 *          - Обучение на корпусе текстов с динамическим расширением словаря
 *          - Потокобезопасное кодирование/декодирование (shared_mutex)
 *          - Сохранение/загрузка модели в текстовом и бинарном форматах
 *          - Пакетная обработка для повышения производительности
 *          - Поддержка специальных токенов (<UNK>, <PAD>, <BOS>, <EOS>)
 * 
 * @note В byte-level режиме словарь инициализируется всеми 256 байтами
 * @warning При обучении требуется достаточно оперативной памяти для хранения корпуса
 * 
 * @see BPETokenizer
 * @see Vocabulary
 */

#include "bpe_tokenizer.hpp"

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

namespace bpe {

// ==================== Конструкторы и деструктор ====================

BPETokenizer::BPETokenizer() 
    : vocab_()
    , merges_()
    , byte_level_(false)
    , unknown_token_("<UNK>")
    , max_token_length_(1000)
    , vocab_size_(8000) {
    // Пустой конструктор - словарь инициализируется при загрузке или обучении
}

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
    
    std::cout << "Created tokenizer with " << vocab_.size() << " base tokens" << std::endl;
}

BPETokenizer::~BPETokenizer() = default;

// ==================== Публичные методы ====================

token_id_t BPETokenizer::unknown_token_id() const {
    token_id_t id = vocab_.token_to_id(unknown_token_);
    return (id == INVALID_TOKEN) ? 0 : id;
}

bool BPETokenizer::load_from_files(const std::string& vocab_path, const std::string& merges_path) {
    std::unique_lock lock(mutex_);
    
    // Загрузка словаря
    if (!vocab_.load(vocab_path)) {
        std::cerr << "Failed to load vocabulary from " << vocab_path << std::endl;
        return false;
    }
    
    // Отладка: вывод загруженных токенов
    std::cout << "Loaded vocabulary with " << vocab_.size() << " tokens:" << std::endl;
    auto tokens = vocab_.get_all_tokens();
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << "  " << i << ": " << tokens[i] << std::endl;
    }
    
    // Загрузка правил слияния
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file: " << merges_path << std::endl;
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
    
    return true;
}

bool BPETokenizer::save_to_files(const std::string& vocab_path, const std::string& merges_path) const {
    std::shared_lock lock(mutex_);
    
    // Сохранение словаря
    if (!vocab_.save(vocab_path)) {
        std::cerr << "Failed to save vocabulary to " << vocab_path << std::endl;
        return false;
    }
    
    // Сохранение правил слияния
    std::ofstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file for writing: " << merges_path << std::endl;
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
// Сохранение в бинарный формат (один файл)
// ======================================================================

bool BPETokenizer::save_binary(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    
    try {
        // Магическое число для проверки формата
        uint32_t magic = 0x42504542;  // "BPEB"
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        
        // Версия формата
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Размер словаря
        uint32_t vocab_size = static_cast<uint32_t>(vocab_.size());
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        
        // Токены
        auto tokens = vocab_.get_all_tokens();
        for (const auto& token : tokens) {
            uint32_t len = static_cast<uint32_t>(token.size());
            file.write(reinterpret_cast<const char*>(&len), sizeof(len));
            file.write(token.data(), len);
        }
        
        // Количество мерджей
        uint32_t merges_count = static_cast<uint32_t>(merges_.size());
        file.write(reinterpret_cast<const char*>(&merges_count), sizeof(merges_count));
        
        // Мерджи (сначала левый, потом правый токен)
        for (const auto& [pair, rank] : merges_) {
            uint32_t left_len = static_cast<uint32_t>(pair.left.size());
            uint32_t right_len = static_cast<uint32_t>(pair.right.size());
            
            file.write(reinterpret_cast<const char*>(&left_len), sizeof(left_len));
            file.write(pair.left.data(), left_len);
            file.write(reinterpret_cast<const char*>(&right_len), sizeof(right_len));
            file.write(pair.right.data(), right_len);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка бинарного сохранения: " << e.what() << std::endl;
        return false;
    }
}

// ======================================================================
// Загрузка из бинарного формата (один файл)
// ======================================================================

bool BPETokenizer::load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    try {
        // Проверка магического числа
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != 0x42504542) return false;
        
        // Проверка версии
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version != 1) return false;
        
        // Очищаем текущие данные
        vocab_.clear();
        merges_.clear();
        
        // Загрузка словаря
        uint32_t vocab_size;
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        for (uint32_t i = 0; i < vocab_size; ++i) {
            uint32_t len;
            file.read(reinterpret_cast<char*>(&len), sizeof(len));
            
            std::string token(len, '\0');
            file.read(&token[0], len);
            
            vocab_.add_token(token);
        }
        
        // Загрузка мерджей
        uint32_t merges_count;
        file.read(reinterpret_cast<char*>(&merges_count), sizeof(merges_count));
        
        for (uint32_t i = 0; i < merges_count; ++i) {
            uint32_t left_len, right_len;
            
            file.read(reinterpret_cast<char*>(&left_len), sizeof(left_len));
            std::string left(left_len, '\0');
            file.read(&left[0], left_len);
            
            file.read(reinterpret_cast<char*>(&right_len), sizeof(right_len));
            std::string right(right_len, '\0');
            file.read(&right[0], right_len);
            
            MergePair pair{left, right};
            merges_[pair] = i;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка бинарной загрузки: " << e.what() << std::endl;
        return false;
    }
}

std::vector<token_id_t> BPETokenizer::encode(const std::string& text) const {
    std::shared_lock lock(mutex_);
    
    std::vector<token_id_t> result;
    token_id_t unk_id = unknown_token_id();
    
    if (byte_level_) {
        // Byte-level режим: каждый байт кодируется отдельно
        auto bytes = utf8_to_bytes(text);
        result.reserve(bytes.size());
        
        for (uint8_t b : bytes) {
            std::string byte_str(1, static_cast<char>(b));
            token_id_t id = vocab_.token_to_id(byte_str);
            result.push_back((id != INVALID_TOKEN) ? id : unk_id);
        }
    } else {
        // Обычный режим: применяем BPE к словам
        auto words = pre_tokenize(text);
        result.reserve(words.size() * 2);  // Примерная оценка
        
        for (const auto& word : words) {
            if (word == " ") {
                continue;
            }
            
            auto chars = split_into_chars(word);
            auto merged = apply_merges(chars);
            
            for (const auto& token : merged) {
                token_id_t id = vocab_.token_to_id(token);
                result.push_back((id != INVALID_TOKEN) ? id : unk_id);
            }
        }
    }
    
    return result;
}

std::string BPETokenizer::decode(const std::vector<token_id_t>& tokens) const {
    std::shared_lock lock(mutex_);
    
    std::vector<uint8_t> bytes;
    bytes.reserve(tokens.size() * 4);  // Примерная оценка для UTF-8
    
    for (token_id_t id : tokens) {
        if (!vocab_.contains_id(id)) {
            continue;
        }
        
        std::string token = vocab_.id_to_token(id);
        
        // Преобразуем токен обратно в байты
        for (char c : token) {
            bytes.push_back(static_cast<uint8_t>(c));
        }
    }
    
    if (bytes.empty()) {
        return "";
    }
    
    // Конвертируем байты обратно в строку
    return std::string(bytes.begin(), bytes.end());
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

void BPETokenizer::train(const std::vector<std::string>& corpus) {
    std::unique_lock lock(mutex_);
    
    std::cout << "Training BPE tokenizer on " << corpus.size() << " examples..." << std::endl;
    
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
    
    // 4. Основной цикл обучения BPE
    size_t num_merges = vocab_size_ - vocab_.size();
    std::cout << "Starting " << num_merges << " merge operations..." << std::endl;
    
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
            std::cout << "No more pairs to merge at step " << merge_step << std::endl;
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
                    ++i;  // Пропускаем следующий токен
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
        if ((merge_step + 1) % 1000 == 0) {
            std::cout << "Merge " << (merge_step + 1) << "/" << num_merges 
                      << ": '" << best_pair.left << "' + '" << best_pair.right 
                      << "' -> '" << new_token << "' (freq: " << best_freq << ")" << std::endl;
        }
    }
    
    std::cout << "Training complete! Vocabulary size: " << vocab_.size() << std::endl;
}

// ==================== Приватные вспомогательные методы ====================

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
                ++i;  // Пропускаем следующий элемент
            } else {
                new_parts.push_back(current[i]);
            }
        }
        
        current = std::move(new_parts);
    }
    
    return current;
}

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
                ++i;  // Пропускаем правый элемент пары
            } else {
                new_chars.push_back(chars[i]);
            }
        }
        
        // Собираем обратно в строку
        std::string new_word;
        new_word.reserve(word.size());  // Примерная оценка размера
        
        for (const auto& c : new_chars) {
            new_word += c;
        }
        new_corpus.push_back(std::move(new_word));
    }
    
    return new_corpus;
}

} // namespace bpe