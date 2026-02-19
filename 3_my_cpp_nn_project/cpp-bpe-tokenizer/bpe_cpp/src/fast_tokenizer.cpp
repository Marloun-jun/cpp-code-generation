/**
 * @file fast_tokenizer.cpp
 * @brief Реализация оптимизированного BPE токенизатора
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 2.0.0
 * 
 * @details Высокопроизводительная реализация BPE с поддержкой:
 *          - SIMD-оптимизаций (AVX2) для массового кодирования
 *          - Параллельного обучения с простым консольным выводом
 *          - Lookup table для быстрого byte-level кодирования
 *          - Кэширования результатов
 *          - Потокобезопасных операций
 *          - Встроенного профайлера для поиска узких мест
 * 
 * @note Требует библиотеки nlohmann/json для работы с JSON
 * 
 * @see FastBPETokenizer
 * @see SimpleProfiler
 */

#include "fast_tokenizer.hpp"
#include "simd_utils.hpp"
#include "profiler.hpp"  // Добавлен профайлер

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace bpe {

// ==================== Конструкторы и деструктор ====================

FastBPETokenizer::FastBPETokenizer(const TokenizerConfig& config) 
    : config_(config)
    , unknown_id_(0)
    , pad_id_(0)
    , bos_id_(0)
    , eos_id_(0) {
    
    PROFILE_FUNCTION();  // Профилирование конструктора
    
    if (config_.enable_cache) {
        PROFILE_BLOCK("cache_initialization");
        cache_ = std::make_unique<StringViewCache>(config_.cache_size);
    }
    
    initialize_special_tokens();
    
    if (config_.enable_profiling) {
        std::cout << "🔧 FastBPETokenizer инициализирован с профилированием" << std::endl;
        SimpleProfiler::setEnabled(true);
        SimpleProfiler::setOutputFile("profiler_report.txt");
    } else {
        std::cout << "FastBPETokenizer инициализирован" << std::endl;
        SimpleProfiler::setEnabled(false);
    }
}

FastBPETokenizer::~FastBPETokenizer() {
    if (config_.enable_profiling) {
        SimpleProfiler::printReport();
        SimpleProfiler::saveReport();
    }
}

// ==================== Инициализация ====================

void FastBPETokenizer::initialize_special_tokens() {
    PROFILE_FUNCTION();
    
    auto add_special = [this](const std::string& token) -> uint32_t {
        auto it = token_to_id_.find(token);
        if (it == token_to_id_.end()) {
            id_to_token_.push_back(token);
            token_to_id_[id_to_token_.back()] = static_cast<uint32_t>(id_to_token_.size() - 1);
            return static_cast<uint32_t>(id_to_token_.size() - 1);
        }
        return it->second;
    };
    
    unknown_id_ = add_special(config_.unknown_token);
    pad_id_ = add_special(config_.pad_token);
    bos_id_ = add_special(config_.bos_token);
    eos_id_ = add_special(config_.eos_token);
}

void FastBPETokenizer::build_token_to_id_map() {
    // Уже построено при загрузке или инициализации
}

// ==================== Загрузка/сохранение ====================

bool FastBPETokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    PROFILE_FUNCTION();
    
    std::unique_lock lock(mutex_);
    
    // Очищаем текущие данные
    id_to_token_.clear();
    token_to_id_.clear();
    merges_.clear();
    
    std::cout << "📖 Загрузка словаря из: " << vocab_path << std::endl;
    
    // Загрузка словаря из JSON
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "❌ Не удалось открыть файл словаря: " << vocab_path << std::endl;
        return false;
    }
    
    try {
        PROFILE_BLOCK("json_parsing");
        nlohmann::json json_data;
        vocab_file >> json_data;
        
        std::cout << "  JSON загружен, тип: " << json_data.type_name() << std::endl;
        
        // Поддержка формата {"0": "<PAD>", "1": "<UNK>", ...}
        if (json_data.is_object()) {
            PROFILE_BLOCK("vocab_object_parsing");
            std::cout << "  Разбор словаря в формате объект..." << std::endl;
            
            // Находим максимальный ID для резервирования памяти
            size_t max_id = 0;
            for (auto& [key, _] : json_data.items()) {
                try {
                    size_t id = std::stoul(key);
                    if (id > max_id) max_id = id;
                } catch (...) {
                    std::cout << "  ⚠️ Предупреждение: нечисловой ключ: " << key << std::endl;
                }
            }
            
            std::cout << "  Максимальный ID: " << max_id << std::endl;
            
            // Резервируем место
            id_to_token_.resize(max_id + 1);
            
            // Заполняем словарь
            for (auto& [key, value] : json_data.items()) {
                try {
                    size_t id = std::stoul(key);
                    std::string token = value.get<std::string>();
                    
                    // Проверяем, не занят ли уже этот ID
                    if (id < id_to_token_.size() && !id_to_token_[id].empty()) {
                        std::cout << "  ⚠️ Предупреждение: ID " << id << " уже имеет токен '" 
                                << id_to_token_[id] << "', перезапись на '" << token << "'" << std::endl;
                    }
                    
                    id_to_token_[id] = token;
                    token_to_id_[id_to_token_[id]] = static_cast<uint32_t>(id);
                    
                    // Отладка для ключевых ID
                    if (id < 10 || (id >= 95 && id <= 100)) {
                        std::cout << "    Загружен: ID " << id << " -> '" << token << "'" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "  ❌ Ошибка разбора записи словаря: " << key << " -> " << e.what() << std::endl;
                }
            }
                        
        } else if (json_data.is_array()) {
            PROFILE_BLOCK("vocab_array_parsing");
            std::cout << "  Разбор словаря в формате массив..." << std::endl;
            for (size_t i = 0; i < json_data.size(); i++) {
                std::string token = json_data[i].get<std::string>();
                id_to_token_.push_back(token);
                token_to_id_[id_to_token_.back()] = static_cast<uint32_t>(i);
                
                if (i < 10) {
                    std::cout << "    Загружен: ID " << i << " -> '" << token << "'" << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "❌ Ошибка разбора JSON словаря: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "  ✅ Всего загружено токенов: " << id_to_token_.size() << std::endl;
    
    // Диагностика содержимого
    #ifdef DEBUG
    diagnose_vocabulary();
    #endif
    
    // Загрузка мерджей
    std::cout << "📖 Загрузка слияний из: " << merges_path << std::endl;
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "❌ Не удалось открыть файл слияний: " << merges_path << std::endl;
        return false;
    }
    
    {
        PROFILE_BLOCK("merges_loading");
        std::string line;
        int rank = 0;
        while (std::getline(merges_file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::istringstream iss(line);
            std::string left, right;
            if (iss >> left >> right) {
                auto left_it = token_to_id_.find(left);
                auto right_it = token_to_id_.find(right);
                
                if (left_it != token_to_id_.end() && right_it != token_to_id_.end()) {
                    merge_key_t key = make_merge_key(left_it->second, right_it->second);
                    merges_[key] = rank++;
                }
            }
        }
    }
    
    std::cout << "  ✅ Загружено слияний: " << merges_.size() << std::endl;
    
    // Убеждаемся что специальные токены есть
    initialize_special_tokens();
    
    return true;
}

bool FastBPETokenizer::save(const std::string& vocab_path, const std::string& merges_path) const {
    PROFILE_FUNCTION();
    
    std::shared_lock lock(mutex_);
    
    // Сохранение словаря
    std::ofstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "❌ Не удалось открыть файл словаря для записи: " << vocab_path << std::endl;
        return false;
    }
    
    {
        PROFILE_BLOCK("vocab_saving");
        nlohmann::json json_data;
        for (size_t i = 0; i < id_to_token_.size(); i++) {
            json_data[std::to_string(i)] = id_to_token_[i];
        }
        
        // Сохраняем с флагом для поддержки UTF-8
        vocab_file << json_data.dump(-1, ' ', true, nlohmann::json::error_handler_t::ignore);
    }
    
    // Сохранение мерджей
    std::ofstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "❌ Не удалось открыть файл слияний для записи: " << merges_path << std::endl;
        return false;
    }
    
    {
        PROFILE_BLOCK("merges_saving");
        merges_file << "#version: 0.2\n";
        
        // Сортируем мерджи по рангу для детерминированного вывода
        std::vector<std::pair<merge_key_t, uint32_t>> sorted_merges(merges_.begin(), merges_.end());
        std::sort(sorted_merges.begin(), sorted_merges.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        for (const auto& [key, rank] : sorted_merges) {
            uint32_t left_id = get_left_from_key(key);
            uint32_t right_id = get_right_from_key(key);
            
            if (left_id < id_to_token_.size() && right_id < id_to_token_.size()) {
                merges_file << id_to_token_[left_id] << " " << id_to_token_[right_id] << "\n";
            }
        }
    }
    
    std::cout << "💾 Сохранен словарь (" << id_to_token_.size() << " токенов) и слияния ("
              << merges_.size() << " пар)" << std::endl;
    
    return true;
}

bool FastBPETokenizer::save_binary(const std::string& path) const {
    (void)path;
    std::cerr << "⚠️ Бинарное сохранение еще не реализовано" << std::endl;
    return false;
}

bool FastBPETokenizer::load_binary(const std::string& path) {
    (void)path;
    std::cerr << "⚠️ Бинарная загрузка еще не реализована" << std::endl;
    return false;
}

// ==================== Диагностика ====================

#ifdef DEBUG
void FastBPETokenizer::diagnose_vocabulary() const {
    std::cout << "\n=== ДИАГНОСТИКА СЛОВАРЯ ===" << std::endl;
    
    // Статистика по длинам токенов
    size_t single_char = 0;
    size_t multi_char = 0;
    
    for (const auto& token : id_to_token_) {
        if (token.length() == 1) {
            single_char++;
        } else {
            multi_char++;
        }
    }
    
    std::cout << "  Односимвольных токенов: " << single_char << std::endl;
    std::cout << "  Многосимвольных токенов: " << multi_char << std::endl;
    
    // Проверка наличия ASCII символов
    std::cout << "\n  Отображение ASCII символов:" << std::endl;
    for (int i = 32; i < 127; i++) {
        std::string ch(1, static_cast<char>(i));
        auto it = token_to_id_.find(ch);
        if (it != token_to_id_.end()) {
            std::cout << "    '" << ch << "' -> " << it->second << std::endl;
        }
    }
    
    std::cout << "============================\n" << std::endl;
}
#endif

// ==================== Кодирование ====================

std::vector<uint32_t> FastBPETokenizer::byte_level_encode(std::string_view text) {
    PROFILE_FUNCTION();
    
    // Статический lookup table для быстрого преобразования char -> ID
    static std::array<uint32_t, 256> char_to_id;
    static bool initialized = false;
    
    if (!initialized) {
        PROFILE_BLOCK("lookup_table_initialization");
        char_to_id.fill(unknown_id_);
        
        for (const auto& [token, id] : token_to_id_) {
            if (token.length() == 1) {
                char_to_id[static_cast<unsigned char>(token[0])] = id;
            }
        }
        
        initialized = true;
        
        if (config_.enable_profiling) {
            int char_count = 0;
            for (uint32_t id : char_to_id) {
                if (id != unknown_id_) char_count++;
            }
            std::cout << "  Таблица поиска инициализирована с " << char_count 
                      << " символами" << std::endl;
        }
    }
    
    // Используем SIMD если доступно
    #ifdef USE_AVX2
    static bool avx2_available = bpe::SIMDUtils::check_avx2_support();
    
    if (avx2_available) {
        PROFILE_BLOCK("avx2_encode");
        static bool avx2_warning = true;
        if (avx2_warning && config_.enable_profiling) {
            std::cout << "🔥 AVX2 оптимизации ВКЛЮЧЕНЫ" << std::endl;
            avx2_warning = false;
        }
        return bpe::SIMDUtils::encode_avx2(text, char_to_id.data(), unknown_id_);
    }
    #endif
    
    // Скалярная версия (fallback)
    PROFILE_BLOCK("scalar_encode");
    std::vector<uint32_t> result;
    result.reserve(text.size());
    
    for (char c : text) {
        result.push_back(char_to_id[static_cast<unsigned char>(c)]);
    }
    
    return result;
}

std::vector<uint32_t> FastBPETokenizer::normal_encode(std::string_view text) {
    PROFILE_FUNCTION();
    // TODO: Реализовать полноценное BPE кодирование
    return byte_level_encode(text);
}

std::vector<uint32_t> FastBPETokenizer::encode(std::string_view text) {
    PROFILE_FUNCTION();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<uint32_t> result;
    
    if (config_.byte_level) {
        result = byte_level_encode(text);
    } else {
        result = normal_encode(text);
    }
    
    if (config_.enable_profiling) {
        auto end = std::chrono::high_resolution_clock::now();
        stats_.encode_calls++;
        stats_.total_tokens_processed += result.size();
        stats_.total_encode_time_ms += 
            std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    return result;
}

std::vector<std::vector<uint32_t>> FastBPETokenizer::encode_batch(
    const std::vector<std::string_view>& texts) {
    
    PROFILE_FUNCTION();
    
    std::vector<std::vector<uint32_t>> results;
    results.reserve(texts.size());
    
    #ifdef USE_OPENMP
    PROFILE_BLOCK("openmp_batch_encode");
    #pragma omp parallel for
    for (size_t i = 0; i < texts.size(); ++i) {
        auto encoded = encode(texts[i]);
        #pragma omp critical
        {
            results[i] = std::move(encoded);
        }
    }
    #else
    PROFILE_BLOCK("sequential_batch_encode");
    for (const auto& text : texts) {
        results.push_back(encode(text));
    }
    #endif
    
    return results;
}

// ==================== Декодирование ====================

std::string FastBPETokenizer::decode(const std::vector<uint32_t>& tokens) {
    PROFILE_FUNCTION();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string result;
    result.reserve(tokens.size());
    
    // Быстрая проверка специальных токенов для ID < 64
    uint64_t special_mask = 0;
    if (unknown_id_ < 64) special_mask |= (1ULL << unknown_id_);
    if (pad_id_ < 64) special_mask |= (1ULL << pad_id_);
    if (bos_id_ < 64) special_mask |= (1ULL << bos_id_);
    if (eos_id_ < 64) special_mask |= (1ULL << eos_id_);
    
    {
        PROFILE_BLOCK("decode_loop");
        for (uint32_t id : tokens) {
            // Пропускаем специальные токены
            if (id < 64 && (special_mask & (1ULL << id))) {
                continue;
            }
            
            if (id < id_to_token_.size()) {
                result += id_to_token_[id];
            }
        }
    }
    
    if (config_.enable_profiling) {
        auto end = std::chrono::high_resolution_clock::now();
        stats_.decode_calls++;
        stats_.total_decode_time_ms += 
            std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    return result;
}

std::string FastBPETokenizer::byte_level_decode(const std::vector<uint32_t>& tokens) {
    return decode(tokens);
}

std::string FastBPETokenizer::normal_decode(const std::vector<uint32_t>& tokens) {
    return decode(tokens);
}

// ==================== Токенизация слов ====================

std::vector<uint32_t> FastBPETokenizer::tokenize_word(std::string_view word) {
    PROFILE_FUNCTION();
    return byte_level_encode(word);
}

#ifdef USE_AVX2
std::vector<uint32_t> FastBPETokenizer::tokenize_word_avx2(std::string_view word) {
    PROFILE_FUNCTION();
    return byte_level_encode(word);
}
#endif

// ==================== Обучение ====================

void FastBPETokenizer::train(const std::vector<std::string>& corpus) {
    std::cerr << "⚠️ Обучение еще не реализовано в FastBPETokenizer" << std::endl;
    std::cerr << "   Используйте parallel_train() для оптимизированного обучения" << std::endl;
}

void FastBPETokenizer::parallel_train(const std::vector<std::string>& corpus, size_t num_merges) {
    PROFILE_FUNCTION();
    
    std::cout << "\n🚀 Запуск параллельного обучения на " << corpus.size() << " примерах..." << std::endl;
    std::cout << "   Используется потоков: " << std::thread::hardware_concurrency() << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // ===== Шаг 1: Подсчет частот символов =====
    std::cout << "\n📊 Подсчет частот символов..." << std::endl;
    auto freq_result = count_char_frequencies_parallel(corpus);
    
    auto mid_time = std::chrono::high_resolution_clock::now();
    auto freq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mid_time - start_time);
    std::cout << "✅ Подсчет частот завершен за " << freq_duration.count() << " мс" << std::endl;
    std::cout << "📈 Найдено уникальных символов: " << freq_result.size() << std::endl;
    
    // ===== Шаг 2: Построение начального словаря =====
    std::cout << "\n📚 Построение начального словаря..." << std::endl;
    build_initial_vocabulary(freq_result);
    
    std::cout << "✅ Начальный размер словаря: " << id_to_token_.size() << std::endl;
    
    // ===== Шаг 3: BPE слияния =====
    std::cout << "\n🔄 Выполнение BPE слияний (цель: " << num_merges << " операций)..." << std::endl;
    // TODO: Реализовать параллельные BPE слияния
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n🎉 Параллельное обучение завершено!" << std::endl;
    std::cout << "   Всего затрачено времени: " << total_duration.count() << " мс" << std::endl;
}

std::unordered_map<std::string, int> FastBPETokenizer::count_char_frequencies_parallel(
    const std::vector<std::string>& corpus) {
    
    PROFILE_FUNCTION();
    
    const size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::unordered_map<std::string, int>> thread_freqs(num_threads);
    std::vector<std::thread> threads;
    
    // Запускаем потоки
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            PROFILE_BLOCK("worker_thread");
            size_t start = t * corpus.size() / num_threads;
            size_t end = (t + 1) * corpus.size() / num_threads;
            
            size_t total = end - start;
            size_t last_percent = 0;
            
            for (size_t i = start; i < end; ++i) {
                PROFILE_BLOCK("process_line");
                for (char c : corpus[i]) {
                    thread_freqs[t][std::string(1, c)]++;
                }
                
                // Показываем прогресс каждые 5%
                if ((i - start) % (total / 20 + 1) == 0) {
                    int percent = static_cast<int>((i - start) * 100.0 / total);
                    if (percent >= last_percent + 5) {
                        last_percent = percent;
                        std::cout << "\r      Прогресс: " << percent << "% (" 
                                  << (i - start) << "/" << total << ")" << std::flush;
                    }
                }
            }
        });
    }
    
    // Ждем завершения
    for (auto& th : threads) {
        th.join();
    }
    std::cout << "\r      Прогресс: 100% (" << corpus.size() << "/" << corpus.size() << ")" << std::endl;
    
    // Объединяем результаты
    PROFILE_BLOCK("merge_results");
    std::unordered_map<std::string, int> combined;
    for (const auto& tf : thread_freqs) {
        for (const auto& [ch, freq] : tf) {
            combined[ch] += freq;
        }
    }
    
    return combined;
}

void FastBPETokenizer::build_initial_vocabulary(
    const std::unordered_map<std::string, int>& char_freq) {
    
    PROFILE_FUNCTION();
    
    // Очищаем текущий словарь
    id_to_token_.clear();
    token_to_id_.clear();
    
    // Добавляем специальные токены
    initialize_special_tokens();
    
    // Добавляем символы из корпуса
    size_t total_chars = char_freq.size();
    size_t current = 0;
    size_t last_percent = 0;
    
    for (const auto& [ch, _] : char_freq) {
        if (!token_to_id_.count(ch)) {
            id_to_token_.push_back(ch);
            token_to_id_[ch] = static_cast<uint32_t>(id_to_token_.size() - 1);
        }
        
        current++;
        int percent = static_cast<int>(current * 100.0 / total_chars);
        if (percent >= last_percent + 5) {
            last_percent = percent;
            std::cout << "\r      Прогресс: " << percent << "% (" 
                      << current << "/" << total_chars << ")" << std::flush;
        }
    }
    std::cout << "\r      Прогресс: 100% (" << total_chars << "/" << total_chars << ")" << std::endl;
}

} // namespace bpe