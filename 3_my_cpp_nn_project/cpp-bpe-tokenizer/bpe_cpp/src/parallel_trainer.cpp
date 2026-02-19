/**
 * @file parallel_trainer.cpp
 * @brief Реализация параллельного обучения BPE токенизатора
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Реализация многопоточных алгоритмов для BPE обучения:
 *          - Использование OpenMP для параллельных циклов
 *          - Разделение корпуса на независимые чанки
 *          - Сбор и объединение результатов из потоков
 * 
 * @note Для производительности использует thread-local storage
 */

#include "parallel_trainer.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <omp.h>

namespace bpe {

// ==================== Конструктор/деструктор ====================

ParallelTrainer::ParallelTrainer(size_t num_threads) {
    if (num_threads == 0) {
        num_threads_ = std::thread::hardware_concurrency();
    } else {
        num_threads_ = num_threads;
    }
    
    #ifdef USE_OPENMP
        omp_set_num_threads(static_cast<int>(num_threads_));
    #endif
    
    std::cout << "ParallelTrainer инициализирован с " 
              << num_threads_ << " потоками" << std::endl;
}

ParallelTrainer::~ParallelTrainer() = default;

// ==================== Основной метод обучения ====================

bool ParallelTrainer::train(const std::vector<std::string>& corpus,
                            size_t target_size,
                            Vocabulary& vocab,
                            std::unordered_map<merge_key_t, int>& merges) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Запуск параллельного обучения на " << corpus.size() 
              << " примерах..." << std::endl;
    
    // 1. Разбиваем корпус на чанки
    auto chunks = split_corpus(corpus);
    
    // 2. Подсчет частот символов
    std::cout << "Подсчет частот символов..." << std::endl;
    auto freq_start = std::chrono::high_resolution_clock::now();
    
    auto char_freq = count_char_frequencies_parallel(chunks);
    
    auto freq_end = std::chrono::high_resolution_clock::now();
    stats_.freq_time_sec = std::chrono::duration<double>(freq_end - freq_start).count();
    
    std::cout << "   Найдено уникальных символов: " << char_freq.size() << std::endl;
    
    // 3. Инициализация словаря
    vocab.clear();
    for (const auto& [ch, _] : char_freq) {
        vocab.add_token(ch);
    }
    vocab.add_special_tokens({"<UNK>", "<PAD>", "<BOS>", "<EOS>"});
    
    std::cout << "Начальный размер словаря: " << vocab.size() << std::endl;
    
    // 4. Основной цикл слияний
    size_t initial_size = vocab.size();
    size_t merges_needed = target_size - initial_size;
    
    std::cout << "Выполнение " << merges_needed << " слияний..." << std::endl;
    
    for (size_t step = 0; step < merges_needed; ++step) {
        if (cancel_.load(std::memory_order_relaxed)) {
            std::cout << "Обучение прервано" << std::endl;
            return false;
        }
        
        // Подсчет частот пар
        auto pair_freq = count_pair_frequencies_parallel(chunks, vocab);
        
        if (pair_freq.empty()) {
            std::cout << "Нет больше пар для слияния на шаге " << step << std::endl;
            break;
        }
        
        // Поиск самой частой пары
        auto best_pair = std::max_element(
            pair_freq.begin(), pair_freq.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        merge_key_t merge_key = best_pair->first;
        uint32_t left_id = get_left_from_key(merge_key);
        uint32_t right_id = get_right_from_key(merge_key);
        
        std::string new_token = vocab.id_to_token(left_id) + vocab.id_to_token(right_id);
        
        // Применяем слияние
        auto merge_start = std::chrono::high_resolution_clock::now();
        
        apply_merge_parallel(chunks, merge_key, new_token, vocab);
        
        auto merge_end = std::chrono::high_resolution_clock::now();
        stats_.merge_time_sec += std::chrono::duration<double>(merge_end - merge_start).count();
        
        // Добавляем в словарь и мерджи
        vocab.add_token(new_token);
        merges[merge_key] = static_cast<int>(step);
        
        // Обновляем прогресс
        update_progress(step + 1, merges_needed);
        
        // Логирование каждые 100 шагов
        if ((step + 1) % 100 == 0 || step == 0) {
            std::cout << "\r   Прогресс: " << std::fixed << std::setprecision(1)
                      << (step + 1) * 100.0 / merges_needed << "%" 
                      << " (шаг " << (step + 1) << "/" << merges_needed << ")" 
                      << std::flush;
        }
    }
    
    std::cout << "\nОбучение завершено!" << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.total_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    stats_.total_merges = merges.size();
    
    std::cout << "   Итоговый размер словаря: " << vocab.size() << std::endl;
    std::cout << "   Время обучения: " << std::fixed << std::setprecision(2)
              << stats_.total_time_sec << " сек" << std::endl;
    
    return true;
}

// ==================== Разбиение корпуса ====================

std::vector<ParallelTrainer::CorpusChunk> ParallelTrainer::split_corpus(
    const std::vector<std::string>& corpus) {
    
    std::vector<CorpusChunk> chunks(num_threads_);
    size_t chunk_size = corpus.size() / num_threads_;
    
    for (size_t i = 0; i < num_threads_; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads_ - 1) ? corpus.size() : (i + 1) * chunk_size;
        
        chunks[i].start_idx = start;
        chunks[i].end_idx = end;
        chunks[i].texts.reserve(end - start);
        
        for (size_t j = start; j < end; ++j) {
            chunks[i].texts.push_back(corpus[j]);
        }
    }
    
    return chunks;
}

// ==================== Подсчет частот символов ====================

std::unordered_map<std::string, size_t> ParallelTrainer::count_char_frequencies_parallel(
    const std::vector<CorpusChunk>& chunks) {
    
    std::vector<std::unordered_map<std::string, size_t>> thread_freqs(num_threads_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < chunks.size(); ++i) {
        for (const auto& text : chunks[i].texts) {
            for (char c : text) {
                thread_freqs[i][std::string(1, c)]++;
            }
        }
    }
    
    // Объединяем результаты
    std::unordered_map<std::string, size_t> combined;
    for (const auto& tf : thread_freqs) {
        for (const auto& [ch, freq] : tf) {
            combined[ch] += freq;
        }
    }
    
    return combined;
}

// ==================== Подсчет частот пар ====================

std::unordered_map<merge_key_t, size_t> ParallelTrainer::count_pair_frequencies_parallel(
    const std::vector<CorpusChunk>& chunks,
    const Vocabulary& vocab) {
    
    std::vector<std::unordered_map<merge_key_t, size_t>> thread_freqs(num_threads_);
    
    #pragma omp parallel for
    for (size_t i = 0; i < chunks.size(); ++i) {
        for (const auto& text : chunks[i].texts) {
            if (text.length() < 2) continue;
            
            for (size_t j = 0; j < text.length() - 1; ++j) {
                std::string left = std::string(1, text[j]);
                std::string right = std::string(1, text[j + 1]);
                
                auto left_id = vocab.token_to_id(left);
                auto right_id = vocab.token_to_id(right);
                
                if (left_id != INVALID_TOKEN && right_id != INVALID_TOKEN) {
                    merge_key_t key = make_merge_key(left_id, right_id);
                    thread_freqs[i][key]++;
                }
            }
        }
    }
    
    // Объединяем результаты
    std::unordered_map<merge_key_t, size_t> combined;
    for (const auto& tf : thread_freqs) {
        for (const auto& [pair, freq] : tf) {
            combined[pair] += freq;
        }
    }
    
    return combined;
}

// ==================== Применение слияния ====================

void ParallelTrainer::apply_merge_parallel(
    std::vector<CorpusChunk>& chunks,
    merge_key_t merge_pair,
    const std::string& new_token,
    Vocabulary& vocab) {
    
    uint32_t left_id = get_left_from_key(merge_pair);
    uint32_t right_id = get_right_from_key(merge_pair);
    
    std::string left_str = vocab.id_to_token(left_id);
    std::string right_str = vocab.id_to_token(right_id);
    
    #pragma omp parallel for
    for (size_t i = 0; i < chunks.size(); ++i) {
        for (auto& text : chunks[i].texts) {
            std::string result;
            result.reserve(text.size());
            
            for (size_t j = 0; j < text.size(); ++j) {
                if (j < text.size() - 1 && 
                    std::string(1, text[j]) == left_str && 
                    std::string(1, text[j + 1]) == right_str) {
                    result += new_token;
                    ++j;  // Пропускаем правый символ
                } else {
                    result += text[j];
                }
            }
            
            text = std::move(result);
        }
    }
}

// ==================== Прогресс ====================

void ParallelTrainer::update_progress(size_t current, size_t total) {
    float progress = static_cast<float>(current) / total;
    progress_.store(progress, std::memory_order_relaxed);
}

} // namespace bpe