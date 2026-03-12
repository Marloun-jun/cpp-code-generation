/**
 * @file test_parallel.cpp
 * @brief Модульные тесты для класса ParallelTrainer
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Набор тестов для проверки функциональности параллельного обучения,
 *          который используется для ускорения обучения BPE токенизатора.
 * 
 *          **Проверяемые аспекты:**
 * 
 *          1) **Создание и инициализация**
 *             - Создание тренера с разным количеством потоков
 *             - Обработка некорректных параметров
 *             - Автоматическое определение числа потоков
 * 
 *          2) **Разбиение корпуса**
 *             - Равномерное распределение данных между потоками
 *             - Обработка пустого и маленького корпуса
 *             - Разные размеры чанков
 * 
 *          3) **Параллельный подсчет частот**
 *             - Корректность подсчета символов
 *             - Работа на пустом и большом корпусе
 *             - Суммарное совпадение с ожиданием
 * 
 *          4) **Управление обучением**
 *             - Отслеживание прогресса
 *             - Отмена обучения
 *             - Сброс статистики
 * 
 *          5) **Многопоточность**
 *             - Безопасность при одновременном доступе
 *             - Отсутствие гонок данных
 * 
 *          6) **Производительность**
 *             - Время создания тренера
 *             - Масштабирование с числом потоков
 * 
 * @note Тесты требуют наличия OpenMP для полноценной проверки
 * @see ParallelTrainer
 */

#include <gtest/gtest.h>

#include "parallel_trainer.hpp"
#include "test_helpers.hpp"

#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <iomanip>

using namespace bpe;

// ======================================================================
// Константы
// ======================================================================

namespace {
    constexpr int TEST_THREADS_2 = 2;
    constexpr int TEST_THREADS_4 = 4;
    constexpr int TEST_THREADS_8 = 8;
    constexpr int TEST_THREADS_16 = 16;
    constexpr int TEST_THREADS_MAX = 1000;
    
    constexpr size_t SMALL_CORPUS_SIZE = 10;
    constexpr size_t MEDIUM_CORPUS_SIZE = 100;
    constexpr size_t LARGE_CORPUS_SIZE = 10000;
    constexpr size_t DEFAULT_LINE_LENGTH = 50;
    constexpr size_t LONG_LINE_LENGTH = 100;
    
    constexpr int MULTITHREAD_TEST_THREADS = 4;
    constexpr int CANCEL_DELAY_MS = 100;
    constexpr int CANCEL_CHECK_DELAY_MS = 50;
    constexpr int PERFORMANCE_ITERATIONS = 10;
    constexpr int64_t MAX_CREATION_TIME_US = 1000000;    // 1 секунда
    constexpr double MAX_PERFORMANCE_DEGRADATION = 2.0;  // не более чем в 2 раза медленнее
    constexpr int MAX_MERGES_AFTER_CANCEL = 10;          // максимальное число слияний после отмены
}

// ======================================================================
// Вспомогательные функции
// ======================================================================

/**
 * @brief Создать тестовый корпус указанного размера
 */
std::vector<std::string> create_test_corpus(size_t num_lines, size_t line_length = DEFAULT_LINE_LENGTH) {
    std::vector<std::string> corpus;
    corpus.reserve(num_lines);
    
    for (size_t i = 0; i < num_lines; ++i) {
        std::string line;
        line.reserve(line_length);
        for (size_t j = 0; j < line_length; ++j) {
            line += static_cast<char>('a' + (i + j) % 26);
        }
        corpus.push_back(line);
    }
    
    return corpus;
}

/**
 * @brief Создать тестовый корпус с повторяющимися паттернами
 */
std::vector<std::string> create_pattern_corpus(size_t num_lines, 
                                               const std::vector<std::string>& patterns) {
    std::vector<std::string> corpus;
    corpus.reserve(num_lines);
    
    for (size_t i = 0; i < num_lines; ++i) {
        corpus.push_back(patterns[i % patterns.size()] + " // " + std::to_string(i));
    }
    
    return corpus;
}

/**
 * @brief Подсчет частот символов последовательно (для проверки)
 */
std::unordered_map<std::string, size_t> count_frequencies_sequential(
    const std::vector<std::string>& corpus) {
    
    std::unordered_map<std::string, size_t> freqs;
    
    for (const auto& line : corpus) {
        for (char c : line) {
            freqs[std::string(1, c)]++;
        }
    }
    
    return freqs;
}

// ======================================================================
// Тесты создания и инициализации
// ======================================================================

TEST(ParallelTrainerTest, Creation) {
    ParallelTrainer trainer1(TEST_THREADS_2);
    EXPECT_EQ(trainer1.num_threads(), TEST_THREADS_2);
    
    ParallelTrainer trainer2(0);
    EXPECT_GT(trainer2.num_threads(), 0);
    EXPECT_LE(trainer2.num_threads(), std::thread::hardware_concurrency());
    
    ParallelTrainer trainer3(-1);
    EXPECT_GT(trainer3.num_threads(), 0);
    EXPECT_LE(trainer3.num_threads(), std::thread::hardware_concurrency());
}

TEST(ParallelTrainerTest, InvalidCreation) {
    ParallelTrainer trainer1(TEST_THREADS_MAX);
    EXPECT_LE(trainer1.num_threads(), std::thread::hardware_concurrency());
    
    ParallelTrainer trainer2(0);
    EXPECT_GT(trainer2.num_threads(), 0);
}

// ======================================================================
// Тесты разбиения корпуса
// ======================================================================

TEST(ParallelTrainerTest, SplitCorpus) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    std::vector<std::string> empty_corpus;
    auto empty_chunks = trainer.split_corpus(empty_corpus, TEST_THREADS_2);
    EXPECT_EQ(empty_chunks.size(), 0);
    
    std::vector<std::string> small_corpus = {"line1"};
    auto small_chunks = trainer.split_corpus(small_corpus, TEST_THREADS_2);
    EXPECT_EQ(small_chunks.size(), 1);
    EXPECT_EQ(small_chunks[0].texts.size(), 1);
    
    std::vector<std::string> normal_corpus = {
        "line1", "line2", "line3", "line4", "line5",
        "line6", "line7", "line8", "line9", "line10"
    };
    
    auto normal_chunks = trainer.split_corpus(normal_corpus, TEST_THREADS_2);
    EXPECT_EQ(normal_chunks.size(), TEST_THREADS_2);
    EXPECT_EQ(normal_chunks[0].texts.size(), normal_corpus.size() / TEST_THREADS_2);
    EXPECT_EQ(normal_chunks[1].texts.size(), normal_corpus.size() / TEST_THREADS_2);
}

TEST(ParallelTrainerTest, SplitCorpusDifferentThreads) {
    std::vector<std::string> corpus;
    for (int i = 0; i < MEDIUM_CORPUS_SIZE; ++i) {
        corpus.push_back("line" + std::to_string(i));
    }
    
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    for (int num_threads : thread_counts) {
        if (num_threads > static_cast<int>(corpus.size())) continue;
        
        ParallelTrainer trainer(num_threads);
        auto chunks = trainer.split_corpus(corpus, num_threads);
        
        EXPECT_EQ(chunks.size(), static_cast<size_t>(num_threads));
        
        size_t total_lines = 0;
        for (const auto& chunk : chunks) {
            total_lines += chunk.texts.size();
        }
        EXPECT_EQ(total_lines, corpus.size());
    }
}

// ======================================================================
// Тесты подсчета частот
// ======================================================================

TEST(ParallelTrainerTest, CountCharFrequencies) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    std::vector<std::string> corpus = {
        "hello",
        "world",
        "test"
    };
    
    auto freqs = trainer.count_char_frequencies_parallel(corpus);
    auto expected = count_frequencies_sequential(corpus);
    
    EXPECT_EQ(freqs.size(), expected.size());
    
    for (const auto& [ch, count] : expected) {
        EXPECT_EQ(freqs[ch], count) << "Неверная частота для символа '" << ch << "'";
    }
}

TEST(ParallelTrainerTest, CountFrequenciesEmptyCorpus) {
    ParallelTrainer trainer(TEST_THREADS_2);
    std::vector<std::string> empty_corpus;
    
    auto freqs = trainer.count_char_frequencies_parallel(empty_corpus);
    
    EXPECT_EQ(freqs.size(), 0);
}

TEST(ParallelTrainerTest, CountFrequenciesLargeCorpus) {
    auto corpus = create_test_corpus(LARGE_CORPUS_SIZE, DEFAULT_LINE_LENGTH);
    
    ParallelTrainer sequential_trainer(1);
    ParallelTrainer parallel_trainer(TEST_THREADS_4);
    
    std::cout << "\nПодсчет частот на " << corpus.size() << " строках:" << std::endl;
    
    auto seq_start = std::chrono::high_resolution_clock::now();
    auto seq_freqs = sequential_trainer.count_char_frequencies_parallel(corpus);
    auto seq_end = std::chrono::high_resolution_clock::now();
    
    auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start);
    std::cout << "  Последовательно (1 поток): " << seq_time.count() << " мс" << std::endl;
    
    auto par_start = std::chrono::high_resolution_clock::now();
    auto par_freqs = parallel_trainer.count_char_frequencies_parallel(corpus);
    auto par_end = std::chrono::high_resolution_clock::now();
    
    auto par_time = std::chrono::duration_cast<std::chrono::milliseconds>(par_end - par_start);
    std::cout << "  Параллельно (4 потока): " << par_time.count() << " мс" << std::endl;
    
    if (par_time.count() > 0) {
        double speedup = static_cast<double>(seq_time.count()) / par_time.count();
        std::cout << "  Ускорение: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    EXPECT_GT(seq_freqs.size(), 0);
    EXPECT_GT(par_freqs.size(), 0);
    EXPECT_EQ(seq_freqs.size(), par_freqs.size());
    
    for (const auto& [ch, count] : seq_freqs) {
        auto it = par_freqs.find(ch);
        ASSERT_NE(it, par_freqs.end()) << "Символ '" << ch << "' отсутствует в параллельных результатах!";
        EXPECT_EQ(it->second, count) << "Несовпадение частоты для символа '" << ch << "'";
    }
    
    size_t total_chars = 0;
    for (const auto& line : corpus) {
        total_chars += line.size();
    }
    
    size_t seq_counted = 0;
    for (const auto& [_, count] : seq_freqs) {
        seq_counted += count;
    }
    
    size_t par_counted = 0;
    for (const auto& [_, count] : par_freqs) {
        par_counted += count;
    }
    
    EXPECT_EQ(total_chars, seq_counted);
    EXPECT_EQ(total_chars, par_counted);
    
    std::cout << "  Всего символов: " << total_chars << std::endl;
    std::cout << "  Уникальных символов: " << seq_freqs.size() << std::endl;
    
    if (par_time.count() > 0) {
        EXPECT_LE(par_time.count(), seq_time.count() * MAX_PERFORMANCE_DEGRADATION) 
            << "Параллельная версия слишком медленная!";
    }
}

// ======================================================================
// Тесты прогресса и отмены
// ======================================================================

TEST(ParallelTrainerTest, ProgressTracking) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    float progress = trainer.progress();
    EXPECT_GE(progress, 0.0f);
    EXPECT_LE(progress, 1.0f);
}

TEST(ParallelTrainerTest, CancelTraining) {
    ParallelTrainer trainer(TEST_THREADS_4);
    auto corpus = create_test_corpus(MEDIUM_CORPUS_SIZE, LONG_LINE_LENGTH);
    
    Vocabulary vocab;
    std::unordered_map<merge_key_t, int> merges;
    
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    
    std::thread training_thread([&]() {
        started = true;
        trainer.train(corpus, 1000, vocab, merges);
        finished = true;
    });
    
    while (!started) {
        std::this_thread::yield();
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(CANCEL_DELAY_MS));
    
    auto stats_before = trainer.stats();
    std::cout << "\nСлияний до отмены: " << stats_before.total_merges << std::endl;
    
    trainer.cancel();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(CANCEL_CHECK_DELAY_MS));
    
    auto stats_after = trainer.stats();
    std::cout << "Слияний после отмены: " << stats_after.total_merges << std::endl;
    
    EXPECT_LE(stats_after.total_merges, stats_before.total_merges + MAX_MERGES_AFTER_CANCEL) 
        << "Статистика продолжает расти после отмены!";
    
    training_thread.join();
    
    auto stats_final = trainer.stats();
    EXPECT_LT(stats_final.total_merges, 1000) << "Обучение должно было быть отменено раньше!";
    
    std::cout << "Финальное количество слияний: " << stats_final.total_merges << std::endl;
    
    EXPECT_TRUE(finished) << "Обучение не завершилось после отмены!";
}

TEST(ParallelTrainerTest, StatsReset) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    auto stats = trainer.stats();
    EXPECT_EQ(stats.total_merges, 0);
    EXPECT_EQ(stats.total_time_sec, 0.0);
    EXPECT_EQ(stats.freq_time_sec, 0.0);
    EXPECT_EQ(stats.merge_time_sec, 0.0);
    EXPECT_EQ(stats.peak_memory_bytes, 0);
    
    std::vector<std::string> corpus = {
        "ab", "ab", "ab", "ab", "ab", "ab", "ab", "ab", "ab", "ab"
    };
    
    Vocabulary vocab;
    std::unordered_map<merge_key_t, int> merges;
    
    vocab.add_token("a");
    vocab.add_token("b");
    vocab.add_token("<UNK>");
    vocab.add_token("<PAD>");
    vocab.add_token("<BOS>");
    vocab.add_token("<EOS>");
    
    std::cout << "\nНачальный размер словаря: " << vocab.size() << std::endl;
    
    size_t target_size = 7;
    std::cout << "Целевой размер: " << target_size << std::endl;
    
    trainer.train(corpus, target_size, vocab, merges);
    
    stats = trainer.stats();
    EXPECT_GT(stats.total_merges, 0) << "Должны быть выполнены слияния!";
    EXPECT_GT(stats.total_time_sec, 0.0) << "Время обучения должно быть > 0!";
    
    std::cout << "Выполнено слияний:          " << stats.total_merges << std::endl;
    std::cout << "Итоговый размер словаря:    " << vocab.size() << std::endl;
    
    trainer.reset_stats();
    
    stats = trainer.stats();
    EXPECT_EQ(stats.total_merges, 0);
    EXPECT_EQ(stats.total_time_sec, 0.0);
    EXPECT_EQ(stats.freq_time_sec, 0.0);
    EXPECT_EQ(stats.merge_time_sec, 0.0);
    EXPECT_EQ(stats.peak_memory_bytes, 0);
}

// ======================================================================
// Тесты многопоточности
// ======================================================================

TEST(ParallelTrainerTest, MultithreadedAccess) {
    ParallelTrainer trainer(TEST_THREADS_4);
    auto corpus = create_test_corpus(MEDIUM_CORPUS_SIZE, DEFAULT_LINE_LENGTH);
    
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    
    Vocabulary shared_vocab;
    std::unordered_map<merge_key_t, int> shared_merges;
    std::mutex mtx;
    
    for (int i = 0; i < MULTITHREAD_TEST_THREADS; ++i) {
        threads.emplace_back([&trainer, &corpus, &errors, &shared_vocab, &shared_merges, &mtx, i]() {
            try {
                if (i % 2 == 0) {
                    Vocabulary local_vocab;
                    std::unordered_map<merge_key_t, int> local_merges;
                    
                    size_t start = i * corpus.size() / MULTITHREAD_TEST_THREADS;
                    size_t end = (i + 1) * corpus.size() / MULTITHREAD_TEST_THREADS;
                    std::vector<std::string> sub_corpus(
                        corpus.begin() + start, 
                        corpus.begin() + end
                    );
                    
                    trainer.train(sub_corpus, 50, local_vocab, local_merges);
                    
                    std::lock_guard<std::mutex> lock(mtx);
                    auto tokens = local_vocab.get_all_tokens();
                    for (size_t j = 0; j < tokens.size(); ++j) {
                        shared_vocab.add_token(tokens[j]);
                    }
                    for (const auto& [key, rank] : local_merges) {
                        shared_merges[key] = rank;
                    }
                } else {
                    for (int j = 0; j < PERFORMANCE_ITERATIONS; ++j) {
                        trainer.progress();
                        trainer.stats();
                        trainer.cancel();  // Безопасно вызывать в любом потоке
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Ошибка в потоке " << i << ": " << e.what() << std::endl;
                errors++;
            } catch (...) {
                errors++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(errors, 0) << "Обнаружены ошибки при многопоточном доступе!";
    EXPECT_GT(shared_vocab.size(), 0) << "Словарь не был заполнен!";
    EXPECT_GT(shared_merges.size(), 0) << "Слияния не были созданы!";
}

// ======================================================================
// Тесты производительности
// ======================================================================

TEST(ParallelTrainerTest, CreationPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    
    ParallelTrainer trainer(TEST_THREADS_4);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\nВремя создания: " << duration.count() << " мкс" << std::endl;
    EXPECT_LT(duration.count(), MAX_CREATION_TIME_US);
}

TEST(ParallelTrainerTest, PerformanceComparison) {
    auto corpus = create_test_corpus(LARGE_CORPUS_SIZE, LONG_LINE_LENGTH);
    
    std::vector<int> thread_counts = {1, 2, 4, 8};
    std::vector<long long> times;
    
    std::cout << "\nСравнение производительности:" << std::endl;
    
    for (int num_threads : thread_counts) {
        if (num_threads > static_cast<int>(std::thread::hardware_concurrency())) continue;
        
        ParallelTrainer trainer(num_threads);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto freqs = trainer.count_char_frequencies_parallel(corpus);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        times.push_back(duration.count());
        
        std::cout << "  " << num_threads << " потоков: " << duration.count() << " мс" << std::endl;
        
        EXPECT_GT(freqs.size(), 0);
    }
    
    if (times.size() >= 2) {
        for (size_t i = 1; i < times.size(); ++i) {
            EXPECT_LE(times[i], times[0] * MAX_PERFORMANCE_DEGRADATION) 
                << thread_counts[i] << " потока должны быть не более чем в " 
                << MAX_PERFORMANCE_DEGRADATION << " раза медленнее 1 потока!";
        }
    }
}

TEST(ParallelTrainerTest, PatternFrequency) {
    ParallelTrainer trainer(TEST_THREADS_4);
    
    std::vector<std::string> patterns = {
        "int main()",
        "std::cout",
        "return 0;"
    };
    
    auto corpus = create_pattern_corpus(LARGE_CORPUS_SIZE, patterns);
    
    auto freqs = trainer.count_char_frequencies_parallel(corpus);
    
    EXPECT_GT(freqs.size(), 0);
    
    std::string all_patterns;
    for (const auto& p : patterns) {
        all_patterns += p;
    }
    
    for (char c : all_patterns) {
        if (c != ' ') {
            std::string key(1, c);
            EXPECT_GT(freqs[key], 0) << "Символ '" << c << "' должен присутствовать!";
        }
    }
}