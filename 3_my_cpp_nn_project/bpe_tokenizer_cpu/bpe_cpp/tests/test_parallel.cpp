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
 *          ┌──────────────────────┬───────────────────────────────────┐
 *          │ Создание и           │ Конструкторы с разным числом      │
 *          │ инициализация        │ потоков, обработка ошибок         │
 *          ├──────────────────────┼───────────────────────────────────┤
 *          │ Разбиение корпуса    │ Равномерное распределение между   │
 *          │                      │ потоками, пустой и малый корпус   │
 *          ├──────────────────────┼───────────────────────────────────┤
 *          │ Подсчет частот       │ Корректность подсчета, сравнение  │
 *          │                      │ с последовательной версией        │
 *          ├──────────────────────┼───────────────────────────────────┤
 *          │ Управление обучением │ Отслеживание прогресса, отмена,   │
 *          │                      │ сброс статистики                  │
 *          ├──────────────────────┼───────────────────────────────────┤
 *          │ Многопоточность      │ Безопасность при одновременном    │
 *          │                      │ доступе, отсутствие гонок данных  │
 *          ├──────────────────────┼───────────────────────────────────┤
 *          │ Производительность   │ Масштабирование с числом потоков, │
 *          │                      │ время создания тренера            │
 *          └──────────────────────┴───────────────────────────────────┘
 * 
 * @note Тесты требуют наличия OpenMP для полноценной проверки
 * @see ParallelTrainer
 */

#include <gtest/gtest.h>

#include "parallel_trainer.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace bpe;

// ============================================================================
// Константы и настройки
// ============================================================================

namespace {
    // Количество потоков для тестов
    constexpr int TEST_THREADS_2 = 2;
    constexpr int TEST_THREADS_4 = 4;
    constexpr int TEST_THREADS_8 = 8;
    constexpr int TEST_THREADS_16 = 16;
    constexpr int TEST_THREADS_MAX = 1000;
    
    // Размеры корпуса
    constexpr size_t SMALL_CORPUS_SIZE = 10;
    constexpr size_t MEDIUM_CORPUS_SIZE = 100;
    constexpr size_t LARGE_CORPUS_SIZE = 10000;
    constexpr size_t DEFAULT_LINE_LENGTH = 50;
    constexpr size_t LONG_LINE_LENGTH = 100;
    
    // Параметры многопоточных тестов
    constexpr int MULTITHREAD_TEST_THREADS = 4;
    constexpr int CANCEL_DELAY_MS = 100;
    constexpr int CANCEL_CHECK_DELAY_MS = 50;
    constexpr int PERFORMANCE_ITERATIONS = 10;
    
    // Пороговые значения производительности
    constexpr int64_t MAX_CREATION_TIME_US = 1000000;      // 1 секунда
    constexpr double MAX_PERFORMANCE_DEGRADATION = 2.0;    // не более чем в 2 раза медленнее
    constexpr int MAX_MERGES_AFTER_CANCEL = 10;            // максимальное число слияний после отмени
    
    // Цвета для вывода (опционально)
    const std::string RESET = "\033[0m";
    const std::string GREEN = "\033[32m";
    const std::string CYAN = "\033[36m";
    const std::string YELLOW = "\033[33m";
    const std::string RED = "\033[31m";
    const std::string BOLD = "\033[1m";
}

// ============================================================================
// Вспомогательные функции
// ============================================================================

/**
 * @brief Создать тестовый корпус указанного размера
 * 
 * @param num_lines Количество строк
 * @param line_length Длина каждой строки
 * @return std::vector<std::string> Корпус для тестов
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
 * 
 * @param num_lines Количество строк
 * @param patterns Вектор паттернов для повторения
 * @return std::vector<std::string> Корпус с повторениями
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
 * 
 * @param corpus Корпус текстов
 * @return std::unordered_map<std::string, size_t> Частоты символов
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

// ============================================================================
// Тесты создания и инициализации
// ============================================================================

/**
 * @test Создание тренера с разными параметрами
 * 
 * Проверяет, что конструктор корректно обрабатывает:
 * - Явное указание числа потоков
 * - Автоматическое определение (0)
 * - Отрицательные значения
 */
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

/**
 * @test Обработка некорректных значений при создании
 * 
 * Проверяет, что слишком большие значения корректируются до разумных пределов.
 */
TEST(ParallelTrainerTest, InvalidCreation) {
    ParallelTrainer trainer1(TEST_THREADS_MAX);
    EXPECT_LE(trainer1.num_threads(), std::thread::hardware_concurrency());
    
    ParallelTrainer trainer2(0);
    EXPECT_GT(trainer2.num_threads(), 0);
}

// ============================================================================
// Тесты разбиения корпуса
// ============================================================================

/**
 * @test Разбиение корпуса на чанки
 * 
 * Проверяет, что split_corpus правильно распределяет данные:
 * - Пустой корпус
 * - Малый корпус (меньше числа потоков)
 * - Обычный корпус
 */
TEST(ParallelTrainerTest, SplitCorpus) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    // Пустой корпус
    std::vector<std::string> empty_corpus;
    auto empty_chunks = trainer.split_corpus(empty_corpus, TEST_THREADS_2);
    EXPECT_EQ(empty_chunks.size(), 0);
    
    // Малый корпус
    std::vector<std::string> small_corpus = {"line1"};
    auto small_chunks = trainer.split_corpus(small_corpus, TEST_THREADS_2);
    EXPECT_EQ(small_chunks.size(), 1);
    EXPECT_EQ(small_chunks[0].texts.size(), 1);
    
    // Обычный корпус (10 строк)
    std::vector<std::string> normal_corpus = {
        "line1", "line2", "line3", "line4", "line5",
        "line6", "line7", "line8", "line9", "line10"
    };
    
    auto normal_chunks = trainer.split_corpus(normal_corpus, TEST_THREADS_2);
    EXPECT_EQ(normal_chunks.size(), TEST_THREADS_2);
    EXPECT_EQ(normal_chunks[0].texts.size(), normal_corpus.size() / TEST_THREADS_2);
    EXPECT_EQ(normal_chunks[1].texts.size(), normal_corpus.size() / TEST_THREADS_2);
}

/**
 * @test Разбиение с разным числом потоков
 * 
 * Проверяет, что суммарное количество строк во всех чанках
 * равно исходному размеру корпуса для любого числа потоков.
 */
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

// ============================================================================
// Тесты подсчета частот
// ============================================================================

/**
 * @test Корректность подсчета частот на малом корпусе
 * 
 * Сравнивает результаты параллельного подсчета с последовательным.
 */
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
        EXPECT_EQ(freqs[ch], count) << "Неверная частота для символа '" << ch << "'!";
    }
}

/**
 * @test Подсчет частот на пустом корпусе
 * 
 * Проверяет, что пустой корпус даёт пустой результат.
 */
TEST(ParallelTrainerTest, CountFrequenciesEmptyCorpus) {
    ParallelTrainer trainer(TEST_THREADS_2);
    std::vector<std::string> empty_corpus;
    
    auto freqs = trainer.count_char_frequencies_parallel(empty_corpus);
    
    EXPECT_EQ(freqs.size(), 0);
}

/**
 * @test Подсчет частот на большом корпусе с сравнением производительности
 * 
 * Сравнивает:
 * - Результаты параллельного и последовательного подсчета
 * - Производительность 1 потока vs 4 потока
 */
TEST(ParallelTrainerTest, CountFrequenciesLargeCorpus) {
    auto corpus = create_test_corpus(LARGE_CORPUS_SIZE, DEFAULT_LINE_LENGTH);
    
    ParallelTrainer sequential_trainer(1);
    ParallelTrainer parallel_trainer(TEST_THREADS_4);
    
    std::cout << "\n" << CYAN << "Подсчет частот на " << corpus.size() << " строках:" << RESET << std::endl;
    
    // Последовательный подсчет (1 поток)
    auto seq_start = std::chrono::high_resolution_clock::now();
    auto seq_freqs = sequential_trainer.count_char_frequencies_parallel(corpus);
    auto seq_end = std::chrono::high_resolution_clock::now();
    
    auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(seq_end - seq_start);
    std::cout << "- Последовательно (1 поток): " << seq_time.count() << " мс" << std::endl;
    
    // Параллельный подсчет (4 потока)
    auto par_start = std::chrono::high_resolution_clock::now();
    auto par_freqs = parallel_trainer.count_char_frequencies_parallel(corpus);
    auto par_end = std::chrono::high_resolution_clock::now();
    
    auto par_time = std::chrono::duration_cast<std::chrono::milliseconds>(par_end - par_start);
    std::cout << "- Параллельно (4 потока):    " << par_time.count() << " мс" << std::endl;
    
    if (par_time.count() > 0) {
        double speedup = static_cast<double>(seq_time.count()) / par_time.count();
        std::cout << "- Ускорение:                 " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Проверка корректности
    EXPECT_GT(seq_freqs.size(), 0);
    EXPECT_GT(par_freqs.size(), 0);
    EXPECT_EQ(seq_freqs.size(), par_freqs.size());
    
    // Проверка, что все символы совпадают
    for (const auto& [ch, count] : seq_freqs) {
        auto it = par_freqs.find(ch);
        ASSERT_NE(it, par_freqs.end()) << "Символ '" << ch << "' отсутствует в параллельных результатах!";
        EXPECT_EQ(it->second, count) << "Несовпадение частоты для символа '" << ch << "'!";
    }
    
    // Проверка суммарного количества символов
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
    
    std::cout << "Всего символов:      " << total_chars << std::endl;
    std::cout << "Уникальных символов: " << seq_freqs.size() << std::endl;
    
    // Проверка производительности (параллельная версия не должна быть медленнее последовательной)
    if (par_time.count() > 0) {
        EXPECT_LE(par_time.count(), seq_time.count() * MAX_PERFORMANCE_DEGRADATION) 
            << "Параллельная версия слишком медленная!";
    }
}

// ============================================================================
// Тесты прогресса и отмены
// ============================================================================

/**
 * @test Отслеживание прогресса
 * 
 * Проверяет, что метод progress() возвращает значения в диапазоне [0, 1].
 */
TEST(ParallelTrainerTest, ProgressTracking) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    float progress = trainer.progress();
    EXPECT_GE(progress, 0.0f);
    EXPECT_LE(progress, 1.0f);
}

/**
 * @test Отмена обучения
 * 
 * Проверяет, что вызов cancel() действительно прерывает обучение
 * и статистика перестаёт расти.
 */
TEST(ParallelTrainerTest, CancelTraining) {
    ParallelTrainer trainer(TEST_THREADS_4);
    auto corpus = create_test_corpus(MEDIUM_CORPUS_SIZE, LONG_LINE_LENGTH);
    
    Vocabulary vocab;
    std::unordered_map<merge_key_t, int> merges;
    
    std::atomic<bool> started{false};
    std::atomic<bool> finished{false};
    
    // Запускаем обучение в отдельном потоке
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
    std::cout << "\n" << YELLOW << "Слияний до отмены: " << stats_before.total_merges << RESET << std::endl;
    
    // Отменяем обучение
    trainer.cancel();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(CANCEL_CHECK_DELAY_MS));
    
    auto stats_after = trainer.stats();
    std::cout << YELLOW << "Слияний после отмены: " << stats_after.total_merges << RESET << std::endl;
    
    // Проверяем, что статистика почти не изменилась после отмены
    EXPECT_LE(stats_after.total_merges, stats_before.total_merges + MAX_MERGES_AFTER_CANCEL) 
        << "Статистика продолжает расти после отмены!";
    
    training_thread.join();
    
    auto stats_final = trainer.stats();
    EXPECT_LT(stats_final.total_merges, 1000) << "Обучение должно было быть отменено раньше!";
    
    std::cout << YELLOW << "Финальное количество слияний: " << stats_final.total_merges << RESET << std::endl;
    
    EXPECT_TRUE(finished) << "Обучение не завершилось после отмены!";
}

/**
 * @test Сброс статистики
 * 
 * Проверяет, что reset_stats() обнуляет все метрики.
 */
TEST(ParallelTrainerTest, StatsReset) {
    ParallelTrainer trainer(TEST_THREADS_2);
    
    // Проверка начального состояния
    auto stats = trainer.stats();
    EXPECT_EQ(stats.total_merges, 0);
    EXPECT_EQ(stats.total_time_sec, 0.0);
    EXPECT_EQ(stats.freq_time_sec, 0.0);
    EXPECT_EQ(stats.merge_time_sec, 0.0);
    EXPECT_EQ(stats.peak_memory_bytes, 0);
    
    // Создаем корпус с повторяющимися паттернами для гарантированного создания слияний
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
    
    std::cout << "\n" << CYAN << "Начальный размер словаря: " << vocab.size() << RESET << std::endl;
    
    size_t target_size = 7;
    std::cout << CYAN << "Целевой размер:           " << target_size << RESET << std::endl;
    
    // Запускаем обучение
    trainer.train(corpus, target_size, vocab, merges);
    
    // Проверяем, что статистика изменилась
    stats = trainer.stats();
    EXPECT_GT(stats.total_merges, 0) << "Должны быть выполнены слияния!";
    EXPECT_GT(stats.total_time_sec, 0.0) << "Время обучения должно быть > 0!";
    
    std::cout << GREEN << "Выполнено слияний:       " << stats.total_merges << RESET << std::endl;
    std::cout << GREEN << "Итоговый размер словаря: " << vocab.size() << RESET << std::endl;
    
    // Сбрасываем статистику
    trainer.reset_stats();
    
    // Проверяем, что статистика обнулилась
    stats = trainer.stats();
    EXPECT_EQ(stats.total_merges, 0);
    EXPECT_EQ(stats.total_time_sec, 0.0);
    EXPECT_EQ(stats.freq_time_sec, 0.0);
    EXPECT_EQ(stats.merge_time_sec, 0.0);
    EXPECT_EQ(stats.peak_memory_bytes, 0);
}

// ============================================================================
// Тесты многопоточности
// ============================================================================

/**
 * @test Многопоточный доступ к тренеру
 * 
 * Запускает несколько потоков, которые одновременно:
 * - Обучают модели на разных частях корпуса
 * - Читают прогресс и статистику
 * - Отменяют обучение
 * 
 * Проверяет отсутствие гонок данных и корректность результатов.
 */
TEST(ParallelTrainerTest, MultithreadedAccess) {
    ParallelTrainer trainer(TEST_THREADS_4);
    
    // Создаем корпус с повторяющимися паттернами для гарантированного создания слияний
    std::vector<std::string> patterns = {
        "ab", "ab", "ab", "ab", "ab",     // Повторяющиеся "ab"
        "abc", "abc", "abc",              // Повторяющиеся "abc"
        "abcd", "abcd", "abcd", "abcd"    // Повторяющиеся "abcd"
    };
    auto corpus = create_pattern_corpus(MEDIUM_CORPUS_SIZE, patterns);
    
    std::cout << "\n" << CYAN << "Создан корпус из " << corpus.size() 
              << " строк с повторяющимися паттернами" << RESET << std::endl;
    
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};
    
    Vocabulary shared_vocab;
    std::unordered_map<merge_key_t, int> shared_merges;
    std::mutex mtx;
    
    // Инициализируем общий словарь базовыми символами
    {
        for (const auto& text : corpus) {
            for (char c : text) {
                std::string token(1, c);
                if (!shared_vocab.contains(token)) {
                    shared_vocab.add_token(token);
                }
            }
        }
        std::cout << GREEN << "Инициализирован общий словарь: " << shared_vocab.size() 
                  << " символов" << RESET << std::endl;
    }
    
    for (int i = 0; i < MULTITHREAD_TEST_THREADS; ++i) {
        threads.emplace_back([&trainer, &corpus, &errors, &shared_vocab, &shared_merges, &mtx, i]() {
            try {
                if (i % 2 == 0) {
                    // Для четных потоков - обучение
                    Vocabulary local_vocab;
                    
                    // Копируем токены из общего словаря в локальный
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        auto tokens = shared_vocab.get_all_tokens();
                        for (const auto& token : tokens) {
                            local_vocab.add_token(token);
                        }
                    }
                    
                    std::unordered_map<merge_key_t, int> local_merges;
                    
                    // Выделяем подкорпус для этого потока
                    size_t start = i * corpus.size() / MULTITHREAD_TEST_THREADS;
                    size_t end = (i + 1) * corpus.size() / MULTITHREAD_TEST_THREADS;
                    std::vector<std::string> sub_corpus(
                        corpus.begin() + start, 
                        corpus.begin() + end
                    );
                    
                    std::cout << "Поток " << i << ": обучение на " << sub_corpus.size() << " строках" << std::endl;
                    std::cout << "Начальный размер словаря: " << local_vocab.size() << std::endl;
                    
                    size_t target_size = local_vocab.size() + 5;
                    std::cout << "Целевой размер:           " << target_size << std::endl;
                    
                    // Запускаем обучение
                    trainer.train(sub_corpus, target_size, local_vocab, local_merges);
                    
                    // Добавляем результаты в общие контейнеры
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        
                        auto tokens = local_vocab.get_all_tokens();
                        std::cout << "Локальный словарь содержит " << tokens.size() << " токенов" << std::endl;
                        
                        for (const auto& token : tokens) {
                            if (!shared_vocab.contains(token)) {
                                shared_vocab.add_token(token);
                                std::cout << "Добавлен новый токен: '" << token << "'" << std::endl;
                            }
                        }
                        
                        std::cout << "Локальных слияний: " << local_merges.size() << std::endl;
                        for (const auto& [key, rank] : local_merges) {
                            if (shared_merges.find(key) == shared_merges.end()) {
                                shared_merges[key] = rank;
                                std::cout << "Добавлено новое слияние: key=" << key << std::endl;
                            }
                        }
                    }
                } else {
                    // Нечетные потоки - читают и отменяют с задержкой
                    for (int j = 0; j < PERFORMANCE_ITERATIONS; ++j) {
                        trainer.progress();
                        trainer.stats();
                        
                        // Отменяем только после небольшой задержки
                        if (j == PERFORMANCE_ITERATIONS / 2) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                            trainer.cancel();
                            std::cout << "Поток " << i << ": вызван cancel()" << std::endl;
                        }
                        
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << RED << "Ошибка в потоке " << i << ": " << e.what() << RESET << std::endl;
                errors++;
            } catch (...) {
                errors++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "\n" << CYAN << BOLD << "РЕЗУЛЬТАТЫ:" << RESET << std::endl;
    std::cout << "Ошибок:                   " << (errors == 0 ? GREEN : RED) << errors << RESET << std::endl;
    std::cout << "Общий размер словаря:     " << GREEN << shared_vocab.size() << RESET << std::endl;
    std::cout << "Общее количество слияний: " << GREEN << shared_merges.size() << RESET << std::endl;
    
    EXPECT_EQ(errors, 0) << "Обнаружены ошибки при многопоточном доступе!";
    EXPECT_GT(shared_vocab.size(), 0) << "Словарь не был заполнен!";
    EXPECT_GT(shared_merges.size(), 0) << "Слияния не были созданы!";
}

// ============================================================================
// Тесты производительности
// ============================================================================

/**
 * @test Время создания тренера
 * 
 * Проверяет, что создание объекта ParallelTrainer не занимает слишком много времени.
 */
TEST(ParallelTrainerTest, CreationPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    
    ParallelTrainer trainer(TEST_THREADS_4);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "\n" << CYAN << "Время создания: " << duration.count() << " мкс" << RESET << std::endl;
    EXPECT_LT(duration.count(), MAX_CREATION_TIME_US);
}

/**
 * @test Сравнение производительности с разным числом потоков
 * 
 * Измеряет время подсчета частот для 1, 2, 4 и 8 потоков
 * и проверяет, что параллельные версии не медленнее последовательной.
 */
TEST(ParallelTrainerTest, PerformanceComparison) {
    auto corpus = create_test_corpus(LARGE_CORPUS_SIZE, LONG_LINE_LENGTH);
    
    std::vector<int> thread_counts = {1, 2, 4, 8};
    std::vector<long long> times;
    
    std::cout << "\n" << CYAN << "Сравнение производительности:" << RESET << std::endl;
    
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
    
    // Проверка, что многопоточные версии не медленнее последовательной
    if (times.size() >= 2) {
        for (size_t i = 1; i < times.size(); ++i) {
            EXPECT_LE(times[i], times[0] * MAX_PERFORMANCE_DEGRADATION) 
                << thread_counts[i] << " потока должны быть не более чем в " 
                << MAX_PERFORMANCE_DEGRADATION << " раза медленнее 1 потока!";
        }
    }
}

/**
 * @test Подсчет частот на корпусе с повторяющимися паттернами
 * 
 * Проверяет, что все ожидаемые символы присутствуют в результатах.
 */
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
    
    // Проверяем, что все символы из паттернов присутствуют
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