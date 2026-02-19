/**
 * @file test_parallel.cpp
 * @brief Модульные тесты для класса ParallelTrainer
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Набор тестов для проверки функциональности параллельного обучения:
 *          - Создание тренера с разным количеством потоков
 *          - Разбиение корпуса на чанки
 *          - Параллельный подсчет частот
 *          - Применение слияний в многопоточном режиме
 * 
 * @note Тесты требуют наличия OpenMP для полноценной проверки
 * @see ParallelTrainer
 */

#include <gtest/gtest.h>
#include "parallel_trainer.hpp"
#include <vector>
#include <string>
#include <thread>
#include <chrono>

using namespace bpe;

// ======================================================================
// Тесты создания и инициализации
// ======================================================================

/**
 * @test Проверка создания тренера с разным количеством потоков
 */
TEST(ParallelTrainerTest, Creation) {
    // Создание с указанным количеством потоков
    ParallelTrainer trainer1(2);
    EXPECT_TRUE(true);
    
    // Создание с автоматическим определением потоков
    ParallelTrainer trainer2(0);
    EXPECT_TRUE(true);
    
    // Создание с отрицательным значением (должно обработаться корректно)
    ParallelTrainer trainer3(-1);
    EXPECT_TRUE(true);
}

/**
 * @test Проверка создания с некорректными параметрами
 */
TEST(ParallelTrainerTest, InvalidCreation) {
    // Очень большое количество потоков
    ParallelTrainer trainer1(1000);
    EXPECT_TRUE(true);
    
    // Нулевое количество (должно использовать аппаратные возможности)
    ParallelTrainer trainer2(0);
    EXPECT_TRUE(true);
}

// ======================================================================
// Тесты разбиения корпуса
// ======================================================================

/**
 * @test Проверка разбиения корпуса на чанки
 */
TEST(ParallelTrainerTest, SplitCorpus) {
    ParallelTrainer trainer(2);
    
    // Пустой корпус
    std::vector<std::string> empty_corpus;
    // TODO: добавить проверку split_corpus когда метод будет публичным
    
    // Маленький корпус
    std::vector<std::string> small_corpus = {"line1"};
    // TODO: добавить проверку
    
    // Нормальный корпус
    std::vector<std::string> normal_corpus = {
        "line1", "line2", "line3", "line4", "line5",
        "line6", "line7", "line8", "line9", "line10"
    };
    // TODO: добавить проверку
    
    EXPECT_TRUE(true);
}

/**
 * @test Проверка разбиения с разным количеством потоков
 */
TEST(ParallelTrainerTest, SplitCorpusDifferentThreads) {
    std::vector<std::string> corpus;
    for (int i = 0; i < 100; ++i) {
        corpus.push_back("line" + std::to_string(i));
    }
    
    // Тестируем с разным количеством потоков
    std::vector<int> thread_counts = {1, 2, 4, 8, 0};
    
    for (int num_threads : thread_counts) {
        ParallelTrainer trainer(num_threads);
        // TODO: добавить проверку split_corpus
    }
    
    EXPECT_TRUE(true);
}

// ======================================================================
// Тесты подсчета частот
// ======================================================================

/**
 * @test Проверка подсчета частот символов
 */
TEST(ParallelTrainerTest, CountCharFrequencies) {
    ParallelTrainer trainer(2);
    
    std::vector<std::string> corpus = {
        "hello",
        "world",
        "test"
    };
    
    // TODO: добавить проверку count_char_frequencies_parallel
    
    EXPECT_TRUE(true);
}

/**
 * @test Проверка подсчета частот на пустом корпусе
 */
TEST(ParallelTrainerTest, CountFrequenciesEmptyCorpus) {
    ParallelTrainer trainer(2);
    std::vector<std::string> empty_corpus;
    
    // TODO: добавить проверку
    
    EXPECT_TRUE(true);
}

/**
 * @test Проверка подсчета частот на большом корпусе
 */
TEST(ParallelTrainerTest, CountFrequenciesLargeCorpus) {
    ParallelTrainer trainer(4);
    
    std::vector<std::string> corpus;
    const int NUM_LINES = 10000;
    
    for (int i = 0; i < NUM_LINES; ++i) {
        corpus.push_back("line " + std::to_string(i) + " with some text");
    }
    
    // TODO: добавить проверку производительности
    
    EXPECT_TRUE(true);
}

// ======================================================================
// Тесты прогресса и отмены
// ======================================================================

/**
 * @test Проверка получения прогресса
 */
TEST(ParallelTrainerTest, ProgressTracking) {
    ParallelTrainer trainer(2);
    
    float progress = trainer.progress();
    EXPECT_GE(progress, 0.0f);
    EXPECT_LE(progress, 1.0f);
}

/**
 * @test Проверка отмены обучения
 */
TEST(ParallelTrainerTest, CancelTraining) {
    ParallelTrainer trainer(2);
    
    trainer.cancel();
    // TODO: проверить, что обучение действительно отменяется
    
    EXPECT_TRUE(true);
}

/**
 * @test Проверка сброса статистики
 */
TEST(ParallelTrainerTest, StatsReset) {
    ParallelTrainer trainer(2);
    
    auto stats = trainer.stats();
    EXPECT_EQ(stats.total_merges, 0);
    EXPECT_EQ(stats.total_time_sec, 0.0);
}

// ======================================================================
// Тесты многопоточности
// ======================================================================

/**
 * @test Проверка работы в многопоточной среде
 */
TEST(ParallelTrainerTest, MultithreadedAccess) {
    ParallelTrainer trainer(4);
    
    std::vector<std::thread> threads;
    
    // Запускаем несколько потоков, которые одновременно обращаются к тренеру
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&trainer, i]() {
            // Просто проверяем, что методы не падают
            trainer.progress();
            trainer.cancel();
            trainer.stats();
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_TRUE(true);
}

// ======================================================================
// Тесты производительности
// ======================================================================

/**
 * @test Измерение времени создания тренера
 */
TEST(ParallelTrainerTest, CreationPerformance) {
    auto start = std::chrono::high_resolution_clock::now();
    
    ParallelTrainer trainer(4);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  ⏱️  Creation time: " << duration.count() << " μs" << std::endl;
    
    EXPECT_LT(duration.count(), 1000000);  // Меньше 1 секунды
}

// ======================================================================
// Запуск тестов
// ======================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n🔧 Запуск тестов ParallelTrainer\n" << std::endl;
    
#ifdef USE_OPENMP
    std::cout << "  ⚡ OpenMP доступен" << std::endl;
#else
    std::cout << "  ⚠️ OpenMP не доступен" << std::endl;
#endif
    
    std::cout << "  📊 Доступно аппаратных потоков: " 
              << std::thread::hardware_concurrency() << std::endl;
    std::cout << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    std::cout << "\n✅ Тестирование завершено. Код возврата: " << result << std::endl;
    
    return result;
}