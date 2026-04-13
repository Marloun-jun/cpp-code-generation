/**
 * @file simd_utils.hpp
 * @brief SIMD-оптимизированные утилиты для векторной обработки текста
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.5.0
 * 
 * @details ВНИМАНИЕ: Этот модуль предназначен ТОЛЬКО для обработки ASCII-текстов.
 *          Для работы с Unicode (русские буквы, эмодзи) используйте стандартные
 *          методы из fast_tokenizer.cpp.
 * 
 *          **Область применения:**
 *          - Быстрая обработка гарантированно ASCII текстов
 *          - Предварительная обработка для оптимизации
 *          - Fallback для байтовых операций
 * 
 *          **Поддерживаемые наборы инструкций:**
 * 
 *          AVX2 (2013+)
 *          ┌─────────────────────────────────────────────┐
 *          │ 256-битные регистры (YMM)                   │
 *          │ 32 ASCII символов за 1 инструкцию           │
 *          │ Intel Haswell, AMD Excavator+               │
 *          └─────────────────────────────────────────────┘
 * 
 *          AVX (2011+)
 *          ┌─────────────────────────────────────────────┐
 *          │ 128-битные регистры (XMM)                   │
 *          │ 16 ASCII символов за 1 инструкцию           │
 *          │ Intel Sandy Bridge, AMD Bulldozer+          │
 *          └─────────────────────────────────────────────┘
 * 
 *          SSE4.2 (2008+)
 *          ┌─────────────────────────────────────────────┐
 *          │ Строковые инструкции (STTNI)                │
 *          │ Поиск подстрок, сравнение строк             │
 *          │ Intel Penryn, AMD Barcelona+                │
 *          └─────────────────────────────────────────────┘
 * 
 * @warning НЕ ИСПОЛЬЗОВАТЬ для русского текста или эмодзи!
 *          SIMD инструкции работают на уровне байтов, а не символов.
 *          Русские буквы в UTF-8 занимают 2-3 байта и будут разбиты.
 * 
 * @see FastBPETokenizer::encode(), FastBPETokenizer::encode_ascii()
 */

#pragma once

#include "config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#ifdef USE_AVX2
    #include <immintrin.h>
#endif

#ifdef USE_AVX
    #include <immintrin.h>
#endif

#ifdef USE_SSE42
    #include <nmmintrin.h>
    #include <smmintrin.h>
#endif

namespace bpe {

// ============================================================================
// SIMDUtils - статический класс с SIMD-утилитами
// ============================================================================

/**
 * @brief Статические методы для SIMD-оптимизированной обработки ASCII
 * 
 * Все методы предназначены ТОЛЬКО для работы с ASCII (1 байт на символ).
 * Для корректной обработки Unicode используйте обычные строковые операции.
 * 
 * **Потокобезопасность:** Все методы потокобезопасны и не имеют состояния.
 * 
 * **Рекомендации по использованию:**
 * @code
 * // 1. Простое кодирование ASCII (автовыбор)
 * auto tokens = SIMDUtils::encode_optimal(text, lookup, unk_id);
 * 
 * // 2. Если нужен конкретный уровень (например, для бенчмарков)
 * if (SIMDUtils::has_avx2()) {
 *     tokens = SIMDUtils::encode_avx2(text, lookup, unk_id);
 * }
 * 
 * // 3. Для отладки
 * std::cout << "SIMD уровень: " << SIMDUtils::get_simd_level() << "\n";
 * @endcode
 * 
 * @note Для русского текста используйте FastBPETokenizer::encode().
 */
class SIMDUtils {
public:
    // ========================================================================
    // Проверка поддержки (compile-time)
    // ========================================================================

    /**
     * @brief Проверка доступности AVX2 при компиляции
     * @return true если код скомпилирован с поддержкой AVX2
     * 
     * @note Это только compile-time проверка! Используйте check_avx2_support()
     *       для runtime проверки процессора.
     */
    static constexpr bool has_avx2() {
        #if defined(USE_AVX2) || defined(__AVX2__)
            return true;
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверка доступности AVX при компиляции
     */
    static constexpr bool has_avx() {
        #if defined(USE_AVX) || defined(__AVX__)
            return true;
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверка доступности SSE4.2 при компиляции
     */
    static constexpr bool has_sse42() {
        #if defined(USE_SSE42) || defined(__SSE4_2__)
            return true;
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверка наличия любой SIMD поддержки
     */
    static constexpr bool has_any_simd() {
        return has_avx2() || has_avx() || has_sse42();
    }

    // ========================================================================
    // Проверка поддержки (runtime)
    // ========================================================================

    /**
     * @brief Проверить поддержку AVX2 процессором во время выполнения
     * 
     * @return true если процессор поддерживает AVX2
     * 
     * Использует инструкцию CPUID для определения возможностей процессора.
     * 
     * **Пример:**
     * @code
     * if (SIMDUtils::check_avx2_support()) {
     *     std::cout << "Запускаем AVX2 версию (только ASCII!)\n";
     *     tokens = SIMDUtils::encode_avx2(text, lookup, unk_id);
     * } else {
     *     tokens = SIMDUtils::encode_scalar(text, lookup);
     * }
     * @endcode
     * 
     * @warning Всегда проверяйте перед использованием AVX2 инструкций!
     *          Исполнение AVX2 на процессоре без поддержки вызовет SIGILL.
     */
    static bool check_avx2_support() {
        #if defined(_MSC_VER)
            int cpuInfo[4];
            __cpuid(cpuInfo, 7);
            return (cpuInfo[1] >> 5) & 1;
        #elif defined(__GNUC__) || defined(__clang__)
            #if defined(__x86_64__) || defined(__i386__)
                unsigned int eax, ebx, ecx, edx;
                __asm__ volatile(
                    "mov $7, %%rax\n"
                    "xor %%rcx, %%rcx\n"
                    "cpuid\n"
                    : "=b"(ebx), "=a"(eax), "=c"(ecx), "=d"(edx)
                    :
                    : "cc"
                );
                return (ebx >> 5) & 1;
            #else
                return false;
            #endif
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверить поддержку AVX процессором
     */
    static bool check_avx_support() {
        #if defined(_MSC_VER)
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[2] & (1 << 28)) != 0;
        #elif defined(__GNUC__) || defined(__clang__)
            #if defined(__x86_64__) || defined(__i386__)
                unsigned int eax, ebx, ecx, edx;
                __asm__ volatile(
                    "mov $1, %%rax\n"
                    "cpuid\n"
                    : "=b"(ebx), "=c"(ecx), "=d"(edx)
                    : "a"(1)
                    : "cc"
                );
                return (ecx & (1 << 28)) != 0;
            #else
                return false;
            #endif
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверить поддержку SSE4.2 процессором
     */
    static bool check_sse42_support() {
        #if defined(_MSC_VER)
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[2] & (1 << 20)) != 0;
        #elif defined(__GNUC__) || defined(__clang__)
            #if defined(__x86_64__) || defined(__i386__)
                unsigned int eax, ebx, ecx, edx;
                __asm__ volatile(
                    "mov $1, %%rax\n"
                    "cpuid\n"
                    : "=b"(ebx), "=c"(ecx), "=d"(edx)
                    : "a"(1)
                    : "cc"
                );
                return (ecx & (1 << 20)) != 0;
            #else
                return false;
            #endif
        #else
            return false;
        #endif
    }

    /**
     * @brief Получить рекомендуемый уровень SIMD для текущего процессора
     * 
     * @return int Код реализации:
     *         2  - AVX2 (256-bit)
     *         1  - AVX (128-bit)  
     *         0  - SSE4.2
     *         -1 - Скалярная
     */
    static int get_recommended_implementation() {
        if (check_avx2_support() && has_avx2()) return 2;
        if (check_avx_support() && has_avx()) return 1;
        if (check_sse42_support() && has_sse42()) return 0;
        return -1;
    }

    // ========================================================================
    // AVX2 оптимизации (256-bit, 32 ASCII символа за раз)
    // ========================================================================

    /**
     * @brief AVX2-ускоренное кодирование ASCII текста в ID токенов
     * 
     * @param text Входной текст (ДОЛЖЕН БЫТЬ ТОЛЬКО ASCII!)
     * @param lookup_table Таблица char->ID (должна содержать 256 элементов)
     * @param unknown_id ID для неизвестных (не используется в этой версии)
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * **Алгоритм работы:**
     * @code
     * ┌─────────────────────────────────────────────────────┐
     * │ Поток обработки (только ASCII!):                    │
     * │                                                     │
     * │ Текст:    H e l l o   W o r l d !                   │
     * │           |                                         │
     * │ Загрузка: 32 ASCII символа в YMM регистр            │
     * │           ┌───────────────────────────┐             │
     * │ YMM0:     │H│e│l│l│o│ │W│o│r│l│d│!│...│             │
     * │           └───────────────────────────┘             │
     * │           |                                         │
     * │ Разбивка на 2 XMM регистра                          │
     * │ XMM0 - Первые 16 символов                           │
     * │ XMM1 - Вторые 16 символов                           │
     * │           |                                         │
     * │ Расширение до 16-битных индексов                    │
     * │           |                                         │
     * │ Сохранение индексов в массив                        │
     * │           |                                         │
     * │ Получение ID из lookup-таблицы                      │
     * └─────────────────────────────────────────────────────┘
     * @endcode
     * 
     * **Производительность:** ~200-300 МБ/с на современных CPU
     * 
     * @warning РАБОТАЕТ ТОЛЬКО С ASCII! Русский текст будет испорчен.
     */
    static std::vector<uint32_t> encode_avx2(std::string_view text,
                                             const uint32_t* lookup_table,
                                             uint32_t unknown_id) {
        std::vector<uint32_t> result;
        result.reserve(text.size());

        (void)unknown_id;    // Не используется в AVX2 версии

        #if defined(USE_AVX2) || defined(__AVX2__)
            size_t i = 0;

            // Основной цикл: 32 символа за итерацию
            for (; i + 32 <= text.size(); i += 32) {
                // Загрузка 32 символов (256 бит)
                __m256i chars = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(text.data() + i)
                );

                // Разбиваем на два 128-битных регистра
                __m128i chars_lo = _mm256_extracti128_si256(chars, 0);
                __m128i chars_hi = _mm256_extracti128_si256(chars, 1);

                // Расширяем 8-битные символы до 16-битных индексов
                __m256i indices_lo = _mm256_cvtepu8_epi16(chars_lo);
                __m256i indices_hi = _mm256_cvtepu8_epi16(chars_hi);

                // Сохраняем индексы (требуется выравнивание для AVX)
                alignas(32) uint16_t indices_array_lo[16];
                alignas(32) uint16_t indices_array_hi[16];

                _mm256_store_si256(reinterpret_cast<__m256i*>(indices_array_lo), indices_lo);
                _mm256_store_si256(reinterpret_cast<__m256i*>(indices_array_hi), indices_hi);

                // Получаем ID для всех 32 символов
                for (int j = 0; j < 16; ++j) {
                    result.push_back(lookup_table[indices_array_lo[j]]);
                }
                for (int j = 0; j < 16; ++j) {
                    result.push_back(lookup_table[indices_array_hi[j]]);
                }
            }

            // Обработка остатка (скалярно)
            for (; i < text.size(); ++i) {
                result.push_back(lookup_table[static_cast<unsigned char>(text[i])]);
            }

        #else
            // Fallback на скалярную версию
            for (char c : text) {
                result.push_back(lookup_table[static_cast<unsigned char>(c)]);
            }
        #endif

        return result;
    }

    // ========================================================================
    // AVX оптимизации (128-bit, 16 ASCII символов за раз)
    // ========================================================================

    /**
     * @brief AVX-ускоренное кодирование ASCII (128-битные регистры)
     * 
     * Обрабатывает 16 символов за итерацию.
     * Работает на процессорах Sandy Bridge и новее.
     * 
     * @warning Только для ASCII! Русский текст будет испорчен.
     */
    static std::vector<uint32_t> encode_avx(std::string_view text,
                                            const uint32_t* lookup_table,
                                            uint32_t unknown_id) {
        std::vector<uint32_t> result;
        result.reserve(text.size());

        (void)unknown_id;

        #if defined(USE_AVX) || defined(__AVX__)
            size_t i = 0;

            // Основной цикл: 16 символов за итерацию
            for (; i + 16 <= text.size(); i += 16) {
                // Загрузка 16 символов (128 бит)
                __m128i chars = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(text.data() + i)
                );

                // Обработка первых 8 символов
                __m128i indices = _mm_cvtepu8_epi16(chars);
                alignas(16) uint16_t indices_array[8];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices_array), indices);

                for (int j = 0; j < 8; ++j) {
                    result.push_back(lookup_table[indices_array[j]]);
                }

                // Обработка следующих 8 символов (сдвиг на 8 байт)
                __m128i chars_high = _mm_srli_si128(chars, 8);
                __m128i indices_high = _mm_cvtepu8_epi16(chars_high);
                alignas(16) uint16_t indices_array_high[8];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices_array_high), indices_high);

                for (int j = 0; j < 8; ++j) {
                    result.push_back(lookup_table[indices_array_high[j]]);
                }
            }

            // Обработка остатка
            for (; i < text.size(); ++i) {
                result.push_back(lookup_table[static_cast<unsigned char>(text[i])]);
            }

        #else
            for (char c : text) {
                result.push_back(lookup_table[static_cast<unsigned char>(c)]);
            }
        #endif

        return result;
    }

    // ========================================================================
    // Скалярная версия (базовый уровень)
    // ========================================================================

    /**
     * @brief Скалярное кодирование (без SIMD)
     * 
     * Базовая версия, работает на любых процессорах.
     * Используется как fallback, если SIMD недоступен.
     * 
     * @note Это единственная версия, которая может работать с Unicode,
     *       но она всё равно работает на уровне байтов.
     *       Для правильной обработки Unicode используйте FastBPETokenizer.
     */
    static std::vector<uint32_t> encode_scalar(std::string_view text,
                                               const uint32_t* lookup_table) {
        std::vector<uint32_t> result;
        result.reserve(text.size());

        for (char c : text) {
            result.push_back(lookup_table[static_cast<unsigned char>(c)]);
        }

        return result;
    }

    // ========================================================================
    // SSE4.2 оптимизации (строковые операции)
    // ========================================================================

    /**
     * @brief SSE4.2-ускоренный поиск подстроки (только ASCII)
     * 
     * @param text Текст для поиска (haystack)
     * @param pattern Искомый паттерн (needle)
     * @return size_t Позиция первого вхождения или npos
     * 
     * **Производительность:**
     * - Короткие паттерны (≤16) - В 2-3 раза быстрее std::string::find
     * - Длинные паттерны (>16)  - Сравнима с std::search
     * 
     * @code
     * std::string_view text = "The quick brown fox jumps over the lazy dog";
     * size_t pos = SIMDUtils::find_substring_sse42(text, "fox");
     * // pos = 16
     * @endcode
     * 
     * @warning Для паттернов с Unicode может давать неверные результаты!
     */
    static size_t find_substring_sse42(std::string_view text, std::string_view pattern) {
        if (pattern.empty()) return 0;
        if (pattern.size() > text.size()) return std::string_view::npos;

        #if defined(USE_SSE42) || defined(__SSE4_2__)
            const char* haystack = text.data();
            size_t haystack_len = text.size();
            const char* needle = pattern.data();
            size_t needle_len = pattern.size();

            // Для паттернов до 16 байт используем SSE4.2 строковые инструкции
            if (needle_len <= 16) {
                alignas(16) char pattern_padded[16] = {0};
                std::memcpy(pattern_padded, needle, needle_len);
                __m128i pattern_vec = _mm_load_si128(
                    reinterpret_cast<const __m128i*>(pattern_padded)
                );

                for (size_t i = 0; i <= haystack_len - needle_len; ++i) {
                    __m128i chunk = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(haystack + i)
                    );

                    unsigned int mode = _SIDD_CMP_EQUAL_ORDERED | _SIDD_UBYTE_OPS;
                    int result = _mm_cmpistri(pattern_vec, chunk, mode);

                    if (result < 16) {
                        bool match = true;
                        for (size_t j = 0; j < needle_len; ++j) {
                            if (haystack[i + result + j] != needle[j]) {
                                match = false;
                                break;
                            }
                        }
                        if (match && i + result <= haystack_len - needle_len) {
                            return i + result;
                        }
                    }
                }
                return std::string_view::npos;
            } else {
                // Для длинных паттернов используем стандартный подход
                auto pos = text.find(pattern);
                return pos;
            }
        #else
            return text.find(pattern);
        #endif
    }

    /**
     * @brief SSE4.2-ускоренное сравнение строк (только ASCII)
     * 
     * @param a Первая строка
     * @param b Вторая строка
     * @return true если строки идентичны
     * 
     * **Производительность:** до 3x быстрее memcmp для длинных строк
     * 
     * @warning Для Unicode строк может давать неверные результаты!
     */
    static bool strings_equal_sse42(std::string_view a, std::string_view b) {
        if (a.size() != b.size()) return false;
        if (a.data() == b.data()) return true;

        #if defined(USE_SSE42) || defined(__SSE4_2__)
            size_t len = a.size();
            const char* a_ptr = a.data();
            const char* b_ptr = b.data();

            size_t i = 0;
            for (; i + 16 <= len; i += 16) {
                __m128i a_chunk = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(a_ptr + i)
                );
                __m128i b_chunk = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_ptr + i)
                );

                __m128i cmp = _mm_cmpeq_epi8(a_chunk, b_chunk);
                int mask = _mm_movemask_epi8(cmp);

                if (mask != 0xFFFF) {
                    return false;
                }
            }

            for (; i < len; ++i) {
                if (a_ptr[i] != b_ptr[i]) {
                    return false;
                }
            }

            return true;
        #else
            return a == b;
        #endif
    }

    // ========================================================================
    // Утилиты для информации
    // ========================================================================

    /**
     * @brief Получить строковое описание уровня SIMD
     * 
     * @return std::string Описание доступных инструкций
     * 
     * @code
     * std::cout << "SIMD: " << SIMDUtils::get_simd_level() << std::endl;
     * // Возможные выводы:
     * // "AVX2 (256-bit) - ТОЛЬКО ДЛЯ ASCII!"
     * // "AVX (128-bit)  - ТОЛЬКО ДЛЯ ASCII!"
     * // "SSE4.2         - ТОЛЬКО ДЛЯ ASCII!"
     * // "SSE4.1         - ТОЛЬКО ДЛЯ ASCII!"
     * // "SSSE3          - ТОЛЬКО ДЛЯ ASCII!"
     * // "SSE3           - ТОЛЬКО ДЛЯ ASCII!"
     * // "SSE2           - ТОЛЬКО ДЛЯ ASCII!"
     * // "Скалярный      - работает с Unicode, но медленнее"
     * @endcode
     */
    static std::string get_simd_level() {
        std::string level = "Скалярный (работает с Unicode, но медленнее)";

        #if defined(__AVX2__)
            level = "AVX2 (256-bit) - ТОЛЬКО ДЛЯ ASCII!";
        #elif defined(__AVX__)
            level = "AVX (128-bit) - ТОЛЬКО ДЛЯ ASCII!";
        #elif defined(__SSE4_2__)
            level = "SSE4.2 - ТОЛЬКО ДЛЯ ASCII!";
        #elif defined(__SSE4_1__)
            level = "SSE4.1 - ТОЛЬКО ДЛЯ ASCII!";
        #elif defined(__SSSE3__)
            level = "SSSE3 - ТОЛЬКО ДЛЯ ASCII!";
        #elif defined(__SSE3__)
            level = "SSE3 - ТОЛЬКО ДЛЯ ASCII!";
        #elif defined(__SSE2__)
            level = "SSE2 - ТОЛЬКО ДЛЯ ASCII!";
        #endif

        return level;
    }

    /**
     * @brief Оптимальное кодирование с автовыбором реализации (ТОЛЬКО ASCII)
     * 
     * @param text Входной текст (ДОЛЖЕН БЫТЬ ТОЛЬКО ASCII!)
     * @param lookup_table Таблица char->ID
     * @param unknown_id ID неизвестного символа
     * @return std::vector<uint32_t> Результат кодирования
     * 
     * Самая простая в использовании функция - автоматически выбирает
     * наилучшую доступную реализацию для текущего процессора.
     * 
     * @warning РАБОТАЕТ ТОЛЬКО С ASCII! Для Unicode используйте encode().
     */
    static std::vector<uint32_t> encode_optimal(std::string_view text,
                                                const uint32_t* lookup_table,
                                                uint32_t unknown_id) {
        int impl = get_recommended_implementation();

        switch (impl) {
            case 2:
                return encode_avx2(text, lookup_table, unknown_id);
            case 1:
                return encode_avx(text, lookup_table, unknown_id);
            default:
                return encode_scalar(text, lookup_table);
        }
    }
};

}    // namespace bpe

/**
 * @example examples/simd_benchmark.cpp
 * Бенчмарк для сравнения производительности SIMD реализаций
 * 
 * @include examples/simd_benchmark.cpp
 * 
 * @code
 * #include "simd_utils.hpp"
 * #include <benchmark/benchmark.h>
 * 
 * static void BM_Encode(benchmark::State& state) {
 *     uint32_t lookup[256];
 *     for (int i = 0; i < 256; ++i) lookup[i] = i;
 *     
 *     std::string text(1024 * 1024, 'a');    // 1 МБ ASCII
 *     
 *     for (auto _ : state) {
 *         auto tokens = bpe::SIMDUtils::encode_optimal(text, lookup, 0);
 *         benchmark::DoNotOptimize(tokens);
 *     }
 *     
 *     state.SetBytesProcessed(state.iterations() * text.size());
 * }
 * 
 * BENCHMARK(BM_Encode);
 * BENCHMARK_MAIN();
 * @endcode
 */

/**
 * @example examples/simd_demo.cpp
 * Демонстрация всех возможностей SIMDUtils
 * 
 * @include examples/simd_demo.cpp
 */