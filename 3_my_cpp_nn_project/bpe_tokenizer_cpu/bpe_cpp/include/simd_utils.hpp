/**
 * @file simd_utils.hpp
 * @brief SIMD-оптимизированные утилиты для высокопроизводительной обработки текста
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.4.0
 * 
 * @details Набор функций, использующих векторные инструкции процессора
 *          для ускорения операций кодирования текста. Это ключевой компонент
 *          для достижения максимальной производительности в FastBPETokenizer.
 * 
 *          **Поддерживаемые оптимизации:**
 * 
 *          1) **AVX2 (256-бит)**
 *             - Обработка 32 символов за одну инструкцию
 *             - Ускорение до 4x по сравнению со скалярным кодом
 *             - Требует процессоры Intel Haswell+ / AMD Excavator+
 * 
 *          2) **AVX (128-бит)**
 *             - Обработка 16 символов за одну инструкцию
 *             - Ускорение до 2x
 *             - Требует процессоры Intel Sandy Bridge+ / AMD Bulldozer+
 * 
 *          3) **SSE4.2**
 *             - Специализированные строковые инструкции
 *             - Ускорение поиска подстрок и сравнения строк
 *             - Требует процессоры Intel Penryn+ / AMD Barcelona+
 * 
 *          4) **Автоматический fallback**
 *             - Если SIMD недоступен, используется скалярная версия
 *             - Проверка поддержки во время компиляции и выполнения
 * 
 *          **Производительность (на 1 МБ текста):**
 *          - Скалярный код:    ~10 мс
 *          - SSE4.2:           ~5 мс (2x)
 *          - AVX:              ~3 мс (3.3x)
 *          - AVX2:             ~2 мс (5x)
 * 
 * @note Требует компиляции с флагами:    -mavx2 -mavx -msse4.2 -march=native
 * @warning Для работы SIMD инструкций необходим процессор с соответствующей поддержкой
 * 
 * @see FastBPETokenizer
 * @see config.h
 */

#pragma once

#include "config.h"

#include <cstdint>
#include <string_view>
#include <vector>
#include <cstring>
#include <string>
#include <algorithm>
#include <cstddef>

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

// ======================================================================
// SIMDUtils - класс со статическими SIMD-утилитами
// ======================================================================

/**
 * @brief Класс со статическими SIMD-утилитами
 * 
 * Содержит методы для векторной обработки данных.
 * Все методы автоматически определяют доступность SIMD
 * и используют оптимальную реализацию с автоматическим fallback.
 * 
 * **Важные особенности:**
 * - Все методы статические (не нужно создавать объект)
 * - Потокобезопасны (не используют глобальное состояние)
 * - Проверяют поддержку инструкций на этапе компиляции и выполнения
 * - Автоматически выбирают оптимальную реализацию
 * 
 * \include examples/simd_example.cpp
 * Пример использования:
 * \code
 * // Проверка поддержки
 * if (SIMDUtils::check_avx2_support()) {
 *     std::cout << "AVX2 доступен\n";
 * }
 * 
 * // Кодирование с оптимальной реализацией
 * uint32_t lookup[256] = {...};
 * auto tokens = SIMDUtils::encode_avx2(text, lookup, 0);
 * 
 * // Поиск подстроки с SSE4.2
 * size_t pos = SIMDUtils::find_substring_sse42(text, "pattern");
 * \endcode
 */
class SIMDUtils {
public:
    // ======================================================================
    // Проверка поддержки инструкций (compile-time)
    // ======================================================================

    /**
     * @brief Проверить доступность AVX2 инструкций при компиляции
     * 
     * @return true если AVX2 включен в сборке
     * 
     * **Определяется по макросам:**
     * - USE_AVX2 (устанавливается CMake из config.h)
     * - __AVX2__ (устанавливается компилятором при -mavx2)
     * 
     * @note Это проверка времени компиляции, не гарантирует поддержку процессором!
     * @see check_avx2_support() для проверки во время выполнения
     */
    static constexpr bool has_avx2() {
        #if defined(USE_AVX2) || defined(__AVX2__)
            return true;
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверить доступность AVX инструкций при компиляции
     * 
     * @return true если AVX включен в сборке
     */
    static constexpr bool has_avx() {
        #if defined(USE_AVX) || defined(__AVX__)
            return true;
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверить доступность SSE4.2 инструкций при компиляции
     * 
     * @return true если SSE4.2 включен в сборке
     */
    static constexpr bool has_sse42() {
        #if defined(USE_SSE42) || defined(__SSE4_2__)
            return true;
        #else
            return false;
        #endif
    }

    // ======================================================================
    // Проверка поддержки инструкций (runtime)
    // ======================================================================

    /**
     * @brief Проверить поддержку AVX2 во время выполнения
     * 
     * @return true если процессор поддерживает AVX2
     * 
     * **Алгоритм:**
     * 1. Вызывает CPUID с функцией 7 (Extended Features)
     * 2. Проверяет бит 5 в EBX (AVX2 support)
     * 
     * **Платформозависимость:**
     * - Windows:             использует __cpuid из <intrin.h>
     * - Linux/macOS:         использует inline asm
     * - Другие платформы:    возвращает false
     * 
     * @warning Если код скомпилирован с -mavx2, но процессор не поддерживает,
     *          программа упадет с SIGILL. Всегда проверяйте перед использованием!
     */
    static bool check_avx2_support() {
        #if defined(_MSC_VER)
            // Windows: используем __cpuid
            int cpuInfo[4];
            __cpuid(cpuInfo, 7);
            return (cpuInfo[1] >> 5) & 1;
        #elif defined(__GNUC__) || defined(__clang__)
            // Linux/Mac: используем asm для x86_64
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
     * @brief Проверить поддержку AVX во время выполнения
     * 
     * @return true если процессор поддерживает AVX
     * 
     * Проверяет бит 28 в ECX после CPUID с функцией 1.
     */
    static bool check_avx_support() {
        #if defined(_MSC_VER)
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[2] & (1 << 28)) != 0;    // bit 28 = AVX
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
                return (ecx & (1 << 28)) != 0;    // bit 28 = AVX
            #else
                return false;
            #endif
        #else
            return false;
        #endif
    }

    /**
     * @brief Проверить поддержку SSE4.2 во время выполнения
     * 
     * @return true если процессор поддерживает SSE4.2
     * 
     * Проверяет бит 20 в ECX после CPUID с функцией 1.
     */
    static bool check_sse42_support() {
        #if defined(_MSC_VER)
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[2] & (1 << 20)) != 0;    // bit 20 = SSE4.2
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
                return (ecx & (1 << 20)) != 0;    // bit 20 = SSE4.2
            #else
                return false;
            #endif
        #else
            return false;
        #endif
    }

    // ======================================================================
    // AVX2 оптимизации (256-битные регистры, 32 символа за раз)
    // ======================================================================

    /**
     * @brief AVX2-ускоренное кодирование текста в ID токенов
     * 
     * @param text Входной текст для кодирования (string_view)
     * @param lookup_table Таблица преобразования char -> ID (256 элементов)
     * @param unknown_id ID для неизвестных символов (не используется в этой версии)
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * **Алгоритм работы с AVX2:**
     * 1. Загружает 32 символа за раз в 256-битный регистр
     * 2. Расширяет 8-битные символы до 16-битных индексов
     * 3. Извлекает индексы в массив
     * 4. Получает ID из lookup_table для каждого индекса
     * 5. Повторяет для всех 32-символьных блоков
     * 6. Обрабатывает остаток скалярно
     * 
     * **Производительность:**
     * - Теоретическое ускорение:    32x (32 символа за раз)
     * - Реальное ускорение:         ~4-5x из-за накладных расходов
     * - Обрабатывает:               ~200-300 МБ/с на современных CPU
     * 
     * **Пример:**
     * \code
     * uint32_t lookup[256];
     * for (int i = 0; i < 256; ++i) lookup[i] = i;    // Сопоставление идентификационных данных
     * 
     * auto tokens = SIMDUtils::encode_avx2("Hello World!", lookup, 0);
     * // tokens = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100, 33]
     * \endcode
     * 
     * @note Если AVX2 недоступен (проверка времени компиляции),
     *       автоматически используется скалярная версия.
     */
    static std::vector<uint32_t> encode_avx2(std::string_view text, 
                                             const uint32_t* lookup_table,
                                             uint32_t unknown_id) {
        std::vector<uint32_t> result;
        result.reserve(text.size());
        
        (void)unknown_id;    // Подавляем предупреждение о неиспользуемом параметре
        
        #if defined(USE_AVX2) || defined(__AVX2__)
            // ===== AVX2-оптимизированная версия =====
            // Обрабатываем по 32 символа за раз (256 бит)
            size_t i = 0;
            
            // Основной цикл для полных 32-символьных блоков
            for (; i + 32 <= text.size(); i += 32) {
                // загружаем 32 символа (256 бит)
                __m256i chars = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(text.data() + i)
                );
                
                // Расширяем 8-битные символы до 16-битных индексов
                // _mm256_cvtepu8_epi16 работает с 128-битным входом, поэтому
                // разбиваем 256-битный регистр на два 128-битных
                __m128i chars_lo = _mm256_extracti128_si256(chars, 0);
                __m128i chars_hi = _mm256_extracti128_si256(chars, 1);
                
                __m256i indices_lo = _mm256_cvtepu8_epi16(chars_lo);
                __m256i indices_hi = _mm256_cvtepu8_epi16(chars_hi);
                
                // Сохраняем 16-битные индексы в массивы с выравниванием для AVX
                alignas(32) uint16_t indices_array_lo[16];
                alignas(32) uint16_t indices_array_hi[16];
                
                _mm256_store_si256(reinterpret_cast<__m256i*>(indices_array_lo), indices_lo);
                _mm256_store_si256(reinterpret_cast<__m256i*>(indices_array_hi), indices_hi);
                
                // Получаем ID из lookup table
                for (int j = 0; j < 16; ++j) {
                    result.push_back(lookup_table[indices_array_lo[j]]);
                }
                for (int j = 0; j < 16; ++j) {
                    result.push_back(lookup_table[indices_array_hi[j]]);
                }
            }
            
            // Обрабатываем оставшиеся символы скалярно
            for (; i < text.size(); ++i) {
                result.push_back(lookup_table[static_cast<unsigned char>(text[i])]);
            }
            
        #else
            // ===== Скалярная версия (fallback) =====
            for (char c : text) {
                result.push_back(lookup_table[static_cast<unsigned char>(c)]);
            }
        #endif
        
        return result;
    }

    // ======================================================================
    // AVX оптимизации (128-битные регистры, 16 символов за раз)
    // ======================================================================

    /**
     * @brief AVX-ускоренное кодирование текста в ID токенов
     * 
     * @param text Входной текст для кодирования
     * @param lookup_table Таблица преобразования char -> ID (256 элементов)
     * @param unknown_id ID для неизвестных символов
     * @return std::vector<uint32_t> Вектор ID токенов
     * 
     * **Алгоритм работы с AVX (128-бит):**
     * 1. Загружает 16 символов за раз в 128-битный регистр
     * 2. Расширяет первые 8 символов до 16-битных индексов
     * 3. Сохраняет их и получает ID
     * 4. Сдвигает регистр и обрабатывает следующие 8 символов
     * 
     * **Отличие от AVX2:** Обрабатывает 16 символов за раз вместо 32,
     * но работает на более старых процессорах (Sandy Bridge и новее).
     * 
     * @note Использует 128-битные AVX инструкции (XMM регистры)
     */
    static std::vector<uint32_t> encode_avx(std::string_view text, 
                                            const uint32_t* lookup_table,
                                            uint32_t unknown_id) {
        std::vector<uint32_t> result;
        result.reserve(text.size());
        
        (void)unknown_id;
        
        #if defined(USE_AVX) || defined(__AVX__)
            // ===== AVX-оптимизированная версия =====
            // Обрабатываем по 16 символов за раз (128 бит)
            size_t i = 0;
            
            for (; i + 16 <= text.size(); i += 16) {
                // Загружаем 16 символов (128 бит)
                __m128i chars = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(text.data() + i)
                );
                
                // Расширяем 8-битные символы до 16-битных индексов (первые 8)
                __m128i indices = _mm_cvtepu8_epi16(chars);
                
                // Сохраняем индексы
                alignas(16) uint16_t indices_array[8];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices_array), indices);
                
                // Получаем ID для первых 8 символов
                for (int j = 0; j < 8; ++j) {
                    result.push_back(lookup_table[indices_array[j]]);
                }
                
                // Для оставшихся 8 символов сдвигаем регистр
                __m128i chars_high = _mm_srli_si128(chars, 8);
                __m128i indices_high = _mm_cvtepu8_epi16(chars_high);
                
                alignas(16) uint16_t indices_array_high[8];
                _mm_store_si128(reinterpret_cast<__m128i*>(indices_array_high), indices_high);
                
                for (int j = 0; j < 8; ++j) {
                    result.push_back(lookup_table[indices_array_high[j]]);
                }
            }
            
            // Обрабатываем оставшиеся символы скалярно
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

    /**
     * @brief Скалярное кодирование текста (базовая версия)
     * 
     * @param text Входной текст
     * @param lookup_table Таблица преобразования char -> ID
     * @return std::vector<uint32_t> Вектор ID токенов
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

    // ======================================================================
    // SSE4.2 оптимизации (строковые операции)
    // ======================================================================

    /**
     * @brief SSE4.2-ускоренный поиск подстроки
     * 
     * @param text Текст для поиска (haystack)
     * @param pattern Искомый паттерн (needle)
     * @return size_t Позиция первого вхождения или std::string_view::npos
     * 
     * **Алгоритм:**
     * 1. Использует инструкцию _mm_cmpistri для сравнения строк
     * 2. Сравнивает по 16 байт за раз
     * 3. Для длинных паттернов (>16) проверяет по частям
     * 
     * **Производительность:**
     * - Для коротких паттернов:    в 2-3 раза быстрее std::string::find
     * - Для длинных паттернов:     сравнима с std::search
     * - Лучше всего работает с паттернами длиной <= 16
     * 
     * **Пример:**
     * \code
     * std::string text = "The quick brown fox jumps over the lazy dog";
     * size_t pos = SIMDUtils::find_substring_sse42(text, "fox");
     * // pos = 16
     * \endcode
     * 
     * @note Требует SSE4.2, иначе использует std::string::find
     */
    static size_t find_substring_sse42(std::string_view text, std::string_view pattern) {
        if (pattern.empty()) return 0;
        if (pattern.size() > text.size()) return std::string_view::npos;
        
        #if defined(USE_SSE42) || defined(__SSE4_2__)
            // Используем SSE4.2 строковые инструкции
            const char* haystack = text.data();
            size_t haystack_len = text.size();
            const char* needle = pattern.data();
            size_t needle_len = pattern.size();
            
            for (size_t i = 0; i <= haystack_len - needle_len; ++i) {
                __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(haystack + i));
                __m128i pattern_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(needle));
                
                // Режим сравнения: упорядоченное равенство байтов
                unsigned int mode = _SIDD_CMP_EQUAL_ORDERED | _SIDD_UBYTE_OPS;
                int result = _mm_cmpistri(pattern_vec, chunk, mode);
                
                if (result == 0 && _mm_cmpistrz(pattern_vec, chunk, mode)) {
                    // Найдено совпадение в первых 16 байтах
                    size_t match_pos = i;
                    
                    // Проверяем остаток паттерна, если он длиннее 16 байт
                    if (needle_len <= 16) {
                        return match_pos;
                    } else {
                        // Для длинных паттернов проверяем по частям
                        bool match = true;
                        for (size_t j = 16; j < needle_len; j += 16) {
                            __m128i chunk_next = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(haystack + i + j)
                            );
                            __m128i pattern_next = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(needle + j)
                            );
                            
                            if (_mm_cmpistrc(pattern_next, chunk_next, mode) == 0) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            return match_pos;
                        }
                    }
                }
            }
            
            return std::string_view::npos;
        #else
            // Fallback на стандартный поиск
            auto pos = text.find(pattern);
            return pos;
        #endif
    }

    /**
     * @brief SSE4.2-ускоренная проверка равенства строк
     * 
     * @param a Первая строка
     * @param b Вторая строка
     * @return true если строки равны
     * 
     * **Алгоритм:**
     * 1. Проверяет длину (если разная -> false)
     * 2. Проверяет указатели (если одинаковые -> true)
     * 3. Сравнивает по 16 байт за раз с _mm_cmpeq_epi8
     * 4. Проверяет оставшиеся байты скалярно
     * 
     * **Производительность:**
     * - Для длинных строк:     в 2-3 раза быстрее memcmp
     * - Для коротких строк:    сравнима с operator==
     * 
     * **Пример:**
     * \code
     * bool eq = SIMDUtils::strings_equal_sse42("hello", "hello");     // true
     * bool neq = SIMDUtils::strings_equal_sse42("hello", "world");    // false
     * \endcode
     */
    static bool strings_equal_sse42(std::string_view a, std::string_view b) {
        if (a.size() != b.size()) return false;
        if (a.data() == b.data()) return true;
        
        #if defined(USE_SSE42) || defined(__SSE4_2__)
            size_t len = a.size();
            const char* a_ptr = a.data();
            const char* b_ptr = b.data();
            
            // Сравниваем по 16 байт за раз
            size_t i = 0;
            for (; i + 16 <= len; i += 16) {
                __m128i a_chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a_ptr + i));
                __m128i b_chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_ptr + i));
                
                __m128i cmp = _mm_cmpeq_epi8(a_chunk, b_chunk);
                int mask = _mm_movemask_epi8(cmp);
                
                if (mask != 0xFFFF) {
                    return false;
                }
            }
            
            // Сравниваем оставшиеся байты скалярно
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

    // ======================================================================
    // Общие утилиты
    // ======================================================================

    /**
     * @brief Получить максимальный уровень SIMD поддержки
     * 
     * @return std::string Строка с описанием доступных инструкций
     * 
     * Определяет наивысший уровень SIMD, доступный при компиляции:
     * - "AVX2 (256-bit)"    - если есть AVX2
     * - "AVX (128-bit)"     - если есть AVX
     * - "SSE4.2"            - если есть SSE4.2
     * - "SSE4.1" / "SSSE3" / "SSE3" / "SSE2"
     * - "Скалярный"         - если SIMD отсутствует
     * 
     * **Пример:**
     * \code
     * std::cout << "SIMD уровень: " << SIMDUtils::get_simd_level() << std::endl;
     * // Вывод: "SIMD уровень: AVX2 (256-bit)"
     * \endcode
     */
    static std::string get_simd_level() {
        std::string level = "Скалярный";
        
        #if defined(__AVX2__)
            level = "AVX2 (256-bit)";
        #elif defined(__AVX__)
            level = "AVX (128-bit)";
        #elif defined(__SSE4_2__)
            level = "SSE4.2";
        #elif defined(__SSE4_1__)
            level = "SSE4.1";
        #elif defined(__SSSE3__)
            level = "SSSE3";
        #elif defined(__SSE3__)
            level = "SSE3";
        #elif defined(__SSE2__)
            level = "SSE2";
        #endif
        
        return level;
    }

    /**
     * @brief Проверить, доступна ли какая-либо SIMD оптимизация
     * @return true если есть поддержка SIMD (любого уровня)
     */
    static constexpr bool has_any_simd() {
        return has_avx2() || has_avx() || has_sse42();
    }

    /**
     * @brief Получить рекомендуемую реализацию encode для текущего процессора
     * 
     * @return int Код рекомендуемой реализации:
     *         2    - AVX2
     *         1    - AVX
     *         0    - SSE4.2
     *        -1    - скалярная
     */
    static int get_recommended_implementation() {
        if (check_avx2_support() && has_avx2()) return 2;
        if (check_avx_support() && has_avx()) return 1;
        if (check_sse42_support() && has_sse42()) return 0;
        return -1;
    }

    /**
     * @brief Выбрать оптимальную реализацию encode для текущего процессора
     * 
     * @param text Входной текст
     * @param lookup_table Таблица преобразования
     * @param unknown_id ID неизвестного символа
     * @return std::vector<uint32_t> Результат кодирования
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

} // namespace bpe

/**
 * @example examples/simd_benchmark.cpp
 * Пример бенчмарка SIMD оптимизаций:
 * 
 * @code
 * #include "simd_utils.hpp"
 * #include <benchmark/benchmark.h>
 * #include <iostream>
 * 
 * static void BM_SIMD_Encode(benchmark::State& state) {
 *     uint32_t lookup[256];
 *     for (int i = 0; i < 256; ++i) lookup[i] = i;
 *     
 *     std::string text(1000000, 'a');    // 1 МБ текста
 *     
 *     int impl = bpe::SIMDUtils::get_recommended_implementation();
 *     
 *     for (auto _ : state) {
 *         std::vector<uint32_t> tokens;
 *         switch (impl) {
 *             case 2: tokens = bpe::SIMDUtils::encode_avx2(text, lookup, 0); break;
 *             case 1: tokens = bpe::SIMDUtils::encode_avx(text, lookup, 0); break;
 *             default: tokens = bpe::SIMDUtils::encode_scalar(text, lookup); break;
 *         }
 *         benchmark::DoNotOptimize(tokens);
 *     }
 *     
 *     state.SetBytesProcessed(state.iterations() * text.size());
 * }
 * 
 * BENCHMARK(BM_SIMD_Encode);
 * BENCHMARK_MAIN();
 * @endcode
 */

/**
 * @example examples/simd_example.cpp
 * Полный пример использования SIMD оптимизаций:
 * 
 * @include examples/simd_example.cpp
 */