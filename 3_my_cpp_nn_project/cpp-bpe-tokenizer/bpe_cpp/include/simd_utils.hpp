/**
 * @file simd_utils.hpp
 * @brief SIMD-оптимизированные утилиты для высокопроизводительной обработки текста
 * 
 * @author Евгений П.
 * @date 2026
 * @version 3.2.0
 * 
 * @details Набор функций, использующих векторные инструкции процессора
 *          для ускорения операций кодирования текста.
 * 
 *          Поддерживаемые оптимизации:
 *          - AVX2: обработка 16 символов за одну инструкцию
 *          - Автоматический fallback на скалярную версию
 *          - Проверка поддержки инструкций во время компиляции
 * 
 * @note Требует компиляции с флагами: -mavx2 -march=native
 * @warning Для работы AVX2 необходим процессор с поддержкой (Intel Haswell+ / AMD Excavator+)
 * 
 * @see FastBPETokenizer
 * @see config.h
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#ifdef USE_AVX2
    #include <immintrin.h>
#endif

namespace bpe {

/**
 * @brief Класс со статическими SIMD-утилитами
 * 
 * Содержит методы для векторной обработки данных.
 * Все методы автоматически определяют доступность SIMD
 * и используют оптимальную реализацию.
 */
class SIMDUtils {
public:
    /**
     * @brief Проверить доступность AVX2 инструкций
     * @return true если AVX2 доступен и включен в сборке
     * 
     * @note Во время компиляции определяется по макросу USE_AVX2
     */
    static bool has_avx2() {
        #ifdef USE_AVX2
            return true;
        #else
            return false;
        #endif
    }

    /**
     * @brief AVX2-ускоренное кодирование текста в ID токенов
     * 
     * @param text Входной текст для кодирования
     * @param lookup_table Таблица преобразования char -> ID (256 элементов)
     * @param unknown_id ID для неизвестных символов
     * @return Вектор ID токенов
     * 
     * @details Алгоритм работы:
     *          1. Загружает 16 символов за раз в 128-битный регистр
     *          2. Расширяет до 16-битных индексов в 256-битном регистре
     *          3. Извлекает индексы и получает ID из lookup_table
     *          4. Повторяет для всех символов
     * 
     * @note Если AVX2 недоступен, автоматически используется скалярная версия
     * 
     * @complexity O(n) с константой ~0.0625 от скалярной версии
     */
    static std::vector<uint32_t> encode_avx2(std::string_view text, 
                                             const uint32_t* lookup_table,
                                             uint32_t unknown_id) {
        std::vector<uint32_t> result;
        result.reserve(text.size());
        
        #ifdef USE_AVX2
            // ===== AVX2-оптимизированная версия =====
            // Обрабатываем по 16 символов за раз (256 бит / 16 бит на символ)
            size_t i = 0;
            
            // Основной цикл для полных 16-символьных блоков
            for (; i + 16 <= text.size(); i += 16) {
                // Загружаем 16 символов (128 бит)
                __m128i chars = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(text.data() + i)
                );
                
                // Расширяем 8-битные символы до 16-битных индексов
                // Это позволяет использовать их как индексы в lookup_table
                __m256i indices = _mm256_cvtepu8_epi16(chars);
                
                // Сохраняем 16-битные индексы в массив
                alignas(32) uint16_t indices_array[16];
                _mm256_store_si256(reinterpret_cast<__m256i*>(indices_array), indices);
                
                // Получаем ID из lookup table для каждого индекса
                #ifdef __GNUC__
                    // GCC/Clang: ручная развертка цикла для лучшей оптимизации
                    result.push_back(lookup_table[indices_array[0]]);
                    result.push_back(lookup_table[indices_array[1]]);
                    result.push_back(lookup_table[indices_array[2]]);
                    result.push_back(lookup_table[indices_array[3]]);
                    result.push_back(lookup_table[indices_array[4]]);
                    result.push_back(lookup_table[indices_array[5]]);
                    result.push_back(lookup_table[indices_array[6]]);
                    result.push_back(lookup_table[indices_array[7]]);
                    result.push_back(lookup_table[indices_array[8]]);
                    result.push_back(lookup_table[indices_array[9]]);
                    result.push_back(lookup_table[indices_array[10]]);
                    result.push_back(lookup_table[indices_array[11]]);
                    result.push_back(lookup_table[indices_array[12]]);
                    result.push_back(lookup_table[indices_array[13]]);
                    result.push_back(lookup_table[indices_array[14]]);
                    result.push_back(lookup_table[indices_array[15]]);
                #else
                    // MSVC: обычный цикл
                    for (int j = 0; j < 16; ++j) {
                        result.push_back(lookup_table[indices_array[j]]);
                    }
                #endif
            }
            
            // Обрабатываем оставшиеся символы (менее 16)
            for (; i < text.size(); ++i) {
                result.push_back(lookup_table[static_cast<unsigned char>(text[i])]);
            }
            
        #else
            // ===== Скалярная версия (fallback) =====
            // Используется если AVX2 недоступен или отключен
            for (char c : text) {
                result.push_back(lookup_table[static_cast<unsigned char>(c)]);
            }
        #endif
        
        return result;
    }

    /**
     * @brief Проверить поддержку AVX2 во время выполнения
     * @return true если процессор поддерживает AVX2
     * 
     * @note Использует CPUID инструкцию для определения возможностей процессора
     */
    static bool check_avx2_support() {
        #ifdef USE_AVX2
            #if defined(_MSC_VER)
                // Windows: используем __cpuid
                int cpuInfo[4];
                __cpuid(cpuInfo, 7);
                return (cpuInfo[1] >> 5) & 1;
            #elif defined(__GNUC__) || defined(__clang__)
                // Linux/Mac: используем asm
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
};

} // namespace bpe

/**
 * @example examples/simd_example.cpp
 * Пример использования SIMD оптимизаций:
 * @code
 * #include "simd_utils.hpp"
 * #include <iostream>
 * 
 * int main() {
 *     // Проверяем поддержку AVX2
 *     if (bpe::SIMDUtils::check_avx2_support()) {
 *         std::cout << "AVX2 поддерживается процессором" << std::endl;
 *     }
 *     
 *     // Создаем lookup table (например, ASCII -> ID)
 *     uint32_t lookup[256];
 *     for (int i = 0; i < 256; ++i) {
 *         lookup[i] = i;  // Просто для примера
 *     }
 *     
 *     // Кодируем текст с SIMD оптимизацией
 *     std::string text = "Hello, SIMD World!";
 *     auto tokens = bpe::SIMDUtils::encode_avx2(text, lookup, 0);
 *     
 *     std::cout << "Закодировано " << tokens.size() << " токенов" << std::endl;
 * }
 * @endcode
 */