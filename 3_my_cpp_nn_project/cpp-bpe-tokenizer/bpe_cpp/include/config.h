/**
 * @file config.h.in
 * @brief Шаблон конфигурационного файла для CMake
 * 
 * @author Ваше Имя
 * @date 2024
 * @version 1.0.0
 * 
 * @details Этот файл обрабатывается CMake для генерации config.h.
 *          Он определяет макросы, основанные на результатах проверки системы
 *          и опциях, выбранных пользователем при конфигурации.
 * 
 *          Процесс обработки CMake:
 *          1. CMake читает этот шаблон
 *          2. Заменяет  на соответствующие значения
 *          3. Создает config.h в бинарной директории
/* #undef  */
 * 
 * @note Этот файл НЕ должен включаться напрямую в исходный код.
 *       Используйте сгенерированный config.h из директории сборки.
 * 
 * @see CMakeLists.txt
 */

#ifndef BPE_TOKENIZER_CONFIG_H
#define BPE_TOKENIZER_CONFIG_H

/**
 * @name Информация о версии
 * Версия проекта, определяемая в корневом CMakeLists.txt
 */
/**@{*/
#define PROJECT_VERSION "0.1.0"  ///< Версия проекта в формате "major.minor.patch"
/**@}*/

/**
 * @name Опции параллелизации
 * Включают поддержку различных фреймворков для параллельных вычислений
 */
/**@{*/
#define USE_OPENMP  ///< Использовать OpenMP для параллельных циклов
/* #undef USE_TBB */
/**@}*/

/**
 * @name SIMD оптимизации
 * Включают поддержку векторных инструкций процессора
 */
/**@{*/
/* #undef USE_AVX2 */
/* #undef USE_SSE42 */
/**@}*/

/**
 * @name Инструменты разработчика
 * Опции для отладки и профилирования
 */
/**@{*/
/* #undef BUILD_WITH_PROFILING */
/**@}*/

#endif // BPE_TOKENIZER_CONFIG_H

/**
 * @example examples/check_features.cpp
 * Пример проверки доступных оптимизаций:
 * 
 * @code
 * #include "config.h"
 * 
 * void check_features() {
 *     #ifdef USE_AVX2
 *         std::cout << "AVX2 оптимизации доступны" << std::endl;
 *     #endif
 *     
 *     #ifdef USE_OPENMP
 *         std::cout << "OpenMP параллелизация доступна" << std::endl;
 *     #endif
 *     
 *     std::cout << "Версия проекта: " << PROJECT_VERSION << std::endl;
 * }
 * @endcode
 */
