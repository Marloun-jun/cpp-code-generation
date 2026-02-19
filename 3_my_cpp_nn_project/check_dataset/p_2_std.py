def find_std_in_using_namespace_std(filename):
# Поиск строк с using_namespace_std, где есть std::
    violations = []
    with open(filename, 'r', encoding='utf-8') as f:
        line_num = 0
        for raw_line in f:
            line_num += 1
            raw_line = raw_line.rstrip('\n')
            if not raw_line.strip() or raw_line.strip().startswith('#'):
                continue
            if 'using_namespace_std' in raw_line and 'std::' in raw_line:
                violations.append(line_num)
    return violations

def fix_std_violations(filename, output_filename):
# Удаление std:: в строках с using_namespace_std
    fixes_applied = 0
    with open(filename, 'r', encoding='utf-8') as fin, \
        open(output_filename, 'w', encoding='utf-8') as fout:
        for line in fin:
            if 'using_namespace_std' in line and 'std::' in line:
                line = line.replace('std::cout', 'cout')
                line = line.replace('std::endl', 'endl')
                line = line.replace('std::cin', 'cin')
                line = line.replace('std::string', 'string')
                line = line.replace('std::vector', 'vector')
                line = line.replace('std::sort', 'sort')
                line = line.replace('std::boolalpha', 'boolalpha')
                line = line.replace('std::byte', 'byte')
                line = line.replace('std::abs', 'abs')
                line = line.replace('std::is_arithmetic_v', 'is_arithmetic_v')
                line = line.replace('std::to_string', 'to_string')
                line = line.replace('std::complex', 'complex')
                line = line.replace('std::array', 'array')
                line = line.replace('std::function', 'function')
                line = line.replace('std::bind', 'bind')
                line = line.replace('std::integer_sequence', 'integer_sequence')
                line = line.replace('std::make_integer_sequence', 'make_integer_sequence')
                line = line.replace('std::index_sequence', 'index_sequence')
                line = line.replace('std::make_index_sequence', 'make_index_sequence')
                line = line.replace('std::index_sequence_for', 'index_sequence_for')
                line = line.replace('std::any', 'any')
                line = line.replace('std::tuple', 'tuple')
                line = line.replace('std::span', 'span')
                line = line.replace('std::variant', 'variant')
                line = line.replace('std::visit', 'visit')
                line = line.replace('std::bitset', 'bitset')
                line = line.replace('std::swap', 'swap')
                line = line.replace('std::pow', 'pow')
                line = line.replace('std::is_arithmetic', 'is_arithmetic')
                line = line.replace('std::count', 'count')
                line = line.replace('std::runtime_error', 'runtime_error')
                line = line.replace('std::ofstream', 'ofstream')
                line = line.replace('std::list', 'list')
                line = line.replace('std::map', 'map')
                line = line.replace('std::future', 'future')
                line = line.replace('std::async', 'async')
                line = line.replace('std::atomic', 'atomic')
                line = line.replace('std::ifstream', 'ifstream')
                line = line.replace('std::thread', 'thread')
                line = line.replace('std::move', 'move')
                line = line.replace('std::fstream', 'fstream')
                line = line.replace('std::remove_if', 'remove_if')
                line = line.replace('std::unordered_set', 'unordered_set')
                line = line.replace('std::unique_ptr', 'unique_ptr')
                line = line.replace('std::mutex', 'mutex')
                line = line.replace('std::copy', 'copy')
                line = line.replace('std::true_type', 'true_type')
                line = line.replace('std::bad_alloc', 'bad_alloc')
                line = line.replace('std::shared_ptr', 'shared_ptr')
                line = line.replace('std::optional', 'optional')
                line = line.replace('std::type_info', 'type_info')
                line = line.replace('std::stack', 'stack')
                line = line.replace('std::queue', 'queue')
                line = line.replace('std::find', 'find')
                line = line.replace('std::out_of_range', 'out_of_range')
                line = line.replace('std::invalid_argument', 'invalid_argument')
                line = line.replace('std::isupper', 'isupper')
                line = line.replace('std::islower', 'islower')
                line = line.replace('std::isdigit', 'isdigit')
                line = line.replace('std::isalnum', 'isalnum')
                line = line.replace('std::isnan', 'isnan')
                line = line.replace('std::isinf', 'isinf')
                line = line.replace('std::numeric_limits', 'numeric_limits')
                line = line.replace('std::logic_error', 'logic_error')
                line = line.replace('std::exception', 'exception')
                # ... (дополнить при необходимости)
                fixes_applied += 1
            fout.write(line)
    print(f"Исправлено строк: {fixes_applied}")
    print(f"Сохранено в: {output_filename}")


if __name__ == "__main__":
    filename = '3_my_cpp_nn_project/check_dataset/2_cpp_code_generation_dataset.csv'
    violations = find_std_in_using_namespace_std(filename)
    if violations:
        print(f"Найдено строк: {len(violations)}")
        print("Номера строк с using_namespace_std где есть std:::")
        print(", ".join(map(str, violations)))
    else:
        print("Нарушений не найдено")
"""
    fix_std_violations(
    '3_my_cpp_nn_project/check_dataset/1_cpp_code_generation_dataset.csv',
    '3_my_cpp_nn_project/check_dataset/1_cpp_code_generation_dataset_fixed.csv'
)
#"""
# должны остаться примеры 9031, 9474, 10307, 10313, 10349, 10370, 10632, 11076, 11264, 11933, 12610, 13025 - их корректировать не нужно!