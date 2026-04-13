#!/bin/bash
echo ''
echo '============================================================'
echo 'ПОСЛЕДНИЕ РЕЗУЛЬТАТЫ БЕНЧМАРКОВ'
echo '============================================================'
echo ''
ls -lt '/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks'/*.json 2>/dev/null | head -5 || echo 'Нет результатов в /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/'
echo ''
