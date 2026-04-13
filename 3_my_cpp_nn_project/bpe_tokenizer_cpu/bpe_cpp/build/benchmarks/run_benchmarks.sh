#!/bin/bash
echo ''
echo '============================================================'
echo 'ЗАПУСК ВСЕХ БЕНЧМАРКОВ (ТИХИЙ РЕЖИМ)'
echo '============================================================'
echo ''
echo 'Результаты будут сохранены в: /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/'
echo ''
echo 'Запуск bench_tokenizer...'
/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/benchmarks/bench_tokenizer --benchmark_out=/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/bench_tokenizer_20260325_153922.json --benchmark_out_format=json > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo 'bench_tokenizer завершен, результаты: /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/bench_tokenizer_20260325_153922.json'
else
  echo 'bench_tokenizer завершился с ошибкой!'
fi
echo 'Запуск bench_fast_tokenizer...'
/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/benchmarks/bench_fast_tokenizer --benchmark_out=/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/bench_fast_tokenizer_20260325_153922.json --benchmark_out_format=json > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo 'bench_fast_tokenizer завершен, результаты: /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/bench_fast_tokenizer_20260325_153922.json'
else
  echo 'bench_fast_tokenizer завершился с ошибкой!'
fi
echo 'Запуск bench_comparison...'
/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/benchmarks/bench_comparison --benchmark_out=/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/bench_comparison_20260325_153922.json --benchmark_out_format=json > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo 'bench_comparison завершен, результаты: /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/bench_comparison_20260325_153922.json'
else
  echo 'bench_comparison завершился с ошибкой!'
fi
echo ''
echo '------------------------------------------------------------'
echo 'Все бенчмарки завершены!'
echo 'Результаты сохранены в: /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/'
echo '------------------------------------------------------------'
