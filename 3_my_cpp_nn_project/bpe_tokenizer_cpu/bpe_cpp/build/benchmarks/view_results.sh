#!/bin/bash
echo ''
echo '============================================================'
echo 'СОДЕРЖИМОЕ ПОСЛЕДНЕГО JSON ФАЙЛА'
echo '============================================================'
echo ''
LATEST=$(ls -t '/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks'/*.json 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
echo "Файл: $LATEST"
echo ""
python3 -m json.tool "$LATEST" | head -50
else
echo 'Нет результатов в /home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/reports/benchmarks/'
fi
echo ''
