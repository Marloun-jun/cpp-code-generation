# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-src"
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-build"
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix"
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix/tmp"
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp"
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix/src"
  "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/john/Projects/NS/3_my_cpp_nn_project/bpe_tokenizer_cpu/bpe_cpp/build/_deps/fmt-subbuild/fmt-populate-prefix/src/fmt-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
