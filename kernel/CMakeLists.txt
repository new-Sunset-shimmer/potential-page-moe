cmake_minimum_required(VERSION 3.18)
project(libmoe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 1. Python 찾기
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# === [여기부터 수정] ===
# 2. Pybind11 경로를 Python에게 직접 물어보기 (Auto-detect)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_found_result
)

if (NOT "${pybind11_found_result}" EQUAL "0")
    message(FATAL_ERROR "pybind11 not found! Try: pip install pybind11")
endif()

# 찾은 경로를 사용하여 패키지 로드
find_package(pybind11 REQUIRED CONFIG PATHS ${pybind11_DIR})
# === [여기까지 수정] ===

find_package(CUDAToolkit REQUIRED)

add_library(moe SHARED 
    src/moe_core.cpp 
    src/bindings.cpp
)

target_include_directories(moe PRIVATE 
    include 
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(moe PRIVATE 
    pybind11::module 
    Python3::Python 
    CUDA::cudart
)

set_target_properties(moe PROPERTIES 
    OUTPUT_NAME "libmoe"
    PREFIX ""
    SUFFIX ".so"
)