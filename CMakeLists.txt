cmake_minimum_required(VERSION 3.15)
project(lobxp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies from Spack
# Spack's "view: true" ensures these are in CMAKE_PREFIX_PATH automatically
find_package(Eigen3 REQUIRED NO_MODULE)
# find_package(pybind11 REQUIRED)

# Library
add_library(lobxp
    ${PROJECT_SOURCE_DIR}/src/lobxp/kalman_filter.cpp
)
target_include_directories(lobxp
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(lobxp PUBLIC Eigen3::Eigen)

# # Example C++ executable
# add_executable(kf_test src/kf_test.cpp)
# target_link_libraries(kf_test PRIVATE lobxp)

# # Python module
# pybind11_add_module(kalmx src/bindings.cpp)
# target_link_libraries(kalmx PRIVATE lobxp)
