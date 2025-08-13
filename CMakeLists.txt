cmake_minimum_required(VERSION 3.15)
project(lobxp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies from Spack
# Spack's "view: true" ensures these are in CMAKE_PREFIX_PATH automatically
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

# Library
add_library(lobxp
    ${PROJECT_SOURCE_DIR}/src/lobxp/kalman_filter.cpp
)
target_include_directories(lobxp
    PUBLIC ${PROJECT_SOURCE_DIR}/include/
)
target_link_libraries(lobxp PUBLIC Eigen3::Eigen)

# Tests
enable_testing()
add_subdirectory(tests)

# Python module
pybind11_add_module(lobxp_py MODULE
  ${PROJECT_SOURCE_DIR}/bindings/py_kalman.cpp
)
target_link_libraries(lobxp_py PRIVATE lobxp Eigen3::Eigen)
target_include_directories(lobxp_py PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Install targets
include(GNUInstallDirs)
install(TARGETS lobxp_py
        LIBRARY DESTINATION ${Python3_SITEARCH}
        RUNTIME DESTINATION ${Python3_SITEARCH})