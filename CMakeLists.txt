cmake_minimum_required(VERSION 3.15)
project(lobxp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED CONFIG)

# Header-only library: carries include dirs + Eigen usage to dependents
add_library(lobxp INTERFACE)
target_include_directories(lobxp INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(lobxp INTERFACE Eigen3::Eigen)

# Tests
enable_testing()
add_subdirectory(tests)

# Python module (links to header-only target for includes + Eigen)
# pybind11_add_module(pylobxp MODULE
#   ${PROJECT_SOURCE_DIR}/bindings/py_kalman.cpp
# )
# target_link_libraries(pylobxp PRIVATE lobxp)  # brings Eigen + includes

# Install: headers + Python module
# include(GNUInstallDirs)
# install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
#         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(TARGETS pylobxp
#         LIBRARY DESTINATION ${Python3_SITEARCH}
#         RUNTIME DESTINATION ${Python3_SITEARCH})
