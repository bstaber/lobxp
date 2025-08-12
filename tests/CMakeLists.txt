cmake_minimum_required(VERSION 3.18)

# If you build a library for your KF code, link against it here.
# Otherwise, just add the .cpp directly to the test target as shown below.

find_package(Eigen3 REQUIRED)           # already in your env
find_package(Catch2 3 REQUIRED CONFIG)  # provided by Spack

add_executable(kf_tests
  test_kalman_filter.cpp
)
target_link_libraries(kf_tests PRIVATE
  lobxp
  Eigen3::Eigen
  Catch2::Catch2WithMain
)

target_include_directories(kf_tests PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/..
)

include(CTest)
include(Catch)
catch_discover_tests(kf_tests)
