# Default target: build and run the demo
default: configure build install-bindings test

# Configure the project (only needed once or if CMakeLists.txt changes)
configure:
    cmake -S . -B cmake-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/.local

# Build the project
build:
    cmake --build cmake-build -j

# Install the Python bindings
install-bindings:
    cmake --install cmake-build

# Run tests
test:
    ctest --test-dir cmake-build --output-on-failure

# Clean the build folder
clean:
    rm -rf cmake-build

# Rebuild from scratch
rebuild: clean configure build

# Linting and formatting
lint:
    clang-tidy src/lobxp/*.cpp include/*.hpp -- -Iinclude -std=c++17
    clang-format --dry-run --Werror src/lobxp/*.cpp include/*.hpp

fmt:
    clang-format -i src/*.cpp include/kalmx/*.hpp
