# Setup environment with spack

The first time you setup the environment, you need to run:

```bash
spack env create lobxp ./spack.yaml
spack env activate lobxp
spack install
```

Then, whenever you open a fresh terminal, you only need to run `spack env activate lobxp`.

# Building

```bash
mkdir build && cd build
cmake ..
make
```