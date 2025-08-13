# Setup environment with spack

The first time you setup the environment, you need to run:

```bash
spack env create lobxp ./spack.yaml
spack env activate lobxp
spack concretize -f
spack install
```

Then, whenever you open a fresh terminal, you only need to run `spack env activate lobxp`.

# Configure, build, and test

The `Justfile` has all you need:

```bash
just configure
just build
just test
```

And if you want to install the python bindings, run

```bash
just install-bindings
```