# Contributing

## Pull requests

Send all changes as pull requests. Do not push to `main`.

## Gates

Each pull request must pass the gates in `.github/workflows/ci.yml`. Run the
same commands on your machine before you open the pull request.

## Port code

This project is a Rust port of LIBLINEAR (C++). Before you add or change port code:

1. Freeze a reference from the original C code. The reference holds the
   inputs and the outputs of the original.
2. Show that the Rust code gives the same outputs as the original, with the
   tolerance stated for each output.

A pull request that has no frozen reference and no parity result is not
accepted.

## Licence

Your contribution is licensed under BSD-3-Clause, the licence of this project.
