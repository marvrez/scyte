# scyte

Scyte is a neural network framework in C –  it is fast, lightweight and easy to install! The framework is inspired by [TensorFlow](https://www.tensorflow.org/), and is based on computational graphs.

## Building and running scyte

To build scyte execute:
```sh
make -j

```

If no errors popped up, you can run it by executing:
```sh
./scyte
```

## Speeding up the processing

For best performance OpenBLAS is recommended to be installed and used. You can enable it by setting `OPENBLAS=1` in the Makefile.

If you don't want to install OpenBLAS but do have OpenMP installed, you can still speed up the processing by setting `OPENMP=1` in the Makefile.
