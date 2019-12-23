# scyte

Scyte is a neural network framework in C –  it is fast, lightweight and easy to install! The framework is inspired by [TensorFlow](https://www.tensorflow.org/), and is based on computational graphs and [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

## Installation 

To clone and build scyte execute:
```sh
git clone git@github.com:marvrez/scyte.git
cd scyte
make -j
```

## Examples

If no errors popped up during the installation you can run the provided examples.
For instance, if you want to train or run inference a simple model that learns to act as an [XOR-gate](https://en.wikipedia.org/wiki/XOR_gate), you can simply do the following.
```sh
# train XOR-model for 1000 epochs, and save model weights to 'xor.weight'
./scyte xor -e 1000 xor.weight

# test the newly trained XOR-model
./scyte xor -p xor.weight
```

Similarly, you can also train and/or run inference on a model that learns the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The simplest way to do this is to download and extract the data
from [here](https://pjreddie.com/media/files/mnist.tar.gz). You can then do the following steps to train and/or run inference on the model.

```sh
# train mnist model for 100 epochs, and save model weights to 'mnist.weight'
./scyte mnist -e 100 mnist.weight --label_path ./mnist/mnist.labels --data_path ./mnist/mnist.train

# test the newly trained XOR-model
./scyte mnist -p mnist.weight -i <test_image>
```

In general, the `--label_path` file is a new-line seperated list of the labels, and `--data_path` file contains a newline-separated list of the image paths relative to label file.

For more options and help, pass `-h` or the `--help` flag.

## Speeding up the processing

For best performance OpenBLAS is recommended to be installed and used. You can enable it by setting `OPENBLAS=1` in the Makefile.

If you don't want to install OpenBLAS but do have OpenMP installed, you can still speed up the processing by setting `OPENMP=1` in the Makefile.
