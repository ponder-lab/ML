# Sibling of `tf2_test_fashion_mnist_load_data.py` that passes all four arrays
# (`x_train`, `y_train`, `x_test`, `y_test`) into a single 4-arg sink. Captures the
# wala/ML#495 multi-tensor-sink-collapse pattern on fashion_mnist; the companion JUnit
# test asserts the observed (broken) result with a TODO referencing #495.
import numpy as np
import tensorflow as tf


def f(a, b, c, d):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert y_train.shape == (60000,)
assert x_test.shape == (10000, 28, 28)
assert y_test.shape == (10000,)
assert x_train.dtype == np.uint8
assert y_train.dtype == np.uint8
assert x_test.dtype == np.uint8
assert y_test.dtype == np.uint8
f(x_train, y_train, x_test, y_test)
