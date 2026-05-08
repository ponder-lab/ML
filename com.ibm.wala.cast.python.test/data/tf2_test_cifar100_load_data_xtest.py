# Sibling of `tf2_test_cifar100_load_data.py` that passes all four arrays
# (`x_train`, `y_train`, `x_test`, `y_test`) into a single 4-arg sink. Captures the
# wala/ML#495 multi-tensor-sink-collapse pattern on cifar100; the companion JUnit test
# asserts the observed (broken) result with a TODO referencing #495.
import numpy as np
import tensorflow as tf


def f(a, b, c, d):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)
assert x_train.dtype == np.uint8
assert y_train.dtype == np.int64
assert x_test.dtype == np.uint8
assert y_test.dtype == np.int64
f(x_train, y_train, x_test, y_test)
