# Test of tf.keras.datasets.cifar100.load_data(). Shapes match cifar10's:
# x_train (50000, 32, 32, 3), y_train (50000, 1), x_test (10000, 32, 32, 3), y_test (10000, 1).
# Image arrays carry uint8 (matching cifar10). LABEL arrays carry int64 — a divergence from
# cifar10's uint8 labels that the dedicated `Cifar100InputData` generator now models precisely
# (closes wala/ML#487).
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
assert x_test.dtype == np.uint8
# `cifar100`'s `y_train` / `y_test` are `int64` at runtime (vs `uint8` for `cifar10`); see
# wala/ML#487. The dedicated `Cifar100InputData` generator now reports `int64` precisely.
assert y_train.dtype == np.int64
assert y_test.dtype == np.int64
f(x_train, y_train, x_test, y_test)
