# Test of tf.keras.datasets.cifar100.load_data(). Shapes match cifar10's:
# x_train (50000, 32, 32, 3), y_train (50000, 1), x_test (10000, 32, 32, 3), y_test (10000, 1).
# Image arrays carry uint8 (matching cifar10). LABEL arrays carry int64 — a divergence from
# cifar10's uint8 labels that the analyzer's reuse of `Cifar10InputData` doesn't yet capture
# (see wala/ML#487).
import numpy as np
import tensorflow as tf


def f(a, b):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert x_test.shape == (10000, 32, 32, 3)
assert y_test.shape == (10000, 1)
assert x_train.dtype == np.uint8
# `cifar100`'s `y_train` / `y_test` are `int64` at runtime (vs `uint8` for `cifar10`). The
# analyzer reuses `Cifar10InputData` and currently reports `uint8` — see wala/ML#487. The
# Python assert here documents runtime truth; the JUnit expectation captures the analyzer's
# current imprecise answer until #487 lands.
assert y_train.dtype == np.int64
f(x_train, y_train)
