# Test of tf.keras.datasets.reuters.load_data(). Variable-length integer-encoded sequences:
# x_train (8982,) object, y_train (8982,) int64, x_test (2246,) object, y_test (2246,) int64.
import numpy as np
import tensorflow as tf


def f(a, b):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
assert x_train.shape == (8982,)
assert y_train.shape == (8982,)
assert x_test.shape == (2246,)
assert y_test.shape == (2246,)
# `x_train` and `x_test` are numpy `object` arrays at runtime (each element is a
# variable-length list of integer-encoded tokens). The analyzer's `DType` lattice has no
# `object` representation and reports `UNKNOWN` — see wala/ML#488. The Python assert here
# documents runtime truth; the JUnit expectation captures the analyzer's current imprecise
# answer until #488 lands.
assert x_train.dtype == np.dtype("object")
assert y_train.dtype == np.int64
f(x_train, y_train)
