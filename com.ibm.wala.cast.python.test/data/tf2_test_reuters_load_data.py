# Test of tf.keras.datasets.reuters.load_data(). Variable-length integer-encoded sequences:
# x_train (8982,) object, y_train (8982,) int64, x_test (2246,) object, y_test (2246,) int64.
import numpy as np
import tensorflow as tf


def f(a):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
assert x_train.shape == (8982,)
assert y_train.shape == (8982,)
assert x_test.shape == (2246,)
assert y_test.shape == (2246,)
assert y_train.dtype == np.int64
f(y_train)
