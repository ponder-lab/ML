# Test of tf.keras.datasets.imdb.load_data(): four arrays unpacked from a nested tuple.
# x_train/x_test are arrays of variable-length integer sequences (object dtype at runtime);
# y_train/y_test are int64 arrays of binary labels.
import numpy as np
import tensorflow as tf


def f(a, b, c, d):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
assert x_train.shape == (25000,)
assert y_train.shape == (25000,)
assert x_test.shape == (25000,)
assert y_test.shape == (25000,)
# `x_train` and `x_test` are numpy `object` arrays at runtime (each element is a
# variable-length list of integer-encoded tokens, like reuters). The analyzer's `DType`
# lattice represents this as `OBJECT` (wala/ML#488), so the asserts and the JUnit agree.
assert x_train.dtype == np.dtype("object")
assert x_test.dtype == np.dtype("object")
assert y_train.dtype == np.int64
assert y_test.dtype == np.int64
f(x_train, y_train, x_test, y_test)
