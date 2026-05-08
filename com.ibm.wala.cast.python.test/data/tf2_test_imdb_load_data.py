# Test of tf.keras.datasets.imdb.load_data(): four arrays unpacked from a nested tuple.
# x_train/x_test are arrays of variable-length integer sequences (object dtype at runtime);
# y_train/y_test are int64 arrays of binary labels.
import numpy as np
import tensorflow as tf


def f(a):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
assert x_train.shape == (25000,)
assert y_train.shape == (25000,)
assert x_test.shape == (25000,)
assert y_test.shape == (25000,)
assert y_train.dtype == np.int64
assert y_test.dtype == np.int64
f(y_train)
