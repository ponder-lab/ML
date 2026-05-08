# Sibling of `tf2_test_reuters_load_data.py` that passes all four arrays into a single
# 4-arg sink. Captures the wala/ML#495 multi-tensor-sink-collapse pattern on reuters;
# the companion JUnit test asserts the observed (broken) result with a TODO
# referencing #495.
import numpy as np
import tensorflow as tf


def f(a, b, c, d):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
assert x_train.shape == (8982,)
assert y_train.shape == (8982,)
assert x_test.shape == (2246,)
assert y_test.shape == (2246,)
assert x_train.dtype == np.dtype("object")
assert y_train.dtype == np.int64
f(x_train, y_train, x_test, y_test)
