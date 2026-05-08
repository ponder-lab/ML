# Test of tf.keras.datasets.boston_housing.load_data():
# x_train (404, 13) float64, y_train (404,) float64, x_test (102, 13) float64, y_test (102,) float64.
import numpy as np
import tensorflow as tf


def f(a, b, c, d):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
assert x_train.shape == (404, 13)
assert y_train.shape == (404,)
assert x_test.shape == (102, 13)
assert y_test.shape == (102,)
assert x_train.dtype == np.float64
assert y_train.dtype == np.float64
assert x_test.dtype == np.float64
assert y_test.dtype == np.float64
f(x_train, y_train, x_test, y_test)
