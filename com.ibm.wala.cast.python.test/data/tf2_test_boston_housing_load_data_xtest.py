# Sibling of `tf2_test_boston_housing_load_data.py` that passes the test pair
# (`x_test`, `y_test`) into the sink alongside the train pair, exercising the
# wala/ML#495 multi-tensor-sink-collapse pattern. With all four arrays flowing
# into a single sink, dataset-loader fallback paths in `TensorGenerator`
# collapse classification on the unrelated train pair too. The companion JUnit
# test captures the currently-observed (broken) result with a TODO referencing
# #495; when the dataset-loader `TypeReference`s are wired into the fallbacks,
# the assertions should tighten to recover precise types on all four params.
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
