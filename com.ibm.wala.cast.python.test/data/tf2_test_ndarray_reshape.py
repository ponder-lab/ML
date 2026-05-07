import tensorflow as tf
import numpy as np


def consume(tensor):
    pass


# `x_train` is a (60000, 28, 28) uint8 ndarray from mnist. Method-style `reshape`
# on it goes through `numpy.ndarray.reshape.do()`, whose return is a synthetic-method
# summary with an implicit PointerKey. NdarrayReshape's DU-walk receiver lookup is
# what recovers the `(60000, 28, 28)` input shape to resolve the `-1` in `[-1, 784]`.
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_train.dtype == np.uint8

y = x_train.reshape([-1, 784])
assert y.shape == (60000, 784)
assert y.dtype == np.uint8

consume(y)
