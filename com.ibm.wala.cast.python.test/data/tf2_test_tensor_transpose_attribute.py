# Test https://github.com/wala/ML/issues/980: under numpy behavior a `tf.Tensor` has a `.T`
# attribute, which returns a `Tensor`, so its result must read the TensorFlow origin, while an
# ndarray's `.T` stays numpy.
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def consume(x):
    pass


t = tf.ones((2, 3)).T
assert isinstance(t, tf.Tensor) and t.shape == (3, 2)
consume(t)

a = np.ones((2, 3)).T
assert isinstance(a, np.ndarray) and a.shape == (3, 2)
consume(a)
