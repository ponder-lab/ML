# https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/reshape

import tensorflow as tf


def f(a):
    pass


t1 = tf.ones([2, 3])
assert isinstance(t1, tf.Tensor)
assert t1.shape == (2, 3)
assert t1.dtype == tf.float32

t2 = tf.reshape(t1, [6])
assert isinstance(t2, tf.Tensor)
assert t2.shape == (6,)
assert t2.dtype == tf.float32

f(t2)
