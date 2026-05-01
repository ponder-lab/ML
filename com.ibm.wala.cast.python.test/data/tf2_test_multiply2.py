# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/multiply#for_example/

import tensorflow as tf


def f(a):
    pass


arg = tf.math.multiply(7, 6)
assert arg.shape == ()
assert arg.dtype == tf.int32

f(arg)
