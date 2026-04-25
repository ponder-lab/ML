# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/multiply#for_example/

import tensorflow as tf


def f(a):
    pass


x = tf.constant(([1, 2, 3, 4]))
z = tf.math.multiply(x, x)
assert z.shape.as_list() == [4]
assert z.dtype == tf.int32
f(z)
