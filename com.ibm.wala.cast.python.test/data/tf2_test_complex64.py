import tensorflow as tf


def consume(x):
    pass


# `z` is a `complex64` tensor; its dtype comes from the explicit `dtype=` argument. See wala/ML#637.
z = tf.constant([1, 2], dtype=tf.complex64)
assert z.shape == (2,)
assert z.dtype == tf.complex64
consume(z)
