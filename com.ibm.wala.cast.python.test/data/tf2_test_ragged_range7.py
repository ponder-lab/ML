import tensorflow as tf


def f(x):
    pass


r = tf.ragged.range([0, 5, 8], [3, 3, 12], 2)
assert isinstance(r, tf.RaggedTensor)
assert r.shape == (3, None)
assert r.dtype == tf.int32

f(r)
