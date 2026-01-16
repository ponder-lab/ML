import tensorflow as tf


def f(x):
    pass


r = tf.ragged.range(10)
assert isinstance(r, tf.RaggedTensor)
assert r.shape == (1, None)
assert r.dtype == tf.int32
f(r)
