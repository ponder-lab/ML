# From: https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/range#for_example

import tensorflow as tf


def f(a):
    pass


start = 3
limit = 18
delta = 3

r = tf.range(start, limit, delta, tf.float32)
assert isinstance(r, tf.Tensor)
assert r.shape == (5,)
assert r.dtype == tf.float32

for i in r:
    assert isinstance(i, tf.Tensor)
    assert (
        i.dtype == tf.float32
    )  # NOTE: This is getting cast here from the original input.
    assert i.shape == ()
    f(i)
