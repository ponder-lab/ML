# Exercises tf.range's dtype= keyword argument (wala/ML#492).
# tf.range honors an explicit dtype override; the analyzer should infer
# float32 here, not the int32 default.

import tensorflow as tf


def f(a):
    pass


r = tf.range(0, 5, dtype=tf.float32)
assert isinstance(r, tf.Tensor)
assert r.shape == (5,)
assert r.dtype == tf.float32
f(r)
