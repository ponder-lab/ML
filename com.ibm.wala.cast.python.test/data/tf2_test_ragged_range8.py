import tensorflow as tf


def f(x):
    pass


# wala/ML#546 canonical case: all three scalar args are compile-time literals,
# so the analyzer can pin the inner row length statically: ceil((18 - 3) / 3) = 5.
# TF's runtime .shape API conservatively reports the ragged dim as None; the
# inner row's concrete length is exposed via r[0].shape.
r = tf.ragged.range(3, 18, 3)
assert isinstance(r, tf.RaggedTensor)
assert r.shape == (1, None)
assert r[0].shape == (5,)
assert r.dtype == tf.int32
f(r)
