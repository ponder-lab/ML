import tensorflow as tf


def f(a):
    pass


# `images` is a Python list literal (not a `tf.Tensor`); the result should still be
# recognized as a tensor (wala/ML#584).
r = tf.image.extract_patches(
    images=[[[[1]]]],
    sizes=[1, 3, 3, 1],
    strides=[1, 5, 5, 1],
    rates=[1, 1, 1, 1],
    padding="VALID",
)
assert r.dtype == tf.int32
f(r)
