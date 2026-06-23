import tensorflow as tf


def f(a):
    pass


# `images` is a Python list literal (not a `tf.Tensor`); the result is still recognized as a
# tensor (wala/ML#584), and its `(1, 1, 1, 1)` shape is recovered from the literal's nesting.
# A 3x3 patch does not fit the 1x1 image, so VALID padding yields a 0-extent spatial output
# (depth 3*3*1 = 9): shape `(1, 0, 0, 9)` (wala/ML#585).
r = tf.image.extract_patches(
    images=[[[[1]]]],
    sizes=[1, 3, 3, 1],
    strides=[1, 5, 5, 1],
    rates=[1, 1, 1, 1],
    padding="VALID",
)
assert r.dtype == tf.int32
assert r.shape == (1, 0, 0, 9)
f(r)
