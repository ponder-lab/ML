import tensorflow as tf


def f(a):
    pass


# `tf.sort(values, ...)` is Tier 6 of wala/ML#449. The XML routes the call
# through `convert_to_tensor` of `values`, so shape and dtype should pass
# through unchanged.
x = tf.constant([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
y = tf.sort(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (6,)
assert y.dtype == tf.float32

f(y)
