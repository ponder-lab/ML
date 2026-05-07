import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


# Keyword-argument variant of `tf2_test_ceil.py`. Same shape and dtype
# expectation; exercises the keyword arg-resolution path on the
# `PassThroughUnaryTensorGenerator` base.
x = tf.constant([1.0, 2.0, 3.0])
assert x.shape == (3,)
assert x.dtype == tf.float32
y = tf.math.ceil(x=x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y)
g(x)
