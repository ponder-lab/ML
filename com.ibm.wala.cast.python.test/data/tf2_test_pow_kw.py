import tensorflow as tf


def f(a):
    pass


# Keyword-argument variant of `tf2_test_pow.py`. Same shape and dtype
# expectation; exercises the keyword arg-resolution path.
x = tf.constant([2.0, 3.0, 4.0])
y = tf.constant([1.0, 2.0, 3.0])
z = tf.math.pow(x=x, y=y)
assert isinstance(z, tf.Tensor)
assert z.shape == (3,)
assert z.dtype == tf.float32
f(z)
