# Keyword-argument variant of `tf2_test_minimum.py`. Same shape and dtype; exercises the
# kw-arg-resolution path on `tf.math.minimum(x=..., y=...)`.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y_in = tf.constant([3.0, 2.0, 1.0])
result = tf.math.minimum(x=x, y=y_in)
assert isinstance(result, tf.Tensor)
assert result.shape == (3,)
assert result.dtype == tf.float32
f(result)
