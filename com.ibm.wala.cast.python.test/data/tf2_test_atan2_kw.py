# Keyword-argument variant of `tf2_test_atan2.py`. Same shape and dtype expectations;
# exercises the kw-arg-resolution path on `tf.math.atan2(y=..., x=...)`. Per TF's API,
# `atan2`'s first arg is conventionally `y` and the second is `x` (computes arctan(y/x)).
import tensorflow as tf


def f(a):
    pass


x = tf.constant([0.0, 0.5, 1.0])
y_in = tf.constant([1.0, 1.0, 1.0])
result = tf.math.atan2(y=x, x=y_in)
assert isinstance(result, tf.Tensor)
assert result.shape == (3,)
assert result.dtype == tf.float32
f(result)
