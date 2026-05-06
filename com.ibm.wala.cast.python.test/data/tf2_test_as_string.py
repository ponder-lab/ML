# Test of tf.strings.as_string: shape passthrough on `input`, output dtype always string.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0])
y = tf.strings.as_string(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.string
f(y)
