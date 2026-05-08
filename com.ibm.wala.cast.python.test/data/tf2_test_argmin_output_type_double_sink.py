import tensorflow as tf


def f(a):
    pass


# Counterpart of `tf2_test_argmax_output_type_double_sink.py` for `tf.math.argmin`.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert x.shape == (2, 3)
assert x.dtype == tf.float32
y = tf.math.argmin(x, axis=0, output_type=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int32

f(x)
f(y)
