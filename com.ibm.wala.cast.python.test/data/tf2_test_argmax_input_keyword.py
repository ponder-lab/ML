import tensorflow as tf


def f(x, y):
    pass


# Counterpart of `tf2_test_argmax_output_type_positional.py` that passes the
# input tensor by KEYWORD (`tf.math.argmax(input=x, axis=0)`). argmax's input
# parameter is named `input`, not `reduce_mean`'s `input_tensor`, so resolving
# it by the superclass name would fail keyword-argument resolution and throw
# `IllegalStateException` during shape inference.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert x.shape == (2, 3)
assert x.dtype == tf.float32
y = tf.math.argmax(input=x, axis=0)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int64

f(x, y)
