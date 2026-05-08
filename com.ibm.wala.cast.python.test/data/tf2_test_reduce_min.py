import tensorflow as tf


def f(a):
    pass


# `tf.reduce_min(input_tensor, axis=None)`: when axis is None and
# keepdims is False (default), reduces all dims to a scalar. Dtype
# inherits from input. Mirrors `reduce_max`'s semantics.
x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y = tf.reduce_min(x)
assert isinstance(y, tf.Tensor)
assert y.shape == ()
assert y.dtype == tf.float32
f(y)
