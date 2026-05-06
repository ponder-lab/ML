import tensorflow as tf


def f(a):
    pass


# `tf.math.pow(x, y)` is element-wise `x ** y`. Output shape is the
# broadcast of `x` and `y`; dtype is the unified dtype. Routed through
# `ElementWiseOperation`.
x = tf.constant([2.0, 3.0, 4.0])
y = tf.constant([1.0, 2.0, 3.0])
z = tf.math.pow(x, y)
assert isinstance(z, tf.Tensor)
assert z.shape == (3,)
assert z.dtype == tf.float32
f(z)
