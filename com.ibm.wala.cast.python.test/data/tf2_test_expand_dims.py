import tensorflow as tf


def f(a):
    pass


# `tf.expand_dims(input, axis)`: dtype inherits from `input`; shape is
# `input.shape` with a length-1 dim inserted at `axis`. The static
# analysis currently emits ⊤ shape (insertion-at-axis composition is a
# follow-up); dtype passthrough is sound and yields float32.
x = tf.constant([1.0, 2.0, 3.0])
y = tf.expand_dims(x, axis=0)
assert isinstance(y, tf.Tensor)
assert y.shape == (1, 3)
assert y.dtype == tf.float32
f(y)
