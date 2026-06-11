import tensorflow as tf


def consume(y):
    pass


def consume_remaining(z):
    pass


x = tf.constant(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
    dtype=tf.float32,
)
assert x.shape == (3, 4) and x.dtype == tf.float32

# All `size` entries non-negative: the output shape is `size` exactly, independent
# of the input shape. `begin=[0, 1]`, `size=[2, 2]` over `(3, 4)` -> `(2, 2)`.
y = tf.slice(x, [0, 1], [2, 2])
assert y.shape == (2, 2) and y.dtype == tf.float32
consume(y)

# A `size[i]` of `-1` means "all remaining" along axis `i`: `input.shape[i] -
# begin[i]`. `begin=[1, 0]`, `size=[-1, 3]` over `(3, 4)` -> `(3 - 1, 3) = (2, 3)`.
z = tf.slice(x, [1, 0], [-1, 3])
assert z.shape == (2, 3) and z.dtype == tf.float32
consume_remaining(z)
