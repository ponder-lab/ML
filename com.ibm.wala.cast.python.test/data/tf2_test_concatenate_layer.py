import tensorflow as tf


def consume_layer(x):
    pass


def consume_layer_axis(x):
    pass


def consume_functional(x):
    pass


def consume_three(x):
    pass


# The layer object under its default axis of -1: the widths sum, the batch axis survives
# (wala/ML#840).
cat = tf.keras.layers.Concatenate()
out = cat([tf.ones((32, 10)), tf.ones((32, 6))])
assert out.shape == (32, 16), out.shape
assert out.dtype == tf.float32, out.dtype
consume_layer(out)

# An explicit axis stored at construction: the batch extents sum, the width survives.
cat0 = tf.keras.layers.Concatenate(axis=0)
out0 = cat0([tf.ones((32, 10)), tf.ones((8, 10))])
assert out0.shape == (40, 10), out0.shape
consume_layer_axis(out0)

# The functional spelling, whose default axis is likewise -1 rather than `tf.concat`'s 0.
f_out = tf.keras.layers.concatenate([tf.ones((32, 10)), tf.ones((32, 6))])
assert f_out.shape == (32, 16), f_out.shape
consume_functional(f_out)

# More than two inputs sum every entry's axis extent.
t_out = tf.keras.layers.Concatenate()(
    [tf.ones((4, 3)), tf.ones((4, 5)), tf.ones((4, 2))]
)
assert t_out.shape == (4, 10), t_out.shape
consume_three(t_out)
