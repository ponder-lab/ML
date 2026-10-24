import tensorflow as tf


def consume(a):
    pass


def consume2(a):
    pass


def consume_unpacked(a):
    assert a.shape == (2, 3) and a.dtype == tf.float32


def consume_stacked(a):
    assert a.shape == (2, 2, 3) and a.dtype == tf.float32


def consume_concatenated(a):
    assert a.shape == (4, 3) and a.dtype == tf.float32


# `tf.meshgrid(*xi)` returns a list with one tensor per input, each of
# the first input's dtype. The elements are read by constant subscript
# of the returned list rather than by destructuring.
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([10.0, 20.0])
grids = tf.meshgrid(x, y)
assert isinstance(grids, list)
assert len(grids) == 2
assert grids[0].shape == (2, 3)
assert grids[1].shape == (2, 3)
assert grids[0].dtype == tf.float32
assert grids[1].dtype == tf.float32

consume(grids[0])
consume2(grids[1])

# Unpacking reads the elements through the generator; stacking and
# concatenating read the list, whose length is the number of inputs.
X, Y = tf.meshgrid(x, y)
consume_unpacked(X)
consume_stacked(tf.stack(tf.meshgrid(x, y)))
consume_concatenated(tf.concat(tf.meshgrid(x, y), 0))
