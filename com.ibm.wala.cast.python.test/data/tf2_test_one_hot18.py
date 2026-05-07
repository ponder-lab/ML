import tensorflow as tf


def f(a):
    pass


arg1 = [[10, 20, 30], [40, 50, 60]]  # Row 1  # Row 2

assert isinstance(arg1, list)
assert all(isinstance(row, list) for row in arg1)
assert all(isinstance(elem, int) for row in arg1 for elem in row)
assert len(arg1) == 2
assert all(len(row) == 3 for row in arg1)

arg2 = tf.one_hot(arg1, 5, None, None, 1)
assert isinstance(arg2, tf.Tensor)
assert arg2.dtype == tf.float32
assert arg2.shape == (2, 5, 3)

f(arg2)
