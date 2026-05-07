import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.RaggedTensor.from_row_starts([3, 1, 4, 1, 5, 9, 2, 6], [0, 4, 4, 7, 8])
assert arg1.shape == (5, None)
assert arg1.dtype == tf.int32

arg2 = tf.RaggedTensor.from_row_starts([3, 11, 4, 11, 5, 19, 21, 6], [0, 4, 4, 7, 8])
assert arg2.shape == (5, None)
assert arg2.dtype == tf.int32

c = add(arg1, arg2)

assert c.shape == (5, None)
assert c.dtype == tf.int32
