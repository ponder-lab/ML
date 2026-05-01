import tensorflow as tf


def add(a, b):
    return a + b


arg1 = tf.ragged.range(3, 18, 3)
assert isinstance(arg1, tf.RaggedTensor)
assert arg1.dtype == tf.int32
assert arg1.shape == (1, None)

arg2 = tf.ragged.range(6, 21, 3)
assert isinstance(arg2, tf.RaggedTensor)
assert arg2.dtype == tf.int32
assert arg2.shape == (1, None)

c = add(arg1, arg2)
