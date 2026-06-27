import tensorflow as tf


def check_input(i1):
    pass


# Input(shape, batch_size, name, dtype, sparse, tensor); tensor is the 6th arg (index 5).
# When `tensor` is given it wraps that existing tensor, so the declared `shape` is ignored and the
# result takes the wrapped tensor's shape and dtype. See https://github.com/wala/ML/issues/617.
t = tf.ones((2, 3))
input1 = tf.keras.Input((10,), None, None, None, False, t)
assert input1.shape == (2, 3)
assert input1.dtype == tf.float32

check_input(input1)
