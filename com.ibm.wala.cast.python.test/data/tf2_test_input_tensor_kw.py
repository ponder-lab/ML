import tensorflow as tf


def check_input(i1):
    pass


# `tensor` wraps an existing tensor: the result has that tensor's shape and dtype verbatim, with no
# batch dimension prepended. See https://github.com/wala/ML/issues/617.
t = tf.ones((2, 3))
input1 = tf.keras.Input(tensor=t)
assert input1.shape == (2, 3)
assert input1.dtype == tf.float32

check_input(input1)
